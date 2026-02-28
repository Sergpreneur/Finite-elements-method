import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.spatial import Delaunay


@dataclass
class GenMesh:
    nodes: np.ndarray        # (N,2)
    tris: np.ndarray         # (M,3) int (0-based)
    edges: np.ndarray        # (K,2) int (0-based)
    edge_phys: np.ndarray    # (K,) int physical tag for each edge
    phys_names: Dict[int, str]


def _unique_rows(a: np.ndarray) -> np.ndarray:
    """Unique rows preserving approximate floats by rounding."""
    b = np.ascontiguousarray(np.round(a, 12))
    _, idx = np.unique(b.view([('', b.dtype)] * b.shape[1]), return_index=True)
    return a[np.sort(idx)]


def generate_beam_hole_mesh(
    L=200.0, h=40.0, a=8.0, xc=40.0,
    hx=4.0, hy=4.0,
    n_theta=64,
    n_rings=3,
    ring_growth=1.7,
    seed=0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a triangular mesh of a rectangle [0,L]x[-h/2,h/2] with a circular hole.
    Returns (nodes, tris) with 0-based indices.

    hx, hy control base point spacing.
    Refinement near the hole: n_theta points around circle and n_rings additional rings.
    """
    rng = np.random.default_rng(seed)

    # --- 1) Base grid points in rectangle ---
    xs = np.arange(0.0, L + 1e-12, hx)
    ys = np.arange(-h/2, h/2 + 1e-12, hy)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # Add slight jitter (helps Delaunay avoid degeneracy), but keep boundary points later explicitly
    #jitter = 0.05 * min(hx, hy)
    #pts = pts + rng.uniform(-jitter, jitter, size=pts.shape)

    # --- 2) Add rectangle boundary points (clean, no jitter) ---
    # Left & Right edges
    yb = np.linspace(-h/2, h/2, int(max(5, round(h / min(hx, hy))) ) + 1)
    left = np.column_stack([np.zeros_like(yb), yb])
    right = np.column_stack([np.full_like(yb, L), yb])

    # Top & Bottom edges
    xb = np.linspace(0.0, L, int(max(5, round(L / min(hx, hy))) ) + 1)
    top = np.column_stack([xb, np.full_like(xb, h/2)])
    bot = np.column_stack([xb, np.full_like(xb, -h/2)])

    pts = np.vstack([pts, left, right, top, bot])

    # --- 3) Add hole boundary points + refinement rings ---
    theta = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    # exact circle boundary
    hole0 = np.column_stack([xc + a*np.cos(theta), 0.0 + a*np.sin(theta)])

    rings = [hole0]
    r = a
    for k in range(n_rings):
        r = r * ring_growth
        ring = np.column_stack([xc + r*np.cos(theta), 0.0 + r*np.sin(theta)])
        rings.append(ring)

    pts = np.vstack([pts] + rings)

    # --- 4) Remove points inside the hole (strictly) ---
    dx = pts[:, 0] - xc
    dy = pts[:, 1] - 0.0
    inside_hole = (dx*dx + dy*dy) < (a * 0.98)**2  # 0.98 so we keep boundary points
    pts = pts[~inside_hole]

    # --- 5) Clip points to rectangle (safety) ---
    eps = 1e-9
    in_rect = (pts[:, 0] >= -eps) & (pts[:, 0] <= L + eps) & (pts[:, 1] >= -h/2 - eps) & (pts[:, 1] <= h/2 + eps)
    pts = pts[in_rect]

    # --- 6) Deduplicate ---
    pts = _unique_rows(pts)

    # --- 7) Delaunay triangulation ---
    tri = Delaunay(pts)
    tris = tri.simplices.copy()  # (M,3)

    # --- 8) Filter triangles: keep only those inside rectangle and outside hole ---
    # centroid test is good enough for this geometry
    cent = pts[tris].mean(axis=1)
    cx = cent[:, 0]; cy = cent[:, 1]

    in_rect_t = (cx >= 0.0) & (cx <= L) & (cy >= -h/2) & (cy <= h/2)
    out_hole_t = ((cx - xc)**2 + (cy - 0.0)**2) >= (a * 1.001)**2
    keep = in_rect_t & out_hole_t
    tris = tris[keep]

    return pts, tris


def extract_boundary_edges(tris: np.ndarray) -> np.ndarray:
    """
    Boundary edges are those that belong to only 1 triangle.
    Returns edges as (K,2) node indices, sorted (min,max).
    """
    # all edges from tris
    e01 = tris[:, [0, 1]]
    e12 = tris[:, [1, 2]]
    e20 = tris[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])

    # normalize (min,max) for counting
    edges = np.sort(edges, axis=1)

    # count occurrences
    # view trick for unique rows
    b = np.ascontiguousarray(edges)
    v = b.view([('', b.dtype)] * b.shape[1])
    uniq, counts = np.unique(v, return_counts=True)

    boundary = uniq[counts == 1].view(b.dtype).reshape(-1, 2)
    return boundary


def classify_edges_left_right(nodes: np.ndarray, edges: np.ndarray, L: float, tol: float = 1e-6):
    """
    Classify boundary edges into LEFT / RIGHT / OTHER based on x coordinate of both endpoints.
    Returns edge_phys (K,) with tags: 1=LEFT, 2=RIGHT, 99=OTHER
    """
    x1 = nodes[edges[:, 0], 0]
    x2 = nodes[edges[:, 1], 0]

    left = (np.abs(x1 - 0.0) < tol) & (np.abs(x2 - 0.0) < tol)
    right = (np.abs(x1 - L) < tol) & (np.abs(x2 - L) < tol)

    phys = np.full(edges.shape[0], 99, dtype=int)
    phys[left] = 1
    phys[right] = 2
    return phys


def write_msh2_ascii(path: str,
                    nodes: np.ndarray,
                    tris: np.ndarray,
                    edges: np.ndarray,
                    edge_phys: np.ndarray,
                    phys_names: Dict[int, str]) -> None:
    """
    Writes a minimal Gmsh .msh v2.2 ASCII file with:
      - PhysicalNames
      - Nodes
      - Elements: line2 + tri3
    Physical tags:
      1: LEFT, 2: RIGHT, 3: DOMAIN
      99: OTHER boundary (optional)
    """
    # Ensure phys names include DOMAIN
    phys_names = dict(phys_names)
    if 3 not in phys_names:
        phys_names[3] = "DOMAIN"
    if 99 not in phys_names:
        phys_names[99] = "OTHER"

    with open(path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        # Physical names
        # dim 1 for curves, dim 2 for surface
        phys_items = []
        for tag, name in phys_names.items():
            dim = 2 if tag == 3 else 1
            phys_items.append((dim, tag, name))
        f.write("$PhysicalNames\n")
        f.write(f"{len(phys_items)}\n")
        for dim, tag, name in phys_items:
            f.write(f'{dim} {tag} "{name}"\n')
        f.write("$EndPhysicalNames\n")

        # Nodes (1-based ids)
        f.write("$Nodes\n")
        f.write(f"{nodes.shape[0]}\n")
        for i, (x, y) in enumerate(nodes, start=1):
            f.write(f"{i} {x:.12g} {y:.12g} 0\n")
        f.write("$EndNodes\n")

        # Elements
        # Gmsh v2 elements line2 type=1, tri3 type=2
        # Format: elm-number elm-type number-of-tags <tags...> node-number-list
        # We'll use 2 tags: (physical, elementary)
        ne = edges.shape[0] + tris.shape[0]
        f.write("$Elements\n")
        f.write(f"{ne}\n")

        eid = 1
        # lines first
        for k in range(edges.shape[0]):
            n1, n2 = edges[k] + 1
            phys = int(edge_phys[k])
            # elementary tag = phys (fine for our needs)
            f.write(f"{eid} 1 2 {phys} {phys} {n1} {n2}\n")
            eid += 1

        # triangles
        for k in range(tris.shape[0]):
            n1, n2, n3 = tris[k] + 1
            phys = 3  # DOMAIN
            f.write(f"{eid} 2 2 {phys} {phys} {n1} {n2} {n3}\n")
            eid += 1

        f.write("$EndElements\n")


def main():
    # ---- Parameters (mm) ----
    L = 200.0
    h = 40.0
    a = 8.0
    xc = 40.0

    # Base spacing (coarser -> faster)
    hx = 4.0
    hy = 4.0

    # Refinement around hole
    n_theta = 80
    n_rings = 3
    ring_growth = 1.6

    nodes, tris = generate_beam_hole_mesh(
        L=L, h=h, a=a, xc=xc,
        hx=hx, hy=hy,
        n_theta=n_theta,
        n_rings=n_rings,
        ring_growth=ring_growth,
        seed=0
    )

    edges = extract_boundary_edges(tris)
    edge_phys = classify_edges_left_right(nodes, edges, L=L, tol=1e-6)

    phys_names = {
        1: "LEFT",
        2: "RIGHT",
        3: "DOMAIN",
        99: "OTHER"
    }

    out = "beam_hole.msh"
    write_msh2_ascii(out, nodes, tris, edges, edge_phys, phys_names)

    # Quick stats
    print(f"[OK] Wrote {out}")
    print(f"Nodes: {nodes.shape[0]}, Tris: {tris.shape[0]}, Boundary edges: {edges.shape[0]}")
    print(f"LEFT edges: {np.sum(edge_phys==1)}, RIGHT edges: {np.sum(edge_phys==2)}, OTHER: {np.sum(edge_phys==99)}")


if __name__ == "__main__":
    main()
