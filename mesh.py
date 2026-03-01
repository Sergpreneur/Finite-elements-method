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
    b = np.ascontiguousarray(np.round(a, 10))
    _, idx = np.unique(b.view([('', b.dtype)] * b.shape[1]), return_index=True)
    return a[np.sort(idx)]


def generate_beam_hole_mesh(
    L=200.0, h=40.0, a=8.0, xc=40.0,
    hx=4.0, hy=4.0,
    n_theta=64,
    n_rings=5,
    ring_growth=1.4,
    seed=0
) -> Tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    # --- 1) Base grid points in rectangle ---
    xs = np.arange(0.0, L + 1e-12, hx)
    ys = np.arange(-h / 2, h / 2 + 1e-12, hy)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # --- 2) Add rectangle boundary points (clean, no jitter) ---
    n_y_bnd = int(max(5, round(h / min(hx, hy)))) + 1
    n_x_bnd = int(max(5, round(L / min(hx, hy)))) + 1
    yb = np.linspace(-h / 2, h / 2, n_y_bnd)
    xb = np.linspace(0.0, L, n_x_bnd)

    left  = np.column_stack([np.zeros_like(yb), yb])
    right = np.column_stack([np.full_like(yb, L), yb])
    top   = np.column_stack([xb, np.full_like(xb,  h / 2)])
    bot   = np.column_stack([xb, np.full_like(xb, -h / 2)])
    pts = np.vstack([pts, left, right, top, bot])

    # --- 3) Hole boundary + refinement rings (denser spacing) ---
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    hole0 = np.column_stack([xc + a * np.cos(theta), a * np.sin(theta)])
    rings = [hole0]

    r_ring = a
    for k in range(n_rings):
        r_ring *= ring_growth
        n_t_k = max(n_theta, int(round(2 * np.pi * r_ring / hx)))
        theta_k = np.linspace(0.0, 2 * np.pi, n_t_k, endpoint=False)
        ring = np.column_stack([xc + r_ring * np.cos(theta_k),
                                 r_ring * np.sin(theta_k)])
        in_rect = ((ring[:, 0] >= 0.0) & (ring[:, 0] <= L) &
                   (ring[:, 1] >= -h / 2) & (ring[:, 1] <= h / 2))
        rings.append(ring[in_rect])

    pts = np.vstack([pts] + rings)

    dx = pts[:, 0] - xc
    dy = pts[:, 1]
    inside_hole = (dx * dx + dy * dy) < (a * 0.97) ** 2
    pts = pts[~inside_hole]

    eps_c = 1e-9
    in_rect = ((pts[:, 0] >= -eps_c) & (pts[:, 0] <= L + eps_c) &
               (pts[:, 1] >= -h / 2 - eps_c) & (pts[:, 1] <= h / 2 + eps_c))
    pts = pts[in_rect]

    pts = _unique_rows(pts)

    tri = Delaunay(pts)
    tris = tri.simplices.copy()

    cent = pts[tris].mean(axis=1)
    cx = cent[:, 0]; cy = cent[:, 1]
    in_rect_t  = (cx >= 0.0) & (cx <= L) & (cy >= -h / 2) & (cy <= h / 2)
    out_hole_t = ((cx - xc) ** 2 + cy ** 2) >= (a * 1.001) ** 2
    tris = tris[in_rect_t & out_hole_t]

    v1 = pts[tris[:, 1]] - pts[tris[:, 0]]
    v2 = pts[tris[:, 2]] - pts[tris[:, 0]]
    area = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) * 0.5
    tris = tris[area > 1e-14]

    return pts, tris


def extract_boundary_edges(tris: np.ndarray) -> np.ndarray:
    e01 = tris[:, [0, 1]]
    e12 = tris[:, [1, 2]]
    e20 = tris[:, [2, 0]]
    edges = np.sort(np.vstack([e01, e12, e20]), axis=1)

    b = np.ascontiguousarray(edges)
    v = b.view([('', b.dtype)] * b.shape[1])
    uniq, counts = np.unique(v, return_counts=True)
    boundary = uniq[counts == 1].view(b.dtype).reshape(-1, 2)
    return boundary


def classify_edges_left_right(nodes: np.ndarray, edges: np.ndarray,
                               L: float, tol: float = 1e-6):

    x1 = nodes[edges[:, 0], 0]
    x2 = nodes[edges[:, 1], 0]

    left  = (np.abs(x1 - 0.0) < tol) & (np.abs(x2 - 0.0) < tol)
    right = (np.abs(x1 - L)   < tol) & (np.abs(x2 - L)   < tol)

    phys = np.full(edges.shape[0], 99, dtype=int)
    phys[left]  = 1
    phys[right] = 2
    return phys


def write_msh2_ascii(path: str,
                     nodes: np.ndarray,
                     tris: np.ndarray,
                     edges: np.ndarray,
                     edge_phys: np.ndarray,
                     phys_names: Dict[int, str]) -> None:

    phys_names = dict(phys_names)
    if 3 not in phys_names:  phys_names[3]  = "DOMAIN"
    if 99 not in phys_names: phys_names[99] = "OTHER"

    with open(path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        phys_items = [(2 if tag == 3 else 1, tag, name)
                      for tag, name in phys_names.items()]
        f.write("$PhysicalNames\n")
        f.write(f"{len(phys_items)}\n")
        for dim, tag, name in phys_items:
            f.write(f'{dim} {tag} "{name}"\n')
        f.write("$EndPhysicalNames\n")

        f.write("$Nodes\n")
        f.write(f"{nodes.shape[0]}\n")
        for i, (x, y) in enumerate(nodes, start=1):
            f.write(f"{i} {x:.12g} {y:.12g} 0\n")
        f.write("$EndNodes\n")

        ne = edges.shape[0] + tris.shape[0]
        f.write("$Elements\n")
        f.write(f"{ne}\n")
        eid = 1
        for k in range(edges.shape[0]):
            n1, n2 = edges[k] + 1
            ph = int(edge_phys[k])
            f.write(f"{eid} 1 2 {ph} {ph} {n1} {n2}\n")
            eid += 1
        for k in range(tris.shape[0]):
            n1, n2, n3 = tris[k] + 1
            f.write(f"{eid} 2 2 3 3 {n1} {n2} {n3}\n")
            eid += 1
        f.write("$EndElements\n")


def main():
    L, h, a, xc = 200.0, 40.0, 8.0, 40.0
    hx = hy = 4.0
    nodes, tris = generate_beam_hole_mesh(
        L=L, h=h, a=a, xc=xc,
        hx=hx, hy=hy, n_theta=80, n_rings=5, ring_growth=1.4, seed=0)

    edges    = extract_boundary_edges(tris)
    edge_phys = classify_edges_left_right(nodes, edges, L=L, tol=1e-6)
    phys_names = {1: "LEFT", 2: "RIGHT", 3: "DOMAIN", 99: "OTHER"}

    out = "beam_hole.msh"
    write_msh2_ascii(out, nodes, tris, edges, edge_phys, phys_names)
    print(f"[OK] Wrote {out}")
    print(f"Nodes: {nodes.shape[0]}, Tris: {tris.shape[0]}, Boundary edges: {edges.shape[0]}")
    print(f"LEFT: {np.sum(edge_phys==1)}, RIGHT: {np.sum(edge_phys==2)}, OTHER: {np.sum(edge_phys==99)}")


if __name__ == "__main__":
    main()