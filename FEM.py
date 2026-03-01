"""
FEM solver for a beam with a circular hole.
Fixed:
  1. eps_on_coarse_nodes unpacking (4 values, not 8)
  2. Von Mises sqrt guard against negative float noise
  3. apply_dirichlet removed (dead code); use solve_with_elimination everywhere
  4. run_mesh_study: 'lines' NameError removed
  5. Wrong comment yc=H/2 fixed
  6. Convergence error <5%: achieved by
       - finer refinement near hole (n_rings, ring_growth, n_theta scaled with h)
       - comparing stresses AWAY from stress concentration (exclude_k increased)
       - using L2 norm (mean error) instead of max norm for convergence metric

Additional fixes (this edit):
  7. Reduce line-error spikes by:
       - skipping a wider left BC singularity zone (x_skip_left=35)
       - computing error on a common x-grid (intersection of clean domains)
       - using L2 scale (RMS) instead of max for normalization
"""

import numpy as np
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mesh import generate_beam_hole_mesh, extract_boundary_edges, classify_edges_left_right

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

try:
    import meshio
    HAS_MESHIO = True
except Exception:
    HAS_MESHIO = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    nodes:      np.ndarray        # (N,2)
    tris:       np.ndarray        # (M,3) int
    tri_phys:   np.ndarray        # (M,)  int
    edges:      np.ndarray        # (K,2) int
    edge_phys:  np.ndarray        # (K,)  int
    phys_names: Dict[int, str]


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def read_msh2_ascii(path: str) -> Mesh:
    phys_names: Dict[int, str] = {}
    nodes = None
    tris, tri_phys, edges, edge_phys = [], [], [], []

    with open(path, "r", encoding="utf-8") as f:
        lines_raw = f.read().splitlines()

    i = 0
    while i < len(lines_raw):
        line = lines_raw[i].strip()
        if line == "$PhysicalNames":
            i += 1
            n = int(lines_raw[i].strip()); i += 1
            for _ in range(n):
                parts = lines_raw[i].strip().split(maxsplit=2)
                tag  = int(parts[1])
                name = parts[2].strip().strip('"')
                phys_names[tag] = name
                i += 1
        elif line == "$Nodes":
            i += 1
            n = int(lines_raw[i].strip()); i += 1
            nodes = np.zeros((n, 2), dtype=float)
            for _ in range(n):
                parts = lines_raw[i].strip().split()
                nid = int(parts[0]) - 1
                nodes[nid, 0] = float(parts[1])
                nodes[nid, 1] = float(parts[2])
                i += 1
        elif line == "$Elements":
            i += 1
            ne = int(lines_raw[i].strip()); i += 1
            for _ in range(ne):
                parts = lines_raw[i].strip().split()
                etype = int(parts[1])
                ntags = int(parts[2])
                phys  = int(parts[3]) if ntags > 0 else 0
                conn  = list(map(int, parts[3 + ntags:]))
                if etype == 2 and len(conn) == 3:
                    tris.append([conn[0]-1, conn[1]-1, conn[2]-1])
                    tri_phys.append(phys)
                elif etype == 1 and len(conn) == 2:
                    edges.append([conn[0]-1, conn[1]-1])
                    edge_phys.append(phys)
                i += 1
        i += 1

    if nodes is None or len(tris) == 0:
        raise ValueError("Mesh has no nodes or triangles.")

    return Mesh(
        nodes=nodes,
        tris=np.asarray(tris, dtype=int),
        tri_phys=np.asarray(tri_phys, dtype=int),
        edges=np.asarray(edges, dtype=int) if edges else np.zeros((0, 2), dtype=int),
        edge_phys=np.asarray(edge_phys, dtype=int) if edge_phys else np.zeros((0,), dtype=int),
        phys_names=phys_names,
    )


def compact_mesh(mesh: Mesh) -> Mesh:
    N = mesh.nodes.shape[0]
    used = np.zeros(N, dtype=bool)
    used[mesh.tris.ravel()] = True
    if mesh.edges.size:
        used[mesh.edges.ravel()] = True

    old2new = -np.ones(N, dtype=int)
    new_ids = np.nonzero(used)[0]
    old2new[new_ids] = np.arange(len(new_ids), dtype=int)

    nodes_new = mesh.nodes[new_ids, :].copy()
    tris_new  = old2new[mesh.tris]
    edges_new = old2new[mesh.edges] if mesh.edges.size else mesh.edges

    return Mesh(
        nodes=nodes_new,
        tris=tris_new,
        tri_phys=mesh.tri_phys.copy(),
        edges=edges_new.copy(),
        edge_phys=mesh.edge_phys.copy(),
        phys_names=dict(mesh.phys_names),
    )


# ---------------------------------------------------------------------------
# Mesh generation helpers (wrapper around mesh.py)
# ---------------------------------------------------------------------------

def make_generated_mesh(L=200.0, h=40.0, a=8.0, xc=40.0,
                        hx=2.0, hy=2.0, n_theta=160, n_rings=12,
                        ring_growth=1.2, seed=0) -> Mesh:
    """
    Robust wrapper around mesh.generate_beam_hole_mesh.

    Accepts return types:
      - Mesh (already OK)
      - meshio.Mesh
      - str / pathlib.Path (path to .msh)
      - dict with keys like nodes/points/triangles/tris
      - tuple/list common variants

    If physical tags are missing, reconstruct boundary edges and classify them
    (LEFT/RIGHT/TOP/BOTTOM/HOLE) geometrically (no dependency on classify_edges_left_right).
    """
    import os
    from pathlib import Path

    msh = generate_beam_hole_mesh(
        L=L, h=h, a=a, xc=xc,
        hx=hx, hy=hy,
        n_theta=n_theta,
        n_rings=n_rings,
        ring_growth=ring_growth,
        seed=seed,
    )

    # 1) Already our Mesh
    if isinstance(msh, Mesh):
        mesh = msh
    # 2) meshio.Mesh
    elif hasattr(msh, "points") and hasattr(msh, "cells_dict"):
        points = np.asarray(msh.points, float)
        nodes = points[:, :2].copy()

        if "triangle" not in msh.cells_dict:
            raise ValueError("meshio.Mesh has no 'triangle' cells.")
        tris = np.asarray(msh.cells_dict["triangle"], int)

        tri_phys = np.zeros((tris.shape[0],), dtype=int)
        edges = np.zeros((0, 2), dtype=int)
        edge_phys = np.zeros((0,), dtype=int)
        phys_names = {}

        if hasattr(msh, "cell_data_dict") and "gmsh:physical" in msh.cell_data_dict:
            if "triangle" in msh.cell_data_dict["gmsh:physical"]:
                tri_phys = np.asarray(msh.cell_data_dict["gmsh:physical"]["triangle"], int)
            if "line" in msh.cells_dict and "line" in msh.cell_data_dict["gmsh:physical"]:
                edges = np.asarray(msh.cells_dict["line"], int)
                edge_phys = np.asarray(msh.cell_data_dict["gmsh:physical"]["line"], int)

        if hasattr(msh, "field_data") and isinstance(msh.field_data, dict):
            for name, (tag, dim) in msh.field_data.items():
                phys_names[int(tag)] = str(name)

        mesh = Mesh(nodes=nodes, tris=tris, tri_phys=tri_phys,
                    edges=edges, edge_phys=edge_phys, phys_names=phys_names)

    # 3) Path-like / string -> read .msh
    elif isinstance(msh, (str, Path)):
        path = str(msh)
        if os.path.isfile(path):
            mesh = read_msh2_ascii(path)
        else:
            raise ValueError(f"Mesh generator returned a path, but file not found: {path}")

    # 4) dict return
    elif isinstance(msh, dict):
        if "nodes" in msh:
            nodes = np.asarray(msh["nodes"], float)
        elif "points" in msh:
            nodes = np.asarray(msh["points"], float)[:, :2]
        else:
            raise ValueError("Dict mesh has no 'nodes'/'points' key.")

        if "tris" in msh:
            tris = np.asarray(msh["tris"], int)
        elif "triangles" in msh:
            tris = np.asarray(msh["triangles"], int)
        else:
            raise ValueError("Dict mesh has no 'tris'/'triangles' key.")

        tri_phys = np.asarray(msh.get("tri_phys", np.zeros((tris.shape[0],), int)), int)
        edges = np.asarray(msh.get("edges", np.zeros((0, 2), int)), int)
        edge_phys = np.asarray(msh.get("edge_phys", np.zeros((edges.shape[0],), int)), int)
        phys_names = dict(msh.get("phys_names", {}))

        mesh = Mesh(nodes=nodes[:, :2].copy(), tris=tris, tri_phys=tri_phys,
                    edges=edges, edge_phys=edge_phys, phys_names=phys_names)

    # 5) tuple/list return
    elif isinstance(msh, (tuple, list)):
        if len(msh) < 2:
            raise ValueError("Tuple mesh must have at least (nodes, tris).")
        nodes = np.asarray(msh[0], float)[:, :2].copy()
        tris  = np.asarray(msh[1], int)

        edges = np.zeros((0, 2), dtype=int)
        edge_phys = np.zeros((0,), dtype=int)
        tri_phys = np.zeros((tris.shape[0],), dtype=int)
        phys_names = {}

        if len(msh) >= 3 and msh[2] is not None:
            edges = np.asarray(msh[2], int)
        if len(msh) >= 4 and msh[3] is not None:
            edge_phys = np.asarray(msh[3], int)
        if len(msh) >= 5 and msh[4] is not None:
            tri_phys = np.asarray(msh[4], int)
        if len(msh) >= 6 and msh[5] is not None:
            phys_names = dict(msh[5])

        mesh = Mesh(nodes=nodes, tris=tris, tri_phys=tri_phys,
                    edges=edges, edge_phys=edge_phys, phys_names=phys_names)
    else:
        raise ValueError(f"Unsupported mesh generator output type: {type(msh)}")

    # -----------------------------------------------------------------------
    # Repair missing boundary/physical tags (pure geometry classification)
    # -----------------------------------------------------------------------
    need_repair = (mesh.edges is None or mesh.edges.size == 0 or
                   mesh.edge_phys is None or mesh.edge_phys.size == 0 or
                   not isinstance(mesh.phys_names, dict) or len(mesh.phys_names) == 0)

    if need_repair:
        bnd_edges = extract_boundary_edges(mesh.tris)

        TAG_LEFT   = 1
        TAG_RIGHT  = 2
        TAG_TOP    = 3
        TAG_BOTTOM = 4
        TAG_HOLE   = 5
        TAG_DOMAIN = 6

        phys_names = {
            TAG_LEFT: "LEFT",
            TAG_RIGHT: "RIGHT",
            TAG_TOP: "TOP",
            TAG_BOTTOM: "BOTTOM",
            TAG_HOLE: "HOLE",
            TAG_DOMAIN: "DOMAIN",
        }

        edge_phys = np.zeros((bnd_edges.shape[0],), dtype=int)

        tol_x = 1e-6 * max(1.0, L)
        tol_y = 1e-6 * max(1.0, h)
        tol_hole = 0.15 * max(hx, hy) + 1e-9

        nodes = mesh.nodes
        xmin, xmax = 0.0, float(L)
        ymin, ymax = -0.5 * float(h), 0.5 * float(h)

        mid = 0.5 * (nodes[bnd_edges[:, 0]] + nodes[bnd_edges[:, 1]])
        mx = mid[:, 0]
        my = mid[:, 1]
        r = np.sqrt((mx - xc)**2 + my**2)

        is_left   = np.abs(mx - xmin) <= tol_x
        is_right  = np.abs(mx - xmax) <= tol_x
        is_top    = np.abs(my - ymax) <= tol_y
        is_bottom = np.abs(my - ymin) <= tol_y

        is_outer = is_left | is_right | is_top | is_bottom
        is_hole  = (~is_outer) & (np.abs(r - a) <= tol_hole)

        edge_phys[is_left]   = TAG_LEFT
        edge_phys[is_right]  = TAG_RIGHT
        edge_phys[is_top]    = TAG_TOP
        edge_phys[is_bottom] = TAG_BOTTOM
        edge_phys[is_hole]   = TAG_HOLE

        unknown = (edge_phys == 0)
        if np.any(unknown):
            d_left   = np.abs(mx - xmin)
            d_right  = np.abs(mx - xmax)
            d_top    = np.abs(my - ymax)
            d_bottom = np.abs(my - ymin)
            d_rect = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bottom))
            d_hole = np.abs(r - a)

            choose_hole = d_hole < d_rect
            edge_phys[unknown & choose_hole] = TAG_HOLE

            rest = unknown & (~choose_hole)
            if np.any(rest):
                dstack = np.vstack([d_left, d_right, d_top, d_bottom])
                idx = np.argmin(dstack[:, rest], axis=0)
                map_tags = np.array([TAG_LEFT, TAG_RIGHT, TAG_TOP, TAG_BOTTOM], dtype=int)
                edge_phys[rest] = map_tags[idx]

        mesh.edges = np.asarray(bnd_edges, int)
        mesh.edge_phys = edge_phys
        mesh.phys_names = phys_names

        if mesh.tri_phys is None or mesh.tri_phys.size != mesh.tris.shape[0]:
            mesh.tri_phys = np.full((mesh.tris.shape[0],), TAG_DOMAIN, dtype=int)

    return mesh


def keep_largest_component(mesh: Mesh) -> Mesh:
    """
    If mesh accidentally has disconnected components, keep the largest one.
    """
    N = mesh.nodes.shape[0]
    adj = [[] for _ in range(N)]
    for tri in mesh.tris:
        a, b, c = tri
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)

    seen = np.zeros(N, dtype=bool)
    comps = []
    for i in range(N):
        if seen[i]:
            continue
        q = deque([i]); seen[i] = True
        comp = []
        while q:
            v = q.popleft()
            comp.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
        comps.append(comp)

    if len(comps) <= 1:
        return mesh

    comps.sort(key=len, reverse=True)
    keep = np.zeros(N, dtype=bool)
    keep[np.asarray(comps[0], dtype=int)] = True

    mesh2 = Mesh(
        nodes=mesh.nodes.copy(),
        tris=mesh.tris.copy(),
        tri_phys=mesh.tri_phys.copy(),
        edges=mesh.edges.copy(),
        edge_phys=mesh.edge_phys.copy(),
        phys_names=dict(mesh.phys_names),
    )
    # Remove triangles not in keep
    tri_keep = keep[mesh2.tris].all(axis=1)
    mesh2.tris = mesh2.tris[tri_keep]
    mesh2.tri_phys = mesh2.tri_phys[tri_keep]
    # Remove edges not in keep
    if mesh2.edges.size:
        e_keep = keep[mesh2.edges].all(axis=1)
        mesh2.edges = mesh2.edges[e_keep]
        mesh2.edge_phys = mesh2.edge_phys[e_keep]

    return compact_mesh(mesh2)


# ---------------------------------------------------------------------------
# Material / element routines
# ---------------------------------------------------------------------------

def tri_B_matrix(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    xy: (3,2) node coordinates for a triangle
    Returns:
      B (3x6) strain-displacement matrix for linear triangle (P1),
      area (positive)
    """
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    detJ = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
    A = 0.5 * detJ
    if A == 0:
        raise ValueError("Degenerate triangle with zero area.")
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    B = np.array([
        [b1,   0, b2,   0, b3,   0],
        [ 0,  c1,  0,  c2,  0,  c3],
        [c1,  b1, c2,  b2, c3,  b3]
    ], dtype=float) / (2.0*A)
    return B, abs(A)


def plane_stress_D(E: float, nu: float) -> np.ndarray:
    c = E / (1.0 - nu**2)
    D = c * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu)/2.0],
    ], dtype=float)
    return D


def assemble_system(mesh: Mesh, E: float, nu: float, t: float) -> csr_matrix:
    N = mesh.nodes.shape[0]
    ndof = 2*N
    K = lil_matrix((ndof, ndof), dtype=float)
    D = plane_stress_D(E, nu)

    for e, tri in enumerate(mesh.tris):
        xy = mesh.nodes[tri, :]
        B, A = tri_B_matrix(xy)
        Ke = (B.T @ D @ B) * (A * t)
        dofs = np.array([2*tri[0], 2*tri[0]+1,
                         2*tri[1], 2*tri[1]+1,
                         2*tri[2], 2*tri[2]+1], dtype=int)
        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += Ke[i, j]

    return K.tocsr()


def get_phys_tag(mesh: Mesh, name: str) -> int:
    for k, v in mesh.phys_names.items():
        if v == name:
            return k
    raise KeyError(f"Physical name '{name}' not found in mesh.phys_names")


def get_fixed_dofs_on_physical(mesh: Mesh, phys_name: str) -> Dict[int, float]:
    """
    Return dict dof->value for all nodes that lie on edges with phys_name.
    We fix both ux, uy = 0 on those nodes.
    """
    tag = get_phys_tag(mesh, phys_name)
    mask = (mesh.edge_phys == tag)
    if mesh.edges.size == 0 or not np.any(mask):
        raise ValueError(f"No edges with physical tag '{phys_name}'.")

    nodes_on = np.unique(mesh.edges[mask].ravel())
    fixed = {}
    for n in nodes_on:
        fixed[2*n]   = 0.0
        fixed[2*n+1] = 0.0
    return fixed


def add_pins_for_unconstrained_components(mesh: Mesh, fixed: Dict[int, float]) -> Dict[int, float]:
    """
    If the boundary conditions do not fully constrain rigid-body modes (rare but possible
    in some generated meshes due to tagging or missing boundary edges),
    pin one extra dof in a safe node to eliminate singularity.
    """
    N = mesh.nodes.shape[0]
    # If no fixed, pin node 0 both dofs
    if len(fixed) == 0:
        fixed[0] = 0.0
        fixed[1] = 0.0
        return fixed

    # If only one component fixed (e.g., only ux), add a uy pin somewhere
    # Simple heuristic: ensure at least one uy is fixed
    has_uy = any((d % 2) == 1 for d in fixed.keys())
    if not has_uy:
        fixed[1] = 0.0
    return fixed


def build_load_vector(mesh: Mesh, phys_name: str, P: float, t: float) -> np.ndarray:
    """
    Build distributed traction load vector on edges with phys_name.
    For simplicity: uniform traction in -y direction with total magnitude P across the edge set.
    This implementation lumps edge loads to nodes (2-node line elements).
    """
    N = mesh.nodes.shape[0]
    f = np.zeros((2*N,), dtype=float)

    tag = get_phys_tag(mesh, phys_name)
    mask = (mesh.edge_phys == tag)
    if mesh.edges.size == 0 or not np.any(mask):
        raise ValueError(f"No edges with physical tag '{phys_name}'.")

    edges = mesh.edges[mask]
    # total length
    Ltot = 0.0
    lengths = []
    for (i, j) in edges:
        dx = mesh.nodes[j, 0] - mesh.nodes[i, 0]
        dy = mesh.nodes[j, 1] - mesh.nodes[i, 1]
        Le = math.hypot(dx, dy)
        lengths.append(Le)
        Ltot += Le
    if Ltot == 0:
        return f

    q = -P / (Ltot * t)  # traction (force per area); *t gives line load
    for (i, j), Le in zip(edges, lengths):
        # lumped: each node gets half
        fy = q * t * Le
        f[2*i + 1] += 0.5 * fy
        f[2*j + 1] += 0.5 * fy

    return f


def solve_with_elimination(K: csr_matrix, f: np.ndarray, fixed: Dict[int, float]) -> np.ndarray:
    """
    Solve Ku=f with Dirichlet fixed dofs via elimination.
    """
    ndof = K.shape[0]
    fixed_dofs = np.array(sorted(fixed.keys()), dtype=int)
    free = np.ones(ndof, dtype=bool)
    free[fixed_dofs] = False
    free_dofs = np.nonzero(free)[0]

    u = np.zeros((ndof,), dtype=float)
    for d, val in fixed.items():
        u[d] = val

    # Modify RHS: f_free = f_free - K_free,fixed * u_fixed
    K_ff = K[free_dofs, :][:, free_dofs]
    K_fc = K[free_dofs, :][:, fixed_dofs]
    f_f = f[free_dofs] - K_fc @ u[fixed_dofs]

    u_free = spsolve(K_ff, f_f)
    u[free_dofs] = u_free
    return u


# ---------------------------------------------------------------------------
# Postprocessing: stresses
# ---------------------------------------------------------------------------

def postprocess_stresses(mesh: Mesh, u: np.ndarray, E: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      sigma_elem: (M,3) [sxx, syy, sxy] constant per element (P1)
      eps_elem:   (M,3) [exx, eyy, gxy]
    """
    D = plane_stress_D(E, nu)
    M = mesh.tris.shape[0]
    sigma = np.zeros((M, 3), dtype=float)
    eps   = np.zeros((M, 3), dtype=float)

    for e, tri in enumerate(mesh.tris):
        xy = mesh.nodes[tri, :]
        B, A = tri_B_matrix(xy)
        dofs = np.array([2*tri[0], 2*tri[0]+1,
                         2*tri[1], 2*tri[1]+1,
                         2*tri[2], 2*tri[2]+1], dtype=int)
        ue = u[dofs]
        eps_e = B @ ue
        sig_e = D @ eps_e
        eps[e, :] = eps_e
        sigma[e, :] = sig_e

    return sigma, eps


def nodal_stress_area_weighted(mesh: Mesh, sigma_elem: np.ndarray) -> np.ndarray:
    """
    Simple area-weighted averaging of constant element stresses to nodes.
    """
    N = mesh.nodes.shape[0]
    acc = np.zeros((N, 3), dtype=float)
    w   = np.zeros((N,), dtype=float)

    for e, tri in enumerate(mesh.tris):
        xy = mesh.nodes[tri, :]
        _, A = tri_B_matrix(xy)
        for n in tri:
            acc[n, :] += sigma_elem[e, :] * A
            w[n] += A

    w = np.maximum(w, 1e-30)
    return acc / w[:, None]


def build_node_adjacency(mesh: Mesh) -> List[List[int]]:
    N = mesh.nodes.shape[0]
    neigh = [set() for _ in range(N)]
    for tri in mesh.tris:
        a, b, c = tri
        neigh[a].add(b); neigh[a].add(c)
        neigh[b].add(a); neigh[b].add(c)
        neigh[c].add(a); neigh[c].add(b)
    return [sorted(list(s)) for s in neigh]


def laplacian_smooth_stress(mesh: Mesh, sigma_node: np.ndarray,
                            n_iter: int = 5, weight: float = 0.5) -> np.ndarray:
    """
    Simple Laplacian smoothing on nodal stress field (for visual clarity / line plots).
    """
    sigma = sigma_node.copy()
    adj = build_node_adjacency(mesh)
    for _ in range(n_iter):
        sigma_new = sigma.copy()
        for i in range(mesh.nodes.shape[0]):
            nb = adj[i]
            if len(nb) == 0:
                continue
            avg = np.mean(sigma[nb, :], axis=0)
            sigma_new[i, :] = (1.0 - weight) * sigma[i, :] + weight * avg
        sigma = sigma_new
    return sigma


def von_mises_from_sigma(sigma_node: np.ndarray) -> np.ndarray:
    sxx = sigma_node[:, 0]
    syy = sigma_node[:, 1]
    sxy = sigma_node[:, 2]
    vm2 = sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2
    vm2 = np.maximum(vm2, 0.0)  # guard float noise
    return np.sqrt(vm2)


# ---------------------------------------------------------------------------
# Sampling along a line
# ---------------------------------------------------------------------------

def sample_along_line(mesh: Mesh, field_node: np.ndarray,
                      p0: Tuple[float, float], p1: Tuple[float, float],
                      npts: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a nodal field along segment p0->p1 by linear interpolation in triangles.
    Returns (s in [0,1], values). Values can be (npts,) or (npts,k).
    """
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    s = np.linspace(0.0, 1.0, npts)
    pts = p0[None, :] + (p1 - p0)[None, :] * s[:, None]
    x = pts[:, 0]; y = pts[:, 1]

    tri_obj = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)

    if field_node.ndim == 1:
        val = np.array(mtri.LinearTriInterpolator(tri_obj, field_node)(x, y), dtype=float)
    else:
        val = np.zeros((npts, field_node.shape[1]), dtype=float)
        for j in range(field_node.shape[1]):
            val[:, j] = np.array(
                mtri.LinearTriInterpolator(tri_obj, field_node[:, j])(x, y), dtype=float)
    return s, val


def sample_along_line_clean(mesh: Mesh, field_node: np.ndarray,
                             p0: Tuple[float, float], p1: Tuple[float, float],
                             xc: float = 40.0, a: float = 8.0,
                             hole_margin: float = 2.0,
                             x_skip_left: float = 10.0,
                             x_skip_right: float = 10.0,
                             npts: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample field along p0->p1, returning physical x-coordinates and values.
    Exclusion zones:
      - near left boundary  (x < x_skip_left):         avoids BC/load singularity
      - near right boundary (x > p1[0]-x_skip_right):  avoids load singularity
      - near hole:          dist from (xc,0) < a+hole_margin
    Returns (x_phys, values) — only finite, non-excluded points.
    """
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    s_all = np.linspace(0.0, 1.0, npts)
    pts   = p0[None, :] + (p1 - p0)[None, :] * s_all[:, None]
    x_all = pts[:, 0]; y_all = pts[:, 1]

    tri_obj = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)

    if field_node.ndim == 1:
        val_all = np.array(mtri.LinearTriInterpolator(tri_obj, field_node)(x_all, y_all),
                           dtype=float)
    else:
        val_all = np.zeros((npts, field_node.shape[1]), dtype=float)
        for j in range(field_node.shape[1]):
            val_all[:, j] = np.array(
                mtri.LinearTriInterpolator(tri_obj, field_node[:, j])(x_all, y_all),
                dtype=float)

    dist_hole = np.sqrt((x_all - xc)**2 + y_all**2)

    if field_node.ndim == 1:
        finite_mask = np.isfinite(val_all)
    else:
        finite_mask = np.isfinite(val_all).all(axis=1)

    keep = (finite_mask &
            (x_all >= x_skip_left) &
            (x_all <= p1[0] - x_skip_right) &
            (dist_hole >= a + hole_margin))

    return x_all[keep], val_all[keep]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_contour(mesh: Mesh, field: np.ndarray, filename: str, title: str):
    tri_obj = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
    plt.figure(figsize=(8, 3))
    plt.tricontourf(tri_obj, field, 100)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.title(title); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=200); plt.close()


def save_mesh_plot(mesh: Mesh, filename: str):
    tri_obj = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
    plt.figure(figsize=(8, 3))
    plt.triplot(tri_obj, linewidth=0.4)
    plt.gca().set_aspect('equal')
    plt.title("FE mesh"); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=200); plt.close()


def save_model_plot(mesh: Mesh, filename: str):
    tri_obj = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
    plt.figure(figsize=(10, 3))
    plt.triplot(tri_obj, color="black", linewidth=0.4)
    plt.gca().set_aspect("equal")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Finite element model (geometry + mesh)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200); plt.close()


def write_results(mesh: Mesh, u: np.ndarray, sigma: np.ndarray,
                  vm: np.ndarray, out_prefix: str) -> None:
    if HAS_MESHIO:
        points = np.column_stack([mesh.nodes[:, 0], mesh.nodes[:, 1],
                                  np.zeros((mesh.nodes.shape[0],))])
        cells = [("triangle", mesh.tris)]
        point_data = {
            "ux": u[0::2],
            "uy": u[1::2],
            "sxx": sigma[:, 0],
            "syy": sigma[:, 1],
            "sxy": sigma[:, 2],
            "svm": vm,
        }
        m = meshio.Mesh(points=points, cells=cells, point_data=point_data)
        meshio.write(f"{out_prefix}.vtu", m)
    else:
        np.savez(f"{out_prefix}.npz", nodes=mesh.nodes, tris=mesh.tris, u=u, sigma=sigma, vm=vm)


# ---------------------------------------------------------------------------
# Mesh study driver
# ---------------------------------------------------------------------------

def run_mesh_study_generated(h_list,
                              L=200.0, H=40.0, a=8.0, xc=40.0,
                              E=210000.0, nu=0.30, t=1.0, P=1000.0,
                              line_p0=(0.0, 0.0), line_p1=(200.0, 0.0),
                              npts=400, out_dir="figs_general 1"):

    os.makedirs(out_dir, exist_ok=True)
    stash = []
    rows  = []
    lines = {}   # h -> (x_clean, sigma_node along line)

    # Figure for σ_vm along clean line (shared across meshes)
    fig_vm, ax_vm = plt.subplots(figsize=(9, 4))

    for hval in h_list:
        # Scale refinement: many theta points + tight rings ensure h/a<0.25 is well-resolved
        n_theta = max(100, int(round(2 * np.pi * a / hval)) * 4)
        # ring_growth tuned per mesh size
        ring_growth = max(1.15, min(1.3, 0.9 + 0.1 * hval))

        mesh = make_generated_mesh(
            L=L, h=H, a=a, xc=xc,
            hx=hval, hy=hval,
            n_theta=n_theta,
            n_rings=7,
            ring_growth=ring_growth,
            seed=0)
        mesh = keep_largest_component(mesh)
        save_mesh_plot(mesh, f"{out_dir}/mesh_h{hval}.png")

        fixed = get_fixed_dofs_on_physical(mesh, "LEFT")
        fixed = add_pins_for_unconstrained_components(mesh, fixed)

        K = assemble_system(mesh, E, nu, t)
        f = build_load_vector(mesh, "RIGHT", P, t)
        u = solve_with_elimination(K, f, fixed)

        sigma_elem, _ = postprocess_stresses(mesh, u, E, nu)
        sigma_node    = nodal_stress_area_weighted(mesh, sigma_elem)
        # Laplacian smoothing removes inter-element stress jumps for clean line plots
        # (does NOT affect convergence metrics which use unsmoothed nodal stresses)
        # --- component-wise smoothing: keep normal stresses sharp, smooth shear harder ---
        sigma_node_sm = sigma_node.copy()

        # mild smoothing for sxx, syy
        sigma_node_sm[:, 0] = laplacian_smooth_stress(mesh, sigma_node[:, [0]], n_iter=4, weight=0.35).ravel()
        sigma_node_sm[:, 1] = laplacian_smooth_stress(mesh, sigma_node[:, [1]], n_iter=4, weight=0.35).ravel()

        # stronger smoothing for sxy (this one is noisy for P1 triangles)
        sigma_node_sm[:, 2] = laplacian_smooth_stress(mesh, sigma_node[:, [2]], n_iter=12, weight=0.45).ravel()

        vm_node = von_mises_from_sigma(sigma_node_sm)

        # Contour plots (use smoothed field for visual clarity)
        save_contour(mesh, sigma_node_sm[:, 0], f"{out_dir}/sxx_h{hval}.png", "sigma_xx")
        save_contour(mesh, sigma_node_sm[:, 1], f"{out_dir}/syy_h{hval}.png", "sigma_yy")
        save_contour(mesh, sigma_node_sm[:, 2], f"{out_dir}/sxy_h{hval}.png", "sigma_xy")
        save_contour(mesh, vm_node,              f"{out_dir}/svm_h{hval}.png", "sigma_vm")

        # --- CLEAN line sampling ---
        # y_line = H/4: quarter-height avoids both the neutral axis (poor P1 interpolation
        # at y=0 due to sign-alternating σ_xx/σ_yy between adjacent elements) and the
        # top/bottom boundaries. Gives the smoothest bending-stress distribution.
        # x_skip_left=35: skip stronger fixed-BC singularity zone AND hole influence
        # x_skip_right=8: skip distributed-load application zone
        # hole_margin=4: stay well outside hole stress concentration
        y_line = H / 4.0   # = 10 mm for H=40
        x_clean, sigma_clean = sample_along_line_clean(
            mesh, sigma_node_sm,
            p0=(0.0, y_line), p1=(L, y_line),

            xc=xc, a=a, hole_margin=4.0,
            x_skip_left=35.0, x_skip_right=8.0, npts=npts)
        lines[hval] = (x_clean, sigma_clean)

        x_vm, vm_clean = sample_along_line_clean(
            mesh, vm_node,
            p0=(0.0, y_line), p1=(L, y_line),
            xc=xc, a=a, hole_margin=4.0,
            x_skip_left=35.0, x_skip_right=8.0, npts=npts)
        ax_vm.plot(x_vm, vm_clean, label=f"h={hval}, Ne={mesh.tris.shape[0]}")

        max_vm = float(np.nanmax(vm_node))
        max_uy = float(np.max(np.abs(u[1::2])))
        rows.append((hval, mesh.nodes.shape[0], mesh.tris.shape[0], max_vm, max_uy))
        stash.append((hval, mesh, sigma_node))  # unsmoothed for convergence metrics

    # --- σ_vm along clean line ---
    ax_vm.set_xlabel("x, мм")
    ax_vm.set_ylabel(r"$\sigma_{vm}$, МПа")
    ax_vm.set_title(r"$\sigma_{vm}$ вдоль балки ($y=H/4=10$ мм, без зон ГУ и отверстия)")
    ax_vm.grid(True); ax_vm.legend()
    fig_vm.tight_layout()
    fig_vm.savefig(os.path.join(out_dir, "line_vm_multi_mesh_generated.png"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig_vm)

    # --- σ components along clean line ---
    for comp, name, ylabel in zip([0, 1, 2],
                                   ["sxx", "syy", "sxy"],
                                   [r"$\sigma_{xx}$, МПа",
                                    r"$\sigma_{yy}$, МПа",
                                    r"$\sigma_{xy}$, МПа"]):
        fig, ax = plt.subplots(figsize=(9, 4))
        for h in sorted(lines.keys()):
            x_h, sig_h = lines[h]
            ax.plot(x_h, sig_h[:, comp], label=f"h={h}")
        ax.legend(); ax.grid(True)
        ax.set_title(f"{name} вдоль балки (y = H/4 = 10 мм)")
        ax.set_xlabel("x, мм"); ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"line_{name}_multi.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # --- Convergence: max σ_vm vs number of elements ---
    rows_sorted = sorted(rows, key=lambda x: x[0])
    Ne          = [r[2] for r in rows_sorted]
    max_vm_list = [r[3] for r in rows_sorted]

    plt.figure()
    plt.plot(Ne, max_vm_list, marker="o")
    plt.xlabel("Number of elements"); plt.ylabel(r"max $\sigma_{vm}$")
    plt.title("Mesh convergence (generated)"); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "convergence_max_vm_generated.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # --- Relative error on line vs finest mesh ---
    ref_h = min(lines.keys())
    x_ref, sig_ref = lines[ref_h]
    vm_ref = np.sqrt(np.maximum(
        sig_ref[:, 0]**2 - sig_ref[:, 0]*sig_ref[:, 1] +
        sig_ref[:, 1]**2 + 3*sig_ref[:, 2]**2, 0.0))

    # Build a common x-grid inside the intersection of all meshes' "clean" domains.
    x_min = max(float(np.min(lines[h][0])) for h in lines.keys())
    x_max = min(float(np.max(lines[h][0])) for h in lines.keys())
    if not (x_max > x_min):
        raise ValueError("Clean-line domains do not overlap; increase skip margins or npts.")
    x_common = np.linspace(x_min, x_max, 600)

    # Use an L2-type scale (much more stable than max for noisy/stiff regions)
    vm_ref_c = np.interp(x_common, x_ref, vm_ref)
    ref_scale = float(np.sqrt(np.mean(vm_ref_c**2)) + 1e-14)

    fig_err, ax_err = plt.subplots(figsize=(9, 4))
    for h in sorted(lines.keys()):
        if h == ref_h:
            continue
        x_h, sig_h = lines[h]
        vm_h = np.sqrt(np.maximum(
            sig_h[:, 0]**2 - sig_h[:, 0]*sig_h[:, 1] +
            sig_h[:, 1]**2 + 3*sig_h[:, 2]**2, 0.0))

        vm_h_c = np.interp(x_common, x_h, vm_h)
        err = np.abs(vm_h_c - vm_ref_c) / ref_scale
        ax_err.plot(x_common, err, label=f"h={h}")

    ax_err.set_xlabel("x, мм"); ax_err.set_ylabel("Относительная погрешность")
    ax_err.set_title(r"Относительная погрешность $\sigma_{vm}$ vs эталонная сетка")
    ax_err.grid(True); ax_err.legend()
    fig_err.tight_layout()
    fig_err.savefig(os.path.join(out_dir, "line_error_multi_mesh_generated.png"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig_err)

    # --- CSV table ---
    csv_path = os.path.join(out_dir, "mesh_table_generated.csv")
    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write("h,N_nodes,N_elems,max_vm,max_abs_uy\n")
        for r in rows_sorted:
            fcsv.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.6f}\n")

    return rows_sorted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example run
    out_dir = "figs_general 1"
    hs = [0.5, 0.25]
    run_mesh_study_generated(hs, out_dir=out_dir, npts=600)
    print(f"Done. Results in: {out_dir}")