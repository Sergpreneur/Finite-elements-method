import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mesh import generate_beam_hole_mesh, extract_boundary_edges, classify_edges_left_right
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Optional VTK writer
try:
    import meshio
    HAS_MESHIO = True
except Exception:
    HAS_MESHIO = False


@dataclass
class Mesh:
    nodes: np.ndarray          # (N,2)
    tris: np.ndarray           # (M,3) int
    tri_phys: np.ndarray       # (M,) int
    edges: np.ndarray          # (K,2) int
    edge_phys: np.ndarray      # (K,) int
    phys_names: Dict[int, str] # physTag -> name


def read_msh2_ascii(path: str) -> Mesh:
    phys_names: Dict[int, str] = {}
    nodes = None
    tris = []
    tri_phys = []
    edges = []
    edge_phys = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "$PhysicalNames":
            i += 1
            n = int(lines[i].strip()); i += 1
            for _ in range(n):
                parts = lines[i].strip().split(maxsplit=2)
                # dim tag "NAME"
                tag = int(parts[1])
                name = parts[2].strip()
                if name.startswith('"') and name.endswith('"'):
                    name = name[1:-1]
                phys_names[tag] = name
                i += 1
            if lines[i].strip() != "$EndPhysicalNames":
                raise ValueError("Bad msh: expected $EndPhysicalNames")
        elif line == "$Nodes":
            i += 1
            n = int(lines[i].strip()); i += 1
            nodes = np.zeros((n, 2), dtype=float)
            for _ in range(n):
                parts = lines[i].strip().split()
                nid = int(parts[0]) - 1
                x = float(parts[1]); y = float(parts[2])
                nodes[nid, 0] = x
                nodes[nid, 1] = y
                i += 1
            if lines[i].strip() != "$EndNodes":
                raise ValueError("Bad msh: expected $EndNodes")
        elif line == "$Elements":
            i += 1
            ne = int(lines[i].strip()); i += 1
            for _ in range(ne):
                parts = lines[i].strip().split()
                # id type ntags tags... nodes...
                etype = int(parts[1])
                ntags = int(parts[2])
                tags = list(map(int, parts[3:3 + ntags]))
                phys = tags[0] if ntags > 0 else 0
                conn = list(map(int, parts[3 + ntags:]))

                if etype == 2 and len(conn) == 3:  # tri3
                    tris.append([conn[0] - 1, conn[1] - 1, conn[2] - 1])
                    tri_phys.append(phys)
                elif etype == 1 and len(conn) == 2:  # line2
                    edges.append([conn[0] - 1, conn[1] - 1])
                    edge_phys.append(phys)

                i += 1
            if lines[i].strip() != "$EndElements":
                raise ValueError("Bad msh: expected $EndElements")
        i += 1

    if nodes is None or len(tris) == 0:
        raise ValueError("Mesh has no nodes or triangles. Use msh2 ASCII with 2D triangles.")

    return Mesh(
        nodes=nodes,
        tris=np.asarray(tris, dtype=int),
        tri_phys=np.asarray(tri_phys, dtype=int),
        edges=np.asarray(edges, dtype=int) if len(edges) else np.zeros((0, 2), dtype=int),
        edge_phys=np.asarray(edge_phys, dtype=int) if len(edge_phys) else np.zeros((0,), dtype=int),
        phys_names=phys_names
    )
def compact_mesh(mesh):
    N = mesh.nodes.shape[0]
    used = np.zeros(N, dtype=bool)
    used[mesh.tris.ravel()] = True

    n_used = int(used.sum())
    n_unused = int((~used).sum())
    if n_unused == 0:
        return mesh

    new_id = -np.ones(N, dtype=int)
    new_id[used] = np.arange(n_used, dtype=int)

    mesh.nodes = mesh.nodes[used]
    mesh.tris = new_id[mesh.tris]

    if mesh.edges.shape[0] > 0:
        keep_e = used[mesh.edges].all(axis=1)
        mesh.edges = new_id[mesh.edges[keep_e]]
        mesh.edge_phys = mesh.edge_phys[keep_e]

    print(f"[mesh] removed unused nodes: {n_unused}, kept: {n_used}")
    return mesh



def D_plane_stress(E: float, nu: float) -> np.ndarray:
    c = E / (1.0 - nu * nu)
    D = np.zeros((3, 3), dtype=float)
    D[0, 0] = c
    D[0, 1] = c * nu
    D[1, 0] = c * nu
    D[1, 1] = c
    D[2, 2] = c * (1.0 - nu) / 2.0
    return D


def tri_B_matrix(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, float]:
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c

    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    A = 0.5 * det
    if A == 0:
        raise ValueError("Degenerate triangle with zero area")

    denom = det

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    B = np.array([
        [b1, 0,  b2, 0,  b3, 0],
        [0,  c1, 0,  c2, 0,  c3],
        [c1, b1, c2, b2, c3, b3],
    ], dtype=float) / denom

    return B, abs(A)


def assemble_system(mesh: Mesh, E: float, nu: float, thickness: float) -> csr_matrix:
    n = mesh.nodes.shape[0]
    ndof = 2 * n

    D = D_plane_stress(E, nu)

    K = lil_matrix((ndof, ndof), dtype=float)

    for e, tri in enumerate(mesh.tris):
        n1, n2, n3 = tri
        a = mesh.nodes[n1]
        b = mesh.nodes[n2]
        c = mesh.nodes[n3]

        B, A = tri_B_matrix(a, b, c)
        Ke = (B.T @ D @ B) * (thickness * A)  # 6x6

        dofs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1], dtype=int)

        for ii in range(6):
            for jj in range(6):
                K[dofs[ii], dofs[jj]] += Ke[ii, jj]

    return K.tocsr()


def build_load_vector(mesh: Mesh, right_phys_name: str, P_total: float, thickness: float) -> np.ndarray:
    n = mesh.nodes.shape[0]
    f = np.zeros((2*n,), dtype=float)

    right_tags = [tag for tag, name in mesh.phys_names.items() if name == right_phys_name]
    if not right_tags:
        raise ValueError(f"Physical curve '{right_phys_name}' not found in mesh physical names.")
    right_tag = right_tags[0]

    idx = np.where(mesh.edge_phys == right_tag)[0]
    if idx.size == 0:
        raise ValueError(f"No line elements found for physical curve '{right_phys_name}'. "
                         f"Make sure boundary has Physical Curve and mesh exports line elements.")

    Lb = 0.0
    lengths = []
    edge_nodes = []
    for k in idx:
        n1, n2 = mesh.edges[k]
        p1 = mesh.nodes[n1]
        p2 = mesh.nodes[n2]
        ell = float(np.linalg.norm(p2 - p1))
        Lb += ell
        lengths.append(ell)
        edge_nodes.append((n1, n2))

    if Lb <= 0:
        raise ValueError("RIGHT boundary length is zero?")

    q = P_total / (thickness * Lb)

    for (n1, n2), ell in zip(edge_nodes, lengths):
        fy = (-q) * thickness * ell / 2.0
        f[2*n1 + 1] += fy
        f[2*n2 + 1] += fy

    return f


def apply_dirichlet(K: csr_matrix, f: np.ndarray, fixed_dofs: np.ndarray, values: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    K = K.tolil()
    for d, val in zip(fixed_dofs, values):
        # zero row and column
        K.rows[d] = [d]
        K.data[d] = [1.0]
        f[d] = val
    # zero columns (need to sweep all rows)
    fixed_set = set(int(x) for x in fixed_dofs.tolist())
    for r in range(K.shape[0]):
        if r in fixed_set:
            continue
        row_cols = K.rows[r]
        row_data = K.data[r]
        for j in range(len(row_cols)):
            if row_cols[j] in fixed_set:
                row_data[j] = 0.0
    return K.tocsr(), f


def get_fixed_dofs_on_physical(mesh: Mesh, left_phys_name: str) -> np.ndarray:
    left_tags = [tag for tag, name in mesh.phys_names.items() if name == left_phys_name]
    if not left_tags:
        raise ValueError(f"Physical curve '{left_phys_name}' not found.")
    left_tag = left_tags[0]

    idx = np.where(mesh.edge_phys == left_tag)[0]
    if idx.size == 0:
        raise ValueError(f"No line elements found for physical curve '{left_phys_name}'.")

    nodes_set = set()
    for k in idx:
        n1, n2 = mesh.edges[k]
        nodes_set.add(int(n1))
        nodes_set.add(int(n2))

    nodes_list = sorted(nodes_set)
    dofs = []
    for n in nodes_list:
        dofs.append(2*n)     # ux
        dofs.append(2*n + 1) # uy
    return np.array(dofs, dtype=int)


def postprocess_stresses(mesh: Mesh, u: np.ndarray, E: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    D = D_plane_stress(E, nu)
    M = mesh.tris.shape[0]
    sigma = np.zeros((M, 3), dtype=float)
    vm = np.zeros((M,), dtype=float)

    for e, tri in enumerate(mesh.tris):
        n1, n2, n3 = tri
        a = mesh.nodes[n1]; b = mesh.nodes[n2]; c = mesh.nodes[n3]
        B, _A = tri_B_matrix(a, b, c)

        ue = np.array([
            u[2*n1], u[2*n1+1],
            u[2*n2], u[2*n2+1],
            u[2*n3], u[2*n3+1],
        ], dtype=float)

        eps = B @ ue
        s = D @ eps
        sxx, syy, txy = s
        sigma[e, :] = sxx, syy, txy
        vm[e] = math.sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*txy*txy)

    return sigma, vm


def write_results(mesh: Mesh, u: np.ndarray, sigma: np.ndarray, vm: np.ndarray, out_prefix: str) -> None:
    if HAS_MESHIO:
        points = np.column_stack([mesh.nodes[:, 0], mesh.nodes[:, 1], np.zeros(mesh.nodes.shape[0])])
        cells = [("triangle", mesh.tris.copy())]

        point_data = {
            "ux": u[0::2],
            "uy": u[1::2],
            "u_mag": np.sqrt(u[0::2]**2 + u[1::2]**2),
        }
        cell_data = {
            "sxx": [sigma[:, 0]],
            "syy": [sigma[:, 1]],
            "txy": [sigma[:, 2]],
            "von_mises": [vm],
        }
        meshio.write_points_cells(
            f"{out_prefix}.vtu",
            points,
            cells,
            point_data=point_data,
            cell_data=cell_data
        )
        print(f"[OK] Wrote {out_prefix}.vtu (open in ParaView)")
    else:
        np.savez(
            f"{out_prefix}.npz",
            nodes=mesh.nodes,
            tris=mesh.tris,
            u=u,
            sigma=sigma,
            von_mises=vm
        )
        print(f"[OK] meshio not installed; wrote {out_prefix}.npz")
import numpy as np
from collections import deque

def connected_components_from_tris(n_nodes: int, tris: np.ndarray):
    adj = [[] for _ in range(n_nodes)]
    for a, b, c in tris:
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)

    seen = np.zeros(n_nodes, dtype=bool)
    comps = []
    for s in range(n_nodes):
        if seen[s]:
            continue
        # isolated node (no adjacency)
        if len(adj[s]) == 0:
            seen[s] = True
            comps.append(np.array([s], dtype=int))
            continue
        q = deque([s])
        seen[s] = True
        comp = [s]
        while q:
            v = q.popleft()
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
                    comp.append(w)
        comps.append(np.array(comp, dtype=int))
    return comps


def add_pins_for_unconstrained_components(mesh, fixed_dofs: np.ndarray):
    n = mesh.nodes.shape[0]
    fixed_set = set(int(d) for d in fixed_dofs.tolist())
    comps = connected_components_from_tris(n, mesh.tris)

    added = []

    for comp in comps:
        has_fixed = False
        for node in comp:
            if (2*node in fixed_set) or (2*node+1 in fixed_set):
                has_fixed = True
                break
        if has_fixed:
            continue

        xs = mesh.nodes[comp, 0]
        ys = mesh.nodes[comp, 1]
        i1 = comp[np.argmin(xs)]
        y1 = mesh.nodes[i1, 1]
        i2 = comp[np.argmax(np.abs(ys - y1))]
        if i2 == i1 and comp.size > 1:
            i2 = comp[1]

        added.extend([2*i1, 2*i1+1, 2*i2+1])

    if added:
        fixed_new = np.unique(np.concatenate([fixed_dofs, np.array(added, dtype=int)]))
        print(f"[mesh] Added pins for {len(added)//3} unconstrained component(s): +{len(added)} DOF")
        return fixed_new
    return fixed_dofs

import numpy as np
from scipy.sparse.linalg import spsolve

def solve_with_elimination(K, f, fixed_dofs, fixed_vals=None):

    ndof = K.shape[0]
    fixed_dofs = np.unique(fixed_dofs).astype(int)

    if fixed_vals is None:
        fixed_vals = np.zeros_like(fixed_dofs, dtype=float)
    else:
        fixed_vals = np.asarray(fixed_vals, dtype=float)
        assert fixed_vals.shape == fixed_dofs.shape

    free = np.ones(ndof, dtype=bool)
    free[fixed_dofs] = False

    # Reduced system
    K_ff = K[free][:, free]
    f_f = f[free].copy()

    # Move known terms to RHS: f_f -= K_fc * u_c
    if fixed_dofs.size > 0:
        K_fc = K[free][:, fixed_dofs]
        f_f = f_f - K_fc @ fixed_vals

    u = np.zeros(ndof, dtype=float)
    u[fixed_dofs] = fixed_vals
    u[free] = spsolve(K_ff, f_f)
    return u

def nodal_stress_area_weighted(mesh: Mesh, sigma_elem: np.ndarray) -> np.ndarray:
    n = mesh.nodes.shape[0]
    sig_sum = np.zeros((n, 3), dtype=float)
    w_sum = np.zeros(n, dtype=float)

    for e, tri in enumerate(mesh.tris):
        n1, n2, n3 = tri
        a = mesh.nodes[n1]; b = mesh.nodes[n2]; c = mesh.nodes[n3]
        det = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        A = abs(0.5*det)

        for node in (n1, n2, n3):
            sig_sum[node] += A * sigma_elem[e]
            w_sum[node] += A

    sigma_node = np.zeros_like(sig_sum)
    mask = w_sum > 0
    sigma_node[mask] = sig_sum[mask] / w_sum[mask, None]

    return sigma_node

def von_mises_from_sigma(sigma: np.ndarray) -> np.ndarray:
    sxx = sigma[..., 0]
    syy = sigma[..., 1]
    txy = sigma[..., 2]
    return np.sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*txy*txy)

def sample_along_line(mesh: Mesh, field_node: np.ndarray,
                      p0: Tuple[float, float], p1: Tuple[float, float],
                      npts: int = 300) -> Tuple[np.ndarray, np.ndarray]:

    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    s = np.linspace(0.0, 1.0, npts)
    pts = p0[None, :] + (p1 - p0)[None, :] * s[:, None]
    x = pts[:, 0]; y = pts[:, 1]

    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)

    if field_node.ndim == 1:
        interp = mtri.LinearTriInterpolator(tri, field_node)
        val = np.array(interp(x, y), dtype=float)
    else:
        val = np.zeros((npts, field_node.shape[1]), dtype=float)
        for j in range(field_node.shape[1]):
            interp = mtri.LinearTriInterpolator(tri, field_node[:, j])
            val[:, j] = np.array(interp(x, y), dtype=float)

    return s, val

def save_contour(mesh, field, filename, title):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        mesh.tris
    )

    plt.figure(figsize=(6, 4))
    plt.tricontourf(tri, field, 100)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def save_mesh_plot(mesh, filename):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        mesh.tris
    )

    plt.figure(figsize=(6, 4))
    plt.triplot(tri, linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.title("Finite element mesh")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
def save_model_plot(mesh, filename):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(
        mesh.nodes[:,0],
        mesh.nodes[:,1],
        mesh.tris
    )

    plt.figure(figsize=(8,3))
    plt.triplot(tri, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Finite element model (geometry + mesh)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def run_mesh_study(msh_paths: List[str],
                   E=210000.0, nu=0.30, t=1.0, P=1000.0,
                   LEFT_NAME="LEFT", RIGHT_NAME="RIGHT",
                   line_p0=(0.0, 0.0), line_p1=(100.0, 0.0),
                   npts=300, out_dir="figs_general"):

    import os
    os.makedirs(out_dir, exist_ok=True)

    rows = []  # for table
    plt.figure()

    # --- (A) Line plot: sigma_vm along line for multiple meshes ---
    for path in msh_paths:
        mesh = read_msh2_ascii(path)

        # Prefer compact_mesh to remove unused nodes
        mesh = compact_mesh(mesh)

        fixed = get_fixed_dofs_on_physical(mesh, LEFT_NAME)
        fixed = add_pins_for_unconstrained_components(mesh, fixed)

        K = assemble_system(mesh, E, nu, t)
        f = build_load_vector(mesh, RIGHT_NAME, P, t)
        u = solve_with_elimination(K, f, fixed)

        sigma_elem, vm_elem = postprocess_stresses(mesh, u, E, nu)
        sigma_node = nodal_stress_area_weighted(mesh, sigma_elem)
        vm_node = von_mises_from_sigma(sigma_node)

        # max metrics
        max_vm = float(vm_node.max())
        max_uy = float(np.abs(u[1::2]).max())

        rows.append((os.path.basename(path), mesh.nodes.shape[0], mesh.tris.shape[0], max_vm, max_uy))

        # sample along line
        s, vm_line = sample_along_line(mesh, vm_node, line_p0, line_p1, npts=npts)

        plt.plot(s, vm_line, label=f"{os.path.basename(path)}  (Ne={mesh.tris.shape[0]})")

    plt.xlabel("s along line (0..1)")
    plt.ylabel(r"$\sigma_{vm}$")
    plt.title(r"$\sigma_{vm}$ along a line for different meshes")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "line_vm_multi_mesh.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- (B) Mesh convergence: max sigma_vm vs number of elements ---
    rows_sorted = sorted(rows, key=lambda x: x[2])  # sort by Ne
    Ne = [r[2] for r in rows_sorted]
    max_vm_list = [r[3] for r in rows_sorted]

    plt.figure()
    plt.plot(Ne, max_vm_list, marker="o")
    plt.xlabel("Number of elements (triangles)")
    plt.ylabel(r"max $\sigma_{vm}$ (nodal, smoothed)")
    plt.title("Mesh convergence: max von Mises vs number of elements")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "convergence_max_vm.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- Save table as CSV for LaTeX ---
    csv_path = os.path.join(out_dir, "mesh_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("mesh_file,N_nodes,N_elems,max_vm,max_abs_uy\n")
        for r in rows_sorted:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")

    print(f"[OK] Saved: {out_dir}/line_vm_multi_mesh.png")
    print(f"[OK] Saved: {out_dir}/convergence_max_vm.png")
    print(f"[OK] Saved: {out_dir}/mesh_table.csv")

    # ---- relative error vs finest mesh ----
    ref_h = min(lines.keys())
    s_ref, vm_ref = lines[ref_h]

    ref_scale = np.max(np.abs(vm_ref)) + 1e-14

    plt.figure()
    err_rows = []

    for h in sorted(lines.keys()):
        if h == ref_h:
            continue

        s_h, vm_h = lines[h]

        # интерполяция на сетку ref
        vm_h_on_ref = np.interp(s_ref, s_h, vm_h)

        err = np.abs(vm_h_on_ref - vm_ref) / ref_scale
        plt.plot(s_ref, err, label=f"h={h}")

        err_rows.append((h, float(np.max(err)), float(np.mean(err))))

    plt.xlabel("s along line")
    plt.ylabel("relative error")
    plt.title("Relative error along line vs finest mesh")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "line_error_multi_mesh_generated.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    with open(os.path.join(out_dir, "mesh_error_table_generated.csv"), "w", encoding="utf-8") as f:
        f.write("h,max_rel_err,mean_rel_err\n")
        for h, emax, emean in err_rows:
            f.write(f"{h},{emax},{emean}\n")
    return rows_sorted


def main():
    msh_path = "beam_hole.msh"

    E = 210000.0
    nu = 0.30
    t = 1.0
    P = 1000.0

    LEFT_NAME = "LEFT"
    RIGHT_NAME = "RIGHT"

    mesh = read_msh2_ascii(msh_path)
    print(f"Nodes: {mesh.nodes.shape[0]}, Tris: {mesh.tris.shape[0]}, Edges: {mesh.edges.shape[0]}")
    print(f"Physical names: {mesh.phys_names}")

    n = mesh.nodes.shape[0]
    used = np.zeros(n, dtype=bool)
    used[mesh.tris.ravel()] = True
    unused_nodes = np.where(~used)[0]
    if unused_nodes.size > 0:
        print("[mesh] Unused nodes:", unused_nodes.size)

    fixed = get_fixed_dofs_on_physical(mesh, LEFT_NAME)
    print("[bc] fixed dofs from LEFT:", fixed.size)

    if unused_nodes.size > 0:
        extra = np.empty(2 * unused_nodes.size, dtype=int)
        extra[0::2] = 2 * unused_nodes
        extra[1::2] = 2 * unused_nodes + 1
        fixed = np.unique(np.concatenate([fixed, extra]))
        print("[bc] fixed dofs total (LEFT + unused):", fixed.size)

    fixed = add_pins_for_unconstrained_components(mesh, fixed)

    K = assemble_system(mesh, E, nu, t)
    f = build_load_vector(mesh, RIGHT_NAME, P, t)

    row_nnz = np.diff(K.indptr)
    print("[dbg] zero rows in K:", np.sum(row_nnz == 0))

    print("Solving...")
    u = solve_with_elimination(K, f, fixed)
    print("Done.")

    sigma, vm = postprocess_stresses(mesh, u, E, nu)
    uy = u[1::2]
    print(f"Min uy = {uy.min():.6e} mm, Max uy = {uy.max():.6e} mm")
    print(f"Max von Mises (elem) = {vm.max():.6e} MPa")

    write_results(mesh, u, sigma, vm, out_prefix="result_beam_hole")

def make_generated_mesh(L=200.0, h=40.0, a=8.0, xc=40.0,
                        hx=4.0, hy=4.0, n_theta=80,
                        n_rings=3, ring_growth=1.6, seed=0) -> Mesh:
    nodes, tris = generate_beam_hole_mesh(
        L=L, h=h, a=a, xc=xc,
        hx=hx, hy=hy,
        n_theta=n_theta,
        n_rings=n_rings,
        ring_growth=ring_growth,
        seed=seed
    )
    edges = extract_boundary_edges(tris)
    edge_phys = classify_edges_left_right(nodes, edges, L=L, tol=1e-6)

    phys_names = {1: "LEFT", 2: "RIGHT", 3: "DOMAIN", 99: "OTHER"}
    tri_phys = np.full(tris.shape[0], 3, dtype=int)

    mesh = Mesh(nodes=nodes, tris=tris, tri_phys=tri_phys,
                edges=edges, edge_phys=edge_phys, phys_names=phys_names)
    return mesh


def run_mesh_study_generated(h_list,
                             L=200.0, H=40.0, a=8.0, xc=40.0,
                             E=210000.0, nu=0.30, t=1.0, P=1000.0,
                             line_p0=(0.0, 0.0), line_p1=(200.0, 0.0),
                             npts=400, out_dir="figs_general"):

    import os
    os.makedirs(out_dir, exist_ok=True)
    stash = []
    rows = []
    lines = {}  # ← для хранения компонент вдоль линии
    plt.figure()

    for hval in h_list:

        n_theta = max(40, int(2*np.pi*a / hval))

        mesh = make_generated_mesh(L=L, h=H, a=a, xc=xc,
                                   hx=hval, hy=hval,
                                   n_theta=n_theta,
                                   n_rings=3, ring_growth=1.6, seed=0)
        mesh = keep_largest_component(mesh)
        # --- сохраняем сетку ---
        save_mesh_plot(mesh, f"{out_dir}/mesh_h{hval}.png")

        fixed = get_fixed_dofs_on_physical(mesh, "LEFT")
        fixed = add_pins_for_unconstrained_components(mesh, fixed)

        K = assemble_system(mesh, E, nu, t)
        f = build_load_vector(mesh, "RIGHT", P, t)
        u = solve_with_elimination(K, f, fixed)

        sigma_elem, _ = postprocess_stresses(mesh, u, E, nu)
        sigma_node = nodal_stress_area_weighted(mesh, sigma_elem)
        vm_node = von_mises_from_sigma(sigma_node)

        # --- цветовые карты ---
        save_contour(mesh, sigma_node[:,0],
                     f"{out_dir}/sxx_h{hval}.png", "sigma_xx")
        save_contour(mesh, sigma_node[:,1],
                     f"{out_dir}/syy_h{hval}.png", "sigma_yy")
        save_contour(mesh, sigma_node[:,2],
                     f"{out_dir}/sxy_h{hval}.png", "sigma_xy")
        save_contour(mesh, vm_node,
                     f"{out_dir}/svm_h{hval}.png", "sigma_vm")

        # --- компоненты вдоль линии ---
        s, sigma_line = sample_along_line(
            mesh,
            sigma_node,
            line_p0,
            line_p1,
            npts=npts
        )

        lines[hval] = (s, sigma_line)

        max_vm = float(np.nanmax(vm_node))
        max_uy = float(np.max(np.abs(u[1::2])))

        rows.append((hval,
                     mesh.nodes.shape[0],
                     mesh.tris.shape[0],
                     max_vm,
                     max_uy))
        stash.append((hval, mesh, sigma_node))
        # --- σ_vm вдоль линии ---
        s_vm, vm_line = sample_along_line(mesh, vm_node,
                                          line_p0, line_p1,
                                          npts=npts)

        plt.plot(s_vm, vm_line,
                 label=f"h={hval}, Ne={mesh.tris.shape[0]}")

    # ====================================================
    # σ_vm multi mesh
    # ====================================================
    plt.xlabel("s along line (0..1)")
    plt.ylabel(r"$\sigma_{vm}$")
    plt.title(r"$\sigma_{vm}$ along a line (generated)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir,
                "line_vm_multi_mesh_generated.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ====================================================
    # σxx, σyy, σxy multi mesh
    # ====================================================
    for comp, name in zip([0,1,2],
                          ["sxx","syy","sxy"]):

        plt.figure()
        for h in sorted(lines.keys()):
            s, sig = lines[h]
            plt.plot(s, sig[:,comp],
                     label=f"h={h}")
        plt.legend()
        plt.grid(True)
        plt.title(f"{name} along line")
        plt.xlabel("s")
        plt.ylabel(name)
        plt.savefig(os.path.join(out_dir,
                    f"line_{name}_multi.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # ====================================================
    # Сходимость max σ_vm
    # ====================================================
    rows_sorted = sorted(rows, key=lambda x: x[0])

    Ne = [r[2] for r in rows_sorted]
    max_vm_list = [r[3] for r in rows_sorted]

    plt.figure()
    plt.plot(Ne, max_vm_list, marker="o")
    plt.xlabel("Number of elements")
    plt.ylabel(r"max $\sigma_{vm}$")
    plt.title("Mesh convergence (generated)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir,
                "convergence_max_vm_generated.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ====================================================
    # % ошибка h vs h/2
    # ====================================================
    print("\nMesh convergence (h vs h/2):")
    for i in range(len(rows_sorted)-1):
        h1, _, _, vm1, _ = rows_sorted[i]
        h2, _, _, vm2, _ = rows_sorted[i+1]

        if abs(h2 - h1/2) < 1e-6:
            err = abs(vm1 - vm2)/abs(vm2)*100.0
            print(f"h={h1} vs h={h2}: error = {err:.3f} %")

    # ====================================================
    # CSV таблица
    # ====================================================
    csv_path = os.path.join(out_dir,
                            "mesh_table_generated.csv")

    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write("h,N_nodes,N_elems,max_vm,max_abs_uy\n")
        for r in rows_sorted:
            fcsv.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")

    print(f"[OK] Saved generated study to {out_dir}")

    return stash, rows_sorted

def mesh_convergence_percent(rows_sorted):
    """
    rows_sorted: список кортежей (h, N_nodes, N_elems, max_vm, max_abs_uy)
    Возвращает список строк (h, h2, max_vm_h, max_vm_h2, eps_vm_%)
    """
    # превратим в словарь по h
    by_h = {float(r[0]): r for r in rows_sorted}

    pairs = []
    for h in sorted(by_h.keys(), reverse=True):
        h2 = h / 2.0
        if h2 in by_h:
            Qh  = float(by_h[h][3])   # max_vm
            Qh2 = float(by_h[h2][3])  # max_vm at h/2
            eps = abs(Qh - Qh2) / (abs(Qh2) + 1e-30) * 100.0
            pairs.append((h, h2, Qh, Qh2, eps))

    return pairs

import matplotlib.tri as mtri
import numpy as np
def keep_largest_component(mesh: Mesh) -> Mesh:
    """
    Оставляет только крупнейшую связную компоненту по треугольникам.
    Убирает "островки" -> пропадают add_pins и дикие eps.
    """
    n = mesh.nodes.shape[0]

    # компоненты по узлам через треугольники
    comps = connected_components_from_tris(n, mesh.tris)

    # размер компоненты по числу узлов
    sizes = np.array([c.size for c in comps], dtype=int)
    i_big = int(np.argmax(sizes))
    keep_nodes = np.zeros(n, dtype=bool)
    keep_nodes[comps[i_big]] = True

    # оставить только треугольники, целиком лежащие в этой компоненте
    keep_tri = keep_nodes[mesh.tris].all(axis=1)
    mesh.tris = mesh.tris[keep_tri]

    # дальше удалить неиспользуемые узлы и перенумеровать
    mesh = compact_mesh(mesh)

    return mesh
import matplotlib.tri as mtri
import numpy as np

import matplotlib.tri as mtri
import numpy as np

def eps_on_coarse_nodes(mesh_h, sigma_h_node,
                        mesh_h2, sigma_h2_node,
                        h_coarse,
                        a=8.0, xc=40.0, yc=0.0,
                        exclude_k=2.0,
                        L=200.0, H=40.0):

    Xc = mesh_h.nodes[:, 0]
    Yc = mesh_h.nodes[:, 1]
    h = float(h_coarse)

    # --- интерполяция тонкой сетки на узлы грубой ---
    tri2 = mtri.Triangulation(mesh_h2.nodes[:, 0],
                              mesh_h2.nodes[:, 1],
                              mesh_h2.tris)

    def interp_field(field2):
        itp = mtri.LinearTriInterpolator(tri2, field2)
        return np.array(itp(Xc, Yc), dtype=float)

    sxx2 = interp_field(sigma_h2_node[:, 0])
    syy2 = interp_field(sigma_h2_node[:, 1])
    sxy2 = interp_field(sigma_h2_node[:, 2])

    sxx1 = sigma_h_node[:, 0]
    syy1 = sigma_h_node[:, 1]
    sxy1 = sigma_h_node[:, 2]

    r = np.sqrt((Xc - xc)**2 + (Yc - yc)**2)

    # --- базовая маска: только точки, где интерполяция определена ---
    base_ok = np.isfinite(sxx1) & np.isfinite(sxx2) & np.isfinite(syy1) & np.isfinite(syy2) & np.isfinite(sxy1) & np.isfinite(sxy2)

    # --- пробуем наборы масок от строгой к мягкой ---
    masks = []

    # 1) строгая: вырезаем отверстие + границы
    masks.append(
        base_ok &
        (r >= (a + exclude_k * h)) &
        (Xc >= 2*h) & (Xc <= L - 2*h) &
        (Yc >= -H/2 + 2*h) & (Yc <= H/2 - 2*h)
    )

    # 2) мягче: без вырезания границ
    masks.append(
        base_ok &
        (r >= (a + exclude_k * h))
    )

    # 3) совсем мягкая: вообще без вырезания, только где интерполяция есть
    masks.append(base_ok)

    ok = None
    for m in masks:
        if np.any(m):
            ok = m
            break

    if ok is None or not np.any(ok):
        print("Warning: no valid points even after relaxing masks.")
        return np.nan, np.nan, np.nan, np.nan

    def eps_component_max(v1, v2):
        den = np.max(np.abs(v2[ok]))
        if den < 1e-14:
            return 0.0
        num = np.max(np.abs(v1[ok] - v2[ok]))
        return 100.0 * num / den

    def vm(sxx, syy, sxy):
        return np.sqrt(sxx*sxx - sxx*syy + syy*syy + 3*sxy*sxy)

    eps_sxx = eps_component_max(sxx1, sxx2)
    eps_syy = eps_component_max(syy1, syy2)
    eps_sxy = eps_component_max(sxy1, sxy2)

    vm1 = vm(sxx1, syy1, sxy1)
    vm2 = vm(sxx2, syy2, sxy2)
    eps_vm = eps_component_max(vm1, vm2)

    return eps_sxx, eps_syy, eps_sxy, eps_vm
if __name__ == "__main__":

    h_list = [8.0, 4.0, 2.0]

    stash, rows_sorted = run_mesh_study_generated(
        h_list,
        L=200.0, H=40.0, a=8.0, xc=40.0,
        E=210000.0, nu=0.30, t=1.0, P=1000.0,
        line_p0=(0.0, 8.0),
        line_p1=(200.0, 8.0),
        npts=400,
        out_dir="figs_general"
    )

    # если хочешь старую табличку по max_vm
    pairs = mesh_convergence_percent(rows_sorted)
    print("Mesh convergence (percent), Q = max_vm:")
    for h, h2, Qh, Qh2, eps in pairs:
        print(f"h={h:.1f} -> h/2={h2:.1f}: Qh={Qh:.3f}, Qh2={Qh2:.3f}, eps={eps:.2f}%")

    print("\nMesh convergence (percent) by components on coarse nodes:")

    stash_sorted = sorted(stash, key=lambda x: x[0], reverse=True)

    for i in range(len(stash_sorted) - 1):
        h1, mesh1, sigma1 = stash_sorted[i]
        h2, mesh2, sigma2 = stash_sorted[i + 1]

        # проверка именно h -> h/2
        if abs(h2 - h1/2) > 1e-9:
            continue

        eps_sxx, eps_syy, eps_sxy, eps_vm = eps_on_coarse_nodes(
            mesh1, sigma1,
            mesh2, sigma2,
            h_coarse=h1,
            a=8.0,
            xc=40.0,
            yc=0.0,        # <-- если отверстие по центру по y, поменяй на yc=H/2
            exclude_k=2.0  # вырезаем зону концентрации
        )

        (eps_sxx_max, eps_syy_max, eps_sxy_max, eps_vm_max,
         eps_sxx_L2, eps_syy_L2, eps_sxy_L2, eps_vm_L2) = eps_on_coarse_nodes(...)

        #print(f"MAX: sxx={eps_sxx_max:.2f}%, syy={eps_syy_max:.2f}%, sxy={eps_sxy_max:.2f}%, vm={eps_vm_max:.2f}%")
        #print(f"L2 : sxx={eps_sxx_L2:.2f}%, syy={eps_syy_L2:.2f}%, sxy={eps_sxy_L2:.2f}%, vm={eps_vm_L2:.2f}%")

    mesh = read_msh2_ascii("beam_hole.msh")
    save_model_plot(mesh, "figs_general/model_geometry.png")