import numpy as np, math
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import os
import matplotlib.tri as mtri

def generate_mesh_quarter_square_with_hole(L=12.0, a=1.0, N=24, rings=3):

    h = L / N

    xs = np.linspace(0, L, N+1)
    ys = np.linspace(0, L, N+1)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    P = np.column_stack([X.ravel(), Y.ravel()])

    r = np.linalg.norm(P, axis=1)
    P = P[r >= a*0.999]

    dtheta = h / a
    ntheta = max(12, int(math.ceil((math.pi/2)/dtheta))+1)
    thetas = np.linspace(0, math.pi/2, ntheta)
    hole = np.column_stack([a*np.cos(thetas), a*np.sin(thetas)])

    ring_pts = []
    for k in range(1, rings+1):
        rk = a + k*h
        ntheta_k = max(16, int(math.ceil((math.pi/2)*rk/h))+1)
        thk = np.linspace(0, math.pi/2, ntheta_k)
        ring_pts.append(np.column_stack([rk*np.cos(thk), rk*np.sin(thk)]))
    ring_pts = np.vstack(ring_pts) if ring_pts else np.empty((0,2))

    P = np.vstack([P, hole, ring_pts])
    P[P < 0] = 0.0
    P = np.unique(np.round(P, 12), axis=0)

    tri = Delaunay(P)
    T = tri.simplices.copy()

    Pc = P[T].mean(axis=1)
    rc = np.linalg.norm(Pc, axis=1)
    inside = (Pc[:,0] >= -1e-9) & (Pc[:,0] <= L+1e-9) & (Pc[:,1] >= -1e-9) & (Pc[:,1] <= L+1e-9) & (rc >= a*1.0001)
    T = T[inside]

    a0 = P[T[:,0]]; a1 = P[T[:,1]]; a2 = P[T[:,2]]
    area = 0.5*np.abs((a1[:,0]-a0[:,0])*(a2[:,1]-a0[:,1])-(a2[:,0]-a0[:,0])*(a1[:,1]-a0[:,1]))
    T = T[area > 1e-12]

    return P, T, h

def assemble_elasticity_P1(P, T, E=210e9, nu=0.3, plane_stress=True, t=1.0):
    n = P.shape[0]
    ndof = 2*n

    if plane_stress:
        coef = E/(1-nu**2)
        D = coef*np.array([[1, nu, 0],
                           [nu, 1, 0],
                           [0, 0, (1-nu)/2]])
    else:
        coef = E/((1+nu)*(1-2*nu))
        D = coef*np.array([[1-nu, nu, 0],
                           [nu, 1-nu, 0],
                           [0, 0, (1-2*nu)/2]])

    I = []; J = []; V = []

    for tri in T:
        x = P[tri,0]; y = P[tri,1]
        A = 0.5*((x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))
        A = abs(A)

        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        dNdx = b/(2*A)
        dNdy = c/(2*A)

        B = np.zeros((3,6))
        for i in range(3):
            B[0,2*i]   = dNdx[i]
            B[1,2*i+1] = dNdy[i]
            B[2,2*i]   = dNdy[i]
            B[2,2*i+1] = dNdx[i]

        Ke = t*A*(B.T @ D @ B)

        dofs = np.array([2*tri[0],2*tri[0]+1, 2*tri[1],2*tri[1]+1, 2*tri[2],2*tri[2]+1])
        ii, jj = np.meshgrid(dofs, dofs, indexing='ij')
        I.extend(ii.ravel().tolist()); J.extend(jj.ravel().tolist()); V.extend(Ke.ravel().tolist())

    K = coo_matrix((V,(I,J)), shape=(ndof,ndof)).tocsr()
    return K, D

def apply_dirichlet(K, f, bc_dofs, bc_vals):
    bc_dofs = np.asarray(bc_dofs, dtype=int)
    bc_vals = np.asarray(bc_vals, dtype=float)

    f_mod = f - K[:, bc_dofs] @ bc_vals

    K_mod = K.tolil()
    K_mod[bc_dofs,:] = 0
    K_mod[:,bc_dofs] = 0
    K_mod[bc_dofs, bc_dofs] = 1.0
    f_mod[bc_dofs] = bc_vals

    return K_mod.tocsr(), f_mod

def solve_kirsch_fem(N, L=12.0, a=1.0, E=210e9, nu=0.3, sigma_inf=100e6):
    P,T,h = generate_mesh_quarter_square_with_hole(L=L, a=a, N=N, rings=3)
    K,D = assemble_elasticity_P1(P,T,E=E,nu=nu,plane_stress=True)

    n = P.shape[0]
    ndof = 2*n
    f = np.zeros(ndof)

    eps_x = sigma_inf/E
    eps_y = -nu*sigma_inf/E

    x = P[:,0]; y = P[:,1]

    bc_dofs=[]; bc_vals=[]

    on_x0 = np.isclose(x,0.0, atol=1e-12)
    for i in np.where(on_x0)[0]:
        bc_dofs.append(2*i); bc_vals.append(0.0)

    on_y0 = np.isclose(y,0.0, atol=1e-12)
    for i in np.where(on_y0)[0]:
        bc_dofs.append(2*i+1); bc_vals.append(0.0)

    on_outer = np.isclose(x,L, atol=1e-12) | np.isclose(y,L, atol=1e-12)
    for i in np.where(on_outer)[0]:
        bc_dofs.extend([2*i,2*i+1])
        bc_vals.extend([eps_x*x[i], eps_y*y[i]])

    d = {}
    for dof,val in zip(bc_dofs, bc_vals):
        d[dof]=val
    bc_dofs = np.array(sorted(d.keys()), dtype=int)
    bc_vals = np.array([d[k] for k in bc_dofs], dtype=float)

    Kc, fc = apply_dirichlet(K, f, bc_dofs, bc_vals)
    u = spsolve(Kc, fc)

    return P,T,h,u,D

def element_stress(P,T,u,D):
    sig = np.zeros((T.shape[0],3))
    cent = P[T].mean(axis=1)

    for e, tri in enumerate(T):
        x = P[tri,0]; y = P[tri,1]
        A = 0.5*((x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))
        A = abs(A)

        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        dNdx = b/(2*A); dNdy = c/(2*A)

        B = np.zeros((3,6))
        for i in range(3):
            B[0,2*i]   = dNdx[i]
            B[1,2*i+1] = dNdy[i]
            B[2,2*i]   = dNdy[i]
            B[2,2*i+1] = dNdx[i]

        dofs = np.array([2*tri[0],2*tri[0]+1,2*tri[1],2*tri[1]+1,2*tri[2],2*tri[2]+1])
        ue = u[dofs]
        eps = B @ ue
        sig[e] = D @ eps

    return cent, sig  # sig = [sigma_xx, sigma_yy, sigma_xy]

def compute_Kt(P,T,u,D,a=1.0,sigma_inf=100e6,h=0.5):
    cent, sig = element_stress(P,T,u,D)
    x=cent[:,0]; y=cent[:,1]
    r=np.sqrt(x*x+y*y)
    theta=np.arctan2(y,x)

    s = np.sin(theta); c = np.cos(theta)
    hoop = sig[:,0]*s*s + sig[:,1]*c*c - 2*sig[:,2]*s*c

    mask = (r >= a) & (r <= a + 1.5*h)
    mask2 = mask & (theta >= (math.pi/2 - 0.25))  # около 90°
    if not np.any(mask2):
        mask2 = mask

    return float(hoop[mask2].max()/sigma_inf)

def run_refinement(N_list, L=12.0, a=1.0, E=210e9, nu=0.3, sigma_inf=100e6):
    rows=[]
    for N in N_list:
        P,T,h,u,D = solve_kirsch_fem(N,L=L,a=a,E=E,nu=nu,sigma_inf=sigma_inf)
        Kt = compute_Kt(P,T,u,D,a=a,sigma_inf=sigma_inf,h=h)
        err = abs(Kt-3.0)
        rows.append((N, len(P), len(T), h, Kt, err))
    return rows
def save_fig(path, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def nodal_stress_area_weighted(P, T, u, D):

    n = P.shape[0]
    sig_sum = np.zeros((n, 3), dtype=float)
    w_sum = np.zeros(n, dtype=float)

    for tri in T:
        x = P[tri, 0]; y = P[tri, 1]
        A = 0.5*((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        A = abs(A)

        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        dNdx = b/(2*A); dNdy = c/(2*A)

        B = np.zeros((3,6))
        for i in range(3):
            B[0,2*i]   = dNdx[i]
            B[1,2*i+1] = dNdy[i]
            B[2,2*i]   = dNdy[i]
            B[2,2*i+1] = dNdx[i]

        dofs = np.array([2*tri[0],2*tri[0]+1, 2*tri[1],2*tri[1]+1, 2*tri[2],2*tri[2]+1])
        ue = u[dofs]
        eps = B @ ue
        sig_e = D @ eps  # [sxx, syy, sxy]

        for node in tri:
            sig_sum[node] += A * sig_e
            w_sum[node] += A

    sigma_node = sig_sum / w_sum[:, None]
    return sigma_node

def save_sigma_xx_convergence(N_list, L=12.0, a=1.0, E=210e9, nu=0.3,
                              sigma_inf=100e6, alpha_deg=45.0,
                              npts=250, out_dir="figs"):

    alpha = np.deg2rad(alpha_deg)
    plt.figure()

    # будем рисовать Кирша по тем r, где FEM действительно определён
    r_kirsch = None
    R_global = None

    # чтобы корректно взять "самую тонкую" сетку, не зависим от порядка N_list
    N_fine = max(N_list)

    for N in N_list:
        P, T, h, u, D = solve_kirsch_fem(N, L=L, a=a, E=E, nu=nu, sigma_inf=sigma_inf)
        sigma_node = nodal_stress_area_weighted(P, T, u, D)

        # дальняя граница по лучу под углом alpha внутри квадранта
        R = min(L/np.cos(alpha), L/np.sin(alpha))
        R_global = R  # одинаково для всех, но оставим явно

        # старт: чуть снаружи отверстия (привязываем к шагу сетки)
        r0 = a + 0.3*h
        r = np.linspace(r0, R, npts)

        x = r*np.cos(alpha)
        y = r*np.sin(alpha)

        tri = mtri.Triangulation(P[:, 0], P[:, 1], T)
        interp_xx = mtri.LinearTriInterpolator(tri, sigma_node[:, 0])
        sxx_f = np.array(interp_xx(x, y), dtype=float)

        # оставляем только точки, которые реально попали в треугольники
        finder = tri.get_trifinder()
        inside = finder(x, y) != -1

        r_plot = r[inside]
        sxx_plot = sxx_f[inside]

        plt.plot(r_plot / a, sxx_plot / sigma_inf, label=f"FEM N={N}")

        # r для Кирша берём с самой тонкой сетки (тогда Кирш стартует там же)
        if N == N_fine:
            r_kirsch = r_plot.copy()

    # если вдруг по какой-то причине r_kirsch не собрался (не должен, но на всякий)
    if r_kirsch is None:
        # fallback: начнём с a*1.001
        r_kirsch = np.linspace(a*1.001, R_global, npts)

    # аналитика Кирша ТОЛЬКО на тех r, где FEM определён (с тонкой сетки)
    srr, stt, srt = kirsch_polar_stress(r_kirsch, alpha, a=a, sigma_inf=sigma_inf)
    sxx_a, _, _ = polar_to_cart_stress(srr, stt, srt, alpha)

    plt.plot(r_kirsch / a, sxx_a / sigma_inf,
             linestyle="--", linewidth=2, label="Kirsch (analytic)")

    plt.xlabel(r"$r/a$")
    plt.ylabel(r"$\sigma_{xx}/\sigma_\infty$")
    plt.title(fr"Convergence of $\sigma_{{xx}}(r)$ at $\alpha={alpha_deg}^\circ$")
    plt.grid(True)
    plt.legend()

    # стартуем с 1.0 на оси (чтобы картинка выглядела нормально),
    # но кривая FEM/Кирша начнётся там, где реально есть данные
    plt.xlim(1.0, (R_global / a) if R_global is not None else None)

    save_fig(f"{out_dir}/sigma_xx_convergence_alpha{alpha_deg:.0f}.png")

def kirsch_polar_stress(r, theta, a=1.0, sigma_inf=100e6):

    s = sigma_inf
    a2 = a*a
    r2 = r*r
    a4 = a2*a2
    r4 = r2*r2

    c2 = np.cos(2*theta)
    s2 = np.sin(2*theta)

    s_rr = 0.5*s*(1 - a2/r2) + 0.5*s*(1 + 3*a4/r4 - 4*a2/r2)*c2
    s_tt = 0.5*s*(1 + a2/r2) - 0.5*s*(1 + 3*a4/r4)*c2
    s_rt = -0.5*s*(1 - 3*a4/r4 + 2*a2/r2)*s2
    return s_rr, s_tt, s_rt

def polar_to_cart_stress(s_rr, s_tt, s_rt, theta):

    c = np.cos(theta); s = np.sin(theta)
    c2 = c*c; s2 = s*s
    cs = s*c

    s_xx = s_rr*c2 - 2*s_rt*cs + s_tt*s2
    s_yy = s_rr*s2 + 2*s_rt*cs + s_tt*c2
    s_xy = (s_rr - s_tt)*cs + s_rt*(c2 - s2)
    return s_xx, s_yy, s_xy

def save_stress_contours(P, T, sigma_node, out_dir="figs", prefix="N96"):
    tri = mtri.Triangulation(P[:,0], P[:,1], T)

    names = [("sxx", 0, r"$\sigma_{xx}$"),
             ("syy", 1, r"$\sigma_{yy}$"),
             ("sxy", 2, r"$\sigma_{xy}$")]

    for tag, j, title in names:
        plt.figure()
        plt.tricontourf(tri, sigma_node[:, j], levels=30)
        plt.gca().set_aspect("equal")
        plt.xlabel("x"); plt.ylabel("y")
        plt.title(f"{title} (smoothed nodal)")
        plt.colorbar()
        save_fig(f"{out_dir}/{prefix}_{tag}_contour.png")

def save_line_comparison(P, T, sigma_node, L=12.0, a=1.0, sigma_inf=100e6,
                         alpha_deg=45.0, npts=200, out_dir="figs", prefix="N96"):
    alpha = np.deg2rad(alpha_deg)

    R = min(L/np.cos(alpha), L/np.sin(alpha))
    r0 = a * 1.001
    r = np.linspace(r0, R, npts)

    x = r*np.cos(alpha)
    y = r*np.sin(alpha)

    tri = mtri.Triangulation(P[:,0], P[:,1], T)
    interp_xx = mtri.LinearTriInterpolator(tri, sigma_node[:,0])
    interp_yy = mtri.LinearTriInterpolator(tri, sigma_node[:,1])
    interp_xy = mtri.LinearTriInterpolator(tri, sigma_node[:,2])

    sxx_f = np.array(interp_xx(x, y), dtype=float)
    syy_f = np.array(interp_yy(x, y), dtype=float)
    sxy_f = np.array(interp_xy(x, y), dtype=float)

    mask = np.isfinite(sxx_f) & np.isfinite(syy_f) & np.isfinite(sxy_f)
    r = r[mask]
    rn = r / a
    sxx_f = sxx_f[mask];
    syy_f = syy_f[mask];
    sxy_f = sxy_f[mask]

    # Аналитику тоже пересчитываем на том же r (важно!)
    srr, stt, srt = kirsch_polar_stress(r, alpha, a=a, sigma_inf=sigma_inf)
    sxx_a, syy_a, sxy_a = polar_to_cart_stress(srr, stt, srt, alpha)

    # analytic along same line (theta=alpha)
    srr, stt, srt = kirsch_polar_stress(r, alpha, a=a, sigma_inf=sigma_inf)
    sxx_a, syy_a, sxy_a = polar_to_cart_stress(srr, stt, srt, alpha)

    rn = r / a
    sxx_f_n = sxx_f / sigma_inf; syy_f_n = syy_f / sigma_inf; sxy_f_n = sxy_f / sigma_inf
    sxx_a_n = sxx_a / sigma_inf; syy_a_n = syy_a / sigma_inf; sxy_a_n = sxy_a / sigma_inf

    def plot_one(yf, ya, ylabel, tag):
        plt.figure()
        plt.plot(rn, ya, linestyle="--", label="Kirsch (analytic)")
        plt.plot(rn, yf, label="FEM (smoothed)")
        plt.xlabel(r"$r/a$")
        plt.ylabel(ylabel)
        plt.title(fr"Line comparison at $\alpha={alpha_deg}^\circ$")
        plt.grid(True)
        plt.legend()
        save_fig(f"{out_dir}/{prefix}_line_alpha{alpha_deg:.0f}_{tag}.png")

    plot_one(sxx_f_n, sxx_a_n, r"$\sigma_{xx}/\sigma_\infty$", "sxx")
    plot_one(syy_f_n, syy_a_n, r"$\sigma_{yy}/\sigma_\infty$", "syy")
    plot_one(sxy_f_n, sxy_a_n, r"$\sigma_{xy}/\sigma_\infty$", "sxy")

    data = np.column_stack([r, rn, sxx_f_n, sxx_a_n, syy_f_n, syy_a_n, sxy_f_n, sxy_a_n])
    header = "r,r_over_a,sxx_fem,sxx_anal,syy_fem,syy_anal,sxy_fem,sxy_anal"
    np.savetxt(f"{out_dir}/{prefix}_line_alpha{alpha_deg:.0f}.csv", data, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    N_vis = 96
    L = 12.0
    a = 0.25
    E = 210e9
    nu = 0.3
    sigma_inf = 100e6

    P, T, h, u, D = solve_kirsch_fem(N_vis, L=L, a=a, E=E, nu=nu, sigma_inf=sigma_inf)
    sigma_node = nodal_stress_area_weighted(P, T, u, D)

    prefix = f"N{N_vis}"
    save_stress_contours(P, T, sigma_node, out_dir="figs", prefix=prefix)

    # линия под углом (можешь поставить 30, 45, 60)
    save_line_comparison(P, T, sigma_node, L=L, a=a, sigma_inf=sigma_inf,
                         alpha_deg=45.0, npts=250, out_dir="figs", prefix=prefix)

    N_list = [96, 192]

    save_sigma_xx_convergence(
        N_list,
        L=12.0,
        a=a,
        alpha_deg=45.0,
        out_dir="figs"
    )

    print("Saved σxx convergence figure.")

    print("Saved figures to ./figs")

