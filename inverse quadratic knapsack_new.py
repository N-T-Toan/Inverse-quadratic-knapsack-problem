"""
INVERSE QUADRATIC KNAPSACK — Optimized + LaTeX export
Requires: numpy>=1.20, gurobipy>=9.5, pandas>=1.4
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd

# -----------------------------
# Global problem size
# -----------------------------
n = 100
R = 100
SEED = 12345
NUM_RUNS = 50

rng = np.random.default_rng(SEED)

# -----------------------------
# Data generation (vectorized)
# -----------------------------
def generate_data(n: int, R: float, rng: np.random.Generator):
    b = rng.uniform(1.0, R, size=n)
    off = R / 10.0
    a = np.clip(rng.uniform(b - off, b + off, size=n), 1.0, b + off)
    d = np.clip(rng.uniform(b - off, b + off, size=n), 1.0, b + off)
    l = rng.uniform(1.0, R, size=n)
    u = l + rng.uniform(R / 100.0, R, size=n)
    h_plus  = rng.uniform(1.0, R, size=n)
    h_minus = rng.uniform(1.0, R, size=n)
    g_plus  = rng.uniform(1.0, R, size=n)
    g_minus = rng.uniform(1.0, R, size=n)
    x_star = rng.uniform(l - off, u + off, size=n)
    x_star = np.clip(x_star, l, u)
    return {
        "a": a, "d": d, "b": b,
        "l": l, "u": u,
        "h_plus": h_plus, "h_minus": h_minus,
        "g_plus": g_plus, "g_minus": g_minus,
        "x_star": x_star
    }

# -----------------------------
# Labels & index sets
# -----------------------------
def determine_labels(oc):
    l = oc["l"]; u = oc["u"]; x = oc["x_star"]
    T = (oc["a"] - x * oc["d"]) / oc["b"]
    eps = 1e-9
    isL = np.isclose(x, l, atol=eps)
    isU = np.isclose(x, u, atol=eps)
    labels = np.full(l.shape, "M", dtype="<U1")
    labels[isL] = "L"; labels[isU] = "U"
    IL = np.flatnonzero(labels == "L")
    IU = np.flatnonzero(labels == "U")
    IM = np.flatnonzero(labels == "M")
    Lvals = T[IL]; Uvals = T[IU]; Mvals = T[IM]
    mins, maxs = [], []
    if Uvals.size: mins.append(Uvals.min())
    if Mvals.size: mins.append(Mvals.min())
    if Lvals.size: maxs.append(Lvals.max())
    if Mvals.size: maxs.append(Mvals.max())
    z1 = np.min(mins) if mins else -np.inf
    z2 = np.max(maxs) if maxs else np.inf
    delta = z2 - z1 if np.isfinite(z1) and np.isfinite(z2) else R
    return {
        "labels": labels,
        "IL": IL, "IU": IU, "IM": IM,
        "L": Lvals, "U": Uvals, "M": Mvals,
        "T": T, "delta": float(delta)
    }

# -----------------------------
# Upper bounds
# -----------------------------
def upper_bounds(oi, rng: np.random.Generator):
    delta = max(oi["delta"], 1e-9)
    n = oi["T"].size
    lo, hi = 0.5*delta, 1.0*delta
    return {
        "lbd_bar":        rng.uniform(lo, hi, size=n),
        "mu_bar":         rng.uniform(lo, hi, size=n),
        "lbd_bar_prime":  rng.uniform(lo, hi, size=n),
        "mu_bar_prime":   rng.uniform(lo, hi, size=n),
    }

# -----------------------------
# Alpha / Beta
# -----------------------------
def calculate_alpha_beta(oi, ub):
    def _max_or_neg_inf(arr): return np.max(arr) if arr.size else -np.inf
    def _min_or_pos_inf(arr): return np.min(arr) if arr.size else np.inf
    L = oi["L"]; M = oi["M"]; U = oi["U"]
    IL = oi["IL"]; IM = oi["IM"]; IU = oi["IU"]
    mu  = ub["mu_bar"]; mu_p = ub["mu_bar_prime"]
    lb  = ub["lbd_bar"]; lb_p = ub["lbd_bar_prime"]
    a1 = _max_or_neg_inf(L - mu[IL] - lb_p[IL]) if IL.size else -np.inf
    a2 = _max_or_neg_inf(M - mu[IM] - lb_p[IM]) if IM.size else -np.inf
    alpha = max(a1, a2)
    b1 = _min_or_pos_inf(U + mu_p[IU] + lb[IU]) if IU.size else np.inf
    b2 = _min_or_pos_inf(M + mu_p[IM] + lb[IM]) if IM.size else np.inf
    beta = min(b1, b2)
    if not np.isfinite(alpha): alpha = -1e6
    if not np.isfinite(beta):  beta  =  1e6
    if beta < alpha: beta, alpha = alpha + 1.0, alpha
    return {"alpha": float(alpha), "beta": float(beta)}

# -----------------------------
# Build compact LP (only needed rows)
# -----------------------------
def get_data_for_lp(oc, oi, ub, ab):
    h_plus, h_minus = oc['h_plus'], oc['h_minus']
    g_plus, g_minus = oc['g_plus'], oc['g_minus']
    lb  = ub['lbd_bar']; mu  = ub['mu_bar']
    lbp = ub['lbd_bar_prime']; mup = ub['mu_bar_prime']
    IL, IU, IM = oi['IL'], oi['IU'], oi['IM']
    Lvals, Uvals, Mvals = oi['L'], oi['U'], oi['M']
    alpha, beta = ab['alpha'], ab['beta']

    c = np.concatenate([h_plus, h_minus, g_plus, g_minus, np.array([0.0])])
    ubounds = np.concatenate([lb, mu, lbp, mup, np.array([beta])])
    lbounds = np.concatenate([np.zeros(4*n), np.array([alpha])])

    m_up = IL.size + IU.size
    A_up = np.zeros((m_up, 4*n + 1))
    b_up = np.zeros(m_up)

    if IL.size:
        rows = np.arange(IL.size); i = IL
        A_up[rows, -1] = -1.0; A_up[rows, n + i] = -1.0; A_up[rows, 2*n + i] = -1.0
        b_up[rows] = -Lvals
    if IU.size:
        base = IL.size; rows = base + np.arange(IU.size); i = IU
        A_up[rows, -1] =  1.0; A_up[rows, i] = -1.0; A_up[rows, 3*n + i] = -1.0
        b_up[rows] =  Uvals

    A_eq = np.zeros((IM.size, 4*n + 1))
    b_eq = np.zeros(IM.size)
    if IM.size:
        rows = np.arange(IM.size); i = IM
        A_eq[rows, -1] = -1.0; A_eq[rows, i] = 1.0; A_eq[rows, n + i] = -1.0
        A_eq[rows, 2*n + i] = -1.0; A_eq[rows, 3*n + i] = 1.0
        b_eq[:] = -Mvals

    return dict(coef=c, A_up=A_up, b_up=b_up, A_eq=A_eq, b_eq=b_eq,
                lbounds=lbounds, ubounds=ubounds)

# -----------------------------
# Solve LP (fast, compact)
# -----------------------------
def solve_LP(lp):
    m = gp.Model("IQKP_LP")
    m.Params.OutputFlag = 0
    m.Params.Threads = 0
    m.Params.Presolve = 2

    x = m.addMVar(shape=lp["coef"].size, lb=lp["lbounds"], ub=lp["ubounds"], name="x")
    m.setObjective(lp["coef"] @ x, GRB.MINIMIZE)

    if lp["A_up"].shape[0] > 0:
        # Thêm ràng buộc bất đẳng thức
        m.addConstrs(
            (lp["A_up"][i] @ x <= lp["b_up"][i] for i in range(lp["A_up"].shape[0])),
            name="ineq"
        )

    if lp["A_eq"].shape[0] > 0:
        # Thêm ràng buộc đẳng thức
        m.addConstrs(
            (lp["A_eq"][i] @ x == lp["b_eq"][i] for i in range(lp["A_eq"].shape[0])),
            name="eq"
        )

    m.optimize()
    return x.X, float(m.objVal)


# -----------------------------
# Algorithm 1 (vectorized-ish)
# -----------------------------
def solve_OUP(oc, oi, ub, ab):
    n = oc["a"].size
    T = oi["T"]; IL, IU, IM = oi["IL"], oi["IU"], oi["IM"]
    h_p, h_m = oc["h_plus"], oc["h_minus"]
    g_p, g_m = oc["g_plus"], oc["g_minus"]
    lb  = ub["lbd_bar"];  mu  = ub["mu_bar"]
    lbp = ub["lbd_bar_prime"]; mup = ub["mu_bar_prime"]
    alpha, beta = ab["alpha"], ab["beta"]

    x_bar = np.where(h_m <= g_p, mu, lbp)
    y_bar = np.where(h_p <= g_m, lb, mup)
    tL = T - x_bar; tU = T + y_bar

    cand = []
    if oi["L"].size: cand.append(oi["L"])
    if oi["U"].size: cand.append(oi["U"])
    if oi["M"].size: cand.append(oi["M"])
    cand += [tL[IL], tU[IU], tL[IM], tU[IM]]
    if alpha <= beta: cand.append(np.array([alpha, beta]))
    bp = np.unique(np.concatenate([c for c in cand if c.size > 0]))
    bp = bp[(bp >= alpha) & (bp <= beta)]
    if bp.size == 0: bp = np.array([alpha, beta])

    phi_L = T - x_bar; phi_U = T + y_bar

    def SL(i, t):
        return np.where(t < phi_L[i], -np.maximum(h_m[i], g_p[i]), -np.minimum(h_m[i], g_p[i]))
    def SU(i, t):
        return np.where(t <= phi_U[i], np.minimum(h_p[i], g_m[i]),  np.maximum(h_p[i], g_m[i]))

    def slope_at(t):
        mask_U = (T[IU] < t)
        sU = SU(IU, t) if IU.size else 0.0
        sU = sU[mask_U].sum() if np.ndim(sU) else sU
        mask_L = (T[IL] > t)
        sL = SL(IL, t) if IL.size else 0.0
        sL = sL[mask_L].sum() if np.ndim(sL) else sL
        mask_Ml = (T[IM] < t)
        sMl = SU(IM, t) if IM.size else 0.0
        sMl = sMl[mask_Ml].sum() if np.ndim(sMl) else sMl
        mask_Mg = (T[IM] > t)
        sMg = SL(IM, t) if IM.size else 0.0
        sMg = sMg[mask_Mg].sum() if np.ndim(sMg) else sMg
        return float(sU + sL + sMl + sMg)

    lo, hi = 0, bp.size - 1
    def s(idx): return slope_at(bp[idx] + 1e-8)
    while lo < hi:
        mid = (lo + hi) // 2
        if s(mid) < 0: lo = mid + 1
        else: hi = mid
    t_opt = float(bp[lo])

    lbd = np.zeros(n); muv = np.zeros(n); lbpv = np.zeros(n); mupv = np.zeros(n)

    for idx in np.concatenate([IU[T[IU] < t_opt], IM[T[IM] < t_opt]]):
        if t_opt > T[idx] + y_bar[idx]:
            if h_p[idx] <= g_m[idx]:
                lbd[idx] = lb[idx]; mupv[idx] = t_opt - T[idx] - lbd[idx]
            else:
                mupv[idx] = mup[idx]; lbd[idx] = t_opt - T[idx] - mupv[idx]
        else:
            if h_p[idx] <= g_m[idx]:
                lbd[idx] = t_opt - T[idx]; mupv[idx] = 0.0
            else:
                mupv[idx] = t_opt - T[idx]; lbd[idx] = 0.0

    for idx in np.concatenate([IL[T[IL] > t_opt], IM[T[IM] > t_opt]]):
        if t_opt < T[idx] - x_bar[idx]:
            if h_m[idx] <= g_p[idx]:
                muv[idx] = mu[idx]; lbpv[idx] = T[idx] - t_opt - muv[idx]
            else:
                lbpv[idx] = lbp[idx]; muv[idx] = T[idx] - t_opt - lbpv[idx]
        else:
            if h_m[idx] <= g_p[idx]:
                muv[idx] = T[idx] - t_opt; lbpv[idx] = 0.0
            else:
                lbpv[idx] = T[idx] - t_opt; muv[idx] = 0.0

    obj = float(h_p @ lbd + h_m @ muv + g_p @ lbpv + g_m @ mupv)
    return lbd, muv, lbpv, mupv, obj, t_opt

# -----------------------------
# Build instance once
# -----------------------------
oc = generate_data(n, R, rng)
oi = determine_labels(oc)
ub = upper_bounds(oi, rng)
ab = calculate_alpha_beta(oi, ub)
lp_data = get_data_for_lp(oc, oi, ub, ab)

# -----------------------------
# One solve (for sanity check)
# -----------------------------
t0 = time.perf_counter()
x_star_lp, obj_lp = solve_LP(lp_data)
t_lp_once = time.perf_counter() - t0
lbd_lp = x_star_lp[:n]; mu_lp = x_star_lp[n:2*n]
lbp_lp = x_star_lp[2*n:3*n]; mup_lp = x_star_lp[3*n:4*n]; t_var = x_star_lp[-1]

t0 = time.perf_counter()
lbd, muv, lbpv, mupv, obj_alg, t_opt = solve_OUP(oc, oi, ub, ab)
t_alg_once = time.perf_counter() - t0

print("Sanity — LP obj:", obj_lp, "Alg obj:", obj_alg)

# -----------------------------
runs = []
bp_counts = []  # lưu số breakpoint mỗi vòng

for k in range(1, NUM_RUNS + 1):
    # LP
    t0 = time.perf_counter()
    _, obj_lp_k = solve_LP(lp_data)
    t_lp_k = time.perf_counter() - t0

    # Algorithm
    t0 = time.perf_counter()
    _, _, _, _, obj_alg_k, _= solve_OUP(oc, oi, ub, ab)
    t_alg_k = time.perf_counter() - t0

    rel_gap_k = abs(obj_alg_k - obj_lp_k) / (abs(obj_lp_k) + 1e-12)
    runs.append((t_lp_k, t_alg_k, obj_lp_k, obj_alg_k, rel_gap_k))

    #bp_counts.append(bp_count)  # thêm breakpoint vào danh sách

df_all = pd.DataFrame(runs, columns=["time_LP", "time_Alg", "obj_LP", "obj_Alg", "rel_gap"])

print("\n==== Summary ====")
print(f"LP time:   min={df_all['time_LP'].min():.6f}, "
      f"avg={df_all['time_LP'].mean():.6f}, "
      f"max={df_all['time_LP'].max():.6f}")

print(f"Alg time:  min={df_all['time_Alg'].min():.6f}, "
      f"avg={df_all['time_Alg'].mean():.6f}, "
      f"max={df_all['time_Alg'].max():.6f}")

#print(f"Breakpoints considered by algorithm: min={min(bp_counts)}, "
 #     f"avg={sum(bp_counts) / len(bp_counts):.2f}, "
  #    f"max={max(bp_counts)}")


