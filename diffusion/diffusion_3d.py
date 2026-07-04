"""
Refined version of diffusion.py, done together with Claude Opus (Anthropic).
The physics is the same as diffusion.py; this file additionally packages the
solver as a function, sweeps two parameter sets, and generates the 3D and
slice comparison plots used in the report.

Diffusion-equation solver for the Black-Scholes call, plus comparison
plots against the analytical Black-Scholes formula.

Substitutions used (heat-equation reduction):
    S = e^y,  t = T - tau,  u = e^{r tau} C,  x = y + (r - sigma^2/2) tau
=>  u_tau = (sigma^2 / 2) u_xx,
    u(x, 0) = max(e^x - X, 0),
    u(x_min, tau) = 0,
    u(x_max, tau) ~ e^{x_max + sigma^2 tau / 2} - X.

Strang splitting of H = A + B, advancing one sub-step of size tau:
    e^{alpha H tau} ~ e^{alpha A tau/2} e^{alpha B tau} e^{alpha A tau/2}.
Per 2x2 block X = [[1,-1],[-1,1]],  e^{beta X} = 1/2 [[1+e^{2 beta}, 1-e^{2 beta}],
                                                     [1-e^{2 beta}, 1+e^{2 beta}]].
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))


def bs_call_scalar(S, X, r, tau, sigma):
    if tau <= 0:
        return max(S - X, 0.0)
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm_cdf(d1) - X * np.exp(-r * tau) * norm_cdf(d2)


def bs_call_vec(S, X, r, tau, sigma):
    """Vectorised B-S call on arrays S and tau (broadcasted)."""
    S = np.asarray(S)
    tau = np.asarray(tau)
    out = np.empty(np.broadcast(S, tau).shape)
    flat_S = np.broadcast_to(S, out.shape).ravel()
    flat_t = np.broadcast_to(tau, out.shape).ravel()
    flat_o = out.ravel()
    for i, (Si, ti) in enumerate(zip(flat_S, flat_t)):
        flat_o[i] = bs_call_scalar(Si, X, r, ti, sigma)
    return out


def solve_diffusion(T, sigma, r, X, S_min, S_max, L=801, tau_step=2e-4,
                    n_snap=80):
    D = 0.5 * sigma**2
    x_min = np.log(S_min)
    x_max = np.log(S_max)
    x = np.linspace(x_min, x_max, L)
    delta = (x_max - x_min) / (L - 1)

    m = int(round(T / tau_step))
    tau_step = T / m  # snap to exact divisor of T

    alpha = -D / delta**2
    beta = alpha * tau_step

    Ma = 0.5 * np.array([[1 + np.exp(beta),    1 - np.exp(beta)],
                         [1 - np.exp(beta),    1 + np.exp(beta)]])
    Mb = 0.5 * np.array([[1 + np.exp(2 * beta), 1 - np.exp(2 * beta)],
                         [1 - np.exp(2 * beta), 1 + np.exp(2 * beta)]])

    def mulA(v):
        o = v.copy()
        o[-1] *= np.exp(beta)
        for i in range(L // 2):
            o[i*2:i*2 + 2] = Ma @ o[i*2:i*2 + 2]
        return o

    def mulB(v):
        o = v.copy()
        o[0] *= np.exp(2 * beta)
        for i in range(L // 2):
            o[i*2 + 1:i*2 + 3] = Mb @ o[i*2 + 1:i*2 + 3]
        return o

    u = np.maximum(np.exp(x) - X, 0.0)
    snap_i = set(np.unique(np.linspace(0, m, n_snap).astype(int)).tolist())
    snaps, taus = [u.copy()], [0.0]

    for j in range(m):
        u = mulA(mulB(mulA(u)))
        tau_now = (j + 1) * tau_step
        u[0] = 0.0
        u[-1] = np.exp(x[-1] + 0.5 * sigma**2 * tau_now) - X
        if (j + 1) in snap_i:
            snaps.append(u.copy())
            taus.append(tau_now)

    return x, np.array(taus), np.array(snaps)


def numerical_call_surface(x, taus, u, r, sigma):
    """Convert u(x, tau) snapshots into C(S, t) on a (S, tau) grid."""
    X2, T2 = np.meshgrid(x, taus)
    S2 = np.exp(X2 - (r - 0.5 * sigma**2) * T2)
    C2 = np.exp(-r * T2) * u
    return S2, T2, C2


# ============================================================
# Case 1: illustrative parameters T=1, sigma=0.3, r=0.05, X=100
# ============================================================
P1 = dict(T=1.0, sigma=0.30, r=0.05, X=100.0,
          S_min=20.0, S_max=400.0, L=801, tau_step=5e-4)
x1, t1, u1 = solve_diffusion(**P1)
S1, T1, Cnum1 = numerical_call_surface(x1, t1, u1, P1["r"], P1["sigma"])

# Analytical surface on the same (S, tau) grid
Can1 = bs_call_vec(S1, P1["X"], P1["r"], T1, P1["sigma"])

# Mask to a nice display window
mask1 = (S1[0] >= 30) & (S1[0] <= 220)


# ============================================================
# Side-by-side: numerical vs analytical surfaces
# ============================================================
fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
norm = plt.Normalize(vmin=0, vmax=max(Cnum1[:, mask1].max(),
                                      Can1[:, mask1].max()))
ax1.plot_surface(S1[:, mask1], T1[:, mask1], Cnum1[:, mask1],
                 cmap="viridis", norm=norm, linewidth=0,
                 antialiased=True, rstride=2, cstride=4)
ax1.set_xlabel(r"$S$", labelpad=6)
ax1.set_ylabel(r"$\tau$", labelpad=6)
ax1.set_zlabel(r"$C(S,t)$", labelpad=6)
ax1.set_title("Numerical diffusion solver", pad=8)
ax1.view_init(elev=22, azim=-130)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(S1[:, mask1], T1[:, mask1], Can1[:, mask1],
                 cmap="viridis", norm=norm, linewidth=0,
                 antialiased=True, rstride=2, cstride=4)
ax2.set_xlabel(r"$S$", labelpad=6)
ax2.set_ylabel(r"$\tau$", labelpad=6)
ax2.set_zlabel(r"$C(S,t)$", labelpad=6)
ax2.set_title("Analytical Black-Scholes", pad=8)
ax2.view_init(elev=22, azim=-130)

fig.suptitle(r"Call surface: $T=1$, $\sigma=0.30$, $r=0.05$, $X=100$",
             y=0.98, fontsize=13)
plt.tight_layout()
plt.savefig("diffusion_compare.png", dpi=140, bbox_inches="tight")


# ============================================================
# Overlay: analytical surface + numerical scatter
# ============================================================
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(S1[:, mask1], T1[:, mask1], Can1[:, mask1],
                cmap="viridis", alpha=0.85, linewidth=0,
                antialiased=True, rstride=2, cstride=4,
                edgecolor="none")
# numerical samples on a coarser subgrid
ii = np.arange(0, mask1.sum(), 12)
jj = np.arange(0, len(t1), 4)
Si = S1[np.ix_(jj, np.where(mask1)[0][ii])]
Ti = T1[np.ix_(jj, np.where(mask1)[0][ii])]
Ci = Cnum1[np.ix_(jj, np.where(mask1)[0][ii])]
ax.scatter(Si.ravel(), Ti.ravel(), Ci.ravel(),
           s=14, c="crimson", depthshade=False,
           label="numerical samples")
ax.set_xlabel(r"$S$", labelpad=6)
ax.set_ylabel(r"$\tau = T-t$", labelpad=6)
ax.set_zlabel(r"$C(S,t)$", labelpad=6)
ax.set_title(r"Diffusion solver overlaid on analytical surface"
             "\n"
             r"$T=1$, $\sigma=0.30$, $r=0.05$, $X=100$", pad=10)
ax.view_init(elev=22, azim=-125)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("diffusion_overlay.png", dpi=140, bbox_inches="tight")


# ============================================================
# Error surface (numerical - analytical)
# ============================================================
err1 = Cnum1 - Can1
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")
amax = np.max(np.abs(err1[:, mask1]))
err_surf = ax.plot_surface(S1[:, mask1], T1[:, mask1], err1[:, mask1],
                           cmap="coolwarm",
                           vmin=-amax, vmax=amax,
                           linewidth=0, antialiased=True,
                           rstride=2, cstride=4)
ax.set_xlabel(r"$S$", labelpad=6)
ax.set_ylabel(r"$\tau$", labelpad=6)
ax.set_zlabel(r"$C_{\mathrm{num}}-C_{\mathrm{BS}}$", labelpad=6)
ax.set_title(r"Pointwise error of the diffusion solver"
             "\n"
             r"$T=1$, $\sigma=0.30$, $r=0.05$, $X=100$", pad=10)
fig.colorbar(err_surf, shrink=0.55, aspect=15, pad=0.1)
ax.view_init(elev=22, azim=-125)
plt.tight_layout()
plt.savefig("diffusion_error.png", dpi=140, bbox_inches="tight")


# ============================================================
# Slices at three values of tau, numerical vs analytical
# ============================================================
slice_taus = [0.0, 0.25, 1.0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, tslc in zip(axes, slice_taus):
    j = int(np.argmin(np.abs(t1 - tslc)))
    Sline = S1[j, mask1]
    Cnumline = Cnum1[j, mask1]
    Canline = Can1[j, mask1]
    ax.plot(Sline, Canline, "k-", lw=2.0, label="analytical")
    ax.plot(Sline, Cnumline, "r--", lw=1.6, label="numerical")
    ax.axvline(P1["X"], color="gray", lw=0.7, ls=":")
    ax.set_title(rf"$\tau={t1[j]:.3f}$")
    ax.set_xlabel(r"$S$")
    ax.grid(alpha=0.4)
axes[0].set_ylabel(r"$C(S,t)$")
axes[0].legend(loc="upper left")
fig.suptitle(r"Diffusion solver vs Black-Scholes at fixed $\tau$"
             r" ($T=1$, $\sigma=0.30$, $r=0.05$, $X=100$)",
             y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig("diffusion_slices.png", dpi=140, bbox_inches="tight")


# ============================================================
# Case 2: BKX parameters (for the existing report figure)
# ============================================================
P2 = dict(T=17/365, sigma=0.1586, r=0.01, X=37.5,
          S_min=0.5 * 37.5, S_max=10 * 37.5,
          L=1001, tau_step=1e-4)
x2, t2, u2 = solve_diffusion(**P2)
S2, T2, Cnum2 = numerical_call_surface(x2, t2, u2, P2["r"], P2["sigma"])

S0 = 52.93
analytical_BKX = bs_call_scalar(S0, P2["X"], P2["r"], P2["T"], P2["sigma"])
S_at_T = np.exp(x2 - (P2["r"] - 0.5 * P2["sigma"]**2) * P2["T"])
C_at_T = np.exp(-P2["r"] * P2["T"]) * u2[-1]
idx_S0 = int(np.argmin(np.abs(S_at_T - S0)))
Cnum_at_S0 = C_at_T[idx_S0]


# ============================================================
# BKX 2D figure: numerical curve + analytical curve
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.5))
mask_BKX = (S_at_T >= 15) & (S_at_T <= 150)
Sline = S_at_T[mask_BKX]
Cline = C_at_T[mask_BKX]
Caline = np.array([bs_call_scalar(s, P2["X"], P2["r"], P2["T"], P2["sigma"])
                   for s in Sline])
ax.plot(Sline, Caline, "k-", lw=2.0, label="analytical Black--Scholes")
ax.plot(Sline, Cline, "r--", lw=1.7, label="numerical diffusion solver")
ax.axvline(S0, color="green", lw=0.9, ls=":",
           label=fr"$S_0={S0}$")
ax.scatter([S0], [Cnum_at_S0], color="red", zorder=5,
           label=fr"numerical $C(S_0)={Cnum_at_S0:.3f}$")
ax.scatter([S0], [analytical_BKX], color="black", zorder=5,
           label=fr"analytical $C(S_0)={analytical_BKX:.3f}$")
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$C(S,0)$")
ax.set_title(r"BKX European call: diffusion solver vs analytical"
             "\n"
             r"$T=17/365$, $\sigma=0.1586$, $r=0.01$, $X=37.5$")
ax.grid(alpha=0.4)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("diffusion_BKX.png", dpi=140, bbox_inches="tight")


print(f"BKX numerical C(S0={S0}) = {Cnum_at_S0:.4f}")
print(f"BKX analytical C(S0={S0}) = {analytical_BKX:.4f}")
print(f"relative error          = {abs(Cnum_at_S0-analytical_BKX)/analytical_BKX*100:.2f}%")
print("\nSaved figures:")
for name in ("diffusion_compare", "diffusion_overlay", "diffusion_error",
             "diffusion_slices", "diffusion_BKX"):
    print(f"  {name}.png")
