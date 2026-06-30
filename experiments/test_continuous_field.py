"""
experiments/test_continuous_field.py
Consistency verification for the continuous (kernel-integral / DeepONet) ionization
field against the existing grid pipeline (scatter -> fft_convolve_3d -> equilibrium).

Claim under test
----------------
The continuous field  x_HII(x) = equilibrium( [ SUM_i w_i K(|x-x_i|) + (S_unres*K)(x) ] / J_ref )
reproduces the grid pipeline when queried at voxel centres, up to:
  * FFT numerical precision               (on-grid sources)  -> Tests 1, 2
  * voxel-scale trilinear source smoothing (off-grid sources) -> Test 3 (the
    discretization error the continuous form removes; agreement still high).

Run:
    cd LAE_pinn && python experiments/test_continuous_field.py
"""

from __future__ import annotations
import os
import sys
import math

import torch

# Make the LAE_pinn package importable as top-level `physics`
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physics.kernels import MixtureKernel, make_3d_kernel_grid           # noqa: E402
from physics.scatter import scatter_to_grid, fft_convolve_3d            # noqa: E402
from physics.excursion_set import ExcursionSetMapping                   # noqa: E402
from physics.continuous_field import (                                  # noqa: E402
    ContinuousIonizationField,
    grid_centre_coords,
    ionization_front_gradient,
    sampled_field_mse,
)

torch.manual_seed(0)
DEVICE = "cpu"
BOX = 160.0          # cMpc/h
G = 48               # working grid (small for a fast self-contained test)
DT = torch.float64   # double precision so FFT-vs-analytic agreement is unambiguous


# ------------------------------------------------------------------ #
#  Shared physics objects (same modules the model uses)
# ------------------------------------------------------------------ #

def make_physics():
    kernel = MixtureKernel(R_init=8.0, delta_init=2.0, lambda_mfp_init=15.0).to(DT)
    excursion = ExcursionSetMapping(alpha_nH_scale_init=1.3, learnable=True).to(DT)
    cfield = ContinuousIonizationField(
        kernel=kernel, excursion=excursion, box_size=BOX, grid_size_ref=G,
    ).to(DT)
    return kernel, excursion, cfield


def grid_pipeline(pos, w, kernel, excursion, A_obs=1.0, S_unres=None):
    """Replicate LAEPINN.forward (equilibrium branch) on the grid."""
    S_obs = scatter_to_grid(pos, w, G)                 # (G,G,G)
    S_emiss = A_obs * S_obs + (S_unres if S_unres is not None else 0.0)
    kgrid = make_3d_kernel_grid(kernel, G, BOX, DEVICE).to(DT)
    J_total = fft_convolve_3d(S_emiss, kgrid)
    J_ref = J_total.mean().clamp(min=1e-12)
    x = excursion(J_total / J_ref)
    return x, J_total, J_ref, kgrid


def metrics(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    rel_l2 = ((a - b).norm() / (b.norm() + 1e-12)).item()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return max_abs, mean_abs, rel_l2, corr


PASS = True
def check(name, ok, detail):
    global PASS
    PASS = PASS and ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ------------------------------------------------------------------ #
#  Test 1 — on-grid sources, observed-only  (pure kernel-sum vs FFT)
# ------------------------------------------------------------------ #

def test1_on_grid_obs_only():
    print("\nTest 1: on-grid sources, observed-only (kernel-sum vs FFT conv)")
    kernel, excursion, cfield = make_physics()
    N = 150
    vox = torch.randint(0, G, (N, 3), device=DEVICE)
    pos = (vox.to(DT)) / G                         # exactly on voxel centres
    w = torch.rand(N, dtype=DT) + 0.1

    x_grid, _, J_ref, _ = grid_pipeline(pos, w, kernel, excursion)
    x_cont = cfield.evaluate_on_grid(G, pos, w, J_unres_grid=None, J_ref=J_ref)

    ma, mn, rl, cc = metrics(x_cont, x_grid)
    # The continuous field is float64-exact, but the GRID reference is only
    # float32-accurate: physics/scatter.fft_convolve_3d casts its inputs with
    # `.float()`, so the FFT accumulates ~sqrt(G^3)*eps_fp32 ~ 1e-5 noise on J.
    # Hence max|Δ| ~ 1e-5 here is the reference's fp32 floor, NOT a model error.
    # relL2 / corr are the fp32-robust agreement metrics (an exact fp64 match
    # would require dropping the `.float()` cast in fft_convolve_3d).
    check("on-grid x_HII match", rl < 1e-4 and cc > 0.99999,
          f"max|Δ|={ma:.2e}, mean|Δ|={mn:.2e}, relL2={rl:.2e}, corr={cc:.6f} (ref=fp32)")


# ------------------------------------------------------------------ #
#  Test 2 — on-grid sources + HOD unresolved field (diffuse path)
# ------------------------------------------------------------------ #

def test2_on_grid_with_unresolved():
    print("\nTest 2: on-grid sources + unresolved diffuse field (linearity + interp)")
    kernel, excursion, cfield = make_physics()
    N = 150
    vox = torch.randint(0, G, (N, 3), device=DEVICE)
    pos = vox.to(DT) / G
    w = torch.rand(N, dtype=DT) + 0.1
    A_obs = 0.7

    # A smooth unresolved source density (mimics Σ_b f_esc_b ε_b(x))
    xs = torch.linspace(0, 2 * math.pi, G, dtype=DT)
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
    S_unres = 1.0 + 0.5 * torch.sin(gx) * torch.cos(gy) * torch.sin(gz)

    x_grid, _, J_ref, kgrid = grid_pipeline(pos, w, kernel, excursion,
                                            A_obs=A_obs, S_unres=S_unres)
    J_unres_grid = fft_convolve_3d(S_unres, kgrid)         # unresolved part (linearity)
    x_cont = cfield.evaluate_on_grid(
        G, pos, A_obs * w, J_unres_grid=J_unres_grid, J_ref=J_ref,
    )

    ma, mn, rl, cc = metrics(x_cont, x_grid)
    # Same fp32 reference floor as Test 1 (fft_convolve_3d casts to float32).
    check("on-grid + diffuse x_HII match", rl < 1e-4 and cc > 0.99999,
          f"max|Δ|={ma:.2e}, mean|Δ|={mn:.2e}, relL2={rl:.2e}, corr={cc:.6f} (ref=fp32)")


# ------------------------------------------------------------------ #
#  Test 3 — off-grid random sources (only difference = voxel smoothing)
# ------------------------------------------------------------------ #

def test3_off_grid():
    print("\nTest 3: off-grid random sources (continuous removes voxel smoothing)")
    kernel, excursion, cfield = make_physics()
    N = 300
    pos = torch.rand(N, 3, dtype=DT)               # arbitrary sub-voxel positions
    w = torch.rand(N, dtype=DT) + 0.1

    x_grid, _, J_ref, _ = grid_pipeline(pos, w, kernel, excursion)
    x_cont = cfield.evaluate_on_grid(G, pos, w, J_unres_grid=None, J_ref=J_ref)

    ma, mn, rl, cc = metrics(x_cont, x_grid)
    # Off-grid: agreement should remain high; residual = sub-voxel scatter smoothing.
    check("off-grid x_HII agreement", (cc > 0.97 and rl < 0.15),
          f"max|Δ|={ma:.2e}, mean|Δ|={mn:.3e}, relL2={rl:.3f}, corr={cc:.4f}")


# ------------------------------------------------------------------ #
#  Test 4 — resolution-freeness + exact autodiff front gradient
# ------------------------------------------------------------------ #

def test4_superres_and_gradient():
    print("\nTest 4: super-resolution rendering + autodiff front gradient")
    kernel, excursion, cfield = make_physics()
    N = 200
    pos = torch.rand(N, 3, dtype=DT)
    w = torch.rand(N, dtype=DT) + 0.1
    _, _, J_ref, _ = grid_pipeline(pos, w, kernel, excursion)

    # Render at 2x resolution from the SAME sources — no retraining, no re-grid.
    x_hi = cfield.evaluate_on_grid(2 * G, pos, w, J_ref=J_ref)
    ok_shape = tuple(x_hi.shape) == (2 * G, 2 * G, 2 * G)
    ok_range = bool((x_hi >= 0).all() and (x_hi <= 1).all() and torch.isfinite(x_hi).all())
    check("super-resolution render", ok_shape and ok_range,
          f"shape={tuple(x_hi.shape)}, range=[{x_hi.min():.3f},{x_hi.max():.3f}]")

    # Exact spatial gradient |∇x_HII| via autodiff (impossible on the grid without FD)
    ctx = {"src_pos": pos, "src_w": w, "J_unres_grid": None, "J_ref": J_ref}
    q = torch.rand(256, 3, dtype=DT)
    gmag = ionization_front_gradient(cfield, ctx, q)
    ok_grad = bool(torch.isfinite(gmag).all() and (gmag >= 0).all())
    check("autodiff front gradient", ok_grad,
          f"|∇x| mean={gmag.mean():.4f}, max={gmag.max():.4f} per cMpc/h")


def test5_gradient_flow():
    """Autograd must reach the GNN factors (src_w), kernel params and alpha_nH_scale."""
    print("\nTest 5: gradient flow to factors / kernel / alpha_nH_scale")
    kernel, excursion, cfield = make_physics()
    N = 120
    pos = torch.rand(N, 3, dtype=DT)
    w = (torch.rand(N, dtype=DT) + 0.1).requires_grad_(True)   # GNN factors proxy

    # a fake "true" field and the resolution-free sampled data loss
    x_true = torch.rand(G, G, G, dtype=DT)
    ctx = {"src_pos": pos, "src_w": w, "J_unres_grid": None, "J_ref": None}
    loss = sampled_field_mse(cfield, ctx, x_true, n_samples=2048)
    loss.backward()

    g_w = (w.grad is not None) and float(w.grad.abs().sum()) > 0
    g_R = kernel.bubble._R_raw.grad is not None and float(kernel.bubble._R_raw.grad.abs()) > 0
    g_lam = kernel.mfp._lambda_raw.grad is not None and float(kernel.mfp._lambda_raw.grad.abs()) > 0
    g_s = excursion._scale_raw.grad is not None and float(excursion._scale_raw.grad.abs()) > 0
    check("grad → factors w_i", g_w, f"sum|∂L/∂w|={float(w.grad.abs().sum()):.3e}")
    check("grad → kernel R, lambda", g_R and g_lam,
          f"|∂L/∂R_raw|={float(kernel.bubble._R_raw.grad.abs()):.3e}, "
          f"|∂L/∂lam_raw|={float(kernel.mfp._lambda_raw.grad.abs()):.3e}")
    check("grad → alpha_nH_scale", g_s, f"|∂L/∂s_raw|={float(excursion._scale_raw.grad.abs()):.3e}")


def test6_pinn_integration():
    """End-to-end: LAEPINN.forward with continuous vs grid generator agree on-grid."""
    print("\nTest 6: LAEPINN integration — continuous vs grid generator (on-grid)")
    try:
        from torch_geometric.data import Data
        from models.pinn import LAEPINN
    except Exception as e:  # torch_geometric not installed
        print(f"  [SKIP] torch_geometric/model unavailable ({type(e).__name__}: {e})")
        return

    torch.manual_seed(3)
    Gi, N, nb = 16, 60, 2
    pos = torch.randint(0, Gi, (N, 3)).to(DT) / Gi        # on-grid sources
    xfeat = torch.rand(N, 8, dtype=DT)
    src_w = torch.rand(N, dtype=DT) + 0.1

    # simple periodic kNN graph (k=6) from positions
    d = pos[:, None, :] - pos[None, :, :]
    d = d - d.round()
    dist = d.norm(dim=-1) + torch.eye(N, dtype=DT) * 1e3
    k = 6
    nbr = dist.topk(k, largest=False).indices                       # (N, k)
    dst = torch.arange(N).repeat_interleave(k)
    src = nbr.reshape(-1)
    ea = pos[src] - pos[dst]; ea = ea - ea.round()
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.cat([ea, ea.norm(dim=-1, keepdim=True)], dim=-1).to(DT)

    graph = Data(x=xfeat, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    graph.src_weights = src_w
    hod = torch.rand(nb, Gi, Gi, Gi, dtype=DT) + 0.5

    def build(gen):
        torch.manual_seed(7)
        return LAEPINN(
            gnn_in_channels=8, gnn_hidden_dim=16, gnn_out_channels=8,
            gnn_n_layers=2, gnn_heads=2, n_hod_bins=nb,
            grid_size=Gi, box_size=BOX, field_generator=gen,
        ).to(DT)

    mc, mg = build("continuous"), build("grid")
    mg.load_state_dict(mc.state_dict())     # identical weights
    mc.eval(); mg.eval()
    with torch.no_grad():
        oc = mc(graph, hod)["x_hii_pred"]
        og = mg(graph, hod)["x_hii_pred"]
    ma, mn, rl, cc = metrics(oc, og)
    # On-grid: continuous J_obs == scatter+FFT J_obs; difference is the fp32 FFT
    # floor (fft_convolve_3d / make_3d_kernel_grid run in float32).
    check("pinn continuous == grid (on-grid)", rl < 1e-4 and cc > 0.99999,
          f"max|Δ|={ma:.2e}, relL2={rl:.2e}, corr={cc:.6f} (ref=fp32)")

    # off-grid query interface (the new capability)
    cfield, ctx = mc.continuous_field(graph, hod)
    q = torch.rand(128, 3, dtype=DT)
    xq = cfield.forward(q, **ctx)["x_hii"]
    ok = bool(torch.isfinite(xq).all() and (xq >= 0).all() and (xq <= 1).all())
    check("off-grid query interface", ok,
          f"x(q): mean={xq.mean():.3f}, range=[{xq.min():.3f},{xq.max():.3f}]")


def _build_graph_on_grid(Gi, N, n_feat=8):
    """Small periodic kNN graph with on-grid source positions (shared by tests 6-7)."""
    from torch_geometric.data import Data
    pos = torch.randint(0, Gi, (N, 3)).to(DT) / Gi
    xfeat = torch.rand(N, n_feat, dtype=DT)
    src_w = torch.rand(N, dtype=DT) + 0.1
    d = pos[:, None, :] - pos[None, :, :]; d = d - d.round()
    dist = d.norm(dim=-1) + torch.eye(N, dtype=DT) * 1e3
    k = 6
    nbr = dist.topk(k, largest=False).indices
    dst = torch.arange(N).repeat_interleave(k); src = nbr.reshape(-1)
    ea = pos[src] - pos[dst]; ea = ea - ea.round()
    g = Data(x=xfeat, edge_index=torch.stack([src, dst], 0),
             edge_attr=torch.cat([ea, ea.norm(dim=-1, keepdim=True)], -1).to(DT), pos=pos)
    g.src_weights = src_w
    return g


def test7_bubble_integration():
    """LAEPINN bubble core: continuous vs grid generator agree on-grid (same zeta)."""
    print("\nTest 7: LAEPINN bubble — continuous vs grid generator (on-grid)")
    try:
        from models.pinn import LAEPINN
    except Exception as e:
        print(f"  [SKIP] torch_geometric/model unavailable ({type(e).__name__})")
        return

    torch.manual_seed(5)
    Gi, N, nb = 16, 60, 2
    graph = _build_graph_on_grid(Gi, N)
    hod = torch.rand(nb, Gi, Gi, Gi, dtype=DT) + 0.5
    density = torch.rand(Gi, Gi, Gi, dtype=DT) + 0.5

    def build(gen):
        torch.manual_seed(9)
        return LAEPINN(
            gnn_in_channels=8, gnn_hidden_dim=16, gnn_out_channels=8,
            gnn_n_layers=2, gnn_heads=2, n_hod_bins=nb, grid_size=Gi, box_size=BOX,
            excursion_type="bubble", bubble_zeta_init=0.3, field_generator=gen,
        ).to(DT)

    mc, mg = build("continuous"), build("grid")
    mg.load_state_dict(mc.state_dict())
    # Skip auto-calibration so BOTH use identical zeta & A_obs -> pure generator test.
    for m in (mc, mg):
        m._amp_calibrated.fill_(True)
        m.eval()
    with torch.no_grad():
        oc = mc(graph, hod, density_grid=density)["x_hii_pred"]
        og = mg(graph, hod, density_grid=density)["x_hii_pred"]
    ma, mn, rl, cc = metrics(oc, og)
    # Hard top-hat boundary + fp32 FFT -> robust metrics (not machine-eps).
    check("bubble continuous == grid (on-grid)", rl < 2e-2 and cc > 0.999,
          f"max|Δ|={ma:.2e}, relL2={rl:.2e}, corr={cc:.5f}  (xmean c={oc.mean():.3f} g={og.mean():.3f})")

    # off-grid bubble query interface
    ev, ctx = mc.continuous_field(graph, hod)
    xq = ev.forward(torch.rand(128, 3, dtype=DT), **ctx)["x_hii"]
    ok = bool(torch.isfinite(xq).all() and (xq >= 0).all() and (xq <= 1).all())
    check("bubble off-grid query interface", ok,
          f"x(q): mean={xq.mean():.3f}, range=[{xq.min():.3f},{xq.max():.3f}]")


if __name__ == "__main__":
    print("=" * 68)
    print("Continuous ionization field — consistency verification")
    print(f"box={BOX} cMpc/h, grid G={G}, dtype={DT}")
    print("=" * 68)
    test1_on_grid_obs_only()
    test2_on_grid_with_unresolved()
    test3_off_grid()
    test4_superres_and_gradient()
    test5_gradient_flow()
    test6_pinn_integration()
    test7_bubble_integration()
    print("\n" + "=" * 68)
    print(f"RESULT: {'ALL TESTS PASSED' if PASS else 'SOME TESTS FAILED'}")
    print("=" * 68)
    sys.exit(0 if PASS else 1)
