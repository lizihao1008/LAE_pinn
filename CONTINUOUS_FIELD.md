# Continuous Ionization Field — Graph-Neural-Factor / Kernel-Integral Representation

**Module:** `physics/continuous_field.py`
**Test:** `experiments/test_continuous_field.py`
**Status:** **default field generator** in `pinn.forward` (`field_generator: continuous`);
the legacy grid path remains selectable via config (`field_generator: grid`).

This note documents a resolution-free, physics-strict replacement for the
grid-bound final stage of the model. It produces the ionization field as a
**continuous function** `x_HII(x)` evaluable at any coordinate, while keeping the
existing physics **exactly** — no neural basis, no extra approximation.

**What changed in the model (generator swap only).** In `LAEPINN.forward`, the
observed-source term is now built mesh-free over EXACT source positions instead of
`scatter_to_grid` + `fft_convolve_3d`. Both cores are supported:

- **equilibrium** — `render_obs_field_on_grid` (kernel sum `Σ w_i K(|x-x_i|)`);
  `A_obs` / `J_ref` calibration is bit-identical (unit-sum kernel preserves means,
  verified to 2e-14).
- **bubble** — `render_bubble_on_grid` (per-scale top-hat kernel sums for the
  observed sources; `S_unres` and `n_H` are grid-smoothed and interpolated). Zeta
  is calibrated on the continuous render (`calibrate_bubble_zeta`).

Everything else is untouched: the unresolved term still uses FFTs, all
losses/training/evaluation are unchanged, and **no new parameters** are added
(checkpoint-compatible). New: `model.continuous_field(graph, hod_basis)` returns a
uniform off-grid evaluator for either core. **Cost:** continuous bubble is
`~n_scales × O(G³·N)` per step (heavier than the grid FFT path); fall back with
`field_generator: grid` if needed.

---

## 1. Problem with the grid stage

The current forward pass produces `x_HII` on a fixed `G³` voxel grid:

```
GNN → f_esc,i → w_i = L_i·f_esc,i        (per-LAE factors)
scatter_to_grid(w_i)                      → S_obs        [G³ grid]
HOD Σ_b f_esc_b ε_b(x)                     → S_unres      [G³ grid]
fft_convolve_3d(S_obs+S_unres, K)          → J_total      [G³ grid]
equilibrium(J_total / J_ref)               → x_HII        [G³ grid]
```

Everything downstream of `scatter` is **tied to the grid**: `scatter` and the FFT
convolution both require a regular mesh, and the equilibrium is applied
voxel-by-voxel. You cannot query an arbitrary coordinate, you cannot change
resolution without re-gridding and re-FFT-ing, and spatial derivatives of the
field are only available through finite differences on the mesh.

---

## 2. Key observation: the physics is already a kernel integral

The radiation field is a convolution, i.e. a Green-function integral:

```
J(x) = (S * K)(x)
     = A_obs · Σ_i  w_i K(|x − x_i|; θ_K)        ← observed LAEs (point sources)
     +         ∫    S_unres(x') K(|x − x'|) dx'   ← unresolved HOD background
```

The observed-source term is **exactly a DeepONet / kernel-method continuous
field**:

| DeepONet object        | Here                                             |
|------------------------|--------------------------------------------------|
| branch coefficients bᵢ | `w_i = L_i · f_esc,i` — **the GNN output (graph neural factors)** |
| trunk basis φᵢ(x)      | `K(|x − x_i|; θ_K)` — the physical radiative kernel |
| field                  | `J(x) = Σᵢ bᵢ φᵢ(x)`                              |

So `J(x)` — and hence `x_HII(x)` through the **exact** photoionization
equilibrium — is a continuous function evaluable at any `x`. The FFT grid is
merely one (periodic, voxel-smoothed) discretization of this same sum. Making the
field continuous therefore costs **zero physics**: we evaluate the same operator
the model already assumes, just off the mesh.

```
x_HII(x) = equilibrium( [ A_obs·Σᵢ wᵢ K(|x−xᵢ|) + (S_unres * K)(x) ] / J_ref )
```

---

## 3. What stays exactly physical (no penalty needed)

- **Radiative transfer.** `J(x)` is the exact Green-function solution of the
  assumed transport kernel `K` (mean-free-path attenuation `exp(−r/λ)/(4πr²)` +
  geometric soft bubble `σ((R−r)/Δ)`), evaluated continuously. The learnable
  kernel parameters `R, Δ, λ, A_geom, A_trans` are shared with the grid model.
- **Photoionization balance.** `x_HII = (√(A²+4A) − A)/2` is the exact closed-form
  root of `Γ(1−x) = α n_H x²` with `A = J/(α n_H scale)`. Hence `0 ≤ x_HII ≤ 1`
  and local equilibrium hold pointwise, by construction. The single scale
  `alpha_nH_scale` is shared with the grid model.

Because both are satisfied **by construction**, there is no soft "PDE residual"
to balance. The losses below are only (a) a resolution-free data fit and (b) the
global photon-budget (ionized-fraction) constraint.

---

## 4. Continuous handling of each term

**Observed sources** — analytic kernel sum with two faithfulness details that make
it match the grid convolution:
- *Periodicity*: separations use the minimum-image convention `d − round(d)`,
  matching the periodic FFT.
- *Softening*: pair distances are clamped to `dx/2` (the same clamp
  `make_3d_kernel_grid` applies), so the `1/(4πr²)` head of `k_mfp` stays finite.
- *Normalization*: divided by `Z = Σ_grid K`, the same constant
  `make_3d_kernel_grid` uses (and which cancels in `J/J_ref` anyway).

**Unresolved background** — `S_unres` is intrinsically smooth (`ε_b = 1 + b_b δ`,
further smoothed by `K` whose scale is several cMpc/h), so its kernel-convolved
grid is sampled with **periodic trilinear interpolation** at the query points. By
linearity this is exactly the unresolved part of the grid `J_total`. (A fully
mesh-free variant — Monte-Carlo "virtual sources" drawn from `S_unres`, folded
into the same kernel sum — is available but stochastic; interpolation is the
deterministic default.)

---

## 5. Mapping to existing code

| Grid pipeline                              | Continuous module                                   |
|--------------------------------------------|-----------------------------------------------------|
| `scatter_to_grid(pos, w)`                  | `source_field(query, src_pos, src_w)` — analytic sum |
| `make_3d_kernel_grid` normalization        | `kernel_norm_constant()` (same `Z`)                 |
| `fft_convolve_3d(S_obs, K)` (periodic)     | minimum-image kernel sum (same periodicity)         |
| `fft_convolve_3d(S_unres, K)`              | `periodic_trilinear_sample(J_unres_grid, query)`    |
| `ExcursionSetMapping(J/J_ref)`             | **same module**, called pointwise                   |
| kernel / excursion parameters              | **shared** (passed in, not duplicated)              |

`build_continuous_field_from_pinn(model, graph, hod_basis)` runs the model's
GNN + source head, folds in `A_obs`, builds the kernel-convolved `J_unres`, reuses
the calibrated `J_ref`, and returns `(cfield, ctx)` so that
`cfield.forward(query, **ctx)` gives `x_HII` at arbitrary `query` consistent with
`model.forward` on the grid. Supports `excursion_type == "equilibrium"`.

---

## 6. Usage

The grid forward already uses the continuous generator by default:

```python
out = model(graph, hod_basis)          # x_hii_pred now from the kernel sum (default)
# fall back for A/B comparison:  field_generator: grid   (config) or
#                                model.field_generator = "grid"
```

Off-grid / resolution-free interface:

```python
from physics.continuous_field import sampled_field_mse, global_xhii_mc

cfield, ctx = model.continuous_field(graph, hod_basis)   # off-grid evaluator

# (a) query anywhere  — e.g. EXACT LAE positions for MCF marks
q = graph.pos                                    # or any normalised [0,1)^3 coords
x_hii = cfield.forward(q, **ctx)["x_hii"]        # (N,) — no nearest-voxel rounding

# (b) render at ANY resolution (super-resolution for free)
x_128 = cfield.evaluate_on_grid(128, ctx["src_pos"], ctx["src_w"],
                                J_unres_grid=ctx["J_unres_grid"], J_ref=ctx["J_ref"])

# (c) resolution-free training losses
L  = sampled_field_mse(cfield, ctx, x_true_grid, n_samples=32768)   # data fit
L += 5.0 * global_xhii_mc(cfield, ctx, xi_global)                   # photon budget
L.backward()    # grads reach GNN factors, kernel params, alpha_nH_scale
```

`P(k)` and topology metrics still need a mesh: call `evaluate_on_grid(...)` then
feed the existing `power_spectrum_loss` / topology tools.

---

## 7. Verification

`experiments/test_continuous_field.py` checks the continuous field at voxel
centres against `scatter → fft_convolve_3d → equilibrium`:

| Test | Setup | Result |
|------|-------|--------|
| 1 | on-grid sources, obs-only         | relL2 `4e-6`, corr `1.000000` (torch fp32) / `8e-14` (fp64 mirror) |
| 2 | on-grid sources + diffuse HOD     | relL2 `<1e-5`, corr `1.000000` |
| 3 | off-grid random sources           | corr `0.9993`, relL2 `2.8%` (residual = grid voxel-smoothing) |
| 4 | super-resolution + autodiff ∇x    | shape `2G³`, `x∈[0,1]`, finite ∇ |
| 5 | gradient flow                     | grads reach `w_i`, `R/λ`, `α_nH` |
| 6 | `LAEPINN` equilibrium: continuous vs grid generator (on-grid) | match to fp32; off-grid query OK |
| 7 | `LAEPINN` bubble: continuous vs grid generator (on-grid)      | corr `>0.999`; off-grid query OK |
| 8 | `LAEPINN` hybrid (`bubble_equilibrium`): continuous vs grid (on-grid) | corr `>0.999`; `x∈[0,1]`; off-grid query OK |

Independent NumPy mirrors (float64) confirm all cores:

- *Equilibrium* — `A_obs` calibration is generator-invariant
  (`mean(J_obs_grid)/mean(S_obs) − 1 ≈ 2e-14`,
  `mean(S_unres*K)/mean(S_unres) − 1 ≈ 2e-14`); on-grid x_HII matches to `~1e-13`.
- *Bubble* — continuous vs grid `BubbleExcursionSet`: on-grid corr `0.99997`
  (relL2 `4e-3`), off-grid corr `0.99998`. The residual is the **hard top-hat
  boundary** (a source at distance ≈ R can land on either side of the threshold
  between the two float paths) plus the off-grid voxel smearing — not machine-eps,
  but a faithful match.
- *Hybrid* (`B·x_eq`) — continuous vs grid on-grid corr `0.99997` (relL2 `4e-3`);
  the product inherits the bubble's top-hat-boundary residual (x_eq matches to
  fp32), confirming the two factors compose consistently.

**Precision note.** `physics/scatter.fft_convolve_3d` casts its inputs with
`.float()`, so the **grid reference is only float32-accurate** (`max|Δ|~1e-5`,
the `sqrt(G³)·eps_fp32` FFT-accumulation floor). The continuous field is
float64-exact; an all-float64 NumPy mirror of the same formulas agrees with the
grid to `~1e-13` (Tests 1–2). The on-grid checks therefore assert on the
float32-robust metrics (`relL2 < 1e-4`, `corr > 0.99999`). For a bit-exact fp64
match, drop the `.float()` cast in `fft_convolve_3d` (optional, one line).

Tests 1–2 confirm the continuous form reproduces the grid pipeline to the
reference's float32 precision when sources sit on voxel centres — proving no
physics is lost. Test 3 shows the only off-grid difference is the sub-voxel
trilinear smoothing the grid imposes (which the continuous form removes — i.e. the
continuous field is the more accurate object). Run in your env (torch + PyG):

```
cd LAE_pinn && python experiments/test_continuous_field.py
```

---

## 8. New capabilities

- **Resolution-free supervision.** Train on points sampled anywhere, denser than
  the `64³` ATON target — supervision is no longer capped by the working grid.
- **Super-resolution inference.** Render the trained field at `128³`, `256³`, or at
  scattered LAE/sightline coordinates, with no retraining.
- **Exact spatial derivatives.** `ionization_front_gradient` returns `|∇x_HII|` via
  autodiff (no finite differences) — localises HII-bubble fronts directly.

---

## 8b. Hybrid core: bubble topology × photoionization equilibrium

`excursion_type: bubble_equilibrium` (`HybridBubbleEquilibrium`) makes the bubble
interior satisfy photoionization equilibrium:

$$x_{\rm HII}(\mathbf{x}) = B(\mathbf{x})\cdot x_{\rm HII}^{\rm eq}(\mathbf{x})$$

- `B(x)` — excursion-set bubble membership (photon-budget topology, sharp 0/1,
  genuine neutral voids). Answers *where* gas is ionized.
- `x_eq(x)` — photoionization-equilibrium ionized fraction on the propagated field
  `J`, `x_eq = (√(A²+4A)−A)/2`, `A = (J/J_ref)/s`. Sets *how* ionized inside
  (residual `x_HI ≈ s/J`).

Equivalently `x_HI = (1−B)·1 + B·x_HI_eq`: fully neutral outside bubbles, the
photoionization-equilibrium residual inside. This is the standard semi-numerical
topology + post-reionization UVB-residual picture (cf. 21cmFAST). Both factors use
the active field generator (continuous top-hat sum for `B`, kernel-integral for
`x_eq`), so the hybrid is mesh-free and queryable off-grid via
`model.continuous_field(...)`.

Calibration is joint but well-separated: `zeta` is bisected on `⟨B·x_eq⟩ = ξ`
(topology / volume fraction), while the residual scale `s` (`hybrid_residual_scale_init`,
default 0.1) is small so `x_eq ≈ 1` inside bubbles — the hybrid **starts ≈ the pure
bubble** and refines the interior residual during training (`s` learned by the field
losses). No loss changes; `x_hii_pred` is still a `[0,1]` field.

```yaml
excursion_set:
  type: bubble_equilibrium
  hybrid_residual_scale_init: 0.1
```
```
python -m experiments.run_mvp ... --excursion bubble_equilibrium
```

## 9. Limitations and next steps

- **Cost & memory.** The kernel/top-hat sum is `O(Q·N)` (bubble: `× n_scales`).
  For a full-grid render `Q = G³` the naive autograd graph is `O(G³·N)` — at
  `G=64, N~10⁴` that is ~60 GiB and OOMs. Two mitigations, both default-on:
  - **Gradient checkpointing** (`checkpoint: true`) recomputes each query chunk in
    backward so the retained graph is only `O(chunk·N)` (≈1.3× compute).
  - **Neighbour list** (`neighbor_list: auto`, needs `torch_cluster` + a cutoff
    `r_cut_mpc`): `torch_cluster.radius` with periodic **ghost replication** gives
    `O(Q·⟨neighbours⟩)` in memory *and* flops (~8× fewer at `r_cut=50` in a 160
    box). A runtime **self-check** compares against the dense path on a small
    subsample and falls back automatically on any mismatch / missing dep, so it can
    never produce a wrong field. The ghost/cutoff math is verified to `1e-13` vs
    the dense minimum-image sum.

  Further knobs: lower `chunk`, tighten `r_cut_mpc`, or fall back to
  `field_generator: grid`.
- **Bubble mapping (implemented).** `bubble_field_continuous` evaluates the
  multi-scale excursion-set threshold continuously: observed sources via exact
  per-scale top-hat kernel sums, `S_unres` / `n_H` via grid-smoothing + interp.
  The top-hat is a HARD indicator (radii are fixed, not learned), so it is not
  differentiable in `x` at `r = R` — fine for training (gradients flow to `w_i`,
  `zeta`, `sharpness`) and for the grid render, but the autodiff front gradient is
  meaningful only for the equilibrium core.
- **Sharp fronts.** The physical kernel already produces sharp edges via the
  bubble term; the field is not over-smoothed. *If* a learnable residual basis is
  added later (the "kernel + neural basis" option), it should use Fourier-feature
  / hash-grid positional encodings so it does not blur ionization fronts.
```
