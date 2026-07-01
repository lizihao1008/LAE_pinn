# LOS Lyα Transmission Module

**Module:** `physics/los_transmission.py`
**Loss:** `physics.los_transmission.los_transmission_loss` (wired in `training/loss.py`)
**Config:** `config/default.yaml → los_transmission` + `loss_weights.los_transmission`
**Status:** auxiliary constraint (does NOT replace the field loss)

## Motivation

Lyα visibility is a **line-of-sight transmission effect**, not a local ionization
label. The observed Lyα flux / EW / `T_IGM` of an LAE must not be read as the
ionization state of its own voxel. During reionization,

```
T_IGM(ν) = exp[-τ_IGM(ν)] ,   τ_IGM(ν) = ∫ n_HI(s) σ_α[ν(s)] ds
```

with the dominant contribution from the **near-side bubble wall, neutral islands,
self-shielded absorbers, and the damping wing** along the observer direction. So:

- an LAE **inside** an ionized bubble can still be **invisible** if dense neutral
  gas sits in front of it along the LOS;
- a **large LOS bubble radius** can make a source **visible**, because Hubble
  expansion redshifts the Lyα photons off resonance before they reach neutral gas.

## Role separation (the model change)

`T_IGM`, `EW_obs`, `L_Lyα_obs` are **no longer** treated as local `x_HII` labels at
the source voxel. Instead:

1. **Spatial clustering / galaxy positions** constrain the 3D ionization topology
   (the field losses: `field_mse`, `power_spectrum`, `binary_bce`, `global_xHII`).
2. **Lyα marks** constrain the **line-of-sight transmission** from each LAE toward
   the observer (this module + its auxiliary loss).

## What the module computes

After the ionization field `x_HII_hat` is generated, for each LAE at `x_i` a ray is
cast along the observer axis `e_LOS` and the **predicted** neutral fraction is
sampled along it (periodic trilinear interpolation of the predicted grid — fully
differentiable, so gradients reach the whole ionization model):

```
x_HI_hat(s) = 1 - x_HII_hat(x_i + s e_LOS)
τ_i         = Σ_s x_HI_hat(s) · W(s) · Δs ,      s ∈ [0, R_max]
W(s)        = amp · v_ref² / (Δv_i + H(z) s + v_floor)²      (damping-wing detuning)
T_IGM_hat,i = exp[-(τ_i + τ_CGM)]
```

- `H(z)` is the Hubble velocity per comoving Mpc/h (`100·√(Ωm(1+z)³+ΩΛ)/(1+z)`,
  `h` cancels; ≈159 km/s per cMpc/h at z=7.14), so a large LOS bubble redshifts
  photons out of resonance (W drops) — visibility rises with bubble radius.
- `W(s)` decreases with distance ⇒ **near-side neutral gas dominates** τ.
- `Δv_i` is the intrinsic Lyα velocity offset; if unavailable a **learnable
  population** `dv_Lya` is used.
- `τ_CGM` is an optional **learnable** local (circum-galactic) attenuation.
- `v_ref` non-dimensionalises `W` so `amp ~ O(1)` gives `τ ~ O(1)` (avoids a
  saturated init); `amp` is learnable.
- Learnable: `dv_Lya`, `τ_CGM`, `amp` (all softplus-positive).

## Auxiliary loss

Added to the composite loss with weight `loss_weights.los_transmission` (default
0.5), **in addition to** the field losses:

```
target: "tigm"  ->  L = MSE( log T_IGM_hat , log T_IGM_obs )     # tigm = L_obs/L_int
target: "lya"   ->  L = MSE( log L_Lya_obs , log L_Lya_int + log T_IGM_hat )
```

Only sources with a valid (positive) measurement contribute (`tigm > 0`). The raw
marks (`tigm_obs`, `lya_int`, `lya_obs`, `muv`) are carried on the graph by the data
pipeline (`preprocessing.prepare_snapshot` → `graph_builder`), separate from the
(normalised) GNN input features.

Gradients flow to (a) the ionization field along each ray — so Lyα marks now shape
the field **along the LOS** rather than at the source voxel — and (b) the LOS
parameters `dv_Lya`, `τ_CGM`, `amp`.

## Usage

```yaml
los_transmission:
  enabled: true
  target: tigm
  los_axis: null        # null → rsd_correction.los_axis (the observer axis)
  r_max_mpc: 80.0
  n_steps: 64
training:
  loss_weights:
    los_transmission: 0.5
```
```
python -m experiments.run_mvp --config           # LOS on (reads default.yaml)
```
Runs on the `--config` path (`build_pinn_from_config`). Set weight 0 or
`enabled: false` to disable. `model.get_learnable_physics()` reports
`dv_lya_kms`, `tau_cgm`, `los_amp`; the training log prints `los=… dvLya=… τCGM=…`.

## Verification (independent NumPy mirror)

`outputs/verify_los_mirror.py` reproduces the τ / T physics and confirms:

| Check | Result |
|-------|--------|
| fully ionized field | `T ≈ exp(-τ_CGM) = 0.951`, `τ = 0` |
| fully neutral field | `T = 2e-4`, `τ = 8.5` (strong absorption) |
| near-side dominance (neutral slab at d = 5 / 20 / 50 cMpc/h) | `T = 0.156 / 0.754 / 0.908` — rises as neutral gas moves farther |
| **same local x_HII=1, wall vs clear LOS** | `T = 0.28` (wall 8 cMpc/h ahead) vs `0.95` (clear) — **visibility set by the LOS, not the local voxel** |

Torch integration: `experiments/test_continuous_field.py::test9_los_transmission`
(T_IGM_hat ∈ (0,1]; loss finite; gradients reach `dv_Lya`, `amp`, and the field).

## Notes / limitations

- The ray uses periodic wrapping (the box tiles), standard for a periodic sim box.
- The `1/Δv²` kernel captures the damping-wing falloff; the resonant core (full
  Gunn-Peterson trough) is not separately modelled — appropriate for the
  damping-wing / detuned regime this constraint targets.
- The ray samples the produced grid field; with the continuous generator the field
  itself is already mesh-free, and the same ray machinery could query it directly
  if sub-voxel LOS accuracy is later needed.
