# Technical Roadmap: LAE-Conditioned Photon-Budget and Ionization-Topology Inference

**Project:** `LAE_pinn`  
**Stack:** PyTorch, PyTorch Geometric (PyG)  
**Data:** Sherwood-Relics + ATON radiative transfer, 160 cMpc/h box, 512³ ionization grid  
**Snapshots:** z = 7.14 (x_HI ≈ 0.69), 6.6 (x_HI ≈ 0.52), 5.756 (x_HI ≈ 0.13)

---

## 1. Scientific Objective

**Core question:**  
Can the spatial distribution and Lyα marks of observed LAEs constrain the 3D ionization topology of the IGM, and to what degree does the inference break the photon-budget / source-model degeneracy?

**What this is NOT:**  
- A GNN that black-box maps galaxy positions → ionization field  
- A model whose target is the MCF itself  
- A simulation emulator tied to one source prescription  

**What this IS:**  
A physics-constrained inverse problem where:
1. A GNN encodes the LAE environment and infers *effective source strength* per galaxy  
2. A physical radiative kernel (learnable parameters, fixed functional form) converts source strengths to an ionizing radiation field $J_{\rm obs}(\mathbf{x})$  
3. A constrained low-dimensional latent field represents unresolved faint-source emissivity $\epsilon_{\rm unres}(\mathbf{x})$  
4. A photon-budget / excursion-set operator maps total emissivity to $\hat{x}_{\rm HII}(\mathbf{x})$  
5. The model outputs *posteriors* over topology field and source-mixture fractions, not a single deterministic prediction  

---

## 2. Physical Decomposition

### 2.1 Total ionizing emissivity

$$\epsilon_{\rm ion}(\mathbf{x}) = \epsilon_{\rm obs}(\mathbf{x}) + \epsilon_{\rm unres}(\mathbf{x})$$

### 2.2 Observed LAE contribution

$$\epsilon_{\rm obs}(\mathbf{x}) = \sum_{i \in \rm LAE} L_i \cdot \xi_{\rm ion} \cdot f_{{\rm esc},\theta}(M_i, \delta_i, T_i, {\rm EW}_i) \cdot K(|\mathbf{x} - \mathbf{x}_i|;\, \theta_K)$$

- $L_i$: observed Lyα luminosity (proxy for UV luminosity via $M_{\rm UV}$)  
- $\xi_{\rm ion}$: ionizing photon production efficiency (fixed prior or learnable scalar)  
- $f_{{\rm esc},\theta}$: escape fraction — output of GNN encoder, range-constrained to $(0, 1)$  
- $K(\cdot;\theta_K)$: physical radiative kernel with learnable parameters $\theta_K$  

### 2.3 Unresolved source emissivity

$$\epsilon_{\rm unres}(\mathbf{x}) = \sum_b F_b \cdot \epsilon_b(\mathbf{x})$$

where $b \in \{\rm bright, intermediate, faint, diffuse\}$ represents source populations binned by $M_{\rm UV}$ / halo mass. Constraints:

$$F_b \geq 0, \quad \sum_b F_b = 1$$

$\epsilon_b(\mathbf{x})$ is modelled as a smooth bias-weighted density field (not a free neural field):

$$\epsilon_b(\mathbf{x}) \propto b_b \cdot \rho_{\rm dm}(\mathbf{x})^{\alpha_b}$$

with luminosity-function and halo-mass-function priors on each $F_b$ and $\alpha_b$.

### 2.4 Physical radiative kernels

Three candidate kernels (all learnable parameters, fixed form):

**Exponential (mean-free-path / diffuse):**
$$K_{\rm mfp}(r;\, \lambda) = \frac{\exp(-r/\lambda)}{4\pi r^2 + \epsilon}$$

**Soft bubble (geometric topology):**
$$K_{\rm bub}(r;\, R, \Delta) = \sigma\!\left(\frac{R - r}{\Delta}\right), \quad \sigma = \text{sigmoid}$$

**Mixture (default):**
$$K(r) = A_{\rm geom}\, K_{\rm bub}(r;\, R, \Delta) + A_{\rm trans}\, K_{\rm mfp}(r;\, \lambda)$$

with $A_{\rm geom} + A_{\rm trans} = 1$, $A_{\rm geom}, A_{\rm trans} \geq 0$.

### 2.5 Ionization mapping

From total emissivity $J(\mathbf{x}) = \epsilon_{\rm ion} * K$ (convolution on the 3D grid), ionization field via an excursion-set-like soft threshold:

$$\hat{x}_{\rm HII}(\mathbf{x}) = \sigma\!\left(\frac{J(\mathbf{x}) - \mu_{\rm thresh}}{\tau}\right)$$

where $\mu_{\rm thresh}$ is a learnable (or physically motivated) recombination threshold and $\tau$ controls sharpness.  

For the probabilistic version, $\mu_{\rm thresh}$ and $\tau$ are outputs of an amortized inference network, giving per-voxel uncertainty.

---

## 3. Model Architecture

```
LAE catalog
(positions, M_UV, T_IGM, EW, z)
        │
        ▼
┌─────────────────────┐
│  k-NN Graph Build   │  r_link ~ 10–20 cMpc/h
│  node features:     │  → (x,y,z), M_UV, log10(M_h),
│  T_IGM, EW, Lya_obs │    T_IGM, EW_obs, Lya_obs/int
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   GNN Encoder       │  3–4 layers of GATv2Conv or SAGE
│   (PyG)             │  node → environment embedding h_i ∈ R^d
│                     │  aggregates local overdensity,
│                     │  neighbour marks, group topology
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   Source Head (per-node MLP)    │
│   h_i → f_esc,i ∈ (0,1)        │  sigmoid activation
│         ξ_ion,i (optional)      │  softplus, prior-regularised
│         weight w_i = L_i·f_esc  │
└─────────────────────────────────┘
        │                              ┌──────────────────────────┐
        │                              │  Unresolved Source Field  │
        │                              │  F_b (softmax, dim=4)    │
        │                              │  latent bias field        │
        │                              │  ε_unres(x) on 64³ grid  │
        │                              └──────────────────────────┘
        │                                          │
        ▼                                          ▼
┌───────────────────────────────────────────────────────┐
│          Physical Radiative Kernel  K(r; θ_K)         │
│   Scatter: sum_i w_i · K(|x - x_i|) → J_obs(x)       │
│   (on 64³ grid, ~2.5 cMpc/h resolution)               │
│   Add: ε_unres(x) → J_total(x)                        │
│   Learnable: R, Δ, λ_mfp, A_geom, A_trans, threshold  │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Excursion-Set Mapping       │
│  J_total → x̂_HII(x)  ∈[0,1] │
│  + global constraint:        │
│    mean(x̂_HII) ~ ⟨x_i⟩_z   │
└─────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│      Output                           │
│  x̂_HII(x)  64³ ionization field      │
│  F_b        source-budget fractions   │
│  θ_K        kernel parameters         │
│  f_esc,i    per-LAE escape fractions  │
│  (optional uncertainty maps)          │
└───────────────────────────────────────┘
```

---

## 4. Parameters

### 4.1 Learnable (trained by gradient descent)

| Symbol | Description | Constraint |
|--------|-------------|------------|
| GNN weights | Environment encoder | standard |
| $f_{{\rm esc},i}$ | Per-LAE escape fraction (output of GNN head) | sigmoid → (0,1) |
| $R$ | Soft bubble radius | softplus → R > 0 |
| $\Delta$ | Bubble edge sharpness | softplus → Δ > 0 |
| $\lambda_{\rm mfp}$ | Mean free path for $K_{\rm mfp}$ | softplus → λ > 0 |
| $A_{\rm geom}, A_{\rm trans}$ | Kernel mixture weights | softmax → sum=1, ≥0 |
| $F_b$ | Unresolved source-population fractions | softmax → sum=1, ≥0 |
| $\alpha_b$ | Density power-law index per population | real |
| $\mu_{\rm thresh}, \tau$ | Excursion-set threshold + sharpness | real |
| Latent bias field | Low-dimensional unresolved emissivity | constrained smooth field |

### 4.2 Fixed physical form (not learnable)

| Item | Description |
|------|-------------|
| Kernel functional form | Exponential / sigmoid — physics-motivated |
| $\sum_b F_b = 1$ | Photon-budget normalization |
| $0 < f_{\rm esc} < 1$ | Physical range |
| $\epsilon_b(\mathbf{x}) \propto b_b \rho^{\alpha_b}$ | Bias model for unresolved sources |
| Excursion-set monotonicity | $J$ → $x_{\rm HII}$ monotone mapping |

### 4.3 Priors (regularisation terms in loss)

| Prior | Target | Form |
|-------|--------|------|
| LF prior | $F_b$ | KL divergence against LF-integrated luminosity budget |
| HMF prior | source-mass distribution | log-normal penalty |
| Smoothness prior | $\epsilon_{\rm unres}$ | TV or gradient penalty |
| Positivity prior | $\epsilon_b$ | softplus / ReLU enforced |
| Global neutral fraction | mean($\hat{x}_{\rm HII}$) ≈ $\langle x_i\rangle_z$ | MSE penalty on scalar |

---

## 5. Loss Function

$$\mathcal{L} = \mathcal{L}_{\rm field} + \lambda_{\rm MCF}\,\mathcal{L}_{\rm MCF} + \lambda_{\rm xHII}\,\mathcal{L}_{\rm xHII} + \lambda_{\rm prior}\,\mathcal{L}_{\rm prior}$$

### 5.1 Field loss (primary)

$$\mathcal{L}_{\rm field} = \underbrace{\| \hat{x}_{\rm HII} - x_{\rm HII}^{\rm true} \|_2^2}_{\rm voxel\, MSE}
+ \beta_1 \underbrace{\| P_{\hat{x}} - P_{x^{\rm true}} \|_2^2}_{\rm power\, spectrum}
+ \beta_2 \underbrace{{\rm BCE}(\hat{x}_{\rm HII},\, [x_{\rm HII}^{\rm true} > 0.5])}_{\rm binary\, topology}$$

Field-level training targets a coarsened 64³ version of the ATON ground-truth Xbox.

### 5.2 MCF consistency loss (auxiliary, not primary target)

$$\mathcal{L}_{\rm MCF} = \sum_{r_k} \left( M_{\hat{x}}(r_k) - M_{x^{\rm true}}(r_k) \right)^2$$

The MCF of the predicted field should match the true MCF without being the direct training target.

### 5.3 Global constraint

$$\mathcal{L}_{\rm xHII} = \left( \langle \hat{x}_{\rm HII} \rangle - \langle x_i^{\rm true} \rangle_z \right)^2$$

### 5.4 Prior regularisation

$$\mathcal{L}_{\rm prior} = D_{\rm KL}(F_b \| F_b^{\rm LF}) + \gamma_1 \| \nabla \epsilon_{\rm unres} \|_1 + \gamma_2 \mathcal{L}_{\rm HMF}$$

---

## 6. Baselines

| Baseline | Description |
|----------|-------------|
| **MCF only** | Traditional summary statistic; no field prediction |
| **Density-only** | $\hat{x}_{\rm HII} \propto \rho_{\rm dm}$ (pure density bias) |
| **LAE density smooth** | Gaussian-smooth LAE number density, no marks |
| **LAE marks, fixed kernel** | Physical kernel but kernel parameters not learned |
| **Black-box U-Net** | Galaxy density → field, no physics structure |
| **Oracle-source PINN** | Use simulation's true source prescription as fixed input |

---

## 7. Ablation Design

Each ablation isolates one model component. All other components are held fixed.

### 7.1 Kernel ablation

| Variant | Kernel |
|---------|--------|
| A1 | Fixed Gaussian σ = 5 cMpc/h (no learning) |
| A2 | Learned exponential only ($K_{\rm mfp}$) |
| A3 | Learned soft bubble only ($K_{\rm bub}$) |
| A4 | **Learned mixture (default)** |

Expected finding: mixture kernel needed to capture both geometric topology and large-scale transmission coherence.

### 7.2 Unresolved source ablation

| Variant | Unresolved sources |
|---------|--------------------|
| B1 | No unresolved sources: $\epsilon_{\rm unres} = 0$ |
| B2 | Unresolved = free neural field (unconstrained) |
| B3 | **Constrained bias-weighted field (default)** |
| B4 | Oracle: use simulation halo catalog for all sources |

Expected finding: constrained model avoids overfitting unresolved emissivity while still improving topology recovery.

### 7.3 GNN encoder ablation

| Variant | Encoder |
|---------|---------|
| C1 | No GNN: $f_{\rm esc,i}$ = global constant |
| C2 | Node features only (no message passing) |
| C3 | 1-layer GNN |
| C4 | **3-layer GATv2 (default)** |
| C5 | Oracle: $f_{\rm esc,i}$ set to simulation ground truth |

### 7.4 Mark ablation

| Variant | Input marks |
|---------|-------------|
| D1 | Positions only (no Lyα marks) |
| D2 | Positions + $M_{\rm UV}$ only |
| D3 | + $T_{\rm IGM}$ |
| D4 | + EW |
| D5 | **All marks (default)** |

---

## 8. Source Degeneracy Stress Tests

This is the core scientific experiment. All runs use the same model architecture but different source assumptions.

| Experiment | Source model | Training data | Test data |
|------------|-------------|---------------|-----------|
| **S1: observed-only** | Only detected LAEs ($M_{\rm UV} < -17.5$) | Fiducial sim | Fiducial sim |
| **S2: oracle all-halo** | All halos in simulation | Fiducial sim | Fiducial sim |
| **S3: fixed-source oracle** | True simulation source prescription (known) | Fiducial sim | Fiducial sim |
| **S4: wrong-source stress** | Massive-only source model | Faint-dominated sim | Faint-dominated sim |
| **S5: wrong-source reverse** | Faint-galaxy model | Massive-dominated sim | Massive-dominated sim |
| **S6: learned mixture** | **Model infers $F_b$** | Fiducial sim | Fiducial sim |
| **S7: ensemble posterior** | Train on multiple prescriptions; marginalise $F_b$ | Multi-sim | Held-out sim |
| **S8: wrong-z stress** | Apply model trained at z=7.14 to z=6.6 | z=7.14 | z=6.6 |

**Primary degeneracy diagnostic:** Compare recovered $F_b$ posteriors across S1–S7.  
If $F_b$ is poorly constrained even with good field reconstruction, the source model is degenerate with topology.  
If $F_b$ shifts systematically in S4/S5 (wrong source stress), the model has absorbed the error into kernel parameters.

---

## 9. Topology Statistics

Computed on both $\hat{x}_{\rm HII}$ and $x_{\rm HII}^{\rm true}$ at threshold $x_{\rm HII} > 0.5$:

| Statistic | Description | Tool |
|-----------|-------------|------|
| Binary mark MCF | MCF with $m_i = [x_{\rm HII}(\mathbf{x}_i) > 0.5]$ | `corrfunc` + custom |
| TIGM mark MCF | MCF with $m_i = T_{\rm IGM,i}$ | existing pipeline |
| Granulometry $G(r)$ | Volume fraction as bubble erosion radius increases | `scipy.ndimage` |
| Bubble size distribution $P(R_b)$ | From watershed / SDF | custom |
| Percolation fraction | Size of largest connected component / total ionized volume | `scipy.ndimage` |
| Connected components $N_c$ | Number of distinct ionized regions | `scipy.ndimage.label` |
| Ion–density cross-correlation | $\xi_{x\rho}(r)$ | FFT-based |
| MCF vs $M_{\rm UV}$ cut | Repeat at $M_{\rm UV} < -17, -18, -19$ | existing pipeline |
| MCF vs redshift | Repeat at z = 7.14, 6.6, 5.756 | existing pipeline |

---

## 10. Simulation / Oracle Classification

| Mode | Description | Simulation knowledge used |
|------|-------------|--------------------------|
| **simulation-informed** | Uses simulation to train model; no oracle at inference | True $x_{\rm HII}$ for training labels |
| **physics-augmented** | Uses physical kernel form; parameters learned | None at inference |
| **oracle-source** | Source prescription from simulation fed directly | Full halo catalog at inference |
| **oracle-field** | True $x_{\rm HII}$ field provided (upper bound on all statistics) | Full ionization field at inference |
| **observed-only** | Only LAEs above detection threshold | Closest to real observations |

The science answer lives in the gap between observed-only and oracle-field: how much information do the Lyα marks add?

---

## 11. Minimum Viable Implementation (MVP)

**Goal:** validate the physics-constrained pipeline end-to-end before adding complexity.

### MVP Inputs (per snapshot)
```
- LAE positions (N_LAE × 3), cMpc/h
- M_UV (N_LAE,)
- T_IGM = Lobs/Lint (N_LAE,)
- EW_obs (N_LAE,)
- log10(halo mass) (N_LAE,)
- redshift z (scalar)
- downsampled density field (64³)
```

### MVP Outputs
```
- x̂_HII field (64³)                          — primary
- F_b = (F_bright, F_faint, F_diffuse) (3,)  — source fractions
- θ_K = (R, Δ, λ_mfp, A_geom) (4,)           — kernel parameters
- f_esc,i (N_LAE,)                            — per-LAE escape fractions
```

### MVP Architecture (simplified)

1. **Graph build:** k-NN graph, k=16, r_max=15 cMpc/h  
2. **GNN:** 3-layer GATv2Conv, hidden_dim=64, output → $f_{{\rm esc},i}$ via sigmoid  
3. **Scatter:** deposit $w_i = L_i f_{{\rm esc},i}$ onto 64³ grid (trilinear)  
4. **Kernel conv:** 3D FFT convolution with $K(r;\theta_K)$ on 64³ grid  
5. **Unresolved:** $\epsilon_{\rm unres}$ = softmax($F_b$) · [basis density fields, 3 × 64³], density fields from downsampled DM simulation  
6. **Threshold:** soft excursion-set sigmoid  
7. **Loss:** field MSE + power spectrum + global $\langle x_i \rangle$ constraint + prior  

### MVP Validation
- Field: MSE, SSIM, binary IoU vs ATON $x_{\rm HII}$ at 64³  
- Topology: MCF, granulometry, bubble-size distribution, percolation fraction  
- Source: recovered $F_b$ vs true simulation prescription  
- Ablation A1 vs A4, B1 vs B3, C1 vs C4, D1 vs D5  
- Stress test S1 (observed-only) vs S2 (oracle all-halo) vs S4 (wrong-source)  

---

## 12. Directory Structure

```
LAE_pinn/
├── ROADMAP.md                    ← this document
├── config/
│   ├── default.yaml              ← model / training hyperparameters
│   └── ablations/                ← per-ablation yaml overrides
├── data/
│   ├── loader.py                 ← load Sherwood-Relics catalog + grids
│   ├── graph_builder.py          ← k-NN graph construction (PyG)
│   └── preprocessing.py          ← normalisation, downsampling
├── physics/
│   ├── kernels.py                ← K_mfp, K_bub, K_mix (learnable params)
│   ├── scatter.py                ← trilinear deposit: LAE weights → grid
│   ├── excursion_set.py          ← soft photon-budget → x_HII mapping
│   └── unresolved_sources.py     ← F_b · ε_b(x) constrained field
├── models/
│   ├── gnn_encoder.py            ← GATv2 / SAGE environment encoder
│   ├── source_head.py            ← h_i → f_esc, ξ_ion heads
│   └── pinn.py                   ← full PINN: GNN + physics modules
├── training/
│   ├── loss.py                   ← field MSE + PS + MCF + priors
│   ├── train.py                  ← training loop
│   └── schedulers.py             ← LR schedule, warm-up
├── evaluation/
│   ├── topology.py               ← granulometry, BSD, percolation
│   ├── mcf_eval.py               ← MCF on predicted field
│   └── source_recovery.py        ← F_b posterior analysis
├── experiments/
│   ├── run_ablation.py           ← ablation runner (A1–D5)
│   ├── run_stress.py             ← source degeneracy stress tests (S1–S8)
│   └── run_mvp.py                ← end-to-end MVP validation
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_physics_sanity_check.ipynb
    └── 03_results_visualization.ipynb
```

---

## 13. Implementation Phases

### Phase 0 — Data pipeline and physics sanity (1–2 days)
- Load all catalogs and grids for z = 7.14, 6.6, 5.756  
- Downsample 512³ → 64³ Xbox and Dbox  
- Implement all three kernel forms; plot radial profiles  
- Verify: manual scatter of LAE positions with oracle $f_{\rm esc}$ gives roughly correct $J(\mathbf{x})$  
- Verify: soft excursion-set mapping from oracle $J$ gives $\hat{x}_{\rm HII}$ close to ATON truth  

### Phase 1 — MVP GNN + physics pipeline (3–5 days)
- Implement GNN encoder + source head  
- Implement trilinear scatter  
- Implement FFT convolution with learned kernel  
- Implement unresolved source field  
- Wire full forward pass: graph → $\hat{x}_{\rm HII}$, $F_b$, $\theta_K$  
- Train on z = 7.14; validate field loss  

### Phase 2 — Loss engineering and topology metrics (2–3 days)
- Add power spectrum loss  
- Add MCF consistency loss  
- Implement granulometry, BSD, percolation evaluation  
- Confirm topology statistics match ground truth at convergence  

### Phase 3 — Ablations and stress tests (3–5 days)
- Run kernel ablations A1–A4  
- Run source ablations B1–B4  
- Run GNN depth ablations C1–C5  
- Run mark ablations D1–D5  
- Run source degeneracy stress tests S1–S8  

### Phase 4 — Multi-redshift and posterior (3–5 days)
- Train jointly on z = 7.14, 6.6, 5.756  
- Add redshift conditioning  
- Implement ensemble / dropout uncertainty for topology posterior  
- Final: $M_{\rm UV}$ cut analysis and percolation-as-a-function-of-z  

---

## 14. Key Scientific Claims to Support or Refute

1. Lyα marks add information beyond galaxy positions alone for topology inference (D1 vs D5)  
2. An unresolved faint source component is required for accurate topology recovery (B1 vs B3)  
3. The kernel mixture outperforms a single exponential or bubble kernel (A2, A3 vs A4)  
4. The model can partially distinguish bright-dominated from faint-dominated source prescriptions via $F_b$ posteriors (S6 vs S4/S5)  
5. MCF scale is not directly interpretable as bubble radius, but the PINN bubble-radius posterior $p(R|\rm data)$ is (compare $R$ posterior to granulometry BSD)  
6. Observed-only LAE information sets a floor on topology inference achievable with real data (S1 vs S2 gap)  
