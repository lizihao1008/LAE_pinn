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
4. A photoionization equilibrium operator maps total emissivity to $\hat{x}_{\rm HII}(\mathbf{x})$
5. The model is trained by comparing the predicted field to the ATON ground-truth ionization field

---

## 2. Physical Decomposition

### 2.1 Total ionizing emissivity

$\epsilon_{\rm ion}(\mathbf{x}) = \epsilon_{\rm obs}(\mathbf{x}) + \epsilon_{\rm unres}(\mathbf{x})$

### 2.2 Observed LAE source density

$S_{\rm obs}(\mathbf{x}) = \sum_{i \in \rm LAE} w_i \cdot \delta^3(\mathbf{x} - \tilde{\mathbf{x}}_i)$

$w_i = \log_{10}(L_{{\rm Ly}\alpha,i}) \cdot f_{{\rm esc},\theta}(h_i)$

- $\log_{10}(L_{{\rm Ly}\alpha,i})$: log-space Lyα luminosity (avoids float32 overflow at $10^{42}$ erg/s; used as proxy for UV photon production)  
- $f_{{\rm esc},\theta}(h_i) \in (0,1)$: escape fraction predicted by GNN source head from environment embedding $h_i$  
- $\tilde{\mathbf{x}}_i$: scatter position after optional learned RSD correction (see §4)  
- $S_{\rm obs}$ is a **source density** (ionizing luminosity per voxel), not yet a radiation field

### 2.3 Unresolved source density (HOD-based)

$S_{\rm unres}(\mathbf{x}) = \sum_{b=1}^{3} f_{{\rm esc},b} \cdot \epsilon_b(\mathbf{x})$

where $f_{{\rm esc},b} \in (0,1)$ is the **only** learnable parameter per mass bin (via sigmoid), and $\epsilon_b(\mathbf{x})$ is a **fixed**, pre-computed HOD basis field.

$S_{\rm unres}$ is also a **source density** — the same physical quantity as $S_{\rm obs}$, just from unresolved halos rather than individually detected LAEs.

**Why not $\rho^{\alpha_b}$?**  
The previous approach (power-law density bias $\epsilon_b \propto \rho_{\rm dm}^{\alpha_b}$) is an arbitrary polynomial in density with no physical interpretation — it conflates the spatial distribution of halos with the escape fraction of their photons. The HOD approach separates these:

- **Spatial distribution** (fixed, from HOD calibration): where faint halos live
- **Escape fraction** (learned): what fraction of their photons escape

#### HOD basis fields

In linear bias theory:

$\epsilon_b(\mathbf{x}) = 1 + b_b \cdot \delta_{\rm dm}(\mathbf{x})$

where $b_b$ is the linear bias of faint halos in mass bin $b$, measured from cross-correlation with the DM density field. Each $\epsilon_b$ is normalised to unit mean, so $f_{{\rm esc},b}$ absorbs the amplitude.

Three M_UV bins for the faint (unresolved) population:

| Bin | M_UV range | Physical description |
|-----|-----------|---------------------|
| $b=0$ | $(-17.5, -16.5]$ | Near-threshold, brightest unresolved halos |
| $b=1$ | $(-16.5, -15.0]$ | Intermediate |
| $b=2$ | $(-15.0, +\infty)$ | Faintest, most numerous |

#### 2.3.1 Simulation calibration path

With access to the full halo catalog (Sherwood-Relics includes halos well below the detection threshold):

1. Select halos with $M_{\rm UV} > -17.5$ (faint, unresolved)
2. For each mass bin $b$: scatter halo positions to 64³ grid → number density $n_b(\mathbf{x})$
3. Measure linear bias: $b_b = \mathrm{Cov}(n_b, \delta_{\rm dm}) / \mathrm{Var}(\delta_{\rm dm})$
4. Build $\epsilon_b(\mathbf{x}) = 1 + b_b \delta_{\rm dm}(\mathbf{x})$, clip to $\geq 0$, normalise

This is model-free: no HMF assumption, no analytical bias model. The bias is directly measured from the data. Implemented in `data/preprocessing.build_hod_basis_from_simulation()`.

#### 2.3.2 Observations calibration path

For real data where the full halo catalog is unavailable:

1. Measure the auto-correlation function (ACF) of detected LAEs
2. Fit HOD parameters ($M_{\min}, \sigma_{\log M}, M_1, \alpha$) from the ACF
3. Extrapolate the HOD to the faint end ($M_{\rm UV} > -17.5$) using the HMF (Sheth-Tormen / Tinker) to get $n_{\bar{b}}$ and $b_b$ per bin
4. Build $\epsilon_b(\mathbf{x}) = 1 + b_b \delta_{\rm proxy}(\mathbf{x})$ where $\delta_{\rm proxy}$ is a density contrast proxy (smoothed LAE density / bias correction, or WL convergence)

Implemented in `data/preprocessing.build_hod_basis_from_observations()`. **The HODUnresolvedField model is identical in both cases** — only the pre-computed basis arrays differ. This makes the trained $f_{\rm esc}(M_b)$ parameterisation directly transferable from simulation training to real-data inference.

### 2.4 Physical radiative kernels

Three candidates (learnable parameters, fixed functional form):

**Exponential (mean-free-path / diffuse transmission):**
$K_{\rm mfp}(r; \lambda) = \frac{\exp(-r/\lambda)}{4\pi r^2 + \epsilon}$

Represents photon transport with mean free path $\lambda$; appropriate for large-scale coherent transmission.

**Soft bubble (geometric ionization front):**
$K_{\rm bub}(r; R, \Delta) = \sigma\left(\frac{R - r}{\Delta}\right), \quad \sigma = \text{sigmoid}$

Represents a geometrically bounded HII region of radius $R$ with edge width $\Delta$.

**Mixture (default, implemented):**
$K(r; \theta_K) = A_{\rm geom} K_{\rm bub}(r; R, \Delta) + A_{\rm trans} K_{\rm mfp}(r; \lambda)$

$A_{\rm geom} + A_{\rm trans} = 1$ via softmax. Captures both geometric topology and diffuse transmission simultaneously. All five parameters ($R, \Delta, \lambda, A_{\rm geom}, A_{\rm trans}$) are gradient-trained.

The kernel is evaluated on a 64³ grid, FFT-convolved periodically with the source grid (O(G³ log G) per step).

### 2.5 Total ionizing radiation field and normalisation

Both source terms are first summed as **source densities**, then convolved with the same kernel:

$S(\mathbf{x}) = S_{\rm obs}(\mathbf{x}) + S_{\rm unres}(\mathbf{x})$

$J_{\rm total}(\mathbf{x}) = S(\mathbf{x}) * K(r;\theta_K)$

This is physically correct and computationally efficient (one FFT instead of two). By linearity: $S * K = (S_{\rm obs} + S_{\rm unres}) * K = (S_{\rm obs} * K) + (S_{\rm unres} * K)$, so both components are propagated by the same radiative transfer kernel.

Implementation in one forward pass:

1. **Trilinear scatter** $S_{\rm obs}$: deposit $w_i$ onto a 64³ grid via out-of-place `scatter_add` — gradients flow to $w_i$ and hence to $f_{{\rm esc},i}$
2. **HOD source density** $S_{\rm unres}$: compute $\sum_b f_{{\rm esc},b} \epsilon_b(\mathbf{x})$
3. **FFT convolution**: `rfftn(S_obs + S_unres)` × `rfftn(K)` → `irfftn`, periodic boundary conditions

**J normalisation** (applied before excursion mapping):
$J_{\rm norm}(\mathbf{x}) = \frac{J_{\rm total}(\mathbf{x})}{\langle J_{\rm total} \rangle}, \quad \langle J_{\rm total} \rangle\ \text{detached from graph}$

This decouples the absolute scale of $J$ (set by luminosity normalisation and kernel amplitude) from $\alpha_{\rm nHscale}$, ensuring the excursion mapping starts in a physically sensible regime regardless of $J$'s absolute units.

### 2.6 Ionization equilibrium (excursion set mapping)

The photoionization equilibrium equation $\Gamma(1-x_{\rm HII}) = \alpha n_H x_{\rm HII}^2$ has an exact closed-form solution:

$A_i \equiv \frac{J_{{\rm norm},i}}{s}, \qquad \hat{x}_{{\rm HII},i} = \frac{\sqrt{A_i^2 + 4A_i} - A_i}{2}$

where $s = \alpha_{\rm nHscale}$ is a single learnable scalar absorbing $\alpha(T)\cdot\langle n_H \rangle$ and J unit conversion.

Physical limits:

- $A \to 0$: $\hat{x}_{\rm HII} \approx \sqrt{A} \to 0$ (fully neutral, no flux)
- $A = 1$: $\hat{x}_{\rm HII} = \frac{\sqrt{5}-1}{2} \approx 0.618$
- $A \to \infty$: $\hat{x}_{\rm HII} \to 1$ (fully ionized)

After J normalisation with $\langle J_{\rm norm}\rangle = 1$: at initialisation $s=1$, the mean ionization is $\approx 0.618$ — already near typical $\xi_{\rm global}$ values, so $s$ only needs to fine-tune the transition sharpness. The global constraint $\langle\hat{x}_{\rm HII}\rangle = \xi_{\rm global}$ is enforced by the loss term $\mathcal{L}_{\rm xHII}$, which trains $s$ via gradient descent (no binary-search oracle needed).

---

## 3. GNN Message Passing: Physical Interpretation

This is the scientific core of the model. The GNN does not simply interpolate a galaxy → field mapping; each message-passing layer has a specific physical role.

### 3.1 Node features (input to GNN)


| Index | Feature                                 | Physical meaning                                               |
| ----- | --------------------------------------- | -------------------------------------------------------------- |
| 0–2   | $x/L, y/L, z/L$                         | Normalised position in $[0,1]^3$                               |
| 3     | $M_{\rm UV}$                            | UV magnitude → UV photon production rate                       |
| 4     | $\log_{10}(M_h/M_\odot)$                | Halo mass → local dark matter density, host galaxy type        |
| 5     | $T_{\rm IGM} = L_{\rm obs}/L_{\rm int}$ | Direct IGM transmission toward this sightline                  |
| 6     | ${\rm EW}_{\rm obs}$                    | Lyman-alpha equivalent width → resonant transmission signature |
| 7     | $\log_{10}(L_{{\rm Ly}\alpha})$         | Observed Lyα luminosity → lower bound on UV output             |


$T_{\rm IGM}$ and ${\rm EW}_{\rm obs}$ are the two direct windows onto the IGM ionization state along the line of sight.

### 3.2 Edge features

Each directed edge $j \to i$ carries:

- $\boldsymbol{\delta}_{ij} = (\delta x, \delta y, \delta z)$ in cMpc/h: relative position vector (direction + scale information)
- $r_{ij} = |\boldsymbol{\delta}_{ij}|$: separation

The model can therefore learn anisotropic environments (e.g., filaments along LOS vs. transverse structures) from edge geometry.

### 3.3 Message passing layers (3 × GATv2Conv)

**Layer 1 — Immediate neighbourhood ($\sim r_{\rm link}$):**

Each galaxy gathers messages from its $k=16$ nearest neighbours (up to $r_{\rm link} = 15\rm cMpc/h$). A message from galaxy $j$ to galaxy $i$ contains $j$'s node features and the relative position $\boldsymbol{\delta}_{ij}$.

*What gets learned:* Does my immediate neighbourhood consist of bright, high-$T_{\rm IGM}$ galaxies clustered tightly? → I am probably inside an HII bubble. Are my neighbours showing low $T_{\rm IGM}$ (neutral foreground)? → I am probably near a bubble edge or in a neutral region. After Layer 1, $h_i^{(1)}$ encodes the **local galaxy environment** (density, marks of immediate neighbours, edge geometry).

**Layer 2 — Intermediate clustering ($\sim 2r_{\rm link}$):**

Messages from 2-hop neighbours propagate information about the broader environment. The receptive field now extends to $\sim 30\rm cMpc/h$.

*What gets learned:* Is this galaxy part of a protocluster (many bright, closely-clustered neighbours-of-neighbours)? Is it in a cosmic void (sparse environment, low $T_{\rm IGM}$ neighbours throughout)? The 2-hop aggregation allows the model to correlate the **large-scale matter density** (traced by LAE clustering) with the local ionization signal. The GAT attention weights at this layer are particularly informative: they learn which 2-hop neighbours are most relevant for predicting $f_{\rm esc}$ (e.g., a distant bright LAE with very high $T_{\rm IGM}$ tells you you're both in a large connected bubble).

**Layer 3 — Large-scale topology context ($\sim 3r_{\rm link} \approx 45\rm cMpc/h$):**

The effective receptive field now covers a volume comparable to individual large HII bubbles at $z\sim7$ (bubble radii $R \sim 5$–$30\rm cMpc/h$). The model can "see" whether the local overdensity is embedded in a large ionized superstructure or is an isolated bubble.

*What gets learned:* Am I the brightest galaxy in a large overdense region that has collectively ionized a huge bubble? Or am I a bright outlier in a mostly neutral region? This layer builds $h_i^{(3)}$ as a summary of the **multi-scale ionization topology** around galaxy $i$, from which the source head predicts $f_{{\rm esc},i}$.

### 3.4 Physical summary of the GNN

The GNN solves the following inference problem for each LAE:

> *Given this galaxy's intrinsic properties and the observed marks (T_IGM, EW, L_Lyα) of its neighbours at multiple scales, what is the probability that a UV photon produced by this galaxy escapes into the IGM?*

The attention mechanism in GATv2Conv learns to weight the contribution of each neighbour adaptively:

$h_i^{(l+1)} = \text{ELU}\left(\sum_{j\in\mathcal{N}(i)} \alpha_{ij}^{(l)} \mathbf{W}^{(l)} [h_j^{(l)}  e_{ij}]\right)$

where $\alpha_{ij}$ (attention weight) is learned jointly with the rest of the model. A high $\alpha_{ij}$ for a neighbour with very high $T_{\rm IGM}$ means "this neighbour's ionization status is highly informative about my own f_esc." This is physically motivated: in a clustered group where multiple LAEs share a common HII bubble, the inter-galaxy $T_{\rm IGM}$ correlation is a strong signal of shared escape fractions.

### 3.5 Source head outputs


| Output                   | Activation           | Physical range      | Meaning                                              |
| ------------------------ | -------------------- | ------------------- | ---------------------------------------------------- |
| $f_{{\rm esc},i}$        | sigmoid              | $(0, 1)$            | Ionizing photon escape fraction                      |
| $\Delta r_{\parallel,i}$ | tanh × $r_{\rm max}$ | $(\pm 2\rm cMpc/h)$ | Learned LOS position correction (optional; see §2.7) |


---

## 4. Learned RSD Correction (Optional Component)

### 4.1 The problem

`Halo_Position.npy` from Sherwood-Relics is in **redshift space** (positions shifted along LOS by $v_{\rm pec}/(aH)$). The ground-truth `Xbox_grid` is in **real/physical space**. Training the model with RSD-distorted positions against a real-space target creates a systematic mismatch.

In real observations, we cannot invert the RSD per-source (we don't know individual $v_{\rm pec}$). Therefore the solution must be **learned from data**, not applied as a preprocessing step.

### 4.2 Why the GNN can learn the RSD inverse

Peculiar velocities are correlated with the local density field (gravitational collapse): galaxies in overdense regions tend to have convergent velocities (Kaiser compression along LOS), while galaxies in underdense regions have divergent velocities (LOS stretching). The GNN already sees this anisotropy in the k-NN graph geometry: if the same number of neighbours are found in a smaller LOS extent than transverse extent, this signals infall (overdense region, compressed in redshift space).

### 4.3 Architecture

A lightweight MLP branch on the environment embedding $h_i$:

$\Delta r_{\parallel,i} = \tanh\left({\rm MLP}(h_i)\right) \times r_{\rm max}$

with $r_{\rm max} = 2\rm cMpc/h \approx 0.8\rm voxels$ at 64³, which covers the typical RSD amplitude at $z\sim7$ ($\sim 0.5$–$1.5\rm cMpc/h$).

The corrected scatter position:
$\tilde{\mathbf{x}}*i = \mathbf{x}i^{\rm RSD} + \Delta r{\parallel,i}\hat{e}*{\rm LOS}$

### 4.4 Gradient flow

The trilinear scatter is differentiable through the fractional position (sub-voxel part of the position). The gradient chain is:

$\mathcal{L}*{\rm field} \to \hat{x}*{\rm HII} \to J_{\rm total} \to J_{\rm obs}\text{-grid} \xrightarrow{{\rm trilinear}} \tilde{\mathbf{x}}*i \to \Delta r*\parallel \to {\rm MLP}$

For sub-voxel corrections (which RSD shifts are), the trilinear fractional weights provide non-zero gradient. This component is disabled by default (`rsd_correction.enabled: false` in config).

---

## 5. Model Architecture (Current Implementation)

```
LAE catalog (N sources)
Positions: redshift-space (Halo_Position.npy)
Marks: M_UV, log10(M_h), T_IGM, EW_obs, log10(L_Lyα)
        │
        ▼
┌───────────────────────────────┐
│  k-NN Graph Construction      │
│  k=16, r_link ≤ 15 cMpc/h    │
│  Edge attr: δx, δy, δz, |r|   │
│  Node feat: 8-dim (§3.1)      │
└───────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  GNN Encoder (3 × GATv2Conv)           │
│                                        │
│  Layer 1: local neighbourhood          │
│    → who are my neighbours?            │
│    → what are their T_IGM, EW, L_Lyα? │
│  Layer 2: clustering environment       │
│    → am I in a protocluster/void?      │
│    → large-scale density bias          │
│  Layer 3: topology context             │
│    → am I inside a large HII bubble?  │
│    → what's the bubble connectivity?  │
│                                        │
│  Output: h_i ∈ R^32 per LAE           │
└────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Source Head (per-LAE MLP)              │
│                                         │
│  h_i → f_esc,i ∈ (0,1)   [sigmoid]     │
│      → Δr∥_i (optional)  [tanh×r_max]  │
│                                         │
│  Effective weight:                      │
│    w_i = log10(L_Lyα,i) · f_esc,i      │
└─────────────────────────────────────────┘
        │ w_i, x̃_i = x_i^RSD + Δr∥_i ê_LOS
        ▼
┌─────────────────────────────────────────────────────┐
│  Trilinear Scatter  →  S_obs(x)                     │
│                                                     │
│  S_obs(x) = Σ_i w_i δ³(x - x̃_i)                   │
│  out-of-place scatter_add onto 64³ grid             │
│  grad flows: w_i → f_esc,i → source head → GNN      │
└─────────────────────────────────────────────────────┘
        │ S_obs      ┌────────────────────────────────────────┐
        │            │  HOD Unresolved Source Field           │
        │            │                                        │
        │            │  S_unres(x) = Σ_b f_esc_b · ε_b(x)   │
        │            │                                        │
        │            │  ε_b(x) = 1+b_b·δ(x)  [FIXED]        │
        │            │    sim:  measure b_b from halo catalog │
        │            │    obs:  ACF fitting + HOD extrap.     │
        │            │                                        │
        │            │  f_esc_b ∈ (0,1)  [LEARNED]           │
        │            │    b=0: MUV(-17.5,-16.5]               │
        │            │    b=1: MUV(-16.5,-15.0]               │
        │            │    b=2: MUV>-15.0                      │
        │            └────────────────────────────────────────┘
        │ S_obs              │ S_unres
        └──────────┬─────────┘
                   ▼
        S_total(x) = S_obs(x) + S_unres(x)
                   │
                   │  * K(r; θ_K)  [one FFT convolution]
                   │  K = A_geom·K_bub(R,Δ) + A_trans·K_mfp(λ)
                   ▼
        J_total(x) = S_total * K     [64³ radiation field]
                   │
                   │  J_norm = J_total / ⟨J_total⟩  (⟨·⟩ detached)
                   ▼
┌──────────────────────────────────────────────────┐
│  Photoionization Equilibrium                     │
│                                                  │
│  A(x) = J_norm(x) / alpha_nH_scale              │
│  x̂_HII(x) = (√(A²+4A) − A) / 2                │
│                                                  │
│  alpha_nH_scale: trained by global_xHII loss     │
│  (no binary-search oracle; pure gradient descent)│
└──────────────────────────────────────────────────┘
                   │
                   ▼
        x̂_HII(x)  64³ ionization field
        f_esc_b    per-bin HOD escape fractions (b=0,1,2)
        θ_K = (R, Δ, λ, A_geom, A_trans)  kernel parameters
        f_esc,i    per-LAE escape fractions
        alpha_nH_scale  global ionization efficiency
```

---

## 6. Parameters

### 6.1 Learnable (trained by gradient descent)


| Symbol                          | Description                     | Constraint          | Gradient path                                   |
| ------------------------------- | ------------------------------- | ------------------- | ----------------------------------------------- |
| GNN weights                     | Environment encoder ($h_i$)     | standard            | all losses → x̂_HII → J → scatter → w_i → f_esc |
| $f_{{\rm esc},i}$               | Per-LAE escape fraction         | sigmoid → (0,1)     | field MSE, BCE, PS                              |
| $R$                             | Soft bubble radius              | softplus → R > 0    | field MSE → J_obs → K_bub                       |
| $\Delta$                        | Bubble edge width               | softplus → Δ > 0    | field MSE → J_obs → K_bub                       |
| $\lambda_{\rm mfp}$             | Mean free path                  | softplus → λ > 0    | field MSE → J_obs → K_mfp                       |
| $A_{\rm geom}, A_{\rm trans}$   | Kernel mixture weights          | softmax → sum=1, ≥0 | field MSE → J_obs → K                           |
| $f_{{\rm esc},b}$ ($b=0,1,2$)   | HOD escape fraction per mass bin | sigmoid → (0,1)    | field MSE → J_unres → ε_b(x)                    |
| $\alpha_{\rm nHscale}$          | Global ionization efficiency    | softplus → s > 0    | xHII loss → x̂_HII                              |
| $\Delta r_{\parallel,i}$ (opt.) | Per-LAE LOS position correction | tanh × r_max        | field MSE → J → scatter pos                     |


### 6.2 Fixed physical form (not learnable)


| Item                                                           | Description                                            |
| -------------------------------------------------------------- | ------------------------------------------------------ |
| Kernel functional form                                         | Exponential / sigmoid (physics-motivated)              |
| $0 < f_{\rm esc} < 1$                                          | Physical range for all escape fractions                |
| $\epsilon_b(\mathbf{x}) = 1 + b_b \delta_{\rm dm}(\mathbf{x})$ | Linear HOD bias field; calibrated before training     |
| $b_b$ linear bias per bin                                      | Measured from halo catalog or ACF fitting; not learned |
| Exact equilibrium formula                                      | $x = (\sqrt{A^2+4A}-A)/2$, not a sigmoid approximation |


### 6.3 Priors (regularisation terms in loss)


| Prior                   | Target                                                     | Form                                                   |
| ----------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| Smoothness prior        | $\epsilon_{\rm unres}$                                     | TV penalty on unresolved emissivity field               |
| Global neutral fraction | $\langle\hat{x}_{\rm HII}\rangle \approx \xi_{\rm global}$ | MSE penalty, trains $\alpha_{\rm nHscale}$             |
| (future) $f_{\rm esc}$ prior | $f_{{\rm esc},b}$                                   | Physics-motivated: $f_{\rm esc}$ decreasing with $M_h$ |


---

## 7. Loss Function

$\mathcal{L} = w_{\rm field}\mathcal{L}_{\rm field} + w_{\rm PS}\mathcal{L}_{\rm PS} + w_{\rm BCE}\mathcal{L}_{\rm BCE} + w_{\xi}\mathcal{L}_{\xi} + w_{\rm prior}\mathcal{L}_{\rm prior}$

### 7.1 Field loss (primary spatial structure)

$\mathcal{L}_{\rm field} = \|\hat{x}_{\rm HII} - x_{\rm HII}^{\rm true}\|_2^2$

Targets the coarsened 64³ ATON Xbox. This is the primary gradient source for the kernel shape ($R, \Delta, \lambda$) and GNN escape fractions.

### 7.2 Power spectrum loss (spatial scale structure)

$\mathcal{L}_{\rm PS} = \|P_{\hat{x}}(k) - P_{x^{\rm true}}(k)\|_2^2$

Penalises wrong characteristic ionization scales independently of voxel-level errors. Particularly effective at training the kernel radius $R$.

### 7.3 Binary topology loss

$\mathcal{L}_{\rm BCE} = {\rm BCE}(\hat{x}_{\rm HII},\,[x_{\rm HII}^{\rm true} > 0.5])$

Pushes the model toward binary (ionized/neutral) predictions consistent with the sharp bubble topology of EoR. NaN-guarded via `nan_to_num` before clamp.

### 7.4 Global ionization constraint

$\mathcal{L}_{\xi} = \left( \langle \hat{x}_{\rm HII} \rangle - \xi_{\rm global} \right)^2$

**No binary-search oracle.** This loss trains $\alpha_{\rm nHscale}$ directly via gradient descent. The J normalisation ensures fast convergence: at initialisation $\langle J_{\rm norm}\rangle = 1$, $s = 1$, $\langle\hat{x}\rangle \approx 0.62$, already near typical $\xi_{\rm global}$.

### 7.5 Prior regularisation

$\mathcal{L}_{\rm prior} = \gamma \|\nabla \epsilon_{\rm unres}\|_1$

The LF-prior KL term on $F_b$ is removed: the HOD basis already encodes physically motivated clustering; only the TV smoothness penalty on $\epsilon_{\rm unres}$ is retained.

Default weights (`config/default.yaml`):


| Component        | Weight |
| ---------------- | ------ |
| `field_mse`      | 1.0    |
| `power_spectrum` | 0.1    |
| `binary_bce`     | 0.5    |
| `global_xHII`    | 5.0    |
| `prior`          | 0.1    |


---

## 8. Known Gradient Flow Issues (Fixed)


| Bug                                      | Symptom                                                        | Root cause                                                              | Fix                                                                            |
| ---------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| GNN receives no gradient from field loss | f_esc never changes                                            | `scatter_add`_ in-place on leaf tensor breaks autograd                  | Out-of-place `scatter_add` over all 8 trilinear corners                        |
| Kernel parameters receive no gradient    | R, λ stuck at init                                             | `with torch.no_grad():` in `make_3d_kernel_grid` detached kernel params | Removed `no_grad` block                                                        |
| NaN in BCE loss                          | `RuntimeError: all elements should be between 0 and 1`         | NaN in x_pred survives `.clamp()` (NaN comparisons return False)        | `nan_to_num` before clamp                                                      |
| All x_pred NaN every epoch               | xHII loss = 0.347 = log(2)/2 (all NaN)                         | Float32 overflow: raw L_Lyα (~10^42 erg/s) overflows float32 max        | Switched to log10(L_Lyα); clip negatives before log                            |
| alpha_nH_scale never converges           | x_pred ≈ 0 everywhere; xHII loss barely moves                  | Binary-search gain counteracted every gradient step on scale            | Removed gain oracle; pure gradient via xHII loss                               |
| x_pred ≈ 0 even without oracle           | J_total ~ 10⁻³ << alpha_nH_scale ~ 1 (scale mismatch)          | J absolute scale not matched to excursion input range                   | Normalise J_total by its detached spatial mean before excursion                |
| RSD mismatch                             | Predicted field derived from RSD positions vs. real-space Xbox | Halo_Position is in redshift space; Xbox is in physical space           | Optional learned Δr∥ head in SourceHead (tanh-bounded, gradient via trilinear) |


---

## 9. Baselines


| Baseline                    | Description                                                   |
| --------------------------- | ------------------------------------------------------------- |
| **MCF only**                | Traditional summary statistic; no field prediction            |
| **Density-only**            | $\hat{x}*{\rm HII} \propto \rho*{\rm dm}$ (pure density bias) |
| **LAE density smooth**      | Gaussian-smooth LAE number density, no marks                  |
| **LAE marks, fixed kernel** | Physical kernel but kernel parameters not learned             |
| **Black-box U-Net**         | Galaxy density → field, no physics structure                  |
| **Oracle-source PINN**      | Use simulation's true source prescription as fixed input      |


---

## 10. Ablation Design

### 10.1 Kernel ablation


| Variant | Kernel                                    |
| ------- | ----------------------------------------- |
| A1      | Fixed Gaussian σ = 5 cMpc/h (no learning) |
| A2      | Learned exponential only ($K_{\rm mfp}$)  |
| A3      | Learned soft bubble only ($K_{\rm bub}$)  |
| A4      | **Learned mixture (default)**             |


### 10.2 Unresolved source ablation


| Variant | Unresolved sources                                              |
| ------- | --------------------------------------------------------------- |
| B1      | No unresolved sources: $\epsilon_{\rm unres} = 0$               |
| B2      | Unresolved = free neural field (no physics)                     |
| B3      | Power-law density bias $\epsilon_b \propto \rho^{\alpha_b}$ (old) |
| B4      | **HOD linear bias field, learned $f_{\rm esc}$ (default)**      |
| B5      | Oracle: scatter faint halos directly onto grid                  |


### 10.3 GNN encoder ablation


| Variant | Encoder                                              |
| ------- | ---------------------------------------------------- |
| C1      | No GNN: $f_{\rm esc,i}$ = global constant            |
| C2      | Node features only (no message passing)              |
| C3      | 1-layer GNN                                          |
| C4      | **3-layer GATv2 (default)**                          |
| C5      | Oracle: $f_{\rm esc,i}$ from simulation ground truth |


### 10.4 Mark ablation


| Variant | Input marks             |
| ------- | ----------------------- |
| D1      | Positions only          |
| D2      | + $M_{\rm UV}$          |
| D3      | + $T_{\rm IGM}$         |
| D4      | + EW                    |
| D5      | **All marks (default)** |


### 10.5 RSD correction ablation


| Variant | Position treatment                                    |
| ------- | ----------------------------------------------------- |
| E1      | RSD positions, no correction (biased baseline)        |
| E2      | **No correction, real-space loss (default training)** |
| E3      | Learned RSD correction Δr∥ head                       |
| E4      | Oracle: invert RSD using true $v_{\rm pec}$           |


---

## 11. Source Degeneracy Stress Tests


| Experiment                  | Source model                              | Expected diagnostic                                    |
| --------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| **S1: observed-only**       | Detected LAEs only ($M_{\rm UV} < -17.5$) | Floor on inference from real observations              |
| **S2: oracle all-halo**     | All simulation halos                      | Upper bound with complete source catalog               |
| **S3: fixed-source oracle** | True source prescription from simulation  | Upper bound for known $f_{\rm esc}$                    |
| **S4: wrong-source stress** | Massive-only ($M_h > 10^{11}M_\odot$)     | $F_b$ posterior shifts to absorb missing faint sources |
| **S5: wrong-source faint**  | Faint-only ($M_h < 10^{10}M_\odot$)       | Kernel radius expands to compensate?                   |
| **S6: learned mixture**     | Model infers $F_b$ (default)              | $F_b$ posterior width = degeneracy measure             |
| **S7: ensemble posterior**  | Train z=7.14,6.6; test z=5.756            | Generalisation to higher ionization fraction           |
| **S8: wrong-z stress**      | Model trained at z=7.14 applied to z=6.6  | Redshift generalisation                                |


**Primary degeneracy diagnostic:** compare recovered $f_{{\rm esc},b}$ across S1–S6. If $f_{{\rm esc},b}$ is poorly constrained even with good field reconstruction, the source model is degenerate with topology. The HOD basis separates spatial distribution (fixed) from escape fraction (learned), making the degeneracy narrower than the old $F_b$ mixing fractions.

---

## 12. Topology Statistics

Computed on both $\hat{x}*{\rm HII}$ and $x*{\rm HII}^{\rm true}$ at threshold $> 0.5$:


| Statistic                         | Description                                              | Tool                  |
| --------------------------------- | -------------------------------------------------------- | --------------------- |
| Binary mark MCF                   | MCF with $m_i = [\hat{x}_{\rm HII}(\mathbf{x}_i) > 0.5]$ | `corrfunc` + custom   |
| TIGM mark MCF                     | MCF with $m_i = T_{\rm IGM,i}$                           | existing pipeline     |
| Granulometry $G(r)$               | Volume fraction vs. erosion radius                       | `scipy.ndimage`       |
| Bubble size distribution $P(R_b)$ | Watershed / SDF                                          | custom                |
| Percolation fraction              | Largest connected component / total ionized volume       | `scipy.ndimage.label` |
| Ion–density cross-correlation     | $\xi_{x\rho}(r)$                                         | FFT-based             |


---

## 13. Directory Structure

```
LAE_pinn/
├── ROADMAP.md
├── config/
│   └── default.yaml              ← model / training hyperparameters
├── data/
│   ├── loader.py                 ← SimSnapshot dataclass; clip L_Lyα < 0
│   ├── graph_builder.py          ← k-NN PyG graph; edge attrs; hod_basis attachment
│   └── preprocessing.py          ← log10(L_Lyα); feature normalisation; J weights
│                                    HODCalibration; build_hod_basis_from_simulation()
│                                    build_hod_basis_from_observations()
├── physics/
│   ├── kernels.py                ← K_mfp, K_bub, K_mix; gradients through kernel params
│   ├── scatter.py                ← out-of-place trilinear scatter_add; FFT convolve
│   ├── excursion_set.py          ← exact equilibrium x=(√(A²+4A)-A)/2; alpha_nH_scale
│   └── unresolved_sources.py     ← HODUnresolvedField: f_esc_b · ε_b(x); lf_kl_loss()=0
├── models/
│   ├── gnn_encoder.py            ← GATv2Conv / SAGEConv; multi-layer with edge_attr
│   ├── source_head.py            ← f_esc (sigmoid); optional Δr∥ (tanh, RSD correction)
│   └── pinn.py                   ← full forward pass; J normalisation before excursion
├── training/
│   ├── loss.py                   ← field MSE + PS + BCE + xHII + prior
│   └── train.py                  ← training loop; NaN guard; diagnostic log
└── experiments/
    ├── run_mvp.py
    ├── run_ablation.py
    └── run_stress.py
```

---

## 14. Implementation Status


| Phase         | Component                                              | Status     |
| ------------- | ------------------------------------------------------ | ---------- |
| Data pipeline | Loader, graph builder, preprocessing                   | ✅ complete |
| Physics       | Kernels (mfp, bubble, mixture)                         | ✅ complete |
| Physics       | Trilinear scatter (out-of-place, differentiable)       | ✅ complete |
| Physics       | FFT convolution (periodic, grad through kernel params) | ✅ complete |
| Physics       | Excursion set (exact equilibrium, J normalisation)     | ✅ complete |
| Physics       | HOD unresolved source (f_esc_b × linear bias, 2 paths)  | ✅ complete |
| Model         | GNN encoder (GATv2, 3-layer, edge attrs)               | ✅ complete |
| Model         | Source head (f_esc, optional Δr∥ RSD correction)       | ✅ complete |
| Model         | Full PINN forward pass                                 | ✅ complete |
| Training      | Loss function (field + PS + BCE + xHII + prior)        | ✅ complete |
| Training      | Training loop (NaN guard, diagnostics, scheduler)      | ✅ complete |
| Training      | J normalisation fix (alpha_nH_scale regime)            | ✅ complete |
| Evaluation    | Topology statistics (granulometry, BSD, percolation)   | 🔲 pending |
| Experiments   | Ablations A–E                                          | 🔲 pending |
| Experiments   | Source degeneracy stress tests S1–S8                   | 🔲 pending |
| Experiments   | Multi-redshift joint training                          | 🔲 pending |


---

## 15. Key Scientific Claims to Support or Refute

1. Lyα marks add information beyond galaxy positions alone for topology inference (D1 vs D5)
2. An unresolved faint source component is required for accurate topology recovery (B1 vs B3)
3. The mixture kernel outperforms a single exponential or bubble kernel (A2, A3 vs A4)
4. The model can partially distinguish bright-dominated from faint-dominated source prescriptions via $f_{{\rm esc},b}$ posteriors (S6 vs S4/S5); the HOD separation of spatial distribution from escape fraction narrows the degeneracy compared to unconstrained $F_b$ mixing
5. Multi-scale message passing (3-layer) recovers more topology than single-scale (C3 vs C4), with the third layer contributing large-bubble context
6. Observed-only LAE information sets a floor on topology inference achievable with real data (S1 vs S2 gap)
7. The learned RSD correction reduces systematic bias in field topology metrics by $\lesssim 0.5$ voxel (E3 vs E1)

