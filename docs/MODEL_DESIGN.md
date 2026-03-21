# **Design Document: Latent-Space Causal Transformer for Accelerator Physics**

## **1. High-Level Objective**
To build a scalable, flexible, and physically consistent surrogate model that predicts the evolution of a beam distribution (represented as a VAE latent vector $z$) through a variable sequence of accelerator elements.

**Key Features:**
*   **Causal:** Respects the arrow of time (downstream elements do not affect upstream beam).
*   **Continuous:** Handles elements of variable lengths and positions using continuous embeddings.
*   **Stable:** Uses residual learning (delta-dynamics) to minimize drift over long lattices.
*   **Unified:** Handles heterogeneous components (Drifts, Quads, Cavities) via a single unified feature space.

---

## **2. Problem Formulation**

**Inputs:**
1.  **Initial Beam State:** $z_0 \in \mathbb{R}^{d_{latent}}$ (Encoded via pre-trained VAE).
2.  **Lattice Sequence:** An ordered list of $N$ elements, where the $i$-th element is defined by:
    *   Parameters $\mathbf{x}_i$ (Strength, Voltage, etc.)
    *   Length $L_i$
    *   Longitudinal position $s_i$ (cumulative distance).

**Output:**
*   **Trajectory:** The sequence of beam states $\{z_1, z_2, \dots, z_N\}$ at the exit of each element.

---

## **3. Shared Architecture Components**

Modules A and B are shared by both model variants (TrackingTransformer and LatticeTransformer).

### **Module A: The Unified Element Encoder**
*Goal: Transform heterogeneous physical parameters into a dense machine-learning embedding without using discrete "Type IDs".*

**1. Raw Input Vector ($\mathbf{x}_{raw}$)**
Construct a fixed-size union vector for every element. Zeros are used for parameters irrelevant to a specific element type.
$$ \mathbf{x}_{raw} = [ L, \quad K_1, \quad K_2, \quad \text{Angle}, \quad V_{rf}, \quad f_{rf}, \quad \phi_{rf} ] $$

**2. Physics-Informed Preprocessing (Normalization)**
Map values to $\approx \mathcal{O}(1)$.
*   $L \leftarrow L / 1.0 \text{ m}$
*   $K \leftarrow K / 10.0 \text{ m}^{-2}$
*   $V \leftarrow V / 10.0 \text{ MV}$
*   $f \leftarrow f / 1.0 \text{ GHz}$
*   $\phi \leftarrow \phi / 2\pi$

**3. Element Projection**
Map the normalized vector $\mathbf{x}_{norm}$ (dim $D_{in}$) to the model dimension $d_{model}$ using an MLP.
$$ \mathbf{e}_i = \text{MLP}_{enc}(\mathbf{x}_{norm}^{(i)}) $$
*   *Design Note:* This allows a $K=0$ Quadrupole to mathematically equal a Drift.

### **Module B: Continuous Positional Encoding**
*Goal: Encode the absolute longitudinal position $s$ to handle variable element lengths and drift spaces accurately.*

**1. Cumulative Position Calculation**
$$ s_i = \sum_{j=0}^{i-1} L_j \quad \text{(Start of element)} $$

**2. Fourier Feature Basis**
Generate frequencies $\omega_k$ geometrically spaced from $\lambda_{min}=0.01$m to $\lambda_{max}=1000$m.
$$ \mathbf{p}(s_i) = \text{Concat}\left[ \dots, \sin(2\pi \omega_k s_i), \cos(2\pi \omega_k s_i), \dots \right] $$

**3. Integration Strategy**
**Concatenate** position features with element features, then project to mix them.
$$ \mathbf{h}_i = \text{Linear}_{mix}( \text{Concat}[ \mathbf{e}_i, \mathbf{p}(s_i) ] ) $$

---

## **3a. TrackingTransformer Architecture**

### **Module C: The Backbone (GPT-Style Transformer)**
*Goal: Model the interactions between the beam history and the current element.*

**1. Input Sequence Construction (Fusion Modes)**
The Transformer input at step $t$ fuses the previous beam state with the current element embedding. Three fusion strategies are supported:

*   **Add** (legacy): $\text{Token}_t = \text{Project}(z_{t-1}) + \mathbf{h}_t$
*   **Concat**: $\text{Token}_t = \text{Linear}_{fuse}(\text{Concat}[\text{Project}(z_{t-1}),\; \mathbf{h}_t])$
*   **Bilinear**: $\text{Token}_t = \text{Linear}_{fuse}(\text{Concat}[\mathbf{z}_p,\; \mathbf{h}_t,\; \mathbf{z}_p \odot \mathbf{h}_t])$ where $\mathbf{z}_p = \text{Project}(z_{t-1})$

Concat and Bilinear give the network more capacity to model interactions between beam state and element parameters, at the cost of a small additional projection layer.

**2. Causal Attention**
Standard Multi-Head Self-Attention with a **Causal Mask** (upper triangular $-\infty$).
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M_{causal}\right)V $$
This ensures prediction of $z_t$ depends only on $\{z_0 \dots z_{t-1}\}$ and $\{e_1 \dots e_t\}$.

### **Module D: The Prediction Head (Delta-Dynamics)**
*Goal: Stabilize long-term rollouts by predicting changes rather than absolute states.*

**1. Residual Prediction**
The network predicts the *update* vector $\Delta z_t$.
$$ \Delta z_t = \text{MLP}_{out}(\text{TransformerOutput}_t) $$

**2. State Update**
$$ \hat{z}_t = z_{t-1} + \Delta z_t $$

**Why:** Most elements (drifts, weak quads) cause minimal change. Learning "Identity + Correction" is easier than learning the full transform from scratch.

### **Training Strategy**

**1. Autoregressive Rollout Schedule**
To prevent "exposure bias" (where the model fails because it never sees its own errors during training):
*   **Epoch 0-10:** Teacher Forcing (Input $z_{t-1}^{GT}$ to predict $z_t$).
*   **Epoch 10+:** Scheduled Sampling (Input predicted $\hat{z}_{t-1}$ with probability $p$, increasing $p$ over time).

---

## **3b. LatticeTransformer Architecture**

*Goal: Eliminate exposure bias by removing autoregressive dependence on predicted beam states, while retaining causal element-to-element attention.*

The TrackingTransformer feeds each predicted $\hat{z}_{t-1}$ back as input to predict $z_t$. This creates a train/inference mismatch (exposure bias): during training the model sees ground-truth states, but at inference it sees its own predictions, and errors compound. The LatticeTransformer avoids this entirely — the initial beam state $z_0$ conditions the transformer globally, and all elements are processed in a single parallel forward pass.

### **Module C': Adaptive Layer Norm Conditioning**
*Goal: Inject beam-state information into every transformer layer without sequential token dependencies.*

**1. Beam Conditioner**
A small MLP maps $z_0$ to per-layer Adaptive Layer Norm (AdaLN) parameters:
$$ \mathbf{c} = \text{MLP}_{cond}(z_0) \in \mathbb{R}^{d_{model}} $$
$$ [\gamma_1^{(\ell)}, \beta_1^{(\ell)}, \gamma_2^{(\ell)}, \beta_2^{(\ell)}]_{\ell=1}^{L} = \text{Linear}_{proj}(\mathbf{c}) $$

All $4 \times L \times d_{model}$ parameters are produced in a single forward pass.

**2. AdaLN Transformer Layers**
Each layer applies Adaptive Layer Norm instead of standard LayerNorm:
$$ \text{AdaLN}(\mathbf{x}; \gamma, \beta) = \gamma \odot \text{LayerNorm}(\mathbf{x}) + \beta $$

The layer structure is pre-norm with residual connections:
$$ \mathbf{x} \leftarrow \mathbf{x} + \text{Attn}(\text{AdaLN}(\mathbf{x}; \gamma_1, \beta_1)) $$
$$ \mathbf{x} \leftarrow \mathbf{x} + \text{FFN}(\text{AdaLN}(\mathbf{x}; \gamma_2, \beta_2)) $$

**3. Initialization**
The projection layer is zero-initialized with bias terms set so that $\gamma=1, \beta=0$. This means AdaLN begins as standard LayerNorm, and beam conditioning is learned gradually.

**4. Causal Attention**
Same causal mask as TrackingTransformer — downstream elements cannot attend to upstream elements.

### **Module D': Trajectory Recovery via Cumulative Sum**
*Goal: Recover the full beam trajectory from per-element updates without sequential state feeding.*

**1. Parallel Delta Prediction**
The transformer produces all $N$ updates simultaneously:
$$ \Delta z_t = \text{MLP}_{out}(\text{TransformerOutput}_t), \quad t = 1 \dots N $$

**2. Cumulative State Update**
$$ \hat{z}_t = z_0 + \sum_{i=1}^{t} \Delta z_i $$

This replaces the sequential $z_{t-1} + \Delta z_t$ update with a single `cumsum` operation. Every predicted state is anchored to $z_0$, so errors in one element's $\Delta z$ do not feed into subsequent predictions.

### **Training Strategy**
No scheduled sampling needed — the model is fully parallel and sees the same inputs at training and inference time. Standard gradient descent on the trajectory MSE loss is sufficient.

### **Trade-offs vs. TrackingTransformer**

| | TrackingTransformer | LatticeTransformer |
|---|---|---|
| Beam state input | $z_{t-1}$ per step (autoregressive) | $z_0$ globally (AdaLN) |
| Training | Teacher forcing → scheduled sampling | Fully parallel, single mode |
| Inference | Sequential rollout | Single forward pass |
| Exposure bias | Mitigated by scheduled sampling | Eliminated by design |
| Expressiveness | Each step sees updated beam state | Each step sees only $z_0$ |

---

## **4. Implementation Details**

### **Hyperparameters (Suggested Starting Point)**
*   **Latent Dim ($z$):** 32 or 64 (from your VAE).
*   **Model Dim ($d_{model}$):** 256 or 512.
*   **Layers:** 4 to 6 (Deeper is better for nonlinearity, but harder to converge on small data).
*   **Heads:** 8.
*   **Positional Frequencies:** 32 pairs (covering 1cm to 1km).

### **Loss Function**
$$ \mathcal{L} = \sum_{t=1}^N \| z_{t}^{GT} - \hat{z}_t \|^2_2 $$
*   *Optional:* Add a Hamiltonian/Symplectic regularization term if your VAE latent space allows calculating emittance.
*   Used by both architectures.

---

### **Evaluation Plots to Generate:**
1.  **Ground Truth vs. Prediction:** $\sigma_x$ and $\sigma_y$ along the $s$-axis.
2.  **Phase Space Reconstruction:** Decoder($z_{final}$) images comparing GT vs Model.
3.  **One-Knob Scan:** Vary a Quad strength $K$ and show the beam waist moving smoothly.
4.  **Speed Benchmark:** Model inference time vs. IMPACT-T/ELEGANT simulation time.
