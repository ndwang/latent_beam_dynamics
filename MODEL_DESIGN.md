# **Design Document: Latent-Space Causal Transformer for Accelerator Physics**

## **1. High-Level Objective**
To build a scalable, flexible, and physically consistent surrogate model that predicts the evolution of a beam distribution (represented as a VAE latent vector $z$) through a variable sequence of accelerator elements.

**Key Features:**
*   **Causal:** Respects the arrow of time (downstream elements do not affect upstream beam).
*   **Continuous:** Handles elements of variable lengths and positions using continuous embeddings.
*   **Stable:** Uses residual learning (delta-dynamics) to minimize drift over long lattices.
*   **Unified:** Handles heterogeneous components (Drifts, Quads, Cavities) via a single unified feature space.

---

## **2. Problem Formulation**

**Inputs:**
1.  **Initial Beam State:** $z_0 \in \mathbb{R}^{d_{latent}}$ (Encoded via pre-trained VAE).
2.  **Lattice Sequence:** An ordered list of $N$ elements, where the $i$-th element is defined by:
    *   Parameters $\mathbf{x}_i$ (Strength, Voltage, etc.)
    *   Length $L_i$
    *   Longitudinal position $s_i$ (cumulative distance).

**Output:**
*   **Trajectory:** The sequence of beam states $\{z_1, z_2, \dots, z_N\}$ at the exit of each element.

---

## **3. Architecture Pipeline**

### **Module A: The Unified Element Encoder**
*Goal: Transform heterogeneous physical parameters into a dense machine-learning embedding without using discrete "Type IDs".*

**1. Raw Input Vector ($\mathbf{x}_{raw}$)**
Construct a fixed-size union vector for every element. Zeros are used for parameters irrelevant to a specific element type.
$$ \mathbf{x}_{raw} = [ L, \quad K_1, \quad K_2, \quad \text{Angle}, \quad V_{rf}, \quad f_{rf}, \quad \phi_{rf} ] $$

**2. Physics-Informed Preprocessing (Normalization)**
Map values to $\approx \mathcal{O}(1)$.
*   $L \leftarrow L / 1.0 \text{ m}$
*   $K \leftarrow K / 10.0 \text{ m}^{-2}$
*   $V \leftarrow V / 10.0 \text{ MV}$
*   $f \leftarrow f / 1.0 \text{ GHz}$
*   $\phi \leftarrow \phi / 2\pi$

**3. Element Projection**
Map the normalized vector $\mathbf{x}_{norm}$ (dim $D_{in}$) to the model dimension $d_{model}$ using an MLP.
$$ \mathbf{e}_i = \text{MLP}_{enc}(\mathbf{x}_{norm}^{(i)}) $$
*   *Design Note:* This allows a $K=0$ Quadrupole to mathematically equal a Drift.

### **Module B: Continuous Positional Encoding**
*Goal: Encode the absolute longitudinal position $s$ to handle variable element lengths and drift spaces accurately.*

**1. Cumulative Position Calculation**
$$ s_i = \sum_{j=0}^{i-1} L_j + \frac{L_i}{2} \quad \text{(Center of element)} $$

**2. Fourier Feature Basis**
Generate frequencies $\omega_k$ geometrically spaced from $\lambda_{min}=0.01$m to $\lambda_{max}=1000$m.
$$ \mathbf{p}(s_i) = \text{Concat}\left[ \dots, \sin(2\pi \omega_k s_i), \cos(2\pi \omega_k s_i), \dots \right] $$

**3. Integration Strategy**
**Concatenate** position features with element features, then project to mix them.
$$ \mathbf{h}_i = \text{Linear}_{mix}( \text{Concat}[ \mathbf{e}_i, \mathbf{p}(s_i) ] ) $$

---

### **Module C: The Backbone (GPT-Style Transformer)**
*Goal: Model the interactions between the beam history and the current element.*

**1. Input Sequence Construction**
The Transformer input at step $t$ consists of the previous beam state and the current element configuration.
$$ \text{Token}_t = \text{Project}(z_{t-1}) + \mathbf{h}_t $$
*(Alternatively: Cross-Attention where Query=$z_{t-1}$ and Key/Value=$\mathbf{h}_{1:t}$. For a pure GPT decoder, concatenating embeddings is standard.)*

**2. Causal Attention**
Standard Multi-Head Self-Attention with a **Causal Mask** (upper triangular $-\infty$).
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M_{causal}\right)V $$
This ensures prediction of $z_t$ depends only on $\{z_0 \dots z_{t-1}\}$ and $\{e_1 \dots e_t\}$.

---

### **Module D: The Prediction Head (Delta-Dynamics)**
*Goal: Stabilize long-term rollouts by predicting changes rather than absolute states.*

**1. Residual Prediction**
The network predicts the *update* vector $\Delta z_t$.
$$ \Delta z_t = \text{MLP}_{out}(\text{TransformerOutput}_t) $$

**2. State Update**
$$ \hat{z}_t = z_{t-1} + \Delta z_t $$

**Why:** Most elements (drifts, weak quads) cause minimal change. Learning "Identity + Correction" is easier than learning the full transform from scratch.

---

## **4. Implementation Details**

### **Hyperparameters (Suggested Starting Point)**
*   **Latent Dim ($z$):** 32 or 64 (from your VAE).
*   **Model Dim ($d_{model}$):** 256 or 512.
*   **Layers:** 4 to 6 (Deeper is better for nonlinearity, but harder to converge on small data).
*   **Heads:** 8.
*   **Positional Frequencies:** 32 pairs (covering 1cm to 1km).

### **Training Strategy**

**1. Loss Function**
$$ \mathcal{L} = \sum_{t=1}^N \| z_{t}^{GT} - \hat{z}_t \|^2_2 $$
*   *Optional:* Add a Hamiltonian/Symplectic regularization term if your VAE latent space allows calculating emittance.

**2. Autoregressive Rollout (Schedule)**
To prevent "exposure bias" (where the model fails because it never sees its own errors during training):
*   **Epoch 0-10:** Teacher Forcing (Input $z_{t-1}^{GT}$ to predict $z_t$).
*   **Epoch 10+:** Scheduled Sampling (Input predicted $\hat{z}_{t-1}$ with probability $p$, increasing $p$ over time).

---

## **5. Checklist for Publication (The "Story")**

When writing your paper, ensure you highlight these contributions:

1.  **Architecture:** "We propose a **Physics-Aware Causal Transformer** that respects the arrow of time in beam transport."
2.  **Representation:** "We introduce a **Unified Element Embedding** strategy that eliminates discrete type constraints, allowing for continuous topology optimization."
3.  **Stability:** "By utilizing **Delta-Dynamics learning** and **Continuous Fourier Positional Encodings**, we achieve stable rollouts over 100+ meters, outperforming standard RNN baselines."
4.  **Flexibility:** "Unlike fixed-matrix surrogates, our model handles **variable lattice configurations** without retraining."

### **Evaluation Plots to Generate:**
1.  **Ground Truth vs. Prediction:** $\sigma_x$ and $\sigma_y$ along the $s$-axis.
2.  **Phase Space Reconstruction:** Decoder($z_{final}$) images comparing GT vs Model.
3.  **One-Knob Scan:** Vary a Quad strength $K$ and show the beam waist moving smoothly.
4.  **Speed Benchmark:** Model inference time vs. IMPACT-T/ELEGANT simulation time.

This is a model I designed. I have a few questions. In the transformer, if each token is z_{t-1} and h_t, and then the token depends on the previous step.
During training, we have z_{ground truth} and we can train in parallel. But during inference, these will be predicted by the model and we cannot guarantee error won't accumulate and the final prediction deviated from the truth. Our training loss doesn't account for this. If we want to measure this loss, then training must run sequential inference and we lose training efficiency.