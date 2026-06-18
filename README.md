# Comparison of Diffusion Models for Text-to-Image Generation — DDPM · DDIM · LDM

> **Final Project for the cource Mathematical Methods of Machine Learning** — Ukrainian Catholic University, Faculty of Applied Sciences, Business Analytics Program

> **Authors:** Anastasiia Dynia · Yuliia Vistak· Viktoriia Stetsyshyn — *07 April 2025*

This project implements and **comparatively evaluates** three families of **denoising diffusion generative models** for **text-conditioned handwritten-digit generation** on MNIST. The central research question: *how do the sampler (stochastic vs. deterministic), the generation space (pixel vs. latent), and the verbosity of the text prompt affect generation quality?*

| Model | What it is | Where it denoises | Key trade-off |
|-------|-----------|-------------------|----------------|
| **DDPM** | Denoising Diffusion Probabilistic Model | Pixel space (28×28) | Stochastic, Markovian, slow (1000 steps) |
| **DDIM** | Denoising Diffusion Implicit Model | Pixel space (28×28) | Deterministic, non-Markovian, ~20× faster |
| **LDM** | Latent Diffusion Model | Latent space (16-D) | Cheapest compute; **best quality overall** |

Each model is conditioned on text prompts (e.g. `"a handwritten digit seven"`) via a pretrained Transformer text encoder and **cross-attention**, and is evaluated with a full battery of generative metrics: **FID, Inception Score, PRDC (Precision/Recall/Density/Coverage), and CLIP score**.

> **Headline finding:** The **Latent Diffusion Model with long prompts** decisively outperforms all pixel-space configurations on *every* metric (FID **55.68** vs. ≥108 for any pixel-space model). Among pixel-space models, **DDIM with short numeric prompts** is the strongest. Counter-intuitively, **long, verbose prompts hurt pixel-space models** but are handled best by the LDM.

All code lives in a single self-contained notebook: [`final_ddpm_ddim_ldm.ipynb`](final_ddpm_ddim_ldm.ipynb), designed to run on a free **Colab / Kaggle T4 GPU**.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [The Mathematics of Diffusion Models](#-the-mathematics-of-diffusion-models)
   - [DDPM — forward process](#1-ddpm-the-forward-diffusion-process)
   - [DDPM — reverse process & ELBO](#2-ddpm-the-reverse-process--elbo)
   - [DDPM — simplified training loss](#3-ddpm-the-simplified-training-objective)
   - [DDPM sampling](#4-ddpm-sampling)
   - [DDIM — deterministic sampling](#5-ddim-deterministic-non-markovian-sampling)
   - [DDIM — loss derivation & the role of variance](#6-ddim-loss-and-the-role-of-variance-σ)
   - [Latent Diffusion (LDM)](#7-latent-diffusion-models-ldm)
3. [Text Conditioning](#-text-conditioning)
4. [Architecture](#-architecture)
5. [Evaluation Metrics](#-evaluation-metrics)
6. [Experimental Design](#-experimental-design)
7. [Results](#-results)
8. [Repository Structure](#-repository-structure)
9. [Getting Started](#-getting-started)
10. [References](#-references)

---

## Project Overview

Diffusion models generate data by learning to **reverse a gradual noising process**: start from a clean image, slowly destroy it with Gaussian noise until it becomes pure static, and train a neural network to undo that destruction step by step. To generate a *new* image, start from random noise and let the network denoise it back into something realistic.

Tools such as DALL·E, Midjourney, and Stable Diffusion all build on this idea. This project reproduces the three foundational variants at small scale and studies three axes:

1. **Sampler** — the *stochastic, Markovian* DDPM vs. the *deterministic, non-Markovian* DDIM.
2. **Generation space** — *pixel space* (DDPM/DDIM) vs. a learned *latent space* (LDM, the basis of Stable Diffusion).
3. **Prompt granularity** — three ways of describing the same digit:
   - `numeric`  `"7"`
   - `digit`  `"digit 7"`
   - `long`  `"a handwritten digit seven"`

---

## The Mathematics of Diffusion Models

Let $x_0 \sim q(x_0)$ be a real image. Diffusion models define two Markov chains over $T$ timesteps (here $T = 1000$): a fixed *forward* chain that adds noise, and a learned *reverse* chain that removes it.

### 1. DDPM — The Forward (Diffusion) Process

The forward process gradually corrupts $x_0$ into noise according to a fixed **variance schedule** $\beta_1, \dots, \beta_T$ (here **linear**, from $10^{-4}$ to $0.02$). It is a Markov chain:

$$q(x_{1:T} \mid x_0) := \prod_{t=1}^{T} q(x_t \mid x_{t-1}), \qquad q(x_t \mid x_{t-1}) := \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t \mathbf{I}\right)$$

Defining $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$, *any* timestep can be sampled directly in **closed form**:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t)\mathbf{I}\right)
\quad\Longrightarrow\quad
\boxed{\,x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0,\mathbf{I})\,}$$

This single equation is exactly `get_noise_for_timestep`:

```python
def get_noise_for_timestep(self, t, x0):
    noise = torch.randn_like(x0)
    s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)          # √ᾱₜ
    s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) # √(1-ᾱₜ)
    return s1 * x0 + s2 * noise, noise
```

As $t \to T$, $\bar{\alpha}_t \to 0$, so $x_T \approx \mathcal{N}(0, \mathbf{I})$ — pure noise.

### 2. DDPM — The Reverse Process & ELBO

The reverse chain is learned, starting from the prior $p(x_T) = \mathcal{N}(0,\mathbf{I})$:

$$p_\theta(x_{0:T}) := p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t), \qquad p_\theta(x_{t-1} \mid x_t) := \mathcal{N}\!\left(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\right)$$

The marginal $p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}$ is intractable, so the model is trained by maximizing the **Evidence Lower Bound (ELBO)**, which decomposes into a sum of KL divergences between the tractable forward posterior $q(x_{t-1}\mid x_t, x_0)$ and the learned reverse step $p_\theta(x_{t-1}\mid x_t)$:

$$\log p_\theta(x_0) \ge \mathbb{E}_q\Big[\underbrace{-D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{\text{prior matching}} - \sum_{t>1}\underbrace{D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{\text{denoising matching}} + \log p_\theta(x_0\mid x_1)\Big]$$

### 3. DDPM — The Simplified Training Objective

Ho et al. (2020) showed that instead of predicting the mean directly, it is far more effective to **predict the noise** $\epsilon$. With the reparameterization from §1, the ELBO collapses to a simple weighted denoising MSE:

$$\mathcal{L} = \mathbb{E}_{t\sim\mathcal{U}(1,T),\ x_0,\ \epsilon\sim\mathcal{N}(0,\mathbf{I})}\Big[\ \lambda(t)\,\big\lVert \epsilon - \epsilon_\theta(x_t, t, c) \big\rVert^2\ \Big], \qquad x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$

with the weight $\lambda(t)$ set to $1$ and $c$ the **text conditioning** embedding. The training loop is precisely this expectation, sampling one random timestep per image:

```python
t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long()
noisy_x, noise = self.get_noise_for_timestep(t, x0)   # build xₜ
text_emb = self.text_encoder(text)                    # condition c
pred = self.model(noisy_x, t.float() / self.timesteps, text_emb)  # εθ
loss = self.loss_fn(pred, noise)                      # ‖ε − εθ‖²
```

### 4. DDPM Sampling

Generation runs the reverse chain for *all* $T$ steps, from $x_T \sim \mathcal{N}(0,\mathbf{I})$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t, c)\right) + \sqrt{\beta_t}\,z, \qquad z \sim \mathcal{N}(0,\mathbf{I})$$

The added noise $z$ (zero on the final step) makes DDPM a **stochastic** sampler:

```python
for i in reversed(range(self.timesteps)):          # 1000  0
    eps = self.model(x, t.float() / self.timesteps, text_emb)
    noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
    x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * eps) \
        + torch.sqrt(beta) * noise
```

**Cost:** 1000 network evaluations per image.

### 5. DDIM — Deterministic, Non-Markovian Sampling

A key limitation of DDPM is that there is **no single path** from $x_t$ to $x_0$, and 1000 sequential steps are slow. DDIM (Song et al., 2021) reformulates diffusion as a **non-Markovian** process whose marginals match DDPM's — letting us **skip timesteps** (here 50) and sample **deterministically**, with no retraining.

At each step, DDIM first **predicts the clean image** $x_0$ from the current $x_t$ — this is the $f_\theta^{(t)}$ estimate that comes straight from inverting the forward equation:

$$\hat{x}_0 = f_\theta^{(t)}(x_t) = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t, c)}{\sqrt{\bar\alpha_t}}$$

then re-noises it to the *next* (earlier) timestep $t'$ along the chosen sub-sequence:

$$x_{t'} = \sqrt{\bar{\alpha}_{t'}}\,\hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t'} - \sigma_t^2}\;\epsilon_\theta(x_t, t, c) + \sigma_t\,z$$

This maps directly to the implementation (defaults: `ddim_steps=50`, `eta=0.0`):

```python
x0_pred = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
x = alpha_t_next.sqrt() * x0_pred \
    + (1 - alpha_t_next - sigma**2).sqrt() * eps + sigma * noise
```

**Cost:** ~50 network evaluations per image — roughly **20× faster** than DDPM.

### 6. DDIM — Loss and the Role of Variance $\sigma$

During training, the step distribution over $x_{t-1}$ given $x_t, x_0$ is Gaussian; since $x_0$ is unknown the network's noise estimate $\epsilon_\theta$ is used. The training loss is the **KL divergence** between the true posterior (mean built from real noise $\epsilon$) and the predicted one (mean built from $\epsilon_\theta$). Both share variance $\sigma_t^2$, so the KL reduces to a scaled noise-prediction error:

$$\mu_\theta = \frac{x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta}{\sqrt{\alpha_t}}, \qquad
D_{\mathrm{KL}} = \frac{1}{2\sigma_t^2}\,\lVert\mu_q - \mu_\theta\rVert^2 = \frac{1-\alpha_t}{2\sigma_t^2\,\alpha_t}\,\lVert\epsilon_\theta - \epsilon\rVert^2$$

— i.e. the same noise-MSE objective as DDPM, up to a per-step weight. The hyper-parameter $\eta$ scales the stochasticity:

$$\sigma_t = \eta\sqrt{\frac{1-\bar{\alpha}_{t'}}{1-\bar{\alpha}_t}}\sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t'}}}$$

- **$\eta = 0$**  $\sigma_t = 0$  fully **deterministic**: the same initial noise always maps to the same image (the *only* randomness is $x_T$). This is the setting used in all DDIM experiments here.
- **$\eta = 1$**  recovers the DDPM stochastic process.

```python
if eta > 0:
    sigma = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)
    noise = torch.randn_like(x)
else:
    sigma = 0.0          # deterministic
    noise = 0.0
```

### 7. Latent Diffusion Models (LDM)

Pixel-space diffusion is expensive because every step operates on the full image, and many high-frequency pixel details are imperceptible. **LDM** (Rombach et al., 2022 — the basis of Stable Diffusion) first trains an **autoencoder** to compress images into a compact latent code, then runs the *entire* diffusion process in that small space.

**Autoencoder.** A convolutional encoder $E$ maps each image $x \in \mathbb{R}^{1\times28\times28}$ to a **16-dimensional** latent $z = E(x) \in \mathbb{R}^{16}$; a transposed-convolutional decoder reconstructs $\tilde{x} = D(z)$. It is trained with a plain reconstruction loss — **no** adversarial, KL, or VQ regularization:

$$\mathcal{L}_{\text{rec}} = \mathbb{E}_{x}\big[\,\lVert x - D(E(x)) \rVert_2^2\,\big]$$

**Latent normalization.** Since the latent space is unregularized, latents are normalized per batch for numerical stability before diffusion:

$$\hat\mu = \frac1B\sum_{i=1}^B z_i,\quad \hat\sigma^2 = \frac1B\sum_{i=1}^B (z_i-\hat\mu)^2,\qquad z_{\text{scaled}} = \frac{z - \hat\mu}{\hat\sigma}$$

**Latent diffusion.** With $E$ frozen, the identical forward/reverse equations from §1–§6 run on $z$ instead of pixels. A lightweight MLP **`LatentDenoiser`** replaces the convolutional U-Net (the data is now a flat vector):

$$z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad
\mathcal{L}_{\text{LDM}} = \mathbb{E}_{z_0, t, \epsilon}\big[\,\lVert \epsilon - \epsilon_\theta(z_t, t, e_y) \rVert_2^2\,\big]$$

where $e_y$ is the text embedding. At inference, sample $z_T \sim \mathcal{N}(0,\mathbf{I})$, denoise from $T$ to $0$, then decode $\tilde{x} = D(z_0)$.

**Why it matters:** denoising a 16-D vector is dramatically cheaper than a 784-pixel image — and, in this project, also produces the **highest-quality** samples (see [Results](#-results)).

---

## Text Conditioning

All three models are conditioned on **natural-language prompts** rather than raw class labels.

### 1. Text Encoder

```python
class TextEncoder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', output_dim=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.project = nn.Linear(hidden_size, output_dim)   #  256-D
```

A pretrained **DistilBERT** (a distilled BERT) encodes the prompt; the `[CLS]` representation is linearly projected to a **256-dimensional** conditioning vector $e_y$.

### 2. Cross-Attention

The conditioning vector is injected into the U-Net bottleneck through **cross-attention**, where image features are *queries* and the text embedding provides *keys* and *values*:

$$\alpha_i = \mathrm{softmax}\!\left(\frac{Q\cdot K_i^\top}{\sqrt{d_k}}\right),\qquad \mathrm{Context}(q) = \sum_i \alpha_i\,V_i, \qquad Q=W_Q\,q,\ K_i = W_K\,e_{y,i},\ V_i = W_V\,e_{y,i}$$

This lets every spatial location attend to the prompt, steering generation toward the requested digit. For the prompt *"a handwritten digit three"*, attention concentrates on the semantically rich token *"three"*.

### Prompt Granularity Experiment

The same model is trained three times — once per prompt style — to test whether prompt verbosity affects quality:

| `prompt_type` | Example for label `7` |
|---------------|------------------------|
| `numeric`     | `"7"`                  |
| `digit`       | `"digit 7"`            |
| `long`        | `"a handwritten digit seven"` |

```python
def numeric_prompt(label): return str(label)
def digit_prompt(label):   return f"digit {label}"
def long_prompt(label):    return f"a handwritten digit {['zero','one','two','three','four',
                                  'five','six','seven','eight','nine'][label]}"
```

---

## Architecture

**Pixel-space U-Net** (`UNet`, used by DDPM & DDIM):

- **Encoder:** 3 conv blocks `1  64  128  256` channels with BatchNorm + ReLU + MaxPool.
- **Time embedding:** scalar timestep $t/T$  MLP  256-D, broadcast and added at the bottleneck.
- **Cross-attention:** injects the 256-D text embedding at the bottleneck.
- **Decoder:** mirror of the encoder with **skip connections** and bilinear upsampling; a final 1-channel conv predicts the noise $\epsilon$.

**Latent autoencoder + denoiser** (LDM):

- **`Autoencoder`:** `Conv  Conv  Flatten  Linear` to a 16-D latent; decoder is `Linear  ConvTranspose  ConvTranspose  Sigmoid`. Trained for 15 epochs (Adam, `lr=1e-3`).
- **`LatentDenoiser`:** an MLP over the concatenation $[\,z_t \,\Vert\, \gamma(t) \,\Vert\, W_y e_y\,]$ — latent, learned time embedding, and projected text embedding — with two ReLU hidden layers, outputting the 16-D predicted noise.

**Common training setup:** Adam (`lr = 1e-3`), MSE noise-prediction loss, $T = 1000$ timesteps, early stopping on validation loss (`patience = 3`). MNIST is split stratified 80/20 (train/test).

---

## Evaluation Metrics

| Metric | Measures | Direction |
|--------|----------|-----------|
| **FID** (Fréchet Inception Distance) | Distance between real & generated Inception-feature distributions |  lower is better |
| **Inception Score (IS)** | Sample fidelity + class diversity |  higher is better |
| **Precision** | Fraction of generated samples inside the real manifold |  |
| **Recall** | Fraction of the real manifold covered by generated samples |  |
| **Density / Coverage** | Robust (k-NN manifold) variants of precision / recall |  |
| **CLIP Score** | Image–text alignment (does the digit match the prompt?) |  |

**FID** compares two Gaussians fit to InceptionV3 features (2048-D), $\mathcal{N}(\mu_R, \Sigma_R)$ (real) and $\mathcal{N}(\mu_S, \Sigma_S)$ (generated):

$$\text{FID} = \lVert \mu_S - \mu_R \rVert_2^2 + \mathrm{Tr}\!\left(\Sigma_S + \Sigma_R - 2(\Sigma_S \Sigma_R)^{1/2}\right)$$

**Inception Score** rewards confident, diverse class predictions:

$$\text{IS} = \exp\!\left(\mathbb{E}_{x\sim p_{\text{gen}}}\big[\,D_{\mathrm{KL}}\big(p(y\mid x)\,\Vert\,p(y)\big)\big]\right)$$

**Precision / Recall** use a k-NN manifold estimate: each real/fake feature defines a hypersphere of radius equal to its distance to its $k$-th nearest neighbor; a point "belongs" to the manifold if it falls inside any such sphere.

$$\text{Precision} = \frac1{|\Phi_g|}\sum_{\phi_g\in\Phi_g} f(\phi_g, \Phi_r), \qquad \text{Recall} = \frac1{|\Phi_r|}\sum_{\phi_r\in\Phi_r} f(\phi_r, \Phi_g)$$

**CLIP Score** embeds image and prompt into CLIP's shared space and averages cosine similarity — directly validating the conditioning:

$$\text{CLIP Score} = \frac1N\sum_{i=1}^N \cos(v_i, t_i)$$

---

## Experimental Design

The notebook runs a grid of experiments — **three pixel-space configurations × three prompt granularities**, plus the **LDM with long prompts** — each fully evaluated:

```
DDPM   ├── long prompts     FID, IS, PRDC, CLIP
       ├── digit prompts    FID, IS, PRDC, CLIP
       └── numeric prompts  FID, IS, PRDC, CLIP

DDIM   ├── long prompts     FID, IS, PRDC, CLIP
       ├── digit prompts    FID, IS, PRDC, CLIP
       └── numeric prompts  FID, IS, PRDC, CLIP

LDM    └── long prompts     FID, IS, PRDC, CLIP
```

Models are trained **separately** for each prompt type, then evaluated visually and quantitatively. Sampling time is also measured so DDPM and DDIM can be compared on the quality–speed trade-off.

---

## Results

### Full comparison (best values in **bold**)

| Model (Prompt) | FID  | Inception Score  | Precision  | Recall  | Density  | Coverage  | CLIP  |
|----------------|------:|:-----------------:|:-----------:|:--------:|:---------:|:----------:|:------:|
| DDPM (long)    | 171.04 | 1.4758 ± 0.0410 | 0.0330 | 0.0570 | 0.0192 | 0.0330 | 23.99 |
| DDPM (digit)   | 108.33 | 1.7410 ± 0.0667 | 0.2370 | 0.1490 | 0.1550 | 0.2370 | 24.42 |
| DDPM (numeric) | 139.47 | 1.5623 ± 0.0651 | 0.1950 | 0.0180 | 0.1274 | 0.1950 | 23.43 |
| DDIM (long)    | 187.78 | 1.3699 ± 0.0247 | 0.0210 | 0.0030 | 0.0156 | 0.0210 | 25.36 |
| DDIM (digit)   | 122.54 | 1.6236 ± 0.0630 | 0.1200 | 0.1240 | 0.0728 | 0.1200 | 23.91 |
| DDIM (numeric) | 114.69 | 1.4852 ± 0.0417 | 0.2520 | 0.1590 | 0.1838 | 0.2520 | 23.70 |
| **LDM (long)** | **55.68** | **2.1475 ± 0.1253** | **0.3380** | **0.4290** | **0.2410** | **0.3380** | **26.69** |

### Key takeaways

- **LDM wins across the board.** Operating in a structured 16-D latent space, LDM (long) achieves an FID of **55.68** — less than half the best pixel-space FID (108.33). It also leads on IS (2.15), Precision (0.338), Recall (0.429), and CLIP (26.69). Compressing away imperceptible pixel detail before diffusing pays off dramatically.

- **Among pixel-space models, DDIM-numeric is best.** It records the lowest pixel-space FID (114.69) and the highest PRDC values (Precision 0.252, Coverage 0.252) — the most balanced fidelity-and-diversity setup despite the minimal prompts. DDIM-numeric clearly beats DDPM-numeric on every quality metric.

- **DDPM-digit gives the best text alignment among pixel-space models** (CLIP 24.42) with a competitive FID (108.33) and the strongest pixel-space IS (1.74).

- **Long prompts *hurt* pixel-space models.** Both DDPM-long (FID 171.04) and DDIM-long (FID 187.78, Recall just 0.003) are the worst configurations — the excess semantic load seems to complicate pixel-space generation, even though DDIM-long posts a high CLIP score (25.36) that *doesn't* translate into visual quality. Yet the *same* long prompts are handled best by the LDM, suggesting the latent space copes with richer conditioning far more gracefully.

- **Sampler trade-off.** DDIM samples ~20× faster than DDPM (50 vs. 1000 steps) with deterministic ($\eta=0$) output; quality is comparable and often better in pixel space, making it the practical default.

---

## Repository Structure

```
MMML_project/
├── final_ddpm_ddim_ldm.ipynb   #  everything: models, training, sampling, metrics
├── mmml_paper.pdf              # full project report
├── README.md                   # this file
├── LICENSE                     # MIT
└── .gitignore
```

---

## Getting Started

### Option A — Google Colab / Kaggle (recommended)

The notebook is built for a free **T4 GPU**:

1. Upload `final_ddpm_ddim_ldm.ipynb` to [Colab](https://colab.research.google.com/) or Kaggle.
2. Set the runtime/accelerator to **GPU**.
3. `Runtime  Run all`. The first cell installs `torch_fidelity`; MNIST downloads automatically.

### Option B — Local

```bash
git clone <your-repo-url> && cd MMML_project
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision transformers torch_fidelity \
            scikit-learn matplotlib numpy scipy pillow tqdm gdown
jupyter notebook final_ddpm_ddim_ldm.ipynb
```

> A CUDA-capable GPU is strongly recommended — DDPM sampling over 1000 timesteps is slow on CPU.

### Minimal usage example

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the digit-prompt dataset
train_loader, test_loader, train_ds, test_ds = get_mnist_text_dataloaders(prompt_type='digit')

# Train + sample with DDIM (fast, deterministic — best pixel-space setup)
model = run_diffusion_pipeline_with_prompt_type(
    train_ds, test_ds,
    prompt_type='numeric',  # 'numeric' | 'digit' | 'long'
    method='ddim',          # 'ddpm' | 'ddim'
    ddim_steps=50,
    eta=0.0,                # deterministic
    epochs=20,
)

# Generate digits 0–9 from text
prompts = [numeric_prompt(i) for i in range(10)]
samples = model.generate_samples(prompts, use_ddim=True, ddim_steps=50)
```

---

## References

1. **Rombach, Blattmann, Lorenz, Esser, Ommer (2022).** *High-Resolution Image Synthesis with Latent Diffusion Models.* [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
2. **Esser, Rombach, Ommer (2021).** *Taming Transformers for High-Resolution Image Synthesis.* [arXiv:2012.09841](https://arxiv.org/abs/2012.09841)
3. **Ho, Jain, Abbeel (2020).** *Denoising Diffusion Probabilistic Models.* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
4. **Zhang, Zhang, Zhang, Kweon, Kim (2023).** *Text-to-Image Diffusion Models in Generative AI: A Survey.* [arXiv:2303.07909](https://arxiv.org/abs/2303.07909)
5. **Song, Meng, Ermon (2021).** *Denoising Diffusion Implicit Models.* [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
6. **Sanh et al. (2019).** *DistilBERT, a distilled version of BERT.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
7. **Radford et al. (2021).** *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
8. **Naeem et al. (2020).** *Reliable Fidelity and Diversity Metrics for Generative Models (PRDC).* [arXiv:2002.09797](https://arxiv.org/abs/2002.09797)

---