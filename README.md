# Fourier Neural Operator for Darcy Flow

A minimal demonstration of the [**Fourier Neural Operator (FNO)**](https://arxiv.org/abs/2010.08895) (Li et al., ICLR 2021) using the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library.

Trained on 2D Darcy flow at **16×16 resolution**, the model generalizes zero-shot to **32×32** — the key property of FNOs that makes them useful as PDE surrogates. Total training time: **~100 seconds** on a T4 GPU.

> **Note:** This is a companion project to [meshgraphnet-cfd-surrogate](https://github.com/ishrat-jan/meshgraphnet-cfd-surrogate), which implements MeshGraphNet from scratch for mesh-based CFD simulation. Together they demonstrate two complementary approaches to neural PDE surrogates: **grid-based** (FNO, this repo) and **graph-based** (MeshGraphNet).

---

## Why FNOs Matter

Classical numerical PDE solvers are tied to a fixed discretization — retrain or re-solve at every new resolution. FNOs learn operators in **Fourier space**, making them inherently resolution-invariant: train once at a coarse grid, evaluate at any finer grid without retraining. This property makes them attractive for:

- **Multi-scale climate and weather models** where resolution varies by region
- **Real-time engineering surrogates** that need coarse-grained training but fine-grained inference
- **Transfer learning** across different simulation meshes

---

## Architecture

The FNO replaces standard convolutional layers with **spectral convolutions** — learnable filters applied in the Fourier domain via FFT:

```
Input a(x) ──► Lifting ──► [Fourier Layer ×4] ──► Projection ──► Output u(x)
                              │
                              ├── FFT
                              ├── Spectral weights (learnable)
                              ├── Inverse FFT
                              └── + Local linear bypass
```

| Component | Detail |
|---|---|
| **Input** | Coefficient field a(x) on a 16×16 grid |
| **Output** | Solution field u(x) on the same grid |
| **Fourier modes** | 16 (retained frequencies per layer) |
| **Hidden channels** | 64 |
| **Fourier layers** | 4 |
| **Loss** | Lp loss (relative L2) + H1 Sobolev loss |
| **Training samples** | 1,000 |
| **Training time** | ~100 seconds (T4 GPU) |

---

## Results

### Resolution Generalization

The model is trained **only** on 16×16 data but evaluated on both 16×16 and 32×32 — no retraining or fine-tuning.

| Test Resolution | Samples | Relative L2 Error |
|---|---|---|
| 16×16 (seen) | 100 | ~9.5% |
| 32×32 (unseen) | 50 | ~13.6% |

Error increases at the higher resolution (expected — the model has never seen this grid spacing), but it still produces qualitatively correct solutions, demonstrating the resolution-invariance property.

---

## Repository Structure

```
fno-navier-stokes/
├── .gitignore
├── LICENSE
├── README.md
├── fno_darcy_flow.py            # Self-contained training + evaluation script
├── fno_navierstoke.ipynb        # Full Kaggle notebook with outputs
├── fno_darcy_results.png        # Visualization of predictions at 32×32
└── requirements.txt
```

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Train + Evaluate

```bash
python fno_darcy_flow.py
```

This will:
1. Download the Darcy flow dataset (auto-handled by `neuraloperator`)
2. Train the FNO for 30 epochs at 16×16 resolution (~100s on a T4)
3. Evaluate on both 16×16 and 32×32 test sets
4. Save a visualization of 32×32 predictions to `fno_darcy_results.png`

---

## Implementation Notes

- This project uses the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library for the FNO architecture and training loop. The goal here is not a from-scratch implementation (see the MeshGraphNet repo for that) but to demonstrate understanding of neural operator concepts and the resolution-generalization property.
- The **Darcy flow** PDE is a standard benchmark for neural operators — it models steady-state fluid flow through a porous medium.
- The **H1 (Sobolev) loss** penalizes derivative errors in addition to pointwise errors, encouraging physically smoother predictions.

---

## References

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhatt, K., Stuart, A., & Anandkumar, A. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- neuraloperator library: [github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)

---

## License

MIT
