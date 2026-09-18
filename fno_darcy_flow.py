"""
Fourier Neural Operator (FNO) for Darcy Flow
=============================================
Minimal demonstration of the FNO (Li et al., ICLR 2021) trained on 2D Darcy
flow using the neuraloperator library. Trains at 16x16, generalizes zero-shot
to 32x32 — demonstrating the resolution-invariance property of neural operators.

Reference: https://arxiv.org/abs/2010.08895
"""

import torch
import matplotlib.pyplot as plt
from neuralop.models import FNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN = 1000
BATCH_SIZE = 32
TEST_RESOLUTIONS = [16, 32]
TEST_SAMPLES = [100, 50]
EPOCHS = 30
LR = 8e-3

# FNO architecture
N_MODES = (16, 16)       # Fourier modes retained per layer
HIDDEN_CHANNELS = 64     # Width of the spectral layers
N_LAYERS = 4             # Number of Fourier layers


def main():
    print("=" * 60)
    print("Fourier Neural Operator — Darcy Flow Surrogate")
    print("=" * 60)
    print(f"Device: {DEVICE}\n")

    # ──────────────────────────────────────────────────────────────────────
    # 1. Load data
    # ──────────────────────────────────────────────────────────────────────
    print("Loading Darcy flow dataset...")
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=N_TRAIN,
        batch_size=BATCH_SIZE,
        n_tests=TEST_SAMPLES,
        test_resolutions=TEST_RESOLUTIONS,
        test_batch_sizes=[BATCH_SIZE, BATCH_SIZE],
    )
    data_processor = data_processor.to(DEVICE)

    print(f"  Training samples: {N_TRAIN} at 16x16")
    print(f"  Test sets: {TEST_SAMPLES[0]} at 16x16, {TEST_SAMPLES[1]} at 32x32\n")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Build model
    # ──────────────────────────────────────────────────────────────────────
    model = FNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=1,
        out_channels=1,
        n_layers=N_LAYERS,
    )
    model = model.to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"FNO parameters: {param_count:,}\n")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Train
    # ──────────────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Losses: relative L2 + H1 Sobolev
    l2_loss = LpLoss(d=2, p=2)
    h1_loss = H1Loss(d=2)

    trainer = Trainer(
        model=model,
        n_epochs=EPOCHS,
        data_processor=data_processor,
        device=DEVICE,
        verbose=True,
    )

    print(f"Training for {EPOCHS} epochs...")
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=l2_loss,
        eval_losses={"l2": l2_loss, "h1": h1_loss},
    )

    # ──────────────────────────────────────────────────────────────────────
    # 4. Evaluate on both resolutions
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    for res in TEST_RESOLUTIONS:
        model.eval()
        data_processor.eval()
        total_l2 = 0.0
        n = 0
        for batch in test_loaders[res]:
            batch = data_processor.preprocess(batch)
            with torch.no_grad():
                out = model(batch["x"])
            out, batch = data_processor.postprocess(out, batch)
            total_l2 += l2_loss(out, batch["y"]).item() * out.shape[0]
            n += out.shape[0]
        print(f"  Resolution {res}x{res}: relative L2 = {total_l2 / n:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # 5. Visualize at unseen 32x32 resolution
    # ──────────────────────────────────────────────────────────────────────
    model.eval()
    data_processor.eval()
    batch = next(iter(test_loaders[32]))
    batch = data_processor.preprocess(batch)
    with torch.no_grad():
        out = model(batch["x"])
    out, batch = data_processor.postprocess(out, batch)

    x = batch["x"].cpu()
    y = batch["y"].cpu()
    pred = out.detach().cpu()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i in range(3):
        axes[i, 0].imshow(x[i, 0])
        axes[i, 0].set_title("Input a(x)")
        axes[i, 1].imshow(y[i, 0])
        axes[i, 1].set_title("True u(x)")
        axes[i, 2].imshow(pred[i, 0])
        axes[i, 2].set_title("FNO prediction")
    plt.suptitle("FNO — Darcy Flow at 32×32 (unseen resolution)", fontsize=14)
    plt.tight_layout()
    plt.savefig("fno_darcy_results.png", dpi=150)
    plt.show()
    print("\nSaved visualization to fno_darcy_results.png")


if __name__ == "__main__":
    main()
