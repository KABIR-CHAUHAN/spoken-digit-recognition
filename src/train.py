import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from model import AudioCNNGRU


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-5
GRAD_CLIP = 5.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ─────────────────────────────────────────────
# Mixup
# ─────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    """
    Mix two samples and their labels.
    """

    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(
        alpha,
        alpha
    )

    batch_size = x.size(0)

    index = torch.randperm(
        batch_size,
        device=x.device
    )

    mixed_x = (
        lam * x +
        (1 - lam) * x[index]
    )

    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


def mixup_loss(
    criterion,
    pred,
    y_a,
    y_b,
    lam
):
    return (
        lam * criterion(pred, y_a)
        +
        (1 - lam) * criterion(pred, y_b)
    )


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Mixup
        x_mix, y_a, y_b, lam = mixup_data(
            x,
            y,
            alpha=0.4
        )

        optimizer.zero_grad()

        logits = model(x_mix)

        loss = mixup_loss(
            criterion,
            logits,
            y_a,
            y_b,
            lam
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)

        all_preds.extend(
            preds.detach().cpu().numpy()
        )

        all_labels.extend(
            y.detach().cpu().numpy()
        )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    return (
        total_loss / len(loader),
        f1
    )


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        total_loss += loss.item()

        preds = logits.argmax(dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            y.cpu().numpy()
        )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    return (
        total_loss / len(loader),
        f1
    )


# ─────────────────────────────────────────────
# Training Setup
# ─────────────────────────────────────────────

def build_training_objects():

    model = AudioCNNGRU(
        num_classes=10,
        gru_hidden=256
    ).to(DEVICE)

    # Label smoothing cross entropy
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR
    )

    return (
        model,
        criterion,
        optimizer,
        scheduler
    )


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def train_model(train_dataset):

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    (
        model,
        criterion,
        optimizer,
        scheduler
    ) = build_training_objects()

    best_f1 = -1.0

    for epoch in range(EPOCHS):

        loss, f1 = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion
        )

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Macro F1: {f1:.4f} | "
            f"LR: {lr:.2e}"
        )

        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                "best_model.pt"
            )

    print(
        f"\nBest training Macro F1: "
        f"{best_f1:.4f}"
    )

    return model


if __name__ == "__main__":
    print(
        "Import train_model() and provide "
        "an AudioDataset to start training."
    )
