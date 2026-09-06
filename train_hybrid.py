import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slice_dataset import SliceDataset
from model.hybrid_unet_transformer import HybridUNetTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("HYBRID U-NET + TRANSFORMER TRAINING")
print("=" * 60)
print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

print("\nLoading training dataset...")

train_dataset = SliceDataset(
    "fast_cache/train"
)

print("Loading validation dataset...")

val_dataset = SliceDataset(
    "fast_cache/val"
)

print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# MODEL
# ============================================================

print("\nCreating Hybrid U-Net + Transformer model...")

model = HybridUNetTransformer(
    num_classes=1
).to(DEVICE)

print("Model created successfully.")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    smooth = 1e-6

    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        pred.sum() + target.sum() + smooth
    )

    return 1.0 - dice


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    model.train()

    total_train_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(DEVICE).float()
        masks = masks.to(DEVICE).float()

        # Remove unnecessary dimensions if present
        if images.dim() == 5:
            images = images.squeeze(1)

        if masks.dim() == 5:
            masks = masks.squeeze(1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Make sure mask has same shape as output
        if masks.shape != outputs.shape:
            masks = masks.view_as(outputs)

        # Calculate losses
        loss_bce = bce_loss(
            outputs,
            masks
        )

        loss_dice = dice_loss(
            outputs,
            masks
        )

        # Combined loss
        loss = loss_bce + loss_dice

        # Backpropagation
        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        # Print progress
        if batch_idx % 10 == 0:

            print(
                f"Batch "
                f"{batch_idx + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    # Average training loss
    avg_train_loss = (
        total_train_loss / len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE).float()
            masks = masks.to(DEVICE).float()

            if images.dim() == 5:
                images = images.squeeze(1)

            if masks.dim() == 5:
                masks = masks.squeeze(1)

            outputs = model(images)

            if masks.shape != outputs.shape:
                masks = masks.view_as(outputs)

            loss_bce = bce_loss(
                outputs,
                masks
            )

            loss_dice = dice_loss(
                outputs,
                masks
            )

            loss = loss_bce + loss_dice

            total_val_loss += loss.item()

    avg_val_loss = (
        total_val_loss / len(val_loader)
    )


    # ========================================================
    # EPOCH RESULTS
    # ========================================================

    print("\nEpoch completed.")

    print(
        f"Training Loss   : {avg_train_loss:.4f}"
    )

    print(
        f"Validation Loss : {avg_val_loss:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = "hybrid_unet_transformer.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Model saved successfully:\n{MODEL_PATH}"
)