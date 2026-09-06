import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from model.hybrid_unet_transformer import HybridUNetTransformer
from multi_organ_dataset import MultiOrganDataset


# ============================================================
# CONFIGURATION - FAST VERSION
# ============================================================

NUM_CLASSES = 31

BATCH_SIZE = 2

EPOCHS = 1

LEARNING_RATE = 1e-4

# Small sample count for fast CPU training
TRAIN_SAMPLES_PER_EPOCH = 200

# Small validation set for fast checking
VAL_SAMPLES = 100

MODEL_PATH = "hybrid_unet_transformer_multiorgan.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(outputs, targets):

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    targets_one_hot = F.one_hot(
        targets,
        num_classes=NUM_CLASSES
    )

    targets_one_hot = targets_one_hot.permute(
        0, 3, 1, 2
    ).float()

    # Ignore background class
    probabilities = probabilities[:, 1:]

    targets_one_hot = targets_one_hot[:, 1:]

    smooth = 1e-5

    intersection = (
        probabilities * targets_one_hot
    ).sum(dim=(0, 2, 3))

    denominator = (
        probabilities.sum(dim=(0, 2, 3))
        + targets_one_hot.sum(dim=(0, 2, 3))
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


# ============================================================
# COMBINED LOSS
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

    def forward(self, outputs, targets):

        ce_loss = self.ce(
            outputs,
            targets
        )

        d_loss = dice_loss(
            outputs,
            targets
        )

        return ce_loss + d_loss


# ============================================================
# START
# ============================================================

print("=" * 60)
print("FAST MULTI-ORGAN TRAINING")
print("=" * 60)

print("Device:", DEVICE)

print("Classes:", NUM_CLASSES)

print("Epochs:", EPOCHS)

print(
    "Training samples:",
    TRAIN_SAMPLES_PER_EPOCH
)

print(
    "Validation samples:",
    VAL_SAMPLES
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading datasets...")

train_dataset = MultiOrganDataset("train")

val_dataset = MultiOrganDataset("val")

print(
    "Total TRAIN samples:",
    len(train_dataset)
)

print(
    "Total VAL samples:",
    len(val_dataset)
)


# ============================================================
# CREATE BALANCED WEIGHTS
# ============================================================

print("\nPreparing training sampler...")

sample_weights = []

for i in range(len(train_dataset)):

    _, mask = train_dataset[i]

    organ_pixels = torch.sum(
        mask > 0
    ).item()

    unique_classes = torch.unique(mask)

    if organ_pixels > 0:

        weight = 2.0

        # More weight for slices
        # containing multiple organs
        if len(unique_classes) > 2:

            weight = 3.0

    else:

        # Reduce background-only slices
        weight = 0.2

    sample_weights.append(weight)


sample_weights = torch.DoubleTensor(
    sample_weights
)


# ============================================================
# RANDOM BALANCED SAMPLER
# ============================================================

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=min(
        TRAIN_SAMPLES_PER_EPOCH,
        len(train_dataset)
    ),
    replacement=True
)


# ============================================================
# FAST VALIDATION SUBSET
# ============================================================

random.seed(42)

val_indices = random.sample(
    range(len(val_dataset)),
    min(
        VAL_SAMPLES,
        len(val_dataset)
    )
)

val_subset = torch.utils.data.Subset(
    val_dataset,
    val_indices
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\nTraining batches:", len(train_loader))

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating model...")

model = HybridUNetTransformer(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

print(
    "Hybrid U-Net + Transformer created."
)


# ============================================================
# LOSS
# ============================================================

criterion = CombinedLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    print("\n")
    print("=" * 60)

    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # TRAINING MODE
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0


    for batch_idx, (images, masks) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            masks
        )


        loss.backward()


        optimizer.step()


        running_train_loss += loss.item()


        if batch_idx % 25 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )


    train_loss = (
        running_train_loss
        / len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0


    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                masks
            )


            running_val_loss += loss.item()


    val_loss = (
        running_val_loss
        / len(val_loader)
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nEpoch completed.")

    print(
        f"Training Loss   : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "Best model saved!"
        )

    else:

        print(
            "Validation loss did not improve."
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")

print("=" * 60)

print(
    "FAST MULTI-ORGAN TRAINING COMPLETED"
)

print("=" * 60)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Best Validation Loss:",
    f"{best_val_loss:.4f}"
)

print("=" * 60)