import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.unet import UNet
from slice_dataset import SliceDataset


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 2

# First test only
EPOCHS = 1

LEARNING_RATE = 1e-4

MODEL_PATH = "best_unet_brainstem.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BRAINSTEM U-NET TRAINING")
print("=" * 70)

print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):

        smooth = 1e-6

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (
            predictions * targets
        ).sum()

        dice = (
            2.0 * intersection + smooth
        ) / (
            predictions.sum()
            + targets.sum()
            + smooth
        )

        return 1.0 - dice


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(predictions, targets):

    smooth = 1e-6

    predictions = torch.sigmoid(predictions)

    predictions = (
        predictions > 0.5
    ).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (
        predictions * targets
    ).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum()
        + targets.sum()
        + smooth
    )

    return dice.item()


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

train_dataset = SliceDataset("train")

print("\nLoading validation dataset...")

val_dataset = SliceDataset("val")


# ============================================================
# DATALOADERS
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

print("\n" + "=" * 70)

print("DATASET INFORMATION")

print("=" * 70)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Train batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating U-Net model...")

model = UNet(
    num_classes=1
)

model = model.to(DEVICE)

print("Model loaded successfully.")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = DiceLoss()


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

best_val_dice = 0.0


for epoch in range(EPOCHS):

    print("\n")
    print("=" * 70)
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN MODE
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for batch_index, (images, masks) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)


        # Forward pass
        outputs = model(images)


        # BCE loss
        loss_bce = bce_loss(
            outputs,
            masks
        )


        # Dice loss
        loss_dice = dice_loss(
            outputs,
            masks
        )


        # Combined loss
        loss = (
            loss_bce
            + loss_dice
        )


        # Clear gradients
        optimizer.zero_grad()


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        total_train_loss += loss.item()


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            (batch_index + 1) % 50 == 0
            or
            (batch_index + 1) == len(train_loader)
        ):

            print(
                f"Batch "
                f"[{batch_index + 1}/"
                f"{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    # Average training loss
    train_loss = (
        total_train_loss
        / len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nRunning validation...")

    model.eval()

    total_val_loss = 0.0

    total_val_dice = 0.0


    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)


            # Prediction
            outputs = model(images)


            # Loss
            loss_bce = bce_loss(
                outputs,
                masks
            )

            loss_dice = dice_loss(
                outputs,
                masks
            )

            loss = (
                loss_bce
                + loss_dice
            )


            total_val_loss += (
                loss.item()
            )


            # Dice
            total_val_dice += dice_score(
                outputs,
                masks
            )


    # Average validation results
    val_loss = (
        total_val_loss
        / len(val_loader)
    )

    val_dice = (
        total_val_dice
        / len(val_loader)
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Dice   : {val_dice:.4f}"
    )

    print("=" * 70)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "\n✓ Best model saved!"
        )

        print(
            f"✓ Dice Score: "
            f"{best_val_dice:.4f}"
        )

        print(
            f"✓ File: {MODEL_PATH}"
        )

    else:

        print(
            "\nNo improvement "
            "in validation Dice."
        )


# ============================================================
# COMPLETED
# ============================================================

print("\n")

print("=" * 70)

print("TRAINING COMPLETED")

print("=" * 70)

print(
    f"Best Validation Dice: "
    f"{best_val_dice:.4f}"
)

print(
    f"Model file: "
    f"{MODEL_PATH}"
)

print("=" * 70)