import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from multi_organ_dataset import MultiOrganDataset
from model.hybrid_unet_transformer import HybridUNetTransformer


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 31
BATCH_SIZE = 2

EPOCHS = 5

TRAIN_SAMPLES_PER_EPOCH = 1000
VAL_SAMPLES = 300

LEARNING_RATE = 1e-4

MODEL_PATH = "hybrid_unet_transformer_multiorgan.pth"

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Classes:", NUM_CLASSES)
print("Epochs:", EPOCHS)
print("Training samples / epoch:", TRAIN_SAMPLES_PER_EPOCH)
print("Validation samples:", VAL_SAMPLES)


# ============================================================
# DATASET
# ============================================================

train_dataset = MultiOrganDataset("train")
val_dataset = MultiOrganDataset("val")

print("Total TRAIN samples:", len(train_dataset))
print("Total VAL samples:", len(val_dataset))

if len(train_dataset) == 0:
    raise RuntimeError("Training dataset is empty.")

if len(val_dataset) == 0:
    raise RuntimeError("Validation dataset is empty.")


# ============================================================
# CLASS PIXEL COUNTS
# ============================================================

class_pixel_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)

print("\nCalculating class pixel frequencies...")

for i in range(len(train_dataset)):

    _, mask = train_dataset[i]

    values, counts = torch.unique(mask, return_counts=True)

    for value, count in zip(values, counts):

        class_id = int(value)

        if 0 <= class_id < NUM_CLASSES:
            class_pixel_counts[class_id] += float(count)


print("\nClass pixel counts:")

for i in range(NUM_CLASSES):
    print(
        f"Class {i:02d}: "
        f"{int(class_pixel_counts[i].item())}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = torch.ones(NUM_CLASSES, dtype=torch.float32)

nonzero = class_pixel_counts > 0

weights[nonzero] = (
    1.0 /
    torch.sqrt(
        class_pixel_counts[nonzero]
    ).float()
)

# Reduce background importance
weights[0] = 0.25

# Avoid extreme weights
weights = torch.clamp(weights, min=0.01, max=5.0)

# Normalize
weights = weights / weights.mean()

weights = weights.float().to(device)

print("\nClass weights:")
print(weights)


# ============================================================
# SAMPLE WEIGHTS FOR BALANCED SAMPLING
# ============================================================

sample_weights = []

for i in range(len(train_dataset)):

    _, mask = train_dataset[i]

    unique_classes = torch.unique(mask)

    foreground_classes = unique_classes[
        unique_classes != 0
    ]

    number_of_organs = len(foreground_classes)

    if number_of_organs == 0:

        # Background-only image
        sample_weight = 0.10

    elif number_of_organs == 1:

        # One organ
        sample_weight = 2.0

    else:

        # Multiple organs
        sample_weight = 4.0

    sample_weights.append(sample_weight)


sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

num_train_samples = min(
    TRAIN_SAMPLES_PER_EPOCH,
    len(train_dataset)
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=num_train_samples,
    replacement=True
)


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = HybridUNetTransformer(
    num_classes=NUM_CLASSES
).to(device)


# ============================================================
# DICE LOSS
# ============================================================

class ForegroundDiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):

        super().__init__()

        self.smooth = smooth

    def forward(self, logits, target):

        # logits:
        # [B, C, H, W]

        # target:
        # [B, H, W]

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        total_dice = 0.0
        count = 0

        for class_id in range(1, NUM_CLASSES):

            pred = probabilities[:, class_id]

            true = (
                target == class_id
            ).float()

            # Only calculate Dice for
            # classes present in target
            if true.sum() == 0:
                continue

            intersection = (
                pred * true
            ).sum()

            denominator = (
                pred.sum() +
                true.sum()
            )

            dice = (
                2.0 * intersection +
                self.smooth
            ) / (
                denominator +
                self.smooth
            )

            total_dice += dice
            count += 1

        if count == 0:
            return logits.sum() * 0.0

        mean_dice = total_dice / count

        return 1.0 - mean_dice


# ============================================================
# COMBINED LOSS
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(self, class_weights):

        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.05
        )

        self.dice = ForegroundDiceLoss()

    def forward(self, logits, target):

        ce_loss = self.ce(
            logits,
            target
        )

        dice_loss = self.dice(
            logits,
            target
        )

        return (
            0.5 * ce_loss +
            0.5 * dice_loss
        )


criterion = CombinedLoss(weights)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    total_correct = 0
    total_pixels = 0

    dice_values = []
    iou_values = []
    precision_values = []
    recall_values = []

    val_count = min(
        VAL_SAMPLES,
        len(val_dataset)
    )

    indices = list(range(len(val_dataset)))

    random.shuffle(indices)

    indices = indices[:val_count]

    with torch.no_grad():

        for index in indices:

            image, mask = val_dataset[index]

            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            logits = model(image)

            loss = criterion(
                logits,
                mask
            )

            total_loss += loss.item()

            prediction = torch.argmax(
                logits,
                dim=1
            )

            total_correct += (
                prediction == mask
            ).sum().item()

            total_pixels += mask.numel()

            # ---------------------------------------------
            # Foreground metrics
            # ---------------------------------------------

            for class_id in range(1, NUM_CLASSES):

                pred_class = (
                    prediction == class_id
                )

                true_class = (
                    mask == class_id
                )

                true_positive = (
                    pred_class &
                    true_class
                ).sum().item()

                false_positive = (
                    pred_class &
                    (~true_class)
                ).sum().item()

                false_negative = (
                    (~pred_class) &
                    true_class
                ).sum().item()

                if (
                    true_class.sum().item() == 0
                    and pred_class.sum().item() == 0
                ):
                    continue

                dice_denominator = (
                    2 * true_positive +
                    false_positive +
                    false_negative
                )

                if dice_denominator > 0:

                    dice = (
                        2 * true_positive /
                        dice_denominator
                    )

                    dice_values.append(dice)

                iou_denominator = (
                    true_positive +
                    false_positive +
                    false_negative
                )

                if iou_denominator > 0:

                    iou = (
                        true_positive /
                        iou_denominator
                    )

                    iou_values.append(iou)

                precision_denominator = (
                    true_positive +
                    false_positive
                )

                if precision_denominator > 0:

                    precision = (
                        true_positive /
                        precision_denominator
                    )

                    precision_values.append(
                        precision
                    )

                recall_denominator = (
                    true_positive +
                    false_negative
                )

                if recall_denominator > 0:

                    recall = (
                        true_positive /
                        recall_denominator
                    )

                    recall_values.append(
                        recall
                    )


    avg_loss = (
        total_loss / max(1, len(indices))
    )

    accuracy = (
        total_correct /
        max(1, total_pixels)
    )

    mean_dice = (
        np.mean(dice_values)
        if dice_values
        else 0.0
    )

    mean_iou = (
        np.mean(iou_values)
        if iou_values
        else 0.0
    )

    mean_precision = (
        np.mean(precision_values)
        if precision_values
        else 0.0
    )

    mean_recall = (
        np.mean(recall_values)
        if recall_values
        else 0.0
    )

    return (
        avg_loss,
        accuracy,
        mean_dice,
        mean_iou,
        mean_precision,
        mean_recall
    )


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    print(
        f"\nEPOCH {epoch + 1}/{EPOCHS}"
    )

    for batch_index, (images, masks) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Safety for accidental 5D output
        if outputs.ndim == 5:

            outputs = outputs.squeeze(2)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        if batch_index % 50 == 0:

            print(
                f"Batch {batch_index}/"
                f"{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )


    # ========================================================
    # EPOCH TRAIN LOSS
    # ========================================================

    train_loss = (
        running_loss /
        max(1, len(train_loader))
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_loss,
        val_accuracy,
        val_dice,
        val_iou,
        val_precision,
        val_recall
    ) = validate()


    # Scheduler
    scheduler.step(val_loss)


    print("\nEpoch completed")

    print(
        f"Training Loss   : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Foreground Dice : "
        f"{val_dice:.4f}"
    )

    print(
        f"Foreground IoU  : "
        f"{val_iou:.4f}"
    )

    print(
        f"Foreground Precision : "
        f"{val_precision:.4f}"
    )

    print(
        f"Foreground Recall : "
        f"{val_recall:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("\nBest model saved!")

        print(
            "Model:",
            MODEL_PATH
        )


# ============================================================
# FINISHED
# ============================================================

print("\n====================================")
print("MULTI-ORGAN TRAINING COMPLETED")
print("====================================")

print(
    "Best Validation Loss:",
    best_val_loss
)

print(
    "Model saved at:",
    MODEL_PATH
)