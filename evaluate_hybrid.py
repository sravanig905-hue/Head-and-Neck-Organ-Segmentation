import torch
import numpy as np
from torch.utils.data import DataLoader
from model.hybrid_unet_transformer import HybridUNetTransformer
from slice_dataset import SliceDataset


# =========================
# SETTINGS
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "hybrid_unet_transformer.pth"
TEST_PATH = "fast_cache/test"

BATCH_SIZE = 2


# =========================
# METRIC FUNCTIONS
# =========================

def calculate_metrics(pred, target):
    """
    pred   : binary prediction
    target : binary ground truth
    """

    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()

    # True Positives
    tp = (pred * target).sum()

    # False Positives
    fp = (pred * (1 - target)).sum()

    # False Negatives
    fn = ((1 - pred) * target).sum()

    # Dice
    dice = (2 * tp + 1e-7) / (
        2 * tp + fp + fn + 1e-7
    )

    # IoU
    iou = (tp + 1e-7) / (
        tp + fp + fn + 1e-7
    )

    # Precision
    precision = (tp + 1e-7) / (
        tp + fp + 1e-7
    )

    # Recall
    recall = (tp + 1e-7) / (
        tp + fn + 1e-7
    )

    return (
        dice.item(),
        iou.item(),
        precision.item(),
        recall.item()
    )


# =========================
# LOAD DATASET
# =========================

print("=" * 60)
print("HYBRID U-NET + TRANSFORMER EVALUATION")
print("=" * 60)

print("Device:", DEVICE)

test_dataset = SliceDataset(TEST_PATH)

print("Test samples:", len(test_dataset))

if len(test_dataset) == 0:
    raise RuntimeError(
        "Test dataset is empty. Check fast_cache/test."
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# LOAD MODEL
# =========================

model = HybridUNetTransformer(
    num_classes=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# =========================
# EVALUATION
# =========================

total_dice = 0.0
total_iou = 0.0
total_precision = 0.0
total_recall = 0.0

num_batches = 0


with torch.no_grad():

    for batch_idx, (images, masks) in enumerate(test_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Remove extra dimension if present
        if images.dim() == 5:
            images = images.squeeze(1)

        if masks.dim() == 5:
            masks = masks.squeeze(1)

        # Model prediction
        outputs = model(images)

        # Convert logits to probability
        probabilities = torch.sigmoid(outputs)

        # Convert probability to binary mask
        predictions = (
            probabilities > 0.5
        ).float()

        # Calculate metrics
        dice, iou, precision, recall = calculate_metrics(
            predictions,
            masks
        )

        total_dice += dice
        total_iou += iou
        total_precision += precision
        total_recall += recall

        num_batches += 1

        if (batch_idx + 1) % 10 == 0:

            print(
                f"Batch {batch_idx + 1}/{len(test_loader)} | "
                f"Dice: {dice:.4f} | "
                f"IoU: {iou:.4f}"
            )


# =========================
# FINAL RESULTS
# =========================

avg_dice = total_dice / num_batches
avg_iou = total_iou / num_batches
avg_precision = total_precision / num_batches
avg_recall = total_recall / num_batches


print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Dice Score : {avg_dice:.4f}")
print(f"IoU        : {avg_iou:.4f}")
print(f"Precision  : {avg_precision:.4f}")
print(f"Recall     : {avg_recall:.4f}")

print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)