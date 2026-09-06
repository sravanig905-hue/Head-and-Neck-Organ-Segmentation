import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.hybrid_unet_transformer import HybridUNetTransformer
from slice_dataset import SliceDataset


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

TEST_DIR = os.path.join(
    PROJECT_ROOT,
    "fast_cache",
    "test"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "hybrid_unet_transformer.pth"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation_metrics.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print("Test directory:")
print(TEST_DIR)

print("Model:")
print(MODEL_PATH)

print("Device:")
print(DEVICE)


test_dataset = SliceDataset(
    TEST_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print()
print("Test samples:", len(test_dataset))


# ============================================================
# LOAD MODEL
# ============================================================

model = HybridUNetTransformer(
    num_classes=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["state_dict"]
    )
else:
    model.load_state_dict(
        checkpoint
    )

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# METRIC STORAGE
# ============================================================

dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []

total_tp = 0
total_fp = 0
total_fn = 0


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for index, batch in enumerate(test_loader):

        images, masks = batch

        images = images.to(
            DEVICE,
            dtype=torch.float32
        )

        masks = masks.to(
            DEVICE,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        outputs = model(images)

        # Handle possible 5D output
        if outputs.ndim == 5:

            outputs = outputs[
                :, :, 0, :, :
            ]

        # ----------------------------------------------------
        # SIGMOID
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # ----------------------------------------------------
        # FLATTEN
        # ----------------------------------------------------

        pred = predictions.cpu().numpy().astype(
            np.uint8
        )

        gt = masks.cpu().numpy().astype(
            np.uint8
        )

        # Remove unnecessary dimensions
        pred = pred.reshape(-1)
        gt = gt.reshape(-1)

        # ----------------------------------------------------
        # CONFUSION COUNTS
        # ----------------------------------------------------

        tp = np.sum(
            (pred == 1) &
            (gt == 1)
        )

        fp = np.sum(
            (pred == 1) &
            (gt == 0)
        )

        fn = np.sum(
            (pred == 0) &
            (gt == 1)
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # ----------------------------------------------------
        # DICE
        # ----------------------------------------------------

        denominator = (
            2 * tp +
            fp +
            fn
        )

        if denominator > 0:

            dice = (
                2 * tp
            ) / denominator

        else:

            dice = 1.0

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        union = (
            tp +
            fp +
            fn
        )

        if union > 0:

            iou = tp / union

        else:

            iou = 1.0

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        precision_denominator = (
            tp + fp
        )

        if precision_denominator > 0:

            precision = (
                tp /
                precision_denominator
            )

        else:

            precision = 0.0

        # ----------------------------------------------------
        # RECALL
        # ----------------------------------------------------

        recall_denominator = (
            tp + fn
        )

        if recall_denominator > 0:

            recall = (
                tp /
                recall_denominator
            )

        else:

            recall = 0.0

        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (index + 1) % 50 == 0:

            print(
                f"Evaluated "
                f"{index + 1}/"
                f"{len(test_loader)}"
            )


# ============================================================
# FINAL MEAN METRICS
# ============================================================

mean_dice = float(
    np.mean(dice_scores)
)

mean_iou = float(
    np.mean(iou_scores)
)

mean_precision = float(
    np.mean(precision_scores)
)

mean_recall = float(
    np.mean(recall_scores)
)


# ============================================================
# GLOBAL METRICS
# ============================================================

global_dice = (
    2 * total_tp
) / (
    2 * total_tp +
    total_fp +
    total_fn
)

global_iou = (
    total_tp
) / (
    total_tp +
    total_fp +
    total_fn
)

global_precision = (
    total_tp
) / (
    total_tp +
    total_fp
) if (
    total_tp + total_fp
) > 0 else 0.0

global_recall = (
    total_tp
) / (
    total_tp +
    total_fn
) if (
    total_tp + total_fn
) > 0 else 0.0


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "available": True,

    "organ": "Brainstem",

    "dataset": "HaN-Seg",

    "split": "test",

    "num_samples": len(test_dataset),

    "dice": mean_dice,

    "iou": mean_iou,

    "precision": mean_precision,

    "recall": mean_recall,

    "global_dice": float(
        global_dice
    ),

    "global_iou": float(
        global_iou
    ),

    "global_precision": float(
        global_precision
    ),

    "global_recall": float(
        global_recall
    )
}


with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print(
    f"Dice Score   : {mean_dice:.4f}"
)

print(
    f"IoU          : {mean_iou:.4f}"
)

print(
    f"Precision    : {mean_precision:.4f}"
)

print(
    f"Recall       : {mean_recall:.4f}"
)

print()
print("Results saved to:")

print(
    OUTPUT_PATH
)

print("=" * 60)