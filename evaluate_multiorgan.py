import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_organ_dataset import MultiOrganDataset
from model.hybrid_unet_transformer import HybridUNetTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "hybrid_unet_transformer_multiorgan_best.pth"
)

CACHE_PATH = os.path.join(
    PROJECT_ROOT,
    "multi_organ_cache"
)

CLASS_MAPPING_PATH = os.path.join(
    CACHE_PATH,
    "class_mapping.txt"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation_metrics.json"
)


# ============================================================
# SETTINGS
# ============================================================

NUM_CLASSES = 31
BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

class_names = {}

with open(
    CLASS_MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        parts = line.split("->")

        if len(parts) != 2:
            continue

        class_id = int(parts[0].strip())
        class_name = parts[1].strip()

        class_names[class_id] = class_name


print("=" * 60)
print("MULTI-ORGAN MODEL EVALUATION")
print("=" * 60)

print("Project root :", PROJECT_ROOT)
print("Model        :", MODEL_PATH)
print("Device       :", DEVICE)
print("Classes      :", NUM_CLASSES)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = MultiOrganDataset("test")

if len(test_dataset) == 0:

    raise RuntimeError(
        "Test dataset is empty."
    )

print(
    "TEST samples:",
    len(test_dataset)
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = HybridUNetTransformer(
    num_classes=NUM_CLASSES
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# METRIC STORAGE
# ============================================================

intersection = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

pred_count = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

target_count = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

true_positive = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

false_positive = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

false_negative = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

total_correct = 0
total_pixels = 0

processed_batches = 0


# ============================================================
# EVALUATION
# ============================================================

print("\nStarting evaluation...")

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        # Safety for 5D output.
        if outputs.ndim == 5:
            outputs = outputs.squeeze(2)

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # Pixel accuracy
        # ----------------------------------------------------

        total_correct += (
            predictions == masks
        ).sum().item()

        total_pixels += masks.numel()


        # ----------------------------------------------------
        # Per-class statistics
        # ----------------------------------------------------

        for class_id in range(NUM_CLASSES):

            pred_class = (
                predictions == class_id
            )

            target_class = (
                masks == class_id
            )

            inter = (
                pred_class & target_class
            ).sum().item()

            pred_pixels = (
                pred_class.sum().item()
            )

            target_pixels = (
                target_class.sum().item()
            )

            intersection[class_id] += inter

            pred_count[class_id] += pred_pixels

            target_count[class_id] += target_pixels

            true_positive[class_id] += inter

            false_positive[class_id] += (
                pred_class & ~target_class
            ).sum().item()

            false_negative[class_id] += (
                ~pred_class & target_class
            ).sum().item()


        processed_batches += 1

        if processed_batches % 50 == 0:

            print(
                f"Processed batches: "
                f"{processed_batches}/"
                f"{len(test_loader)}"
            )


# ============================================================
# PIXEL ACCURACY
# ============================================================

pixel_accuracy = (
    total_correct /
    max(total_pixels, 1)
)


# ============================================================
# PER-ORGAN METRICS
# ============================================================

per_organ = {}

dice_values = []
iou_values = []
precision_values = []
recall_values = []


for class_id in range(1, NUM_CLASSES):

    name = class_names.get(
        class_id,
        f"Class_{class_id}"
    )

    inter = intersection[class_id]

    pred_pixels = pred_count[class_id]

    target_pixels = target_count[class_id]

    tp = true_positive[class_id]

    fp = false_positive[class_id]

    fn = false_negative[class_id]


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    if (
        pred_pixels == 0
        and target_pixels == 0
    ):

        dice = 1.0

    else:

        dice = (
            2.0 * inter
            /
            max(
                pred_pixels + target_pixels,
                1e-12
            )
        )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    union = (
        pred_pixels
        + target_pixels
        - inter
    )

    if union == 0:

        iou = 1.0

    else:

        iou = (
            inter /
            union
        )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if (
        tp + fp == 0
    ):

        precision = 1.0 if target_pixels == 0 else 0.0

    else:

        precision = (
            tp /
            (tp + fp)
        )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if (
        tp + fn == 0
    ):

        recall = 1.0 if target_pixels == 0 else 0.0

    else:

        recall = (
            tp /
            (tp + fn)
        )


    per_organ[name] = {

        "class_id": class_id,

        "dice": float(dice),

        "iou": float(iou),

        "precision": float(precision),

        "recall": float(recall),

        "predicted_pixels": int(pred_pixels),

        "ground_truth_pixels": int(target_pixels)
    }


    # Include classes that exist in ground truth.
    if target_pixels > 0:

        dice_values.append(dice)

        iou_values.append(iou)

        precision_values.append(precision)

        recall_values.append(recall)


# ============================================================
# MACRO METRICS
# ============================================================

mean_dice = (
    float(np.mean(dice_values))
    if dice_values
    else 0.0
)

mean_iou = (
    float(np.mean(iou_values))
    if iou_values
    else 0.0
)

mean_precision = (
    float(np.mean(precision_values))
    if precision_values
    else 0.0
)

mean_recall = (
    float(np.mean(recall_values))
    if recall_values
    else 0.0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(
    f"Pixel Accuracy : "
    f"{pixel_accuracy:.4f} "
    f"({pixel_accuracy * 100:.2f}%)"
)

print(
    f"Mean Dice      : "
    f"{mean_dice:.4f}"
)

print(
    f"Mean IoU       : "
    f"{mean_iou:.4f}"
)

print(
    f"Mean Precision : "
    f"{mean_precision:.4f}"
)

print(
    f"Mean Recall    : "
    f"{mean_recall:.4f}"
)


# ============================================================
# PER-ORGAN RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("PER-ORGAN RESULTS")
print("=" * 80)

for name, metrics in per_organ.items():

    print(
        f"{name:<20} | "
        f"Dice: {metrics['dice']:.4f} | "
        f"IoU: {metrics['iou']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f}"
    )


# ============================================================
# SAVE JSON
# ============================================================

results = {

    "model":
        "Hybrid U-Net + Transformer",

    "model_file":
        os.path.basename(MODEL_PATH),

    "num_classes":
        NUM_CLASSES,

    "test_samples":
        len(test_dataset),

    "pixel_accuracy":
        float(pixel_accuracy),

    "mean_dice":
        mean_dice,

    "mean_iou":
        mean_iou,

    "mean_precision":
        mean_precision,

    "mean_recall":
        mean_recall,

    "per_organ":
        per_organ
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\n")
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print(
    "Metrics saved to:"
)

print(
    OUTPUT_PATH
)