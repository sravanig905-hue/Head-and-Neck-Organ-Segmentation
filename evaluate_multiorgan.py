import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_organ_dataset import MultiOrganDataset
from model.hybrid_unet_transformer import HybridUNetTransformer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = r"C:\Users\Sravani\Desktop\HaN_Seg_Project"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "hybrid_unet_transformer_multiorgan.pth"
)

MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "multi_organ_cache",
    "class_mapping.txt"
)

NUM_CLASSES = 31
BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

class_mapping = {}

with open(
    MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        class_id, organ = line.split(
            " -> ",
            1
        )

        class_mapping[int(class_id)] = organ


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = MultiOrganDataset(
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = HybridUNetTransformer(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print(
    "Model loaded successfully."
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# METRIC STORAGE
# ============================================================

intersection = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

predicted_pixels = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

ground_truth_pixels = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)


# ============================================================
# EVALUATION
# ============================================================

print("\n==============================================")
print("TEST SET EVALUATION")
print("==============================================")

with torch.no_grad():

    for batch_index, (images, masks) in enumerate(
        test_loader
    ):

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        outputs = model(
            images
        )

        if outputs.dim() == 5:
            outputs = outputs.squeeze(2)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for class_id in range(
            NUM_CLASSES
        ):

            pred = (
                predictions == class_id
            )

            truth = (
                masks == class_id
            )

            intersection[class_id] += (
                pred & truth
            ).sum().item()

            predicted_pixels[class_id] += (
                pred
            ).sum().item()

            ground_truth_pixels[class_id] += (
                truth
            ).sum().item()

        if (
            batch_index + 1
        ) % 20 == 0:

            print(
                f"Processed batches: "
                f"{batch_index + 1}/"
                f"{len(test_loader)}"
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

results = {}

all_dice = []
all_iou = []
all_precision = []
all_recall = []


for class_id in range(
    1,
    NUM_CLASSES
):

    organ = class_mapping.get(
        class_id,
        f"Class_{class_id}"
    )

    inter = intersection[
        class_id
    ]

    pred_count = predicted_pixels[
        class_id
    ]

    truth_count = ground_truth_pixels[
        class_id
    ]

    union = (
        pred_count
        +
        truth_count
        -
        inter
    )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    if (
        pred_count == 0
        and truth_count == 0
    ):

        dice = 1.0

    else:

        dice = (
            2.0 * inter
            /
            (pred_count + truth_count + 1e-8)
        )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union == 0:

        iou = 1.0

    else:

        iou = (
            inter
            /
            (union + 1e-8)
        )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if pred_count == 0:

        precision = 0.0

    else:

        precision = (
            inter
            /
            (pred_count + 1e-8)
        )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if truth_count == 0:

        recall = 0.0

    else:

        recall = (
            inter
            /
            (truth_count + 1e-8)
        )


    results[organ] = {
        "class_id": class_id,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall)
    }


    all_dice.append(dice)
    all_iou.append(iou)
    all_precision.append(precision)
    all_recall.append(recall)


# ============================================================
# OVERALL RESULTS
# ============================================================

overall = {

    "dice": float(
        np.mean(all_dice)
    ),

    "iou": float(
        np.mean(all_iou)
    ),

    "precision": float(
        np.mean(all_precision)
    ),

    "recall": float(
        np.mean(all_recall)
    )
}


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================================")
print("ORGAN-WISE RESULTS")
print("==============================================")

print(
    f"{'ID':<5}"
    f"{'Organ':<25}"
    f"{'Dice':<10}"
    f"{'IoU':<10}"
    f"{'Precision':<12}"
    f"{'Recall':<10}"
)

print("-" * 75)


for organ, metrics in results.items():

    print(
        f"{metrics['class_id']:<5}"
        f"{organ:<25}"
        f"{metrics['dice']:.4f}    "
        f"{metrics['iou']:.4f}    "
        f"{metrics['precision']:.4f}       "
        f"{metrics['recall']:.4f}"
    )


# ============================================================
# OVERALL
# ============================================================

print("\n==============================================")
print("OVERALL TEST RESULTS")
print("==============================================")

print(
    f"Mean Dice      : {overall['dice']:.4f}"
)

print(
    f"Mean IoU       : {overall['iou']:.4f}"
)

print(
    f"Mean Precision : {overall['precision']:.4f}"
)

print(
    f"Mean Recall    : {overall['recall']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    PROJECT_ROOT,
    "evaluation_metrics.txt"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MULTI-ORGAN SEGMENTATION "
        "EVALUATION\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        f"Mean Dice      : "
        f"{overall['dice']:.4f}\n"
    )

    f.write(
        f"Mean IoU       : "
        f"{overall['iou']:.4f}\n"
    )

    f.write(
        f"Mean Precision : "
        f"{overall['precision']:.4f}\n"
    )

    f.write(
        f"Mean Recall    : "
        f"{overall['recall']:.4f}\n\n"
    )

    f.write(
        "ORGAN-WISE RESULTS\n"
    )

    f.write(
        "------------------------------------\n"
    )

    for organ, metrics in results.items():

        f.write(
            f"\n{organ}\n"
        )

        f.write(
            f"Class ID  : "
            f"{metrics['class_id']}\n"
        )

        f.write(
            f"Dice      : "
            f"{metrics['dice']:.4f}\n"
        )

        f.write(
            f"IoU       : "
            f"{metrics['iou']:.4f}\n"
        )

        f.write(
            f"Precision : "
            f"{metrics['precision']:.4f}\n"
        )

        f.write(
            f"Recall    : "
            f"{metrics['recall']:.4f}\n"
        )


print("\nResults saved to:")

print(
    output_file
)

print("\nEVALUATION COMPLETED!")