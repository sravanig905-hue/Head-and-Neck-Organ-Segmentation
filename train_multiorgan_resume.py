import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from multi_organ_dataset import MultiOrganDataset
from model.hybrid_unet_transformer import HybridUNetTransformer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CACHE_PATH = os.path.join(PROJECT_ROOT, "multi_organ_cache")

OLD_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "hybrid_unet_transformer_multiorgan.pth"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "hybrid_unet_transformer_resume_checkpoint.pth"
)

BEST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "hybrid_unet_transformer_multiorgan_best.pth"
)


# ============================================================
# 2. SETTINGS
# ============================================================

NUM_CLASSES = 31

BATCH_SIZE = 2

EPOCHS = 10

TRAIN_SAMPLES_PER_EPOCH = 1000

VAL_SAMPLES = 300

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

SEED = 42


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MULTI-ORGAN HYBRID U-NET + TRANSFORMER TRAINING")
print("=" * 60)

print("Project root :", PROJECT_ROOT)
print("Device       :", DEVICE)
print("Classes      :", NUM_CLASSES)
print("Epochs       :", EPOCHS)
print("Batch size   :", BATCH_SIZE)
print("Train/epoch  :", TRAIN_SAMPLES_PER_EPOCH)
print("Validation   :", VAL_SAMPLES)


# ============================================================
# 5. DATASET
# ============================================================

print("\nLoading datasets...")

train_dataset = MultiOrganDataset("train")
val_dataset = MultiOrganDataset("val")

if len(train_dataset) == 0:
    raise RuntimeError(
        "Training dataset is empty. Check multi_organ_cache/train."
    )

if len(val_dataset) == 0:
    raise RuntimeError(
        "Validation dataset is empty. Check multi_organ_cache/val."
    )

print("\nTotal TRAIN samples:", len(train_dataset))
print("Total VAL samples  :", len(val_dataset))


# ============================================================
# 6. CREATE WEIGHTED SAMPLER
# ============================================================

print("\nCalculating foreground sampling weights...")

sample_weights = []

for index in range(len(train_dataset)):

    _, mask = train_dataset[index]

    foreground_pixels = torch.count_nonzero(mask).item()

    if foreground_pixels > 0:
        weight = 5.0
    else:
        weight = 1.0

    sample_weights.append(weight)


sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=min(
        TRAIN_SAMPLES_PER_EPOCH,
        len(train_dataset)
    ),
    replacement=True
)


# ============================================================
# 7. VALIDATION SUBSET
# ============================================================

val_indices = list(range(len(val_dataset)))

random.seed(SEED)

if len(val_indices) > VAL_SAMPLES:
    val_indices = random.sample(
        val_indices,
        VAL_SAMPLES
    )

val_subset = torch.utils.data.Subset(
    val_dataset,
    val_indices
)


# ============================================================
# 8. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# 9. MODEL
# ============================================================

print("\nCreating model...")

model = HybridUNetTransformer(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# 10. LOSS FUNCTIONS
# ============================================================

class_weights = torch.ones(
    NUM_CLASSES,
    dtype=torch.float32
)

# Give foreground classes more importance.
class_weights[1:] = 2.0

class_weights = class_weights.to(DEVICE)


criterion_ce = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


def dice_loss_multiclass(
    logits,
    targets,
    num_classes=31
):

    probs = torch.softmax(logits, dim=1)

    total_dice = 0.0

    valid_classes = 0

    for class_id in range(1, num_classes):

        target_class = (
            targets == class_id
        ).float()

        probability_class = probs[:, class_id]

        target_sum = target_class.sum()

        # Skip a class if it does not occur
        # in this batch.
        if target_sum.item() == 0:
            continue

        intersection = (
            probability_class * target_class
        ).sum()

        denominator = (
            probability_class.sum()
            + target_sum
            + 1e-6
        )

        dice = (
            2.0 * intersection
            + 1e-6
        ) / denominator

        total_dice += (1.0 - dice)

        valid_classes += 1

    if valid_classes == 0:
        return torch.tensor(
            0.0,
            device=logits.device
        )

    return total_dice / valid_classes


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 12. RESUME / INITIAL MODEL
# ============================================================

start_epoch = 0
best_val_loss = float("inf")


if os.path.exists(CHECKPOINT_PATH):

    print("\nResume checkpoint found.")
    print("Loading:", CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val_loss = checkpoint[
        "best_val_loss"
    ]

    print(
        "Resuming from epoch:",
        start_epoch + 1
    )

    print(
        "Previous best validation loss:",
        best_val_loss
    )


else:

    print("\nNo resume checkpoint found.")

    # Use the existing trained model as
    # starting weights if available.
    if os.path.exists(OLD_MODEL_PATH):

        print(
            "Loading existing model weights:"
        )

        print(OLD_MODEL_PATH)

        state_dict = torch.load(
            OLD_MODEL_PATH,
            map_location=DEVICE
        )

        model.load_state_dict(
            state_dict
        )

        print(
            "Existing model weights loaded."
        )

    else:

        print(
            "Existing model not found."
        )

        print(
            "Training will start from random weights."
        )


# ============================================================
# 13. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    batches = 0

    for batch_index, (images, masks) in enumerate(
        train_loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        if outputs.ndim == 5:
            outputs = outputs.squeeze(2)

        ce_loss = criterion_ce(
            outputs,
            masks
        )

        dice_loss = dice_loss_multiclass(
            outputs,
            masks,
            NUM_CLASSES
        )

        loss = (
            0.7 * ce_loss
            + 0.3 * dice_loss
        )

        loss.backward()

        # Prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        batches += 1

        if (batch_index + 1) % 50 == 0:

            print(
                f"Batch {batch_index + 1}/"
                f"{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return running_loss / max(
        batches,
        1
    )


# ============================================================
# 14. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    batches = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            if outputs.ndim == 5:
                outputs = outputs.squeeze(2)

            ce_loss = criterion_ce(
                outputs,
                masks
            )

            dice_loss = dice_loss_multiclass(
                outputs,
                masks,
                NUM_CLASSES
            )

            loss = (
                0.7 * ce_loss
                + 0.3 * dice_loss
            )

            running_loss += loss.item()

            batches += 1

    return running_loss / max(
        batches,
        1
    )


# ============================================================
# 15. TRAINING LOOP
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)


for epoch in range(
    start_epoch,
    EPOCHS
):

    print("\n")
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("-" * 60)

    train_loss = train_one_epoch()

    val_loss = validate()

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print("\nEpoch completed.")

    print(
        f"Training Loss   : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    print(
        f"Learning Rate   : {current_lr:.7f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("\nBEST MODEL SAVED!")

        print(
            BEST_MODEL_PATH
        )


    # ========================================================
    # SAVE FULL CHECKPOINT
    # ========================================================

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_loss":
            best_val_loss,

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "num_classes":
            NUM_CLASSES
    }


    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )


    print("\nCHECKPOINT SAVED!")

    print(
        CHECKPOINT_PATH
    )


# ============================================================
# 16. FINAL
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "\nBest model:"
)

print(
    BEST_MODEL_PATH
)

print(
    "\nResume checkpoint:"
)

print(
    CHECKPOINT_PATH
)

print("\nYou can now run evaluation.")