from pathlib import Path
import random
import shutil


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg"
)

OUTPUT_PATH = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\split_dataset"
)

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# FIND CASES
# ============================================================

case_folders = sorted([
    folder
    for folder in DATASET_PATH.rglob("*")
    if folder.is_dir()
    and folder.name.startswith("case_")
])

print("Total cases found:", len(case_folders))


if len(case_folders) == 0:
    raise ValueError("No case folders found!")


# ============================================================
# SHUFFLE CASES
# ============================================================

random.seed(SEED)

random.shuffle(case_folders)


# ============================================================
# CALCULATE SPLIT
# ============================================================

total = len(case_folders)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)

train_cases = case_folders[:train_count]

val_cases = case_folders[
    train_count:train_count + val_count
]

test_cases = case_folders[
    train_count + val_count:
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

train_path = OUTPUT_PATH / "train"
val_path = OUTPUT_PATH / "val"
test_path = OUTPUT_PATH / "test"

train_path.mkdir(parents=True, exist_ok=True)
val_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# COPY CASE FOLDERS
# ============================================================

def copy_cases(cases, destination):

    for case in cases:

        target = destination / case.name

        shutil.copytree(
            case,
            target,
            dirs_exist_ok=True
        )


copy_cases(train_cases, train_path)
copy_cases(val_cases, val_path)
copy_cases(test_cases, test_path)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print("\nTraining cases   :", len(train_cases))
print("Validation cases :", len(val_cases))
print("Testing cases    :", len(test_cases))

print("\nTraining:")
for case in train_cases:
    print(" ", case.name)

print("\nValidation:")
for case in val_cases:
    print(" ", case.name)

print("\nTesting:")
for case in test_cases:
    print(" ", case.name)


print("\n" + "=" * 60)
print("SPLIT COMPLETED")
print("=" * 60)