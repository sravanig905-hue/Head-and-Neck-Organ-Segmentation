import os
import random

PROJECT_ROOT = r"C:\Users\Sravani\Desktop\HaN_Seg_Project"

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "HaN-Seg",
    "HaN-Seg",
    "set_1"
)

SPLIT_PATH = os.path.join(
    PROJECT_ROOT,
    "splits"
)

SEED = 42


# ============================================================
# FIND CASES
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("ERROR: Dataset path not found:")
    print(DATASET_PATH)
    exit()

cases = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if folder.startswith("case_")
    and os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])


print("\n==============================================")
print("MULTI-ORGAN DATASET SPLIT")
print("==============================================")

print(
    "Dataset path:",
    DATASET_PATH
)

print(
    "Total cases found:",
    len(cases)
)


if len(cases) == 0:

    print("\nERROR: No case folders found!")
    exit()


# ============================================================
# SHUFFLE
# ============================================================

random.seed(SEED)

random.shuffle(cases)


# ============================================================
# 70 / 15 / 15 SPLIT
# ============================================================

total = len(cases)

train_count = int(
    total * 0.70
)

val_count = int(
    total * 0.15
)

test_count = (
    total
    - train_count
    - val_count
)


train_cases = cases[
    :train_count
]

val_cases = cases[
    train_count:
    train_count + val_count
]

test_cases = cases[
    train_count + val_count:
]


# ============================================================
# CREATE SPLIT DIRECTORY
# ============================================================

os.makedirs(
    SPLIT_PATH,
    exist_ok=True
)


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_split(
    filename,
    case_list
):

    path = os.path.join(
        SPLIT_PATH,
        filename
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for case in sorted(case_list):

            f.write(
                case + "\n"
            )

    print(
        f"Saved {filename}: "
        f"{len(case_list)} cases"
    )


# ============================================================
# SAVE SPLITS
# ============================================================

save_split(
    "train.txt",
    train_cases
)

save_split(
    "val.txt",
    val_cases
)

save_split(
    "test.txt",
    test_cases
)


# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("SPLIT COMPLETED")
print("==============================================")

print(
    f"Train cases : {len(train_cases)}"
)

print(
    f"Val cases   : {len(val_cases)}"
)

print(
    f"Test cases  : {len(test_cases)}"
)

print(
    f"Total       : "
    f"{len(train_cases) + len(val_cases) + len(test_cases)}"
)

print("\nTrain:")
print(sorted(train_cases))

print("\nValidation:")
print(sorted(val_cases))

print("\nTest:")
print(sorted(test_cases))

print("\nSplit files saved in:")
print(SPLIT_PATH)