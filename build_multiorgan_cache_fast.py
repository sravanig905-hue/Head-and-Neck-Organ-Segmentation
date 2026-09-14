import os
import glob
import numpy as np
import SimpleITK as sitk
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "HaN-Seg",
    "HaN-Seg",
    "set_1"
)

SPLIT_ROOT = os.path.join(PROJECT_ROOT, "splits")
CACHE_ROOT = os.path.join(PROJECT_ROOT, "multi_organ_cache")

IMG_SIZE = (256, 256)

# Keep limited background slices
MAX_BACKGROUND_PER_CASE = 50


def resize_image(arr):
    img = Image.fromarray(arr.astype(np.float32))
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32)


def resize_mask(arr):
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize(IMG_SIZE, Image.Resampling.NEAREST)
    return np.array(img, dtype=np.uint8)


def get_organs(case_dir):
    files = glob.glob(
        os.path.join(case_dir, "*_OAR_*.seg.nrrd")
    )

    organs = []

    for f in files:
        name = os.path.basename(f)

        # Example:
        # case_01_OAR_Brainstem.seg.nrrd
        if "_OAR_" in name and name.endswith(".seg.nrrd"):
            organ = name.split("_OAR_")[1]
            organ = organ.replace(".seg.nrrd", "")
            organs.append(organ)

    return sorted(set(organs))


def load_split(split):
    split_file = os.path.join(
        SPLIT_ROOT,
        f"{split}.txt"
    )

    if not os.path.exists(split_file):
        print("ERROR: Split file not found:", split_file)
        return []

    with open(split_file, "r") as f:
        cases = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return cases


def build_class_mapping(all_cases):
    organs = set()

    for case in all_cases:
        case_dir = os.path.join(DATASET_ROOT, case)

        if os.path.exists(case_dir):
            organs.update(get_organs(case_dir))

    organs = sorted(organs)

    mapping = {
        0: "Background"
    }

    for i, organ in enumerate(organs, start=1):
        mapping[i] = organ

    return mapping


def build_case(case_name, split, mapping):

    case_dir = os.path.join(
        DATASET_ROOT,
        case_name
    )

    ct_files = glob.glob(
        os.path.join(case_dir, "*_IMG_CT.nrrd")
    )

    if not ct_files:
        print("  CT not found:", case_name)
        return 0

    ct_file = ct_files[0]

    print(f"Processing {split}/{case_name}")

    # -------------------------
    # Load CT ONCE
    # -------------------------
    ct_img = sitk.ReadImage(ct_file)
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)

    # Normalize entire CT volume
    mean = ct.mean()
    std = ct.std()

    if std > 0:
        ct = (ct - mean) / std

    # -------------------------
    # Create multi-class mask
    # -------------------------
    multi_mask = np.zeros(
        ct.shape,
        dtype=np.uint8
    )

    for class_id, organ in mapping.items():

        if class_id == 0:
            continue

        mask_file = os.path.join(
            case_dir,
            f"{case_name}_OAR_{organ}.seg.nrrd"
        )

        if not os.path.exists(mask_file):
            continue

        mask_img = sitk.ReadImage(mask_file)
        mask = sitk.GetArrayFromImage(mask_img)

        # Binary mask
        mask = mask > 0

        # Assign class ID
        multi_mask[mask] = class_id

    # -------------------------
    # Select useful slices
    # -------------------------
    positive_slices = np.where(
        np.any(multi_mask > 0, axis=(1, 2))
    )[0]

    if len(positive_slices) == 0:
        print("  No positive slices!")
        return 0

    # Add nearby slices
    selected = set()

    for z in positive_slices:
        for offset in range(-3, 4):
            zz = z + offset

            if 0 <= zz < ct.shape[0]:
                selected.add(zz)

    selected = sorted(selected)

    # Limited background slices
    background = [
        z for z in range(ct.shape[0])
        if z not in selected
    ]

    if len(background) > MAX_BACKGROUND_PER_CASE:
        step = max(
            1,
            len(background) // MAX_BACKGROUND_PER_CASE
        )
        background = background[::step][:MAX_BACKGROUND_PER_CASE]

    selected = sorted(
        set(selected + background)
    )

    print(
        f"  CT slices      : {ct.shape[0]}"
    )

    print(
        f"  Positive slices: {len(positive_slices)}"
    )

    print(
        f"  Selected slices: {len(selected)}"
    )

    # -------------------------
    # Output folder
    # -------------------------
    output_dir = os.path.join(
        CACHE_ROOT,
        split,
        case_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    count = 0

    # -------------------------
    # Resize + save
    # -------------------------
    for z in selected:

        image_slice = ct[z]
        mask_slice = multi_mask[z]

        image_resized = resize_image(
            image_slice
        )

        mask_resized = resize_mask(
            mask_slice
        )

        image_file = os.path.join(
            output_dir,
            f"slice_{z:04d}_image.npy"
        )

        mask_file = os.path.join(
            output_dir,
            f"slice_{z:04d}_mask.npy"
        )

        np.save(
            image_file,
            image_resized.astype(np.float16)
        )

        np.save(
            mask_file,
            mask_resized.astype(np.uint8)
        )

        count += 1

    print(
        f"  Saved: {count} slices"
    )

    return count


def main():

    print("=" * 60)
    print("FAST MULTI-ORGAN CACHE BUILDER")
    print("=" * 60)

    os.makedirs(
        CACHE_ROOT,
        exist_ok=True
    )

    # -------------------------
    # Load splits
    # -------------------------
    train_cases = load_split("train")
    val_cases = load_split("val")
    test_cases = load_split("test")

    all_cases = (
        train_cases +
        val_cases +
        test_cases
    )

    print()
    print("Dataset Split:")
    print("-" * 40)
    print("Train cases :", len(train_cases))
    print("Val cases   :", len(val_cases))
    print("Test cases  :", len(test_cases))

    # -------------------------
    # Build class mapping
    # -------------------------
    mapping = build_class_mapping(
        all_cases
    )

    mapping_file = os.path.join(
        CACHE_ROOT,
        "class_mapping.txt"
    )

    with open(
        mapping_file,
        "w"
    ) as f:

        for class_id, organ in mapping.items():
            f.write(
                f"{class_id} -> {organ}\n"
            )

    print()
    print(
        f"Number of classes: {len(mapping)}"
    )

    print(
        f"Class mapping saved to: {mapping_file}"
    )

    # -------------------------
    # Build each split
    # -------------------------
    total = 0

    for split, cases in [
        ("train", train_cases),
        ("val", val_cases),
        ("test", test_cases)
    ]:

        print()
        print("=" * 60)
        print(f"BUILDING {split.upper()} CACHE")
        print("=" * 60)

        for i, case in enumerate(cases, start=1):

            print(
                f"\n[{i}/{len(cases)}] ",
                end=""
            )

            total += build_case(
                case,
                split,
                mapping
            )

    print()
    print("=" * 60)
    print("CACHE BUILD COMPLETED")
    print("=" * 60)

    print(
        "Total slices saved:",
        total
    )


if __name__ == "__main__":
    main()