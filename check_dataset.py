from pathlib import Path
import SimpleITK as sitk


# ============================================================
# HaN-Seg DATASET PATH
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg"
)


# ============================================================
# CHECK DATASET PATH
# ============================================================

if not DATASET_PATH.exists():
    print("ERROR: Dataset path not found!")
    print(DATASET_PATH)
    exit()

print("=" * 70)
print("HaN-Seg DATASET CHECK")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH)


# ============================================================
# FIND CASE FOLDERS
# ============================================================

case_folders = sorted(
    [
        folder
        for folder in DATASET_PATH.rglob("*")
        if folder.is_dir() and folder.name.startswith("case_")
    ]
)


print("\nTotal case folders found:", len(case_folders))


# ============================================================
# CHECK EACH CASE
# ============================================================

for case_index, case_folder in enumerate(case_folders, start=1):

    print("\n")
    print("=" * 70)
    print(f"CASE {case_index}: {case_folder.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # CT FILE
    # --------------------------------------------------------

    ct_files = list(case_folder.glob("*_IMG_CT.nrrd"))

    if len(ct_files) == 0:

        print("CT file: NOT FOUND")

    else:

        ct_file = ct_files[0]

        print("\nCT file:")
        print(ct_file.name)

        try:

            ct_image = sitk.ReadImage(str(ct_file))

            ct_size = ct_image.GetSize()
            ct_spacing = ct_image.GetSpacing()

            print("CT size:", ct_size)
            print("CT spacing:", ct_spacing)

        except Exception as e:

            print("Error reading CT:")
            print(e)


    # --------------------------------------------------------
    # MRI FILE
    # --------------------------------------------------------

    mr_files = list(case_folder.glob("*_IMG_MR_T1.nrrd"))

    if len(mr_files) > 0:

        print("\nMRI file:")
        print(mr_files[0].name)

    else:

        print("\nMRI file: NOT FOUND")


    # --------------------------------------------------------
    # SEGMENTATION FILES
    # --------------------------------------------------------

    seg_files = sorted(
        case_folder.glob("*.seg.nrrd")
    )

    print("\nNumber of segmentation masks:", len(seg_files))

    print("\nOrgan masks:")

    for seg_file in seg_files:

        # Remove case prefix and .seg.nrrd
        organ_name = seg_file.name

        organ_name = organ_name.replace(
            f"{case_folder.name}_OAR_",
            ""
        )

        organ_name = organ_name.replace(
            ".seg.nrrd",
            ""
        )

        print("  -", organ_name)


    # --------------------------------------------------------
    # CHECK MASK DIMENSIONS
    # --------------------------------------------------------

    if len(ct_files) > 0 and len(seg_files) > 0:

        print("\nMask dimension check:")

        try:

            ct_image = sitk.ReadImage(str(ct_files[0]))

            ct_size = ct_image.GetSize()

            matching = 0
            not_matching = 0

            for seg_file in seg_files:

                try:

                    mask_image = sitk.ReadImage(
                        str(seg_file)
                    )

                    mask_size = mask_image.GetSize()

                    if mask_size == ct_size:

                        matching += 1

                    else:

                        not_matching += 1

                        print(
                            "  MISMATCH:",
                            seg_file.name,
                            "->",
                            mask_size
                        )

                except Exception as e:

                    print(
                        "  ERROR:",
                        seg_file.name,
                        e
                    )

            print(
                "\nMatching masks:",
                matching
            )

            print(
                "Non-matching masks:",
                not_matching
            )

        except Exception as e:

            print("Dimension check error:", e)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("DATASET CHECK COMPLETED")
print("=" * 70)