import os
import glob
from collections import defaultdict

DATASET_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1"

organ_cases = defaultdict(list)

case_folders = sorted(glob.glob(os.path.join(DATASET_PATH, "case_*")))

for case_folder in case_folders:
    case_name = os.path.basename(case_folder)

    files = glob.glob(os.path.join(case_folder, "*_OAR_*.seg.nrrd"))

    for file in files:
        filename = os.path.basename(file)

        organ = filename.split("_OAR_", 1)[1]
        organ = organ.replace(".seg.nrrd", "")

        organ_cases[organ].append(case_name)

print("\n" + "=" * 60)
print("        MULTI-ORGAN DATASET ANALYSIS")
print("=" * 60)

print(f"\nTotal patient cases found: {len(case_folders)}")
print(f"Total unique organs found: {len(organ_cases)}")

print("\nOrgan availability:")
print("-" * 60)

for i, organ in enumerate(sorted(organ_cases), 1):
    cases = organ_cases[organ]

    print(
        f"{i:2}. {organ:<30} "
        f"{len(cases):>3} / {len(case_folders)} cases"
    )

print("\n" + "=" * 60)