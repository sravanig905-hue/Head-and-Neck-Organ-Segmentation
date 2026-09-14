import os
import glob

DATASET_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1"

organs = set()

for case_folder in sorted(glob.glob(os.path.join(DATASET_PATH, "case_*"))):
    for file in glob.glob(os.path.join(case_folder, "*_OAR_*.seg.nrrd")):

        filename = os.path.basename(file)

        organ = filename.split("_OAR_", 1)[1]
        organ = organ.replace(".seg.nrrd", "")

        organs.add(organ)

print("\n===================================")
print("      DATASET ORGAN LIST")
print("===================================\n")

for i, organ in enumerate(sorted(organs), start=1):
    print(f"{i}. {organ}")

print("\n===================================")
print(f"Total unique organs: {len(organs)}")
print("===================================")