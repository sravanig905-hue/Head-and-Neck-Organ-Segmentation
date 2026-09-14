from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg"
)


# ============================================================
# ORGAN MASK WE WILL TEST FIRST
# ============================================================

# First, we use ONE organ to verify the complete pipeline.
# Later we will convert this into multi-organ segmentation.

TARGET_ORGAN = "Brainstem"


# ============================================================
# DATASET CLASS
# ============================================================

class HaNSegDataset(Dataset):

    def __init__(self, dataset_path, target_organ="Brainstem"):

        self.dataset_path = Path(dataset_path)
        self.target_organ = target_organ

        self.samples = []

        # Find all case folders
        case_folders = sorted(
            [
                folder
                for folder in self.dataset_path.rglob("*")
                if folder.is_dir()
                and folder.name.startswith("case_")
            ]
        )

        print("Cases found:", len(case_folders))

        # Find CT + selected mask
        for case_folder in case_folders:

            ct_files = list(
                case_folder.glob("*_IMG_CT.nrrd")
            )

            mask_files = list(
                case_folder.glob(
                    f"*_OAR_{target_organ}.seg.nrrd"
                )
            )

            if len(ct_files) == 1 and len(mask_files) == 1:

                self.samples.append(
                    {
                        "case": case_folder.name,
                        "ct": ct_files[0],
                        "mask": mask_files[0]
                    }
                )

        print(
            f"Valid CT + {target_organ} pairs:",
            len(self.samples)
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Read CT
        # ----------------------------------------------------

        ct_image = sitk.ReadImage(
            str(sample["ct"])
        )

        ct_array = sitk.GetArrayFromImage(
            ct_image
        ).astype(np.float32)


        # ----------------------------------------------------
        # Read Mask
        # ----------------------------------------------------

        mask_image = sitk.ReadImage(
            str(sample["mask"])
        )

        mask_array = sitk.GetArrayFromImage(
            mask_image
        ).astype(np.float32)


        # ----------------------------------------------------
        # Convert mask to binary
        # ----------------------------------------------------

        mask_array = (
            mask_array > 0
        ).astype(np.float32)


        # ----------------------------------------------------
        # Normalize CT
        # ----------------------------------------------------

        mean = ct_array.mean()
        std = ct_array.std()

        if std > 0:

            ct_array = (
                ct_array - mean
            ) / std


        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        ct_tensor = torch.from_numpy(
            ct_array
        )

        mask_tensor = torch.from_numpy(
            mask_array
        )


        # ----------------------------------------------------
        # Add channel dimension
        #
        # Original:
        #     [Slices, Height, Width]
        #
        # After:
        #     [1, Slices, Height, Width]
        # ----------------------------------------------------

        ct_tensor = ct_tensor.unsqueeze(0)

        mask_tensor = mask_tensor.unsqueeze(0)


        return {
            "image": ct_tensor,
            "mask": mask_tensor,
            "case": sample["case"]
        }


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HaN-Seg NRRD DATASET TEST")
    print("=" * 70)


    dataset = HaNSegDataset(
        DATASET_PATH,
        TARGET_ORGAN
    )


    print("\nDataset length:", len(dataset))


    if len(dataset) > 0:

        sample = dataset[0]

        print("\nFirst sample information")
        print("-" * 70)

        print("Case:", sample["case"])

        print(
            "CT tensor shape:",
            sample["image"].shape
        )

        print(
            "Mask tensor shape:",
            sample["mask"].shape
        )

        print(
            "CT data type:",
            sample["image"].dtype
        )

        print(
            "Mask data type:",
            sample["mask"].dtype
        )

        print(
            "CT min:",
            sample["image"].min().item()
        )

        print(
            "CT max:",
            sample["image"].max().item()
        )

        print(
            "Mask unique values:",
            torch.unique(sample["mask"])
        )


    # ========================================================
    # DATALOADER TEST
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATALOADER TEST")
    print("=" * 70)


    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )


    for batch in loader:

        print("\nBatch received successfully!")

        print(
            "Batch CT shape:",
            batch["image"].shape
        )

        print(
            "Batch Mask shape:",
            batch["mask"].shape
        )

        print(
            "Batch case:",
            batch["case"]
        )

        break


    print("\n")
    print("=" * 70)
    print("NRRD DATALOADER TEST COMPLETED")
    print("=" * 70)