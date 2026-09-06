from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\split_dataset"
)

TARGET_ORGAN = "Brainstem"

TARGET_SPACING = (1.0, 1.0, 2.0)

TARGET_SIZE = (256, 256)


# ============================================================
# DATASET
# ============================================================

class HaNSegDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        target_organ="Brainstem"
    ):

        self.dataset_path = Path(dataset_path)
        self.target_organ = target_organ

        self.samples = []

        # Find case folders
        case_folders = sorted([
            folder
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
            and folder.name.startswith("case_")
        ])

        print("Cases found:", len(case_folders))

        # Find CT + mask pairs
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

                self.samples.append({
                    "case": case_folder.name,
                    "ct": ct_files[0],
                    "mask": mask_files[0]
                })

        print(
            f"Valid {target_organ} pairs:",
            len(self.samples)
        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)


    # ========================================================
    # RESAMPLE
    # ========================================================

    def resample(
        self,
        image,
        target_spacing,
        is_mask=False
    ):

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(
                original_size[i]
                * original_spacing[i]
                / target_spacing[i]
            ))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()

        resampler.SetOutputSpacing(
            target_spacing
        )

        resampler.SetSize(
            new_size
        )

        resampler.SetOutputDirection(
            image.GetDirection()
        )

        resampler.SetOutputOrigin(
            image.GetOrigin()
        )

        resampler.SetTransform(
            sitk.Transform()
        )

        if is_mask:

            resampler.SetInterpolator(
                sitk.sitkNearestNeighbor
            )

        else:

            resampler.SetInterpolator(
                sitk.sitkLinear
            )

        return resampler.Execute(image)


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]


        # ----------------------------------------------------
        # Read CT
        # ----------------------------------------------------

        ct_image = sitk.ReadImage(
            str(sample["ct"])
        )


        # ----------------------------------------------------
        # Read Mask
        # ----------------------------------------------------

        mask_image = sitk.ReadImage(
            str(sample["mask"])
        )


        # ----------------------------------------------------
        # Resample
        # ----------------------------------------------------

        ct_image = self.resample(
            ct_image,
            TARGET_SPACING,
            False
        )

        mask_image = self.resample(
            mask_image,
            TARGET_SPACING,
            True
        )


        # ----------------------------------------------------
        # NumPy
        # ----------------------------------------------------

        ct = sitk.GetArrayFromImage(
            ct_image
        ).astype(np.float32)

        mask = sitk.GetArrayFromImage(
            mask_image
        ).astype(np.uint8)


        # ----------------------------------------------------
        # Binary mask
        # ----------------------------------------------------

        mask = (mask > 0).astype(np.float32)


        # ----------------------------------------------------
        # CT normalization
        # ----------------------------------------------------

        mean = ct.mean()
        std = ct.std()

        if std > 0:

            ct = (ct - mean) / std


        # ----------------------------------------------------
        # Convert every axial slice to 256 × 256
        # ----------------------------------------------------

        resized_ct = []
        resized_mask = []


        for z in range(ct.shape[0]):

            ct_slice = sitk.GetImageFromArray(
                ct[z]
            )

            mask_slice = sitk.GetImageFromArray(
                mask[z]
            )


            # CT resize
            ct_slice = sitk.Resample(
                ct_slice,
                [256, 256],
                sitk.Transform(),
                sitk.sitkLinear,
                ct_slice.GetOrigin(),
                (
                    ct_slice.GetSize()[0] / 256,
                    ct_slice.GetSize()[1] / 256
                ),
                ct_slice.GetDirection(),
                0.0,
                sitk.sitkFloat32
            )


            # Mask resize
            mask_slice = sitk.Resample(
                mask_slice,
                [256, 256],
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                mask_slice.GetOrigin(),
                (
                    mask_slice.GetSize()[0] / 256,
                    mask_slice.GetSize()[1] / 256
                ),
                mask_slice.GetDirection(),
                0,
                sitk.sitkUInt8
            )


            resized_ct.append(
                sitk.GetArrayFromImage(
                    ct_slice
                )
            )

            resized_mask.append(
                sitk.GetArrayFromImage(
                    mask_slice
                )
            )


        ct = np.stack(resized_ct)

        mask = np.stack(resized_mask)


        # ----------------------------------------------------
        # PyTorch tensors
        # ----------------------------------------------------

        ct_tensor = torch.from_numpy(
            ct
        ).float()

        mask_tensor = torch.from_numpy(
            mask
        ).float()


        # ----------------------------------------------------
        # Add channel
        # ----------------------------------------------------

        ct_tensor = ct_tensor.unsqueeze(0)

        mask_tensor = mask_tensor.unsqueeze(0)


        return {
            "image": ct_tensor,
            "mask": mask_tensor,
            "case": sample["case"]
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    train_path = DATASET_PATH / "train"

    dataset = HaNSegDataset(
        train_path,
        TARGET_ORGAN
    )

    print("\nDataset length:", len(dataset))


    if len(dataset) > 0:

        sample = dataset[0]

        print("\nFirst sample:")

        print(
            "Case:",
            sample["case"]
        )

        print(
            "Image shape:",
            sample["image"].shape
        )

        print(
            "Mask shape:",
            sample["mask"].shape
        )

        print(
            "Image dtype:",
            sample["image"].dtype
        )

        print(
            "Mask dtype:",
            sample["mask"].dtype
        )

        print(
            "Mask values:",
            torch.unique(sample["mask"])
        )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )


    for batch in loader:

        print("\nDataLoader test successful!")

        print(
            "Batch image:",
            batch["image"].shape
        )

        print(
            "Batch mask:",
            batch["mask"].shape
        )

        print(
            "Batch case:",
            batch["case"]
        )

        break