from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\split_dataset"
)

TARGET_ORGAN = "Brainstem"

TARGET_SPACING = (1.0, 1.0, 2.0)

TARGET_SIZE = (256, 256)


# ============================================================
# DATASET CLASS
# ============================================================

class HaNSegDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        target_organ="Brainstem",
        augment=False
    ):

        self.dataset_path = Path(dataset_path)
        self.target_organ = target_organ
        self.augment = augment

        self.samples = []

        # Find cases
        case_folders = sorted([
            folder
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
            and folder.name.startswith("case_")
        ])

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
            f"{self.dataset_path.name}: "
            f"{len(self.samples)} valid cases"
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
    # RESIZE 2D SLICE
    # ========================================================

    def resize_slice(self, array, is_mask=False):

        image = sitk.GetImageFromArray(array)

        interpolation = (
            sitk.sitkNearestNeighbor
            if is_mask
            else sitk.sitkLinear
        )

        resized = sitk.Resample(
            image,
            [TARGET_SIZE[0], TARGET_SIZE[1]],
            sitk.Transform(),
            interpolation,
            image.GetOrigin(),
            (
                image.GetSize()[0] / TARGET_SIZE[0],
                image.GetSize()[1] / TARGET_SIZE[1]
            ),
            image.GetDirection(),
            0,
            sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
        )

        return sitk.GetArrayFromImage(resized)


    # ========================================================
    # AUGMENTATION
    # ========================================================

    def augment_slice(self, image, mask):

        # Horizontal flip
        if np.random.rand() < 0.5:

            image = np.flip(
                image,
                axis=1
            ).copy()

            mask = np.flip(
                mask,
                axis=1
            ).copy()


        # Vertical flip
        if np.random.rand() < 0.2:

            image = np.flip(
                image,
                axis=0
            ).copy()

            mask = np.flip(
                mask,
                axis=0
            ).copy()


        # 90 degree rotation
        if np.random.rand() < 0.2:

            image = np.rot90(
                image
            ).copy()

            mask = np.rot90(
                mask
            ).copy()


        return image, mask


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
        # Read MASK
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
            is_mask=False
        )

        mask_image = self.resample(
            mask_image,
            TARGET_SPACING,
            is_mask=True
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

        mask = (mask > 0).astype(np.uint8)


        # ----------------------------------------------------
        # Normalize CT
        # ----------------------------------------------------

        mean = ct.mean()
        std = ct.std()

        if std > 0:

            ct = (ct - mean) / std


        # ----------------------------------------------------
        # Resize all slices
        # ----------------------------------------------------

        resized_ct = []
        resized_mask = []

        for z in range(ct.shape[0]):

            ct_slice = self.resize_slice(
                ct[z],
                is_mask=False
            )

            mask_slice = self.resize_slice(
                mask[z],
                is_mask=True
            )

            # Keep binary after resize
            mask_slice = (
                mask_slice > 0.5
            ).astype(np.float32)


            # ------------------------------------------------
            # Augmentation ONLY if requested
            # ------------------------------------------------

            if self.augment:

                ct_slice, mask_slice = (
                    self.augment_slice(
                        ct_slice,
                        mask_slice
                    )
                )


            resized_ct.append(
                ct_slice.astype(np.float32)
            )

            resized_mask.append(
                mask_slice.astype(np.float32)
            )


        # ----------------------------------------------------
        # Stack slices
        # ----------------------------------------------------

        ct = np.stack(
            resized_ct
        )

        mask = np.stack(
            resized_mask
        )


        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            ct
        ).float()

        mask_tensor = torch.from_numpy(
            mask
        ).float()


        # ----------------------------------------------------
        # Channel dimension
        # ----------------------------------------------------

        image_tensor = image_tensor.unsqueeze(0)

        mask_tensor = mask_tensor.unsqueeze(0)


        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case": sample["case"]
        }


# ============================================================
# TEST ALL THREE DATASETS
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FINAL HaN-Seg DATASET PIPELINE")
    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_dataset = HaNSegDataset(
        DATASET_PATH / "train",
        TARGET_ORGAN,
        augment=True
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_dataset = HaNSegDataset(
        DATASET_PATH / "val",
        TARGET_ORGAN,
        augment=False
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_dataset = HaNSegDataset(
        DATASET_PATH / "test",
        TARGET_ORGAN,
        augment=False
    )


    print("\nDataset sizes:")

    print(
        "Train      :",
        len(train_dataset)
    )

    print(
        "Validation :",
        len(val_dataset)
    )

    print(
        "Test       :",
        len(test_dataset)
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )


    # ========================================================
    # TEST TRAIN LOADER
    # ========================================================

    print("\nTesting Train DataLoader...")

    for batch in train_loader:

        print(
            "Train image shape:",
            batch["image"].shape
        )

        print(
            "Train mask shape:",
            batch["mask"].shape
        )

        print(
            "Train case:",
            batch["case"]
        )

        print(
            "Mask values:",
            torch.unique(batch["mask"])
        )

        break


    # ========================================================
    # TEST VALIDATION LOADER
    # ========================================================

    print("\nTesting Validation DataLoader...")

    for batch in val_loader:

        print(
            "Validation image shape:",
            batch["image"].shape
        )

        print(
            "Validation mask shape:",
            batch["mask"].shape
        )

        break


    # ========================================================
    # TEST TEST LOADER
    # ========================================================

    print("\nTesting Test DataLoader...")

    for batch in test_loader:

        print(
            "Test image shape:",
            batch["image"].shape
        )

        print(
            "Test mask shape:",
            batch["mask"].shape
        )

        break


    print("\n" + "=" * 70)
    print("FINAL DATASET PIPELINE TEST COMPLETED")
    print("=" * 70)