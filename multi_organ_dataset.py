import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = r"C:\Users\Sravani\Desktop\HaN_Seg_Project"

CACHE_PATH = os.path.join(
    PROJECT_ROOT,
    "multi_organ_cache"
)


class MultiOrganDataset(Dataset):

    def __init__(self, split):

        self.split = split

        split_path = os.path.join(
            CACHE_PATH,
            split
        )

        self.samples = []

        case_folders = sorted(
            glob.glob(
                os.path.join(
                    split_path,
                    "case_*"
                )
            )
        )

        for case_folder in case_folders:

            image_files = sorted(
                glob.glob(
                    os.path.join(
                        case_folder,
                        "*_image.npy"
                    )
                )
            )

            for image_file in image_files:

                mask_file = image_file.replace(
                    "_image.npy",
                    "_mask.npy"
                )

                if os.path.exists(mask_file):

                    self.samples.append(
                        (
                            image_file,
                            mask_file
                        )
                    )

        print(
            f"{split.upper()} samples: "
            f"{len(self.samples)}"
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_file, mask_file = self.samples[index]

        image = np.load(
            image_file
        ).astype(
            np.float32
        )

        mask = np.load(
            mask_file
        ).astype(
            np.int64
        )

        # Add channel dimension
        # [256,256] -> [1,256,256]

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        # Mask remains:
        # [256,256]

        mask = torch.from_numpy(
            mask
        )

        return image, mask


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    train_dataset = MultiOrganDataset(
        "train"
    )

    val_dataset = MultiOrganDataset(
        "val"
    )

    test_dataset = MultiOrganDataset(
        "test"
    )

    print("\n======================================")
    print("MULTI-ORGAN DATASET TEST")
    print("======================================")

    if len(train_dataset) > 0:

        image, mask = train_dataset[0]

        print(
            "Image shape:",
            image.shape
        )

        print(
            "Mask shape :",
            mask.shape
        )

        print(
            "Image dtype:",
            image.dtype
        )

        print(
            "Mask dtype :",
            mask.dtype
        )

        print(
            "Mask classes:",
            torch.unique(mask).tolist()
        )

    # --------------------------------------------------------
    # DataLoader test
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    images, masks = next(
        iter(train_loader)
    )

    print("\nDataLoader test:")

    print(
        "Batch images:",
        images.shape
    )

    print(
        "Batch masks :",
        masks.shape
    )

    print("\nDATASET LOADER READY!")