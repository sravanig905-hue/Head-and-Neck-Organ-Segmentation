import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CACHE_PATH = os.path.join(
    BASE_DIR,
    "multi_organ_cache"
)


# ============================================================
# MULTI-ORGAN DATASET
# ============================================================

class MultiOrganDataset(Dataset):

    def __init__(self, split):

        self.split = split

        self.split_dir = os.path.join(
            CACHE_PATH,
            split
        )

        self.samples = []

        if not os.path.exists(self.split_dir):

            print(
                f"WARNING: Directory not found: "
                f"{self.split_dir}"
            )

            return

        # ----------------------------------------------------
        # Search recursively inside case folders
        # ----------------------------------------------------

        for root, dirs, files in os.walk(
            self.split_dir
        ):

            for filename in files:

                if not filename.endswith(
                    "_image.npy"
                ):
                    continue

                image_path = os.path.join(
                    root,
                    filename
                )

                mask_filename = filename.replace(
                    "_image.npy",
                    "_mask.npy"
                )

                mask_path = os.path.join(
                    root,
                    mask_filename
                )

                if os.path.exists(mask_path):

                    self.samples.append(
                        (
                            image_path,
                            mask_path
                        )
                    )

        # ----------------------------------------------------
        # Sort for reproducibility
        # ----------------------------------------------------

        self.samples.sort()

        print(
            f"{split.upper()} samples:",
            len(self.samples)
        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        image_path, mask_path = (
            self.samples[index]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = np.load(
            image_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask = np.load(
            mask_path
        ).astype(
            np.int64
        )

        # ----------------------------------------------------
        # Image shape
        # [H,W] → [1,H,W]
        # ----------------------------------------------------

        if image.ndim == 2:

            image = np.expand_dims(
                image,
                axis=0
            )

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).long()

        return image, mask


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MULTI-ORGAN DATASET TEST")
    print("=" * 60)

    print(
        "Cache path:",
        CACHE_PATH
    )

    train_dataset = MultiOrganDataset(
        "train"
    )

    val_dataset = MultiOrganDataset(
        "val"
    )

    test_dataset = MultiOrganDataset(
        "test"
    )

    print()
    print(
        "TRAIN samples:",
        len(train_dataset)
    )

    print(
        "VAL samples:",
        len(val_dataset)
    )

    print(
        "TEST samples:",
        len(test_dataset)
    )

    if len(train_dataset) > 0:

        images, masks = train_dataset[0]

        print()
        print(
            "Sample image shape:",
            images.shape
        )

        print(
            "Sample mask shape:",
            masks.shape
        )

        print(
            "Image dtype:",
            images.dtype
        )

        print(
            "Mask dtype:",
            masks.dtype
        )

        print(
            "Mask classes:",
            torch.unique(masks).tolist()
        )

        print()
        print(
            "DATASET LOADER READY!"
        )

    else:

        print()
        print(
            "ERROR: TRAIN DATASET IS EMPTY!"
        )