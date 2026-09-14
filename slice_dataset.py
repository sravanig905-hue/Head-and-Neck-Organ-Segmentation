import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class SliceDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.samples = []

        # Find all case folders
        case_dirs = sorted(
            glob.glob(
                os.path.join(root_dir, "case_*")
            )
        )

        print(f"\nSearching: {root_dir}")
        print(f"Found case folders: {len(case_dirs)}")

        # Find image/mask pairs
        for case_dir in case_dirs:

            image_files = sorted(
                glob.glob(
                    os.path.join(
                        case_dir,
                        "image_*.npy"
                    )
                )
            )

            for image_path in image_files:

                filename = os.path.basename(
                    image_path
                )

                # image_0001.npy -> mask_0001.npy
                slice_id = filename.replace(
                    "image_",
                    ""
                )

                mask_path = os.path.join(
                    case_dir,
                    "mask_" + slice_id
                )

                if os.path.exists(mask_path):

                    self.samples.append(
                        (image_path, mask_path)
                    )

        print(
            f"Found image-mask pairs: "
            f"{len(self.samples)}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        # Load numpy arrays
        image = np.load(image_path)
        mask = np.load(mask_path)

        # Convert to float32
        image = image.astype(
            np.float32
        )

        mask = mask.astype(
            np.float32
        )

        # Convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Add channel dimension
        # [H,W] -> [1,H,W]
        if image.ndim == 2:
            image = image.unsqueeze(0)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Make sure mask is binary
        mask = (mask > 0).float()

        return image, mask


# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------

if __name__ == "__main__":

    dataset = SliceDataset(
        "fast_cache/train"
    )

    print("\nDataset size:", len(dataset))

    if len(dataset) > 0:

        image, mask = dataset[0]

        print(
            "Image shape:",
            image.shape
        )

        print(
            "Mask shape:",
            mask.shape
        )

        print(
            "Image dtype:",
            image.dtype
        )

        print(
            "Mask dtype:",
            mask.dtype
        )

        print(
            "Image min/max:",
            image.min().item(),
            image.max().item()
        )

        print(
            "Mask unique values:",
            torch.unique(mask)
        )