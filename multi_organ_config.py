# ==========================================
# MULTI-ORGAN SEGMENTATION CONFIGURATION
# ==========================================

ORGAN_NAMES = [
    # Replace these 24 names with the exact
    # names 1-24 printed by your dataset script.

    "Brainstem",

    # Example:
    # "Organ_2",
    # "Organ_3",
]

# Background = class 0
NUM_CLASSES = len(ORGAN_NAMES) + 1


# Class ID mapping
CLASS_MAPPING = {
    0: "Background"
}

for class_id, organ in enumerate(ORGAN_NAMES, start=1):
    CLASS_MAPPING[class_id] = organ


# Reverse mapping
ORGAN_TO_CLASS = {
    organ: class_id
    for class_id, organ in CLASS_MAPPING.items()
    if class_id != 0
}


# Display colors for website
# RGB values

ORGAN_COLORS = {
    "Brainstem": (128, 0, 255),

    # Add remaining organs here later
}


if __name__ == "__main__":

    print("=" * 50)
    print("MULTI-ORGAN CONFIGURATION")
    print("=" * 50)

    print("\nClasses:")

    for class_id, organ in CLASS_MAPPING.items():
        print(f"{class_id:2} -> {organ}")

    print("\nTotal classes:", NUM_CLASSES)