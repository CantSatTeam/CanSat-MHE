import os
import json
import random
from pathlib import Path

IMAGE_FOLDER = "../../dataset/validation/image"
OUTPUT_JSON_PATH = "./test-dataset/augmentation-assignments.json"
RATIOS = {
    "none": 0.4,
    "cloud": 0.2,
    "quality": 0.2,
    "motion_blur": 0.2
}
SEED = 67

def assign_augmentations() -> None:
    random.seed(SEED)

    image_files = [f.name for f in Path(IMAGE_FOLDER).iterdir() if f.suffix.lower() == ".png"]
    print(image_files)
    random.shuffle(image_files)

    assignments = {key: [] for key in RATIOS.keys()}
    total_images = len(image_files)

    start_idx = 0
    for aug_type, ratio in RATIOS.items():
        count = int(total_images * ratio)
        if aug_type == list(RATIOS.keys())[-1]: # rounding
            count = total_images - start_idx
        
        assignments[aug_type] = image_files[start_idx:start_idx + count]
        start_idx += count
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(assignments, f, indent=4)

    print(f"Assigned {total_images} images:")
    for aug_type, images in assignments.items():
        print(f"\t{aug_type}: {len(images)} images ({len(images)/total_images*100:.1f}%)")

if __name__ == "__main__":
    assign_augmentations()
