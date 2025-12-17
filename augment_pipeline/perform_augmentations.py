import random, os, json
from PIL import Image

from apply_blur import apply_blur
from apply_quality import apply_quality

INPUT_PATH = "./augment_pipeline/input_images"
OUTPUT_PATH = "./augment_pipeline/output"

def perform_augmentation(augmentation_type: str, image_name: str, seed: int):
    # open input image
    path = INPUT_PATH + "/" + image_name + ".jpg"
    input_image = Image.open(path)

    # apply augmentation
    augment_output = None
    if augmentation_type == "motion_blur":
        augment_output = apply_blur(input_image, seed)
    elif augmentation_type == "quality":
        augment_output = apply_quality(input_image, seed)
    else:
        # ! is this even how you do error descriptions in python
        raise ValueError("augmentation_type is not in [\"motion_blur\", \"quality\"]")

    # log seed
    augment_output.log_data["seed"] = seed

    # write image and log
    output_path = OUTPUT_PATH + "/" + image_name + "/"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    augment_output.output_image.save(output_path + "out.jpg")
    with open(output_path + "log.json", "w") as f:
        json.dump(augment_output.log_data, f, indent=4)

if __name__ == "__main__":
    augmentation_type = "motion_blur"
    image_name = "1"
    seed = random.randrange(0, 2**32)
    perform_augmentation(augmentation_type, image_name, seed)
