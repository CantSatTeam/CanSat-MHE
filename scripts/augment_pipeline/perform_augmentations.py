import random, os, json
from PIL import Image

from apply_blur import apply_blur
from apply_quality import apply_quality
from apply_cloud import apply_cloud

DATA_PATH = "../data/"
INPUT_PATH = DATA_PATH + "augment/input"
OUTPUT_PATH = DATA_PATH + "augment/output"

def perform_augmentation(augmentation_type: str, image_name: str, seed: int) -> None:
    # open input image
    path = INPUT_PATH + "/" + image_name + ".jpg"
    input_image = Image.open(path)

    # apply augmentation
    augment_output = None
    if augmentation_type == "motion_blur":
        augment_output = apply_blur(input_image, seed)
    elif augmentation_type == "quality":
        augment_output = apply_quality(input_image, seed)
    elif augmentation_type == "cloud":
        augment_output = apply_cloud(input_image, seed)
    else:
        # ! is this even how you do error descriptions in python
        raise ValueError("augmentation_type is not in [\"motion_blur\", \"quality\", \"cloud\"]")

    # log seed
    augment_output.log_data["seed"] = seed

    # write image and log
    output_path = OUTPUT_PATH + "/" + image_name + "/"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    augment_output.output_image.save(output_path + "out.jpg")
    with open(output_path + "log.json", "w") as f:
        json.dump(augment_output.log_data, f, indent=4)

def test_augmentation_n_times(augmentation_type: str, image_name: str, n: int) -> None:
    test_output_path = OUTPUT_PATH + "/" + image_name + "_tests_" + augmentation_type + "/"
    if not os.path.isdir(test_output_path):
        os.makedirs(test_output_path)
    for i in range(n):
        seed = random.randrange(0, 2**32)
        perform_augmentation(augmentation_type, image_name, seed)
        res_image = Image.open(OUTPUT_PATH + "/" + image_name + "/out.jpg")
        res_image.save(test_output_path + f"out_{i}.jpg")
        log = json.load(open(OUTPUT_PATH + "/" + image_name + "/log.json"))
        json.dump(log, open(test_output_path + f"log_{i}.json", "w"), indent=4)

if __name__ == "__main__":
    augmentation_type = "quality"
    image_name = "1"
    seed = random.randrange(0, 2**32)
    # perform_augmentation(augmentation_type, image_name, seed)
    test_augmentation_n_times(augmentation_type, image_name, 10)
