from PIL import Image
import random, numpy, json, os

from lib.motion_blur import Kernel
from augment_output import AugmentOutput

INPUT_PATH = "./augment_pipeline/input_images"
OUTPUT_PATH = "./augment_pipeline/output"

def apply_blur(input_image: Image, seed: int) -> AugmentOutput:
    # seed
    random.seed(seed)
    numpy.random.seed(seed) # for motion_blur.Kernel

    # get random values
    kernel_size = random.randint(15, 45)
    intensity = random.uniform(0, 0.7)

    # apply augmentation
    kernel = Kernel(size = (kernel_size, kernel_size), intensity = intensity)
    output_image = kernel.applyTo(input_image, keep_image_dim = True)

    # return
    return AugmentOutput(
        output_image = output_image,
        log_data = {
            "augmentation": "motion_blur",
            "version": "1",
            "kernel_size": kernel_size,
            "intensity": intensity,
        }
    )

if __name__ == "__main__":
    seed = random.randrange(0, 2**32)

    name = "1"
    path = INPUT_PATH + "/" + name + ".jpg"
    input_image = Image.open(path)

    augment_output = apply_blur(input_image, seed)
    augment_output.log_data["seed"] = seed

    output_path = OUTPUT_PATH + "/" + name + "/"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    augment_output.output_image.save(output_path + "out.jpg")
    with open(output_path + "log.json", "w") as f:
        json.dump(augment_output.log_data, f, indent=4)
