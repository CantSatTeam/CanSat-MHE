from lib.motion_blur import Kernel
from PIL import Image
import random

INPUT_PATH = "./augment_pipeline/input_images"
OUTPUT_PATH = "./augment_pipeline/output"

def apply_blur(name: str):
    kernel_size = random.randint(15, 45)
    intensity = random.uniform(0, 0.7)

    path = INPUT_PATH + "/" + name + ".jpg"
    input_image = Image.open(path)
    kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
    output_image = kernel.applyTo(input_image, keep_image_dim=True)

    output_path = OUTPUT_PATH + "/" + name + ".jpg"
    output_image.save(output_path)
    return output_image

if __name__ == "__main__":
    apply_blur("1")