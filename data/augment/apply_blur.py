from PIL import Image
import random, numpy

from lib.motion_blur import Kernel
from augment_output import AugmentOutput

def apply_blur(input_image: Image, seed: int) -> AugmentOutput:
    # seed
    random.seed(seed)
    numpy.random.seed(seed) # for motion_blur.Kernel

    # get random values
    kernel_size = random.randint(12, 17)
    intensity = random.uniform(0, 0.4)

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
