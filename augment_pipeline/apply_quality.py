from PIL import Image
import random, json, os, numpy, io

from augment_output import AugmentOutput

INPUT_PATH = "./augment_pipeline/input_images"
OUTPUT_PATH = "./augment_pipeline/output"

def downscale_upscale(image: Image, scale_factor: float) -> Image:
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    downscaled_image = image.resize(new_size, resample=Image.BICUBIC)
    upscaled_image = downscaled_image.resize((image.width, image.height), resample=Image.BICUBIC)
    return upscaled_image

def gaussian_noise(image: Image, sigma: float) -> Image:
    array = numpy.array(image).astype(numpy.float32)
    noise = numpy.random.normal(0, sigma, array.shape).astype(numpy.float32)
    noisy_array = array + noise
    noisy_array = numpy.clip(noisy_array, 0, 255).astype(numpy.uint8)
    noisy_image = Image.fromarray(noisy_array)
    return noisy_image

# note that higher quality value means less compression, from 95 to 1
def jpeg_compress(image: Image, quality: int) -> Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    compressed_image.load()
    return compressed_image

def apply_quality(input_image: Image, seed: int) -> AugmentOutput:
    # seed
    random.seed(seed)
    numpy.random.seed(seed)

    # get random values
    scale_factor = random.uniform(0.08, 0.15)
    noise_sigma = random.uniform(12, 25)
    jpeg_severity = random.randint(7, 15)

    # apply augmentation
    # output_image = corrupt(numpy.array(input_image), corruption_name="jpeg_compression", severity=jpeg_severity)
    output_image = downscale_upscale(input_image, scale_factor=scale_factor)
    output_image = gaussian_noise(output_image, sigma=noise_sigma)
    output_image = jpeg_compress(output_image, quality=jpeg_severity)

    # return
    return AugmentOutput(
        output_image = output_image,
        log_data = {
            "augmentation": "quality_degradation",
            "version": "1",
            "scale_factor": scale_factor,
            "noise_sd": noise_sigma,
            "jpeg_severity": jpeg_severity
        }
    )

if __name__ == "__main__":
    seed = random.randrange(0, 2**32)

    name = "1"
    path = INPUT_PATH + "/" + name + ".jpg"
    input_image = Image.open(path)

    augment_output = apply_quality(input_image, seed)
    augment_output.log_data["seed"] = seed

    output_path = OUTPUT_PATH + "/" + name + "/"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    augment_output.output_image.save(output_path + "out.jpg")
    with open(output_path + "log.json", "w") as f:
        json.dump(augment_output.log_data, f, indent=4)
