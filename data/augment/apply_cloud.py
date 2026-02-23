from PIL import Image
import random, numpy, math

from augment_output import AugmentOutput

# slightly modified from https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py
def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = numpy.empty((mapsize, mapsize), dtype=numpy.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * numpy.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + numpy.roll(cornerref, shift=-1, axis=0)
        squareaccum += numpy.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + numpy.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + numpy.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + numpy.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + numpy.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def fog(x, severity0, severity1):
    # c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]
    c = (severity0, severity1)

    x = numpy.array(x) / 255.
    max_val = x.max()
    # x += c[0] * plasma_fractal(wibbledecay=c[1])[:32, :32][..., numpy.newaxis]
    mapsize = 2 ** math.ceil(math.log2(max(x.shape[0], x.shape[1])))
    width = x.shape[1]
    height = x.shape[0]
    x += c[0] * plasma_fractal(wibbledecay=c[1], mapsize=mapsize)[:height, :width][..., numpy.newaxis]
    return numpy.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def apply_cloud(input_image: Image, seed: int) -> AugmentOutput:
    # seed
    random.seed(seed)
    numpy.random.seed(seed)

    # get random values
    severity0 = random.uniform(0.4, 1.5)
    severity1 = random.uniform(1.25, 1.5)

    # thing
    numpy_input_image = numpy.array(input_image)
    # print(fog(numpy_input_image, severity0, severity1))
    cloud_image_array = fog(numpy_input_image, severity0, severity1).astype(numpy.uint8)
    cloud_image = Image.fromarray(cloud_image_array).convert("RGB")

    return AugmentOutput(
        output_image = cloud_image,
        log_data = {
            "augmentation": "cloud_overlay",
            "version": "1.1",
            # "lerpAmount": lerpAmount,
            "severity0": severity0,
            "severity1": severity1
        }
    )
