from typing import NamedTuple
from PIL import Image

# namedtuple or dataclass?
class AugmentOutput(NamedTuple):
    output_image: Image
    log_data: dict
