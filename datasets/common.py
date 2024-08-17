import os


class CULaneDIR:
    raw_images_dir = "raw_images"
    images_dir = "images"
    labels_dir = "labels"
    list_dir = "list"


SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png']
COLOR_MAP = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}


def get_culane_root():
    culane_root = os.getenv('CULANEROOT')
    if culane_root is None:
        raise EnvironmentError("\033[01;31mPlease set 'CULANEROOT' environment variable\033[0m")
    else:
        return culane_root
