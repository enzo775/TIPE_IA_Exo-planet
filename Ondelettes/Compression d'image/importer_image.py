from PIL import Image


def charger_image(image, gray=True):
    """
    Nécessite l'extension.
    """
    pre = "./Ondelettes/Images/"
    mode = "L" if gray else "RGB"
    try:
        return Image.open(pre + image).convert(mode)
    except:
        raise FileNotFoundError