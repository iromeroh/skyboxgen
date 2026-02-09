from PIL import Image

from skyboxgen.background import destar_background


def test_destar_background_dimensions():
    img = Image.new("RGB", (64, 32), color=(10, 10, 10))
    out = destar_background(img, percentile=90.0, blur_radius=1.0, mask_expand=1)
    assert out.size == img.size
