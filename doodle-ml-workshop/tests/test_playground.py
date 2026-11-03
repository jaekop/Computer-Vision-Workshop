import numpy as np

from src.playground import sketch_to_pixels


def test_blank_canvas():
    assert np.count_nonzero(sketch_to_pixels(None)) == 0


def test_black_stroke_on_white():
    image = np.full((56, 56, 4), 255, dtype=np.uint8)
    image[20:36, 20:36, :3] = 0
    output = sketch_to_pixels({"composite": image})
    assert output.shape == (1, 784)
    assert output.max() > 0
    assert output.min() == 0

