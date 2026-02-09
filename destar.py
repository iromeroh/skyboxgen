import argparse

import numpy as np
from photutils.background import Background2D, MedianBackground
from PIL import Image


def destar_image(input_path: str, output_path: str, box_size: int, filter_size: int) -> None:
    # Estimate a smooth diffuse background by removing point-like star sources.
    img = Image.open(input_path).convert("RGB")
    data = np.asarray(img, dtype=float)
    background = np.zeros_like(data)

    for i in range(3):
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            data[:, :, i],
            box_size=box_size,
            filter_size=(filter_size, filter_size),
            bkg_estimator=bkg_estimator,
        )
        background[:, :, i] = bkg.background

    background = np.clip(background, 0, 255).astype(np.uint8)
    Image.fromarray(background, mode="RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="De-star an equirectangular 2MASS image")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--box-size", type=int, default=50)
    parser.add_argument("--filter-size", type=int, default=3)
    args = parser.parse_args()

    destar_image(args.input, args.output, args.box_size, args.filter_size)
    print(f"Diffuse background saved as '{args.output}'")


if __name__ == "__main__":
    main()
