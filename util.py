"""Utility functions."""
from pathlib import Path
from shutil import which
from typing import Protocol, Iterable, Any
from PIL import Image, ImageOps
from config import CONV_GREED_EXE_NAME, PMC_EST_EXE_NAME, PMC_GREED_EXE_NAME


class WriterProtocol(Protocol):
    def writerow(self, row: Iterable[Any]) -> Any:
        ...


def executable_check() -> None:
    """
    Check that required executables exist.

    The `and` conditional is used to account for both Windows and Linux.

    Raises an error if any expected program is not found.
    """
    if (which(f"{CONV_GREED_EXE_NAME}") is None) and (
        which(f"./{CONV_GREED_EXE_NAME}") is None
    ):
        raise ValueError("conv_greed program not found in PATH")
    if (which(f"{PMC_EST_EXE_NAME}") is None) and (
        which(f"./{PMC_EST_EXE_NAME}") is None
    ):
        raise ValueError("conv_est program not found in PATH")
    if (which(f"{PMC_GREED_EXE_NAME}") is None) and (
        which(f"./{PMC_GREED_EXE_NAME}") is None
    ):
        raise ValueError("pmc_greed program not found in PATH")


def trim(img_name: Path | str) -> None:
    """
    Trim and save image.

    Idea to invert from https://stackoverflow.com/a/57552536

    Removes white and black borders.
    """
    image = Image.open(img_name)
    black_crop = image.crop(image.getbbox())
    inverted_image = ImageOps.invert(black_crop.convert("RGB"))
    final_crop = image.crop(inverted_image.getbbox())
    final_crop.save(img_name)
