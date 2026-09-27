"""Build a Deep Zoom image pyramid for inspecting a long panorama."""

from math import ceil, log2
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

from PIL import Image


TILE_SIZE = 256


def save_deepzoom(image_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        width, height = source.size
        max_level = ceil(log2(max(width, height)))
        tile_root = output_dir / 'deepzoom_files'
        for level in range(max_level + 1):
            scale = 2 ** (max_level - level)
            size = (ceil(width / scale), ceil(height / scale))
            level_image = source if level == max_level else source.resize(size, Image.Resampling.LANCZOS)
            folder = tile_root / str(level)
            folder.mkdir(parents=True, exist_ok=True)
            for row, top in enumerate(range(0, size[1], TILE_SIZE)):
                for col, left in enumerate(range(0, size[0], TILE_SIZE)):
                    tile = level_image.crop((left, top, min(left + TILE_SIZE, size[0]), min(top + TILE_SIZE, size[1])))
                    tile.save(folder / f'{col}_{row}.jpg', 'JPEG', quality=88, optimize=True)
            if level_image is not source:
                level_image.close()

    xmlns = 'http://schemas.microsoft.com/deepzoom/2008'
    descriptor = Element('Image', {'TileSize': str(TILE_SIZE), 'Overlap': '0', 'Format': 'jpg', 'xmlns': xmlns})
    SubElement(descriptor, 'Size', {'Width': str(width), 'Height': str(height)})
    ElementTree(descriptor).write(output_dir / 'deepzoom.dzi', encoding='utf-8', xml_declaration=True)
