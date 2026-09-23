import argparse
from pathlib import Path
from .pipeline import Settings,unwrap,natural_key

p=argparse.ArgumentParser(description='Rubber track surface panorama')
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('output/result'))
p.add_argument('--width-mm',type=float,required=True)
p.add_argument('--pitch-mm',type=float,required=True)
p.add_argument('--total-links',type=int,default=None)
p.add_argument('--pixels-per-mm',type=float,default=5)
a=p.parse_args()
paths=sorted((x for x in a.input.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}),key=natural_key)
unwrap(paths,a.output,Settings(a.width_mm,a.pitch_mm,a.total_links,a.pixels_per_mm))
