from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from track_unwrap.geometry import analyze

out = Path('output/debug')
out.mkdir(parents=True, exist_ok=True)
for p in sorted(Path('ref_img').glob('*.jpg')):
    f = analyze(p, 450, 86)
    cv2.imwrite(str(out / (p.stem + '_geometry.jpg')), f.debug)
    rect, start = f.render(900, 900*86/450)
    cv2.imwrite(str(out / (p.stem + '_rect.jpg')), rect)
    print(p.name, 'anchors', f.anchors.round(1).tolist(), 'width', (f.right-f.left).min(), (f.right-f.left).max(), 'range', f.pitch_range.tolist(), flush=True)
