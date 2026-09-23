from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track_unwrap.geometry import analyze
from track_unwrap.matching import estimate_pitch_shift
frames=[analyze(p,450,86) for p in sorted(Path('ref_img').glob('*.jpg'))]
rects=[f.render(1000,1000*86/450) for f in frames]
for i in range(len(frames)-1):
 a,sa=rects[i]; b,sb=rects[i+1]
 shift,report=estimate_pitch_shift(a,b,sa,sb,1000*86/450)
 print(frames[i].name,frames[i+1].name,{k:v for k,v in report.items() if 'points' not in k},flush=True)
