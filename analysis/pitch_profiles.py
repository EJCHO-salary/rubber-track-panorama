from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track_unwrap.geometry import analyze
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
for p in sorted(Path('ref_img').glob('*.jpg')):
 f=analyze(p,450,86)
 small=cv2.resize(f.image,(1200,1600))
 l=f.left[::int(1/f.analysis_scale)] if False else np.interp(np.arange(1600)/f.analysis_scale,np.arange(len(f.left)),f.left)*f.analysis_scale
 r=np.interp(np.arange(1600)/f.analysis_scale,np.arange(len(f.right)),f.right)*f.analysis_scale
 mx=(l[:,None]+(r-l)[:,None]*np.linspace(0,1,1000)).astype(np.float32)
 my=np.broadcast_to(np.arange(1600,dtype=np.float32)[:,None],mx.shape).copy()
 g=cv2.cvtColor(cv2.remap(small,mx,my,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
 print(p.name)
 for a,b in [(450,480),(510,540),(530,560),(540,570)]:
  s=gaussian_filter1d(g[:,a:b].mean(axis=1),5)
  contrast=gaussian_filter1d(s,70)-s
  peaks,_=find_peaks(contrast,distance=150,prominence=5)
  print(a,b,'dark',peaks.tolist())
