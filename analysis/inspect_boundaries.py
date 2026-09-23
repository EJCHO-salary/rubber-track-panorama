from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from track_unwrap.geometry import read_image
for p in [Path('ref_img/11.jpg'),Path('ref_img/case2/20250807_091633.jpg')]:
 im=read_image(p)
 if im.shape[1]>im.shape[0]:im=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
 im=cv2.resize(im,(1200,1600));f=cv2.GaussianBlur(im.astype(np.float32),(0,0),2)
 edge=np.linalg.norm(f[:,8:]-f[:,:-8],axis=2);edge=np.pad(edge,((0,0),(4,4)),mode='edge')
 profile=gaussian_filter1d(np.mean(np.sort(edge[:1300],axis=0)[260:1100],axis=0),3)
 peaks,_=find_peaks(profile,distance=20)
 print(p.name,sorted([(int(x),round(float(profile[x]),1)) for x in peaks],key=lambda x:-x[1])[:20])
