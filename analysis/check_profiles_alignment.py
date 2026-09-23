from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
for p in sorted(Path('output/prototype2/debug').glob('*overlap_before.jpg')):
 im=cv2.imread(str(p));w=im.shape[1]//2; h=im.shape[0]
 a,b=[cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float32) for x in [im[:,:w],im[:,w:]]]
 out=[];cap=80
 for x in np.linspace(.05,.95,13)*w:
  xx=int(x)
  ss=[]
  for g in [a,b]:
   s=gaussian_filter1d(np.mean(g[:,max(0,xx-25):min(w,xx+25)],axis=1),2)
   ss.append(s-gaussian_filter1d(s,25))
  scores=[]
  for dy in range(-cap,cap+1):
   sa=ss[0][cap:h-cap];sb=ss[1][cap+dy:h-cap+dy]
   scores.append(float(np.dot(sa,sb)/max(np.linalg.norm(sa)*np.linalg.norm(sb),1e-6)))
  best=np.argmax(scores);out.append([round(x/w,2),best-cap,round(scores[best],2)])
 print(p.name,out)
