"""Generate readable review crops from the actual output, never source retouching."""
from pathlib import Path
import json
import cv2
import numpy as np

for folder in [Path('output/result'),Path('output/case2')]:
    report=json.loads((folder/'quality_report.json').read_text(encoding='utf-8'))
    im=cv2.imread(str(folder/'panorama.png'))
    bands=[]
    for i,part in enumerate(np.array_split(im,4,axis=1)):
        thumb=cv2.resize(part,(1600,round(part.shape[0]*1600/part.shape[1])),interpolation=cv2.INTER_AREA)
        label=np.full((36,1600,3),246,np.uint8)
        cv2.putText(label,f'Section {i+1} / 4',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(40,55,45),1,cv2.LINE_AA)
        bands.extend([label,thumb,np.full((16,1600,3),246,np.uint8)])
    cv2.imwrite(str(folder/'review.jpg'),np.concatenate(bands,axis=0),[cv2.IMWRITE_JPEG_QUALITY,95])
    vertical=cv2.imread(str(folder/'panorama_vertical.jpg'))
    tiles=[]
    for i,seam in enumerate(report['seams']):
        yy=np.median(seam['seam_y_by_x'])-report['crop_start_row']
        half=report['nominal_pitch_px']*.8
        if yy<0 or yy>=len(vertical):continue
        lo=max(0,int(yy-half));hi=min(len(vertical),int(yy+half))
        crop=vertical[lo:hi]
        crop=cv2.resize(crop,(700,300),interpolation=cv2.INTER_AREA)
        title=np.full((32,700,3),245,np.uint8)
        cv2.putText(title,f'Seam {i+1}: {seam["source"]}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1,cv2.LINE_AA)
        tiles.append(np.concatenate([title,crop]))
    for page,start in enumerate(range(0,len(tiles),6)):
        subset=tiles[start:start+6]
        while len(subset)%2:subset.append(np.full_like(subset[0],245))
        contact=np.concatenate([np.concatenate(subset[k:k+2],axis=1) for k in range(0,len(subset),2)])
        cv2.imwrite(str(folder/f'seams_review_{page+1}.jpg'),contact,[cv2.IMWRITE_JPEG_QUALITY,95])
    print(folder,report['output_size_wh'],report['observed_span_pitches'],report['verified_loop_links'])
