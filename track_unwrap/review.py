"""Readable sections of one completed panorama; no changes to image content."""
import cv2
import numpy as np


def save_review(image, path):
    bands=[]
    for i,part in enumerate(np.array_split(image,4,axis=1)):
        thumb=cv2.resize(part,(1600,round(part.shape[0]*1600/part.shape[1])),interpolation=cv2.INTER_AREA)
        label=np.full((36,1600,3),246,np.uint8)
        cv2.putText(label,f'Section {i+1} / 4',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(40,55,45),1,cv2.LINE_AA)
        bands.extend([label,thumb,np.full((16,1600,3),246,np.uint8)])
    cv2.imwrite(str(path),np.concatenate(bands,axis=0),[cv2.IMWRITE_JPEG_QUALITY,95])
