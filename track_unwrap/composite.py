"""Content-aware seams and multiband composition with preserved defect geometry."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def illumination_profile(images,pitch_px):
    profiles=[]
    for im in images:
        low=low_frequency(im,pitch_px)
        profiles.append(np.median(low,axis=0))
    return np.median(profiles,axis=0)


def normalize_illumination(im,profile,pitch_px):
    low=low_frequency(im,pitch_px)
    target=cv2.resize(profile[None,:,:],(im.shape[1],1),interpolation=cv2.INTER_LINEAR)
    gain=np.clip((target+10)/(low+10),.65,1.55)**.9
    return np.clip(im.astype(np.float32)*gain,0,255).astype(np.uint8)


def low_frequency(im,pitch_px):
    scale=min(1.,320/im.shape[1])
    small=cv2.resize(im.astype(np.float32),(round(im.shape[1]*scale),round(im.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    small=cv2.GaussianBlur(small,(0,0),im.shape[1]*.04*scale,sigmaY=pitch_px*.8*scale)
    return cv2.resize(small,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR)


def minimum_seam(a,b,pitch_px):
    h,w = a.shape[:2]
    factor = min(1.,700/w)
    size = (round(w*factor),round(h*factor))
    aa,bb = [cv2.resize(im,size,interpolation=cv2.INTER_AREA).astype(np.float32) for im in (a,b)]
    ga,gb = [cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32) for im in (aa,bb)]
    edges=[]
    for g in (ga,gb):
        smooth=cv2.GaussianBlur(g,(0,0),2)
        mag=cv2.magnitude(cv2.Sobel(smooth,cv2.CV_32F,1,0),cv2.Sobel(smooth,cv2.CV_32F,0,1))
        edges.append(cv2.dilate(mag,np.ones((7,7),np.uint8)))
    diff = np.mean(np.abs(aa-bb),axis=2)
    # Avoid cutting prominent cracks and ridge edges, and avoid overlap ends.
    cost = cv2.GaussianBlur(diff,(0,0),1) + 1.5*(edges[0]+edges[1])
    hh,ww=cost.shape
    border=max(2,round(min(pitch_px*.22,h*.15)*factor))
    cost[:border]=1e5;cost[-border:]=1e5
    cost += 5*(np.linspace(-1,1,hh)[:,None]**2)
    dp=cost[:,0].copy()
    back=np.zeros((hh,ww),np.int8)
    jumps=np.arange(-2,3)
    for x in range(1,ww):
        choices=np.stack([np.roll(dp,int(j))+.9*abs(j) for j in jumps])
        for n,j in enumerate(jumps):
            if j>0:choices[n,:j]=1e9
            if j<0:choices[n,j:]=1e9
        best=choices.argmin(axis=0)
        back[:,x]=jumps[best]
        dp=cost[:,x]+choices[best,np.arange(hh)]
    yy=int(dp.argmin()); path=np.empty(ww,np.float32)
    for x in range(ww-1,-1,-1):
        path[x]=yy;yy-=int(back[yy,x])
    path=np.interp(np.linspace(0,ww-1,w),np.arange(ww),path)/factor
    return gaussian_filter1d(path,2)


def blend_overlap(a,b,seam):
    h,w=a.shape[:2]
    mask_a=(np.arange(h)[:,None]<=seam[None,:]).astype(np.uint8)*255
    mask_b=255-mask_a
    blender=cv2.detail_MultiBandBlender()
    blender.setNumBands(6)
    blender.prepare((0,0,w,h))
    blender.feed(a.astype(np.int16),mask_a,(0,0))
    blender.feed(b.astype(np.int16),mask_b,(0,0))
    result,_=blender.blend(None,None)
    return np.clip(result,0,255).astype(np.uint8)


def merge(canvas, moving, start_row, pitch_px,debug_path=None):
    overlap=min(canvas.shape[0]-start_row,moving.shape[0])
    if overlap < pitch_px*.6:
        raise ValueError("Overlap is insufficient for a verified seam.")
    reference=canvas[start_row:start_row+overlap]
    # Robust per-channel gain from the same physical surface, kept modest.
    ratios=(reference.astype(np.float32)+20)/(moving[:overlap].astype(np.float32)+20)
    gain=np.median(ratios.reshape(-1,3),axis=0).clip(.94,1.06)
    moving=np.clip(moving.astype(np.float32)*gain,0,255).astype(np.uint8)
    if debug_path is not None:
        cv2.imwrite(str(debug_path),np.concatenate([reference,moving[:overlap]],axis=1))
    # Keep defect geometry intact; choose a seam around parallax instead of
    # forcing incompatible raised/deep surfaces to match with a dense warp.
    info={"local_warp":"disabled_to_preserve_defect_shape"}
    seam=minimum_seam(reference,moving[:overlap],pitch_px)
    joined=blend_overlap(reference,moving[:overlap],seam)
    total=max(canvas.shape[0],start_row+moving.shape[0])
    result=np.zeros((total,canvas.shape[1],3),np.uint8)
    result[:len(canvas)]=canvas
    result[start_row:start_row+overlap]=joined
    result[start_row+overlap:start_row+len(moving)]=moving[overlap:]
    info.update({"start_row":start_row,"overlap_rows":overlap,"gain_bgr":gain.tolist(),
                 "seam_y_by_x":(seam+start_row).round(2).tolist()})
    return result,info
