from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter1d,median_filter
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator


class UnwrapError(ValueError):
    """An input cannot be reconstructed reliably."""


def read_image(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def continuous_edge(score, lo, hi):
    """Maximum edge evidence with a smooth row-to-row path."""
    band = score[:, lo:hi].astype(np.float32)
    band /= max(float(np.percentile(band, 95)), 1)
    costs = -np.minimum(band, 3)
    costs += .55*np.linspace(-1,1,band.shape[1],dtype=np.float32)[None,:]**2
    h, w = costs.shape
    back = np.zeros((h, w), np.int8)
    best = costs[0].copy()
    jumps = np.arange(-3, 4)
    for y in range(1, h):
        choices = np.stack([np.roll(best, int(j)) + .16 * abs(j) for j in jumps])
        for n, j in enumerate(jumps):
            if j > 0:
                choices[n, :j] = 1e6
            elif j < 0:
                choices[n, j:] = 1e6
        arg = choices.argmin(axis=0)
        back[y] = jumps[arg]
        best = costs[y] + choices[arg, np.arange(w)]
    path = np.empty(h, np.float32)
    x = int(best.argmin())
    for y in range(h - 1, -1, -1):
        path[y] = x + lo
        x -= int(back[y, x])
    return gaussian_filter1d(path, 9)


def nominal_boundary(path, side):
    """Fit the outer envelope, ignoring local tread/edge shadows."""
    chunks = np.array_split(np.arange(len(path)), 12)
    yy = np.array([a.mean() for a in chunks])
    xx = np.array([np.median(path[a]) for a in chunks])
    keep=np.ones(len(xx),bool)
    for _ in range(3):
        poly=np.polyfit(yy[keep],xx[keep],1)
        residual=xx-np.polyval(poly,yy)
        keep=np.abs(residual)<max(4,2.5*np.median(np.abs(residual)))
    return np.polyval(poly,np.arange(len(path))).astype(np.float32)


def monotone_map(x, y, query):
    q = np.asarray(query)
    result = PchipInterpolator(x, y, extrapolate=True)(np.clip(q, x[0], x[-1]))
    result = np.where(q < x[0], y[0] + (q - x[0]) * (y[1]-y[0])/(x[1]-x[0]), result)
    return np.where(q > x[-1], y[-1] + (q-x[-1]) * (y[-1]-y[-2])/(x[-1]-x[-2]), result)


def refine_foreground_edges(small,left,right,valid_end):
    """Refine a coarse strip using foreground/background color and graph cuts."""
    h,w=small.shape[:2]
    mask=np.zeros((valid_end+1,w),np.uint8)
    for y in range(valid_end+1):
        span=right[y]-left[y]
        lo=max(0,int(left[y]-span*.10));hi=min(w,int(right[y]+span*.10))
        mask[y,lo:hi]=cv2.GC_PR_BGD
        mask[y,max(0,int(left[y])):min(w,int(right[y])+1)]=cv2.GC_PR_FGD
        mask[y,int(left[y]+span*.10):int(right[y]-span*.10)]=cv2.GC_FGD
    cv2.grabCut(small[:valid_end+1],mask,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_MASK)
    foreground=(mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)
    ls=left.copy();rs=right.copy()
    for y in range(valid_end+1):
        center=int((left[y]+right[y])/2)
        # Select the connected row interval containing the definite core.
        left_bg=np.flatnonzero(~foreground[y,:center])
        right_bg=np.flatnonzero(~foreground[y,center:])
        if len(left_bg):ls[y]=left_bg[-1]+1
        if len(right_bg):rs[y]=center+right_bg[0]-1
    ls[:valid_end+1]=gaussian_filter1d(median_filter(ls[:valid_end+1],size=7),2)
    rs[:valid_end+1]=gaussian_filter1d(median_filter(rs[:valid_end+1],size=7),2)
    ls[valid_end+1:]=ls[valid_end];rs[valid_end+1:]=rs[valid_end]
    return ls,rs


@dataclass
class Frame:
    name: str
    image: np.ndarray
    left: np.ndarray
    right: np.ndarray
    anchors: np.ndarray
    center_fraction: float
    period_source_px: float
    analysis_scale: float
    debug: np.ndarray
    origin_pitch: float = 0.0
    input_rotation: int = -1
    valid_y_start: float = 0.0
    valid_y_end: float | None = None
    center_x: np.ndarray | None = None
    mesh_pitch_offsets: np.ndarray | None = None

    @property
    def pitch_range(self):
        y = np.array([self.valid_y_start, self.image.shape[0] - 1. if self.valid_y_end is None else self.valid_y_end])
        bounds=self.pitch_at(y)
        if self.mesh_pitch_offsets is not None:
            low=[np.interp(bounds[0],np.arange(len(self.anchors)),self.mesh_pitch_offsets[:,c]) for c in [0,1]]
            high=[np.interp(bounds[1],np.arange(len(self.anchors)),self.mesh_pitch_offsets[:,c]) for c in [0,1]]
            bounds=bounds+np.array([max(0,*low),min(0,*high)])
        return bounds

    def pitch_at(self, y):
        return monotone_map(self.anchors, np.arange(len(self.anchors)), y)

    def source_y(self, pitch):
        return monotone_map(np.arange(len(self.anchors)), self.anchors, pitch)

    def render(self, width, pitch_px):
        low, high = self.pitch_range
        start = int(np.ceil(low * pitch_px))
        end = int(np.floor(high * pitch_px))
        positions = np.arange(start, end + 1) / pitch_px
        u = np.linspace(0, 1, width, dtype=np.float32)
        if self.mesh_pitch_offsets is None:
            ys = self.source_y(positions).astype(np.float32)
            my=np.broadcast_to(ys[:,None],(len(ys),width)).copy()
        else:
            my=np.empty((len(positions),width),np.float32)
            for col,fraction in enumerate(u):
                edge=0 if fraction<.5 else 1
                weight=abs(fraction-.5)*2
                controls=np.arange(len(self.anchors))+self.mesh_pitch_offsets[:,edge]*weight
                my[:,col]=monotone_map(controls,self.anchors,positions)
        source_rows=np.arange(len(self.left))
        left=np.interp(my,source_rows,self.left)
        right=np.interp(my,source_rows,self.right)
        center=np.interp(my,source_rows,self.center_x) if self.center_x is not None else (left+right)/2
        mx=np.where(u[None,:]<.5,left+(center-left)*(u[None,:]*2),center+(right-center)*((u[None,:]-.5)*2)).astype(np.float32)
        rectified = cv2.remap(self.image, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rectified, start / pitch_px


def analyze(path, width_mm, pitch_mm):
    image = read_image(path)
    rotation=-1
    if image.shape[1]>image.shape[0]:
        image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
        rotation=int(cv2.ROTATE_90_CLOCKWISE)
    oh, ow = image.shape[:2]
    scale = 1600 / oh
    small = cv2.resize(image, (round(ow * scale), 1600), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    smooth = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 2)
    edge = np.linalg.norm(smooth[:, 8:] - smooth[:, :-8], axis=2)
    edge = np.pad(edge, ((0, 0), (4, 4)), mode="edge")
    sample=np.sort(edge[:int(h*.82)],axis=0)
    profile=gaussian_filter1d(np.mean(sample[int(len(sample)*.20):int(len(sample)*.85)],axis=0),3)
    edge_peaks,_=find_peaks(profile,distance=max(5,round(w*.015)))
    left_candidates=edge_peaks[(edge_peaks>w*.025)&(edge_peaks<w*.44)]
    right_candidates=edge_peaks[(edge_peaks>w*.60)&(edge_peaks<w*.99)]
    if not len(left_candidates) or not len(right_candidates):
        raise UnwrapError(f"{Path(path).name}: could not locate both track borders.")
    lc=int(left_candidates[np.argmax(profile[left_candidates])])
    rc=int(right_candidates[np.argmax(profile[right_candidates])])
    margin=round(w*(.025 if rc-lc<w*.65 else .045))
    left_raw=continuous_edge(edge,max(0,lc-margin),min(w,lc+margin))
    right_raw=continuous_edge(edge,max(0,rc-margin),min(w,rc+margin))
    baseline=float(np.median((right_raw-left_raw)[int(h*.10):int(h*.65)]))
    valid_end=h-1
    if baseline < w*.65:
        convergence=gaussian_filter1d(right_raw-left_raw,12)<baseline*.97
        tail=np.flatnonzero(convergence & (np.arange(h)>h*.72))
        if len(tail)>25:
            valid_end=max(int(h*.72),int(tail[0]-baseline*pitch_mm/width_mm*.2))
        else:
            valid_end=int(h*.87)
    def fit_boundary(path,side):
        fit=nominal_boundary(path[:valid_end+1],side)
        return np.pad(fit,(0,h-len(fit)),mode='edge')
    left = fit_boundary(left_raw,"left")
    right = fit_boundary(right_raw,"right")
    if baseline < w*.65:
        left,right=refine_foreground_edges(small,left,right,valid_end)
    # A tiny inward offset avoids mixing background into cubic interpolation.
    left += 2
    right -= 2
    widths = right - left
    if np.min(widths) < w * .18:
        raise UnwrapError(f"{Path(path).name}: track boundaries are uncertain.")
    rw = 1000
    mx = (left[:, None] + widths[:, None] * np.linspace(0, 1, rw)).astype(np.float32)
    my = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], mx.shape).copy()
    rectified = cv2.remap(small, mx, my, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY).astype(np.float32)
    expected = float(np.median(widths) * pitch_mm / width_mm)
    edge_x=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0))
    edge_x=cv2.GaussianBlur(edge_x,(0,0),3,sigmaY=expected*.6)
    central_left=np.argmax(edge_x[:,390:495],axis=1)+390
    central_right=np.argmax(edge_x[:,505:620],axis=1)+505
    centers=gaussian_filter1d((central_left+central_right)/2,expected*.6)
    center_source=left+widths*centers/rw
    cx = 545
    signal = gaussian_filter1d(gray[:, 530:560].mean(axis=1), 5)
    contrast = gaussian_filter1d(signal, expected * .35) - signal
    peaks, props = find_peaks(contrast, distance=max(12, expected * .70), prominence=5)
    # Edge-truncated structures are not used as control points.
    peaks = peaks[(peaks > expected * .10) & (peaks < valid_end - expected * .10)]
    if len(peaks) < 3:
        raise UnwrapError(f"{Path(path).name}: fewer than three reliable pitch anchors.")
    spacings = np.diff(peaks)
    period = float(np.median(spacings))
    # A weak/occluded recess can have no prominent minimum. Keep the lattice
    # continuous using the best local evidence near the missing repeat.
    repaired=[]
    for a,b in zip(peaks[:-1],peaks[1:]):
        repaired.append(float(a))
        steps=int(round((b-a)/period))
        if steps>=2 and abs((b-a)/period-steps)<.35:
            for k in range(1,steps):
                predicted=a+(b-a)*k/steps
                lo=max(int(a+period*.5),int(predicted-period*.18))
                hi=min(int(b-period*.5),int(predicted+period*.18))
                if hi>lo:
                    repaired.append(float(lo+np.argmax(contrast[lo:hi])))
    repaired.append(float(peaks[-1]))
    peaks=np.array(repaired)
    spacings=np.diff(peaks)
    if len(peaks) > 3 and not .83 < spacings[0] / period < 1.17:
        peaks = peaks[1:]
    if len(peaks) > 3 and not .83 < np.diff(peaks)[-1] / period < 1.17:
        peaks = peaks[:-1]
    spacings = np.diff(peaks)
    period = float(np.median(spacings))
    if not .65 * expected < period < 1.45 * expected:
        raise UnwrapError(f"{Path(path).name}: detected repeat is inconsistent with width/pitch.")
    if np.max(np.abs(spacings / period - 1)) > .30:
        raise UnwrapError(f"{Path(path).name}: missing or ambiguous pitch anchor; review needed. Anchors: {peaks.tolist()}; period {period:.1f}")
    phase=peaks-np.arange(len(peaks))*period
    peaks=.55*peaks+.45*(gaussian_filter1d(phase,.8)+np.arange(len(peaks))*period)
    debug = small.copy()
    for points, color in [(left, (0, 255, 100)), (right, (0, 255, 100))]:
        cv2.polylines(debug, [np.stack([points, np.arange(h)], axis=1).astype(np.int32)], False, color, 2)
    for i, y in enumerate(peaks):
        x = int(left[int(y)] + widths[int(y)] * cx / rw)
        cv2.circle(debug, (x, int(y)), 10, (20, 80, 255), 3)
        cv2.putText(debug, str(i), (x + 14, int(y) + 7), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 80, 255), 2)
    frame=Frame(Path(path).name, image,
                 np.interp(np.arange(oh) * scale, np.arange(h), left) / scale,
                 np.interp(np.arange(oh) * scale, np.arange(h), right) / scale,
                 peaks.astype(float) / scale, cx / rw, period / scale, scale, debug,input_rotation=rotation)
    # A visible rounded end has converging borders. Omit its foreshortened
    # portion; overlapping frames supply that physical material in a flat view.
    frame.valid_y_end=valid_end/scale
    frame.center_x=np.interp(np.arange(oh)*scale,np.arange(h),center_source)/scale
    cv2.polylines(frame.debug,[np.stack([center_source,np.arange(h)],axis=1).astype(np.int32)],False,(255,200,20),2)
    if valid_end<h-1:
        cv2.line(frame.debug,(0,valid_end),(w-1,valid_end),(255,150,0),3)
    return frame
