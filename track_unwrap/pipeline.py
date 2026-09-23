import json
import hashlib
import re
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .geometry import analyze,UnwrapError
from .matching import estimate_pitch_shift
from .composite import merge,illumination_profile,normalize_illumination
from .front_view import refine_front_view
from .review import save_review


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)',str(path))]


@dataclass
class Settings:
    width_mm: float
    pitch_mm: float
    total_links: int | None = None
    pixels_per_mm: float = 5.0

    def validate(self):
        if not (np.isfinite(self.width_mm) and 10 <= self.width_mm <= 3000):
            raise ValueError("Track width must be between 10 and 3000 mm.")
        if not (np.isfinite(self.pitch_mm) and 5 <= self.pitch_mm <= self.width_mm):
            raise ValueError("Pitch must be positive and no larger than the track width.")
        if self.total_links is not None and (isinstance(self.total_links,bool) or not isinstance(self.total_links,int) or not 1 <= self.total_links <= 2000):
            raise ValueError("Total links must be omitted or a positive integer up to 2000.")
        if not (np.isfinite(self.pixels_per_mm) and .5 <= self.pixels_per_mm <= 8):
            raise ValueError("Output resolution must be between 0.5 and 8 px/mm.")


def unwrap(paths, output_dir, settings: Settings, progress: Callable = print):
    settings.validate()
    paths=[Path(p) for p in paths]
    if not 1 <= len(paths) <= 100:
        raise ValueError("Provide 1 to 100 ordered images.")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    debug=out/'debug';debug.mkdir(exist_ok=True)
    cv2.setRNGSeed(42);cv2.setNumThreads(4)
    frames=[]
    hashes=set()
    skipped=[]
    for p in paths:
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in hashes:
            skipped.append(p.name);continue
        hashes.add(digest)
        progress(f"Geometry: {p.name}")
        f=analyze(p,settings.width_mm,settings.pitch_mm)
        frames.append(f)
        cv2.imwrite(str(debug/(p.stem+'_geometry.jpg')),f.debug)
    work_width=1000;work_pitch=work_width*settings.pitch_mm/settings.width_mm
    rects=[f.render(work_width,work_pitch) for f in frames]
    tone_profile=illumination_profile([a for a,s in rects],work_pitch)
    pair_reports=[]
    signs=[]
    integer_origins=[0]
    for i in range(len(frames)-1):
        a,sa=rects[i];b,sb=rects[i+1]
        shift,info=estimate_pitch_shift(a,b,sa,sb,work_pitch)
        if shift:signs.append(np.sign(shift))
        frames[i+1].origin_pitch=frames[i].origin_pitch+shift+info['median_phase_error_pitch']
        integer_origins.append(integer_origins[-1]+shift)
        info.update({'source':frames[i].name,'target':frames[i+1].name})
        pair_reports.append(info)
        progress(f"Matched {frames[i].name} -> {frames[i+1].name}: {shift} pitches, {info['accepted_matches']} points")
    if len(set(signs))>1:
        raise UnwrapError("Photographs reverse direction or contain a repeated segment. Review their order.")
    loop_links=None;closing=None
    if len(frames)>2:
        try:
            a,sa=rects[0];b,sb=rects[-1]
            short_shift,closing=estimate_pitch_shift(a,b,sa,sb,work_pitch)
            circumference=abs(frames[-1].origin_pitch-(short_shift+closing['median_phase_error_pitch']))
            candidate=abs(integer_origins[-1]-short_shift)
            if candidate>=max(len(f.anchors) for f in frames) and abs(circumference-candidate)<.75 and closing['accepted_matches']>=50 and closing['x_span_fraction']>.5:
                if settings.total_links is not None and candidate!=settings.total_links:
                    raise UnwrapError(f"Verified loop has {candidate} links, but {settings.total_links} was supplied.")
                loop_links=candidate
                # Integer pitch identity supplies the circumference. Distribute
                # small accumulated phase drift while fixing the closure phase.
                closure_drift=(frames[-1].origin_pitch-integer_origins[-1])-closing['median_phase_error_pitch']
                for j,f in enumerate(frames):
                    f.origin_pitch-=closure_drift*j/(len(frames)-1)
                progress(f"Verified loop: {candidate} pitches; {closing['accepted_matches']} closing correspondences")
        except UnwrapError as exc:
            if str(exc).startswith('Verified loop'):
                raise
    progress('Front view: fitting constrained left/center/right geometry')
    front_view=refine_front_view(frames,rects,pair_reports,work_pitch,work_width,loop_links,closing)
    starts=[f.origin_pitch+f.pitch_range[0] for f in frames]
    ends=[f.origin_pitch+f.pitch_range[1] for f in frames]
    order=np.argsort(starts)
    beginning=min(starts)
    width=round(settings.width_mm*settings.pixels_per_mm)
    pitch=settings.pitch_mm*settings.pixels_per_mm
    estimated_height=round((max(ends)-beginning)*pitch)
    if width*estimated_height > 100_000_000:
        raise UnwrapError("Output exceeds 100 megapixels; reduce output resolution or split the session.")
    canvas=None;seams=[];placements=[]
    for index in order:
        f=frames[index]
        progress(f"Rendering: {f.name}")
        im,start=f.render(width,pitch)
        cv2.imwrite(str(debug/(Path(f.name).stem+'_front_view.jpg')),cv2.resize(im,(720,round(len(im)*720/width))),[cv2.IMWRITE_JPEG_QUALITY,94])
        im=normalize_illumination(im,tone_profile,pitch)
        y=round((f.origin_pitch+start-beginning)*pitch)
        placements.append({'source':f.name,'start_row':y,'height':len(im),
                           'origin_pitch':f.origin_pitch,'anchors_source_y':f.anchors.tolist()})
        if canvas is None:
            canvas=im
        elif y+len(im)<=len(canvas):
            continue
        else:
            canvas,info=merge(canvas,im,y,pitch,debug/(Path(f.name).stem+'_overlap_before.jpg'))
            info['source']=f.name;seams.append(info)
    crop_start=0
    if loop_links is not None:
        target_length=round(loop_links*pitch)
        if len(canvas)>=target_length:
            # Cut at equal canonical pitch phase, not at an arbitrary image edge.
            crop_end=min(len(canvas),round((np.floor(max(ends))-beginning)*pitch))
            crop_start=max(0,crop_end-target_length)
            canvas=canvas[crop_start:crop_start+target_length]
        else:
            loop_links=None
    # A horizontal strip: capture progression increases from left to right.
    rotation=cv2.ROTATE_90_CLOCKWISE if not signs or signs[0]<0 else cv2.ROTATE_90_COUNTERCLOCKWISE
    horizontal=cv2.rotate(canvas,rotation)
    cv2.imwrite(str(out/'panorama.png'),horizontal)
    cv2.imwrite(str(out/'panorama.jpg'),horizontal,[cv2.IMWRITE_JPEG_QUALITY,96])
    save_review(horizontal,out/'review.jpg')
    preview=cv2.resize(horizontal,(min(2200,horizontal.shape[1]),round(horizontal.shape[0]*min(1,2200/horizontal.shape[1]))),interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out/'preview.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,95])
    cv2.imwrite(str(out/'panorama_vertical.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,95])
    observed=canvas.shape[0]/pitch
    report={'settings':asdict(settings),'input_count':len(paths),'used_count':len(frames),'skipped_exact_duplicates':skipped,
            'mode':'provided_images' if settings.total_links is None else 'expected_link_count',
            'output_size_wh':[horizontal.shape[1],horizontal.shape[0]],'nominal_pitch_px':pitch,
            'observed_span_pitches':round(observed,3),'full_loop_verified':loop_links is not None,
            'verified_loop_links':loop_links,'crop_start_row':crop_start,'rotation':int(rotation),
            'coverage_status':'verified_full_loop' if loop_links is not None else ('provided_images_processed' if settings.total_links is None else 'partial_or_unverified_loop'),
            'note':'Observed span includes partial pitches at both ends; it is not a counted number of complete links.',
            'placements':placements,'pair_registration':pair_reports,'seams':seams,'front_view':front_view,
            'limitations':['Single central track, length vertical, edges visible, ordered overlapping photos required.',
                           'Nominal width/pitch rectification is not calibrated metrology.',
                           'Original illumination and occluded details cannot be fully recovered.']}
    if settings.total_links is not None:
        report['expected_length_mm']=settings.total_links*settings.pitch_mm
        report['unobserved_length_lower_bound_mm']=round(max(0,settings.total_links-observed)*settings.pitch_mm,2)
    (out/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    provenance={'settings':asdict(settings),'rotation':int(rotation),'crop_start_row':crop_start,
                'placements':placements,'seams':seams,'images':[{'file':f.name,'source_size_wh':[f.image.shape[1],f.image.shape[0]],
                    'sha256':hashlib.sha256(paths[[p.name for p in paths].index(f.name)].read_bytes()).hexdigest(),
                    'left_x_by_source_y':f.left[::8].round(2).tolist(),'right_x_by_source_y':f.right[::8].round(2).tolist(),
                    'center_x_by_source_y':f.center_x[::8].round(2).tolist(),
                    'mesh_pitch_offsets':None if f.mesh_pitch_offsets is None else f.mesh_pitch_offsets.round(6).tolist(),
                    'input_rotation':f.input_rotation,'valid_y_range':[f.valid_y_start,f.valid_y_end],
                    'boundary_sampling_stride':8,'anchors_y':f.anchors.tolist()} for f in frames]}
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    progress(f"Saved: {out/'panorama.png'}")
    return report
