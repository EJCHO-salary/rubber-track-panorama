"""Regularized front-view mesh with fixed left/center/right width coordinates."""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import lsqr


def refine_front_view(frames,rects,pairs,pitch_px,width_px,loop_links=None,closing=None):
    offsets=np.cumsum([0]+[2*len(f.anchors) for f in frames])
    rows=[];cols=[];vals=[];targets=[]
    def basis(frame_index,u,p):
        n=len(frames[frame_index].anchors)
        t=float(np.clip(p,0,n-1));i=min(int(t),n-2);fraction=t-i
        side=0 if u<.5 else 1
        weight=abs(u-.5)*2
        return [(offsets[frame_index]+2*i+side,weight*(1-fraction)),
                (offsets[frame_index]+2*(i+1)+side,weight*fraction)]
    def add(equation,target,weight=1):
        row=len(targets)
        for col,value in equation:rows.append(row);cols.append(col);vals.append(value*weight)
        targets.append(target*weight)
    links=[(i,i+1,info) for i,info in enumerate(pairs)]
    if loop_links is not None and closing is not None:
        links.append((0,len(frames)-1,closing))
    for ia,ib,info in links:
        for a,b in zip(info['source_points'],info['target_points']):
            ua,ub=a[0]/width_px,b[0]/width_px
            # Fit the rubber faces; the deep center can have incompatible parallax.
            if .42<ua<.60 or .42<ub<.60:continue
            pa=a[1]/pitch_px+rects[ia][1];pb=b[1]/pitch_px+rects[ib][1]
            delta=pb+frames[ib].origin_pitch-pa-frames[ia].origin_pitch
            if loop_links is not None:delta-=round(delta/loop_links)*loop_links
            if abs(delta)>.28:continue
            equation=basis(ia,ua,pa)+[(c,-v) for c,v in basis(ib,ub,pb)]
            add(equation,delta)
    data_count=len(targets)
    if data_count<20:
        return {'status':'insufficient_mesh_constraints','fit_correspondences':data_count}
    for i,f in enumerate(frames):
        for k in range(len(f.anchors)):
            for edge in range(2):
                j=offsets[i]+2*k+edge
                add([(j,1)],0,2.0)
                if k:
                    add([(j,1),(j-2,-1)],0,2.0)
                if k>1:
                    add([(j,1),(j-2,-2),(j-4,1)],0,5.0)
    matrix=coo_matrix((vals,(rows,cols)),shape=(len(targets),offsets[-1])).tocsr()
    target=np.asarray(targets)
    weight=np.ones(len(target))
    solution=np.zeros(offsets[-1])
    for _ in range(4):
        solution=lsqr(diags(np.sqrt(weight))@matrix,target*np.sqrt(weight),atol=1e-7,btol=1e-7,iter_lim=400)[0]
        residual=matrix@solution-target
        weight[:data_count]=np.minimum(1,.035/np.maximum(np.abs(residual[:data_count]),1e-8))
    solution=np.clip(solution,-.18,.18)
    for i,f in enumerate(frames):
        f.mesh_pitch_offsets=solution[offsets[i]:offsets[i+1]].reshape(-1,2)
    residual=(matrix@solution-target)[:data_count]
    return {'status':'applied','fit_correspondences':data_count,
            'before_median_pitch_error':round(float(np.median(np.abs(target[:data_count]))),5),
            'after_median_pitch_error':round(float(np.median(np.abs(residual))),5),
            'maximum_control_displacement_pitch':round(float(np.max(np.abs(solution))),5),
            'width_constraints':'left=0, center=0.5, right=1; monotonic longitudinal controls',
            'measurement_note':'Fitting residual, not independent dimensional accuracy.'}
