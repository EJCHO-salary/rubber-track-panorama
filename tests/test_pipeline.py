from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from track_unwrap.geometry import Frame,monotone_map,UnwrapError
from track_unwrap.front_view import refine_front_view
from track_unwrap.matching import estimate_pitch_shift
from track_unwrap.pipeline import Settings,natural_key
from track_unwrap import api


@pytest.mark.parametrize('value',[None,'','   '])
def test_link_count_can_be_omitted(value):
    assert api.parse_links(value) is None


@pytest.mark.parametrize('value',['0','-1','4.5','nan','x'])
def test_invalid_link_count_rejected(value):
    with pytest.raises(ValueError):api.parse_links(value)


def test_positive_link_count():
    assert api.parse_links(' 72 ')==72


@pytest.mark.parametrize('settings',[Settings(0,86),Settings(450,0),Settings(float('nan'),86),Settings(230,480),Settings(450,86,0)])
def test_invalid_dimensions(settings):
    with pytest.raises(ValueError):settings.validate()


def test_monotonic_geometry_and_control_points():
    anchors=np.array([100.,210.,305.,403.,505.])
    mapped=monotone_map(np.arange(5),anchors,np.linspace(-.5,4.5,500))
    assert np.all(np.diff(mapped)>0)
    np.testing.assert_allclose(monotone_map(np.arange(5),anchors,np.arange(5)),anchors)


def test_natural_filename_order():
    assert sorted(['10.jpg','2.jpg','1.jpg'],key=natural_key)==['1.jpg','2.jpg','10.jpg']


def test_known_overlap_has_one_correct_integer_shift():
    rng=np.random.default_rng(19)
    surface=cv2.GaussianBlur(rng.integers(0,256,(900,500,3),dtype=np.uint8),(3,3),.5)
    first=surface[:650];second=surface[200:850]
    shift,info=estimate_pitch_shift(first,second,0,0,50)
    assert shift==4
    assert info['accepted_matches']>100
    assert abs(info['median_phase_error_pitch'])<.01


def test_no_features_fails_instead_of_fake_stitch():
    blank=np.full((500,400,3),127,np.uint8)
    with pytest.raises(UnwrapError):estimate_pitch_shift(blank,blank,0,0,50)


def test_mesh_fits_known_surface_displacement_without_folding():
    from types import SimpleNamespace
    frames=[SimpleNamespace(anchors=np.arange(7)*100.,origin_pitch=0,mesh_pitch_offsets=None) for _ in range(2)]
    a=[[u*1000,p*100] for u in [.12,.22,.32,.7,.8,.9] for p in np.linspace(.2,5.7,30)]
    b=[[x,y+8] for x,y in a]
    result=refine_front_view(frames,[(None,0),(None,0)],[{'source_points':a,'target_points':b}],100,1000)
    assert result['status']=='applied'
    assert result['after_median_pitch_error']<result['before_median_pitch_error']/2
    for f in frames:
        assert np.abs(f.mesh_pitch_offsets).max()<=.18
        assert np.all(np.diff(np.arange(7)[:,None]+f.mesh_pitch_offsets,axis=0)>0)


def test_rectification_preserves_outer_rails_and_center():
    yy,xx=np.mgrid[:101,:81]
    image=np.stack([xx*3,yy*2,np.zeros_like(xx)],axis=2).astype(np.uint8)
    frame=Frame('synthetic',image,np.full(101,10.),np.full(101,70.),np.array([20.,40.,60.,80.]),.5,20,1,image,
                center_x=np.full(101,45.),mesh_pitch_offsets=np.tile([.1,-.1],(4,1)))
    result,start=frame.render(21,20)
    np.testing.assert_allclose(result[:,0,0],30,atol=1)
    np.testing.assert_allclose(result[:,10,0],135,atol=1)
    np.testing.assert_allclose(result[:,-1,0],210,atol=1)
    expected=(20+(start+np.arange(len(result))/20)*20)*2
    np.testing.assert_allclose(result[:,10,1],expected,atol=1)


@pytest.mark.parametrize('links',[None,'','72'])
def test_multipart_optional_links(tmp_path,monkeypatch,links):
    monkeypatch.setattr(api,'JOBS',tmp_path)
    captured=[]
    monkeypatch.setattr(api.pool,'submit',lambda fn,*args:captured.append(args))
    buf=BytesIO();Image.new('RGB',(120,160),(90,90,90)).save(buf,format='PNG')
    data={'width_mm':'230','pitch_mm':'48'}
    if links is not None:data['total_links']=links
    with TestClient(api.app) as client:
        response=client.post('/api/jobs',data=data,files=[('images',('photo.png',buf.getvalue(),'image/png'))])
        assert response.status_code==202,response.text
        assert captured[0][2].total_links==(72 if links=='72' else None)
        assert client.get('/api/jobs/'+response.json()['id']).status_code==200
        assert client.get('/api/jobs/'+response.json()['id']+'/files/secret.txt').status_code==404
