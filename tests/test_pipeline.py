from io import BytesIO
from pathlib import Path
import json
import threading
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
from track_unwrap.tiles import save_deepzoom


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


def test_user_link_override_and_total_length_survive_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'JOBS',tmp_path)
    captured=[]
    monkeypatch.setattr(api.pool,'submit',lambda fn,*args:captured.append(args))
    buf=BytesIO();Image.new('RGB',(120,160),(90,90,90)).save(buf,format='PNG')
    with TestClient(api.app) as client:
        response=client.post('/api/jobs',data={'width_mm':'230','pitch_mm':'48','total_links':''},
                             files=[('images',('photo.png',buf.getvalue(),'image/png'))])
        assert response.status_code==202
        job_id=response.json()['id']
        api.set_state(job_id,status='complete',result={'settings':{'width_mm':230,'pitch_mm':48,'total_links':None},
                                                      'full_loop_verified':True,'verified_loop_links':72})
        first=client.get('/api/jobs/'+job_id).json()
        assert first['links']['source']=='image'
        assert first['links']['total_length_mm']==3456
        edited=client.patch('/api/jobs/'+job_id+'/links',json={'total_links':74}).json()
        assert edited['links']['total_length_mm']==3552
        assert edited['links']['disagrees_with_image'] is True
        assert client.get('/api/jobs').json()['items'][0]['links']['effective_links']==74
        assert client.patch('/api/jobs/'+job_id+'/links',json={'total_links':0}).status_code==422
    with TestClient(api.app) as client:
        assert client.get('/api/jobs/'+job_id).json()['links']['effective_links']==74
        reset=client.patch('/api/jobs/'+job_id+'/links',json={'total_links':None}).json()
        assert reset['links']['effective_links']==72


def test_partial_photos_do_not_invent_whole_link_count():
    state={'settings':{'pitch_mm':86,'total_links':None},
           'result':{'full_loop_verified':False,'verified_loop_links':None,'observed_span_pitches':25.33}}
    assert api._link_summary(state)['total_length_mm'] is None
    state['user_total_links']=100
    assert api._link_summary(state)['total_length_mm']==8600


def test_deepzoom_tiles_serve_without_path_traversal(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'JOBS',tmp_path)
    job_id='a'*32
    folder=tmp_path/job_id/'result'
    folder.mkdir(parents=True)
    Image.new('RGB',(513,257),(40,70,100)).save(folder/'panorama.jpg')
    save_deepzoom(folder/'panorama.jpg',folder)
    with TestClient(api.app) as client:
        api.jobs[job_id]={'id':job_id,'status':'complete','settings':{'width_mm':230,'pitch_mm':48}}
        assert client.get(f'/api/jobs/{job_id}/files/deepzoom.dzi').status_code==200
        assert client.get(f'/api/jobs/{job_id}/files/deepzoom_files/10/0_0.jpg').status_code==200
        assert client.get(f'/api/jobs/{job_id}/files/deepzoom_files/10/no.jpg').status_code==404


def test_queued_work_is_resubmitted_after_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'JOBS',tmp_path)
    captured=[]
    monkeypatch.setattr(api.pool,'submit',lambda fn,*args:captured.append(args))
    job_id='b'*32
    folder=tmp_path/job_id/'input'
    folder.mkdir(parents=True)
    Image.new('RGB',(120,160),(90,90,90)).save(folder/'001.png')
    state={'id':job_id,'status':'processing','created_at':'2026-01-01T00:00:00+00:00',
           'settings':{'width_mm':230,'pitch_mm':48,'total_links':None,'pixels_per_mm':5.0}}
    (tmp_path/job_id/'job.json').write_text(__import__('json').dumps(state),encoding='utf-8')
    with TestClient(api.app) as client:
        assert client.get('/api/jobs/'+job_id).json()['status']=='queued'
        assert len(captured)==1
        assert captured[0][0]==job_id


def test_legacy_job_detail_includes_settings_and_photo_count(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'JOBS',tmp_path)
    job_id='c'*32
    folder=tmp_path/job_id
    folder.mkdir()
    state={'id':job_id,'status':'complete','message':'완료',
           'result':{'settings':{'width_mm':230,'pitch_mm':48,'total_links':None},
                     'input_count':12,'full_loop_verified':True,'verified_loop_links':72}}
    (folder/'job.json').write_text(json.dumps(state),encoding='utf-8')
    with TestClient(api.app) as client:
        detail=client.get('/api/jobs/'+job_id).json()
        assert detail['settings']['width_mm']==230
        assert detail['input_count']==12
        assert detail['links']['total_length_mm']==3456


def test_delete_job_removes_originals_outputs_and_history(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'JOBS',tmp_path)
    job_id='d'*32
    folder=tmp_path/job_id
    (folder/'input').mkdir(parents=True)
    (folder/'result'/'deepzoom_files'/'10').mkdir(parents=True)
    (folder/'input'/'001.jpg').write_bytes(b'original')
    (folder/'result'/'panorama.png').write_bytes(b'output')
    (folder/'result'/'deepzoom_files'/'10'/'0_0.jpg').write_bytes(b'tile')
    state={'id':job_id,'status':'complete','created_at':'2026-01-01T00:00:00+00:00',
           'settings':{'width_mm':230,'pitch_mm':48,'total_links':None}}
    (folder/'job.json').write_text(json.dumps(state),encoding='utf-8')
    with TestClient(api.app) as client:
        deleted=client.delete('/api/jobs/'+job_id)
        assert deleted.status_code==200
        assert deleted.json()=={'deleted':True,'pending':False}
        assert not folder.exists()
        assert client.get('/api/jobs/'+job_id).status_code==404
        assert client.get('/api/jobs').json()['total']==0
        assert client.get(f'/api/jobs/{job_id}/files/panorama.png').status_code==404
        assert client.delete('/api/jobs/'+job_id).status_code==404
    with TestClient(api.app) as client:
        assert client.get('/api/jobs').json()['total']==0


def test_delete_processing_job_stops_worker_and_cleans_files(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'JOBS',tmp_path)
    started=threading.Event()
    release=threading.Event()
    def slow_unwrap(_paths,_folder,_settings,progress):
        started.set()
        assert release.wait(5)
        progress('Rendering:')
        raise AssertionError('삭제된 작업이 계속 처리됐습니다.')
    monkeypatch.setattr(api,'unwrap',slow_unwrap)
    buf=BytesIO();Image.new('RGB',(120,160),(90,90,90)).save(buf,format='PNG')
    with TestClient(api.app) as client:
        response=client.post('/api/jobs',data={'width_mm':'230','pitch_mm':'48'},
                             files=[('images',('photo.png',buf.getvalue(),'image/png'))])
        job_id=response.json()['id']
        assert started.wait(5)
        deleted=client.delete('/api/jobs/'+job_id)
        assert deleted.status_code==200
        assert deleted.json()=={'deleted':False,'pending':True}
        assert client.get('/api/jobs').json()['total']==0
        assert client.get('/api/jobs/'+job_id).status_code==404
        release.set()
        for _ in range(100):
            if not (tmp_path/job_id).exists():break
            threading.Event().wait(0.05)
        assert not (tmp_path/job_id).exists()
