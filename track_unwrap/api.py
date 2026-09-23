"""Local demo API. Run: python -m uvicorn track_unwrap.api:app --host 127.0.0.1 --port 8765"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from .pipeline import Settings, unwrap

ROOT=Path(__file__).resolve().parents[1]
JOBS=ROOT/'output'/'jobs'
app=FastAPI(title='Rubber Track Unwrapping',version='0.1.0')
pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='track-worker')
lock=threading.Lock()
jobs={}
ARTIFACTS={'panorama.png','panorama.jpg','preview.jpg','review.jpg','panorama_vertical.jpg','quality_report.json','provenance.json'}


def parse_links(value):
    if value is None or str(value).strip()=='':
        return None
    try:
        n=int(str(value).strip())
    except ValueError:
        raise ValueError('전체 링크 수는 양의 정수로 입력하거나 비워두세요.')
    if not 1<=n<=2000:
        raise ValueError('전체 링크 수는 1~2000 범위로 입력하거나 비워두세요.')
    return n


def set_state(job_id,**values):
    with lock:
        jobs[job_id].update(values)
        state=dict(jobs[job_id])
    (JOBS/job_id/'job.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')


def run_job(job_id,paths,settings):
    set_state(job_id,status='processing',message='제품 영역을 확인하고 있습니다.')
    def progress(message):
        names={'Geometry:':'외곽과 피치 검출 중','Matched ':'겹치는 제품 위치 확인 중','Front view:':'외곽·중앙선 기준 정면뷰 보정 중','Verified loop:':'한 바퀴 중복 구간 확인 완료','Rendering:':'원본 합성 중','Saved:':'결과 저장 완료'}
        friendly=next((v for k,v in names.items() if message.startswith(k)),message)
        set_state(job_id,message=friendly)
    try:
        result=unwrap(paths,JOBS/job_id/'result',settings,progress)
        set_state(job_id,status='complete',message='전개 사진을 생성했습니다.',result=result)
    except Exception as exc:
        set_state(job_id,status='needs_review',message=str(exc))


@app.get('/')
def home():
    return FileResponse(ROOT/'web'/'index.html')


@app.post('/api/jobs',status_code=202)
async def create_job(width_mm:float=Form(...),pitch_mm:float=Form(...),total_links:str|None=Form(None),images:list[UploadFile]=File(...)):
    try:
        settings=Settings(width_mm,pitch_mm,parse_links(total_links))
        settings.validate()
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    if not 1<=len(images)<=100:
        raise HTTPException(422,'사진 1~100장을 선택하세요.')
    job_id=uuid4().hex
    folder=JOBS/job_id/'input';folder.mkdir(parents=True)
    paths=[];total=0
    for i,upload in enumerate(images):
        suffix=Path(upload.filename or '').suffix.lower()
        if suffix not in {'.jpg','.jpeg','.png','.webp'}:
            raise HTTPException(422,'JPEG, PNG, WebP 원본을 사용하세요.')
        path=folder/f'{i+1:03d}{suffix}'
        size=0
        with path.open('wb') as dest:
            while chunk:=await upload.read(1024*1024):
                size+=len(chunk);total+=len(chunk)
                if size>30*1024*1024 or total>500*1024*1024:
                    raise HTTPException(413,'사진당 30MB, 작업당 500MB를 초과했습니다.')
                dest.write(chunk)
        try:
            with Image.open(path) as im:
                if im.width*im.height>40_000_000 or min(im.size)<100:
                    raise HTTPException(422,'사진 해상도는 최소 100px, 최대 4천만 화소입니다.')
                im.verify()
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError):
            raise HTTPException(422,'읽을 수 없는 이미지가 포함되어 있습니다.')
        paths.append(path)
    with lock:
        jobs[job_id]={'id':job_id,'status':'queued','message':'처리를 기다리고 있습니다.'}
    set_state(job_id)
    pool.submit(run_job,job_id,paths,settings)
    return {'id':job_id,'status':'queued'}


@app.get('/api/jobs/{job_id}')
def get_job(job_id:str):
    if len(job_id)!=32 or any(c not in '0123456789abcdef' for c in job_id):
        raise HTTPException(404)
    with lock:
        state=jobs.get(job_id)
    if state is None:
        saved=JOBS/job_id/'job.json'
        if not saved.is_file():raise HTTPException(404)
        state=json.loads(saved.read_text(encoding='utf-8'))
        if state['status'] in {'queued','processing'}:
            state.update(status='needs_review',message='서버가 재시작되었습니다. 사진을 다시 제출하세요.')
    return state


@app.get('/api/jobs/{job_id}/files/{name}')
def get_artifact(job_id:str,name:str):
    get_job(job_id)
    if name not in ARTIFACTS:raise HTTPException(404)
    path=JOBS/job_id/'result'/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path)


@app.get('/sample/{name}')
def sample(name:str):
    if name not in ARTIFACTS:raise HTTPException(404)
    path=ROOT/'output'/'result'/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path)


@app.get('/samples/{case_name}/{name}')
def selected_sample(case_name:str,name:str):
    folders={'case1':'result','case2':'case2'}
    if case_name not in folders or name not in ARTIFACTS:raise HTTPException(404)
    path=ROOT/'output'/folders[case_name]/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path)
