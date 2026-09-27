"""Shared local service with persistent work list and editable link count."""
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, StrictInt

from .pipeline import Settings, unwrap


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'output' / 'jobs'
pool = ThreadPoolExecutor(max_workers=max(1, min(4, int(os.getenv('TRACK_WORKERS', '1')))), thread_name_prefix='track-worker')
lock = threading.RLock()
jobs: dict[str, dict] = {}
ARTIFACTS = {'panorama.png', 'panorama.jpg', 'preview.jpg', 'review.jpg', 'panorama_vertical.jpg',
             'quality_report.json', 'provenance.json', 'deepzoom.dzi'}
SAMPLES = {'case1': 'result', 'case2': 'case2'}
TILE_PATTERN = re.compile(r'^[0-9]+_[0-9]+\.jpg$')


def parse_links(value):
    if value is None or str(value).strip() == '':
        return None
    try:
        n = int(str(value).strip())
    except ValueError as exc:
        raise ValueError('전체 링크 수는 양의 정수로 입력하거나 비워두세요.') from exc
    if not 1 <= n <= 2000:
        raise ValueError('전체 링크 수는 1~2000 범위로 입력하거나 비워두세요.')
    return n


def _valid_id(job_id):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise HTTPException(404)


def _compact_result(report):
    keys = ('settings', 'input_count', 'used_count', 'skipped_exact_duplicates', 'mode',
            'output_size_wh', 'observed_span_pitches', 'full_loop_verified', 'verified_loop_links', 'has_deepzoom',
            'coverage_status', 'front_view', 'expected_length_mm', 'input_link_count_disagrees_with_image')
    return {key: report[key] for key in keys if key in report}


def _link_summary(state):
    result = state.get('result') or {}
    settings = state.get('settings') or result.get('settings') or {}
    manual = state.get('user_total_links', settings.get('total_links'))
    inferred = result.get('verified_loop_links') if result.get('full_loop_verified') else None
    effective = manual if manual is not None else inferred
    pitch = settings.get('pitch_mm')
    length_mm = round(effective * pitch, 2) if effective is not None and pitch is not None else None
    return {'suggested_links': inferred, 'entered_links': manual, 'effective_links': effective,
            'source': 'user' if manual is not None else 'image' if inferred is not None else 'unknown',
            'total_length_mm': length_mm, 'total_length_m': round(length_mm / 1000, 3) if length_mm is not None else None,
            'disagrees_with_image': manual is not None and inferred is not None and manual != inferred}


def _sync_quality_report(job_id):
    path = JOBS / job_id / 'result' / 'quality_report.json'
    if not path.is_file():
        return
    with lock:
        summary = _link_summary(jobs[job_id])
        report = json.loads(path.read_text(encoding='utf-8'))
        report['applied_link_count'] = summary['effective_links']
        report['link_count_source'] = summary['source']
        report['user_total_links'] = summary['entered_links']
        report['total_length_mm'] = summary['total_length_mm']
        report['total_length_m'] = summary['total_length_m']
        report['link_count_disagrees_with_image'] = summary['disagrees_with_image']
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)


def _public_state(state):
    data = dict(state)
    data['links'] = _link_summary(state)
    if data['status'] == 'queued':
        with lock:
            pending = sorted((job for job in jobs.values() if job['status'] == 'queued'),
                             key=lambda job: (job.get('created_at', ''), job['id']))
        data['queue_position'] = next((i + 1 for i, job in enumerate(pending) if job['id'] == data['id']), None)
    else:
        data['queue_position'] = None
    return data


def set_state(job_id, **values):
    with lock:
        jobs[job_id].update(values)
        jobs[job_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
        state = dict(jobs[job_id])
        path = JOBS / job_id / 'job.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    return state


def run_job(job_id, paths, settings):
    set_state(job_id, status='processing', message='제품 영역을 확인하고 있습니다.')
    def progress(message):
        labels = {'Geometry:': '외곽과 피치 검출 중', 'Matched ': '겹치는 제품 위치 확인 중',
                  'Front view:': '외곽·중앙선 기준 정면뷰 보정 중', 'Verified loop:': '한 바퀴 중복 구간 확인 완료',
                  'Rendering:': '원본 합성 중', 'Saved:': '결과 저장 완료'}
        friendly = next((value for prefix, value in labels.items() if message.startswith(prefix)), message)
        set_state(job_id, message=friendly)
    try:
        report = unwrap(paths, JOBS / job_id / 'result', settings, progress)
        set_state(job_id, status='complete', message='전개 사진을 생성했습니다.', result=_compact_result(report))
        _sync_quality_report(job_id)
    except Exception as exc:
        set_state(job_id, status='needs_review', message=str(exc))


def _resume_jobs():
    JOBS.mkdir(parents=True, exist_ok=True)
    with lock:
        jobs.clear()
        for path in JOBS.glob('*/job.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
                if state.get('id') != path.parent.name:
                    continue
                state.setdefault('created_at', datetime.fromtimestamp(path.stat().st_ctime, timezone.utc).isoformat())
                jobs[state['id']] = state
            except (ValueError, OSError):
                continue
        pending = sorted((state for state in jobs.values() if state['status'] in {'queued', 'processing'}),
                         key=lambda state: (state.get('created_at', ''), state['id']))
    for state in pending:
        folder = JOBS / state['id'] / 'input'
        paths = sorted((path for path in folder.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})) if folder.is_dir() else []
        if not paths or 'settings' not in state:
            set_state(state['id'], status='needs_review', message='보관된 입력을 찾지 못했습니다. 사진을 다시 제출하세요.')
            continue
        set_state(state['id'], status='queued', message='대기 작업을 복구했습니다.')
        pool.submit(run_job, state['id'], paths, Settings(**state['settings']))


@asynccontextmanager
async def lifespan(_app):
    _resume_jobs()
    yield


app = FastAPI(title='Rubber Track Unwrapping', version='0.2.0', lifespan=lifespan)


class LinkUpdate(BaseModel):
    total_links: StrictInt | None = Field(..., description='Positive count, or null to use image suggestion.')


@app.post('/api/jobs', status_code=202)
async def create_job(width_mm: float = Form(...), pitch_mm: float = Form(...),
                     total_links: str | None = Form(None), images: list[UploadFile] = File(...)):
    try:
        settings = Settings(width_mm, pitch_mm, parse_links(total_links))
        settings.validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not 1 <= len(images) <= 100:
        raise HTTPException(422, '사진 1~100장을 선택하세요.')
    job_id = uuid4().hex
    folder = JOBS / job_id / 'input'
    folder.mkdir(parents=True)
    paths, names = [], []
    total_bytes = 0
    for i, upload in enumerate(images):
        suffix = Path(upload.filename or '').suffix.lower()
        if suffix not in {'.jpg', '.jpeg', '.png', '.webp'}:
            raise HTTPException(422, 'JPEG, PNG, WebP 원본을 사용하세요.')
        path = folder / f'{i + 1:03d}{suffix}'
        size = 0
        with path.open('wb') as dest:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                total_bytes += len(chunk)
                if size > 30 * 1024 * 1024 or total_bytes > 500 * 1024 * 1024:
                    raise HTTPException(413, '사진당 30MB, 작업당 500MB를 초과했습니다.')
                dest.write(chunk)
        try:
            with Image.open(path) as im:
                if im.width * im.height > 40_000_000 or min(im.size) < 100:
                    raise HTTPException(422, '사진 해상도는 최소 100px, 최대 4천만 화소입니다.')
                im.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, '읽을 수 없는 이미지가 포함되어 있습니다.') from exc
        paths.append(path)
        names.append(Path(upload.filename or path.name).name[:200])
    now = datetime.now(timezone.utc).isoformat()
    with lock:
        jobs[job_id] = {'id': job_id, 'status': 'queued', 'message': '처리를 기다리고 있습니다.',
                        'created_at': now, 'updated_at': now, 'settings': asdict(settings),
                        'input_names': names, 'input_count': len(paths), 'user_total_links': settings.total_links}
    set_state(job_id)
    pool.submit(run_job, job_id, paths, settings)
    return {'id': job_id, 'status': 'queued'}


@app.get('/api/jobs')
def list_jobs(limit: int = 50, offset: int = 0):
    if not 1 <= limit <= 100 or offset < 0:
        raise HTTPException(422, '페이지 범위를 확인하세요.')
    with lock:
        ordered = sorted(jobs.values(), key=lambda state: (state.get('created_at', ''), state['id']), reverse=True)
        page = [dict(state) for state in ordered[offset:offset + limit]]
    items = []
    for state in page:
        result = state.get('result') or {}
        items.append({'id': state['id'], 'status': state['status'], 'message': state.get('message'),
                      'created_at': state.get('created_at'), 'updated_at': state.get('updated_at'),
                      'settings': state.get('settings') or result.get('settings'),
                      'input_count': state.get('input_count') or result.get('input_count'),
                      'used_count': result.get('used_count'), 'full_loop_verified': result.get('full_loop_verified'),
                      'links': _link_summary(state)})
    return {'items': items, 'total': len(ordered)}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    _valid_id(job_id)
    with lock:
        state = jobs.get(job_id)
    if state is None:
        path = JOBS / job_id / 'job.json'
        if not path.is_file():
            raise HTTPException(404)
        state = json.loads(path.read_text(encoding='utf-8'))
    return _public_state(state)


@app.patch('/api/jobs/{job_id}/links')
def update_links(job_id: str, update: LinkUpdate):
    _valid_id(job_id)
    get_job(job_id)
    try:
        count = parse_links(update.total_links)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    state = set_state(job_id, user_total_links=count)
    if state['status'] == 'complete':
        _sync_quality_report(job_id)
    return _public_state(state)


def _artifact(path: Path, name: str):
    if name not in ARTIFACTS or not (path / name).is_file():
        raise HTTPException(404)
    return FileResponse(path / name, media_type='application/xml' if name.endswith('.dzi') else None)


def _tile(path: Path, level: int, tile: str):
    if not 0 <= level <= 30 or not TILE_PATTERN.fullmatch(tile):
        raise HTTPException(404)
    target = path / 'deepzoom_files' / str(level) / tile
    if not target.is_file():
        raise HTTPException(404)
    return FileResponse(target, media_type='image/jpeg')


@app.get('/api/jobs/{job_id}/files/{name}')
def get_artifact(job_id: str, name: str):
    get_job(job_id)
    return _artifact(JOBS / job_id / 'result', name)


@app.get('/api/jobs/{job_id}/tiles/{level}/{tile}')
@app.get('/api/jobs/{job_id}/files/deepzoom_files/{level}/{tile}')
def get_job_tile(job_id: str, level: int, tile: str):
    get_job(job_id)
    return _tile(JOBS / job_id / 'result', level, tile)


@app.get('/api/samples')
def list_samples():
    items = []
    for name, folder in SAMPLES.items():
        path = ROOT / 'output' / folder / 'quality_report.json'
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding='utf-8'))
        items.append({'id': name, 'settings': report['settings'], 'input_count': report['input_count'],
                      'used_count': report['used_count'], 'output_size_wh': report['output_size_wh'],
                      'full_loop_verified': report['full_loop_verified'], 'verified_loop_links': report['verified_loop_links'],
                      'has_deepzoom': (path.parent / 'deepzoom.dzi').is_file()})
    return {'items': items}


@app.get('/sample/{name}')
def sample(name: str):
    return _artifact(ROOT / 'output' / 'result', name)


@app.get('/samples/{case_name}/{name}')
def selected_sample(case_name: str, name: str):
    if case_name not in SAMPLES:
        raise HTTPException(404)
    return _artifact(ROOT / 'output' / SAMPLES[case_name], name)


@app.get('/samples/{case_name}/tiles/{level}/{tile}')
@app.get('/samples/{case_name}/deepzoom_files/{level}/{tile}')
def sample_tile(case_name: str, level: int, tile: str):
    if case_name not in SAMPLES:
        raise HTTPException(404)
    return _tile(ROOT / 'output' / SAMPLES[case_name], level, tile)


def _front_page():
    built = ROOT / 'frontend' / 'dist' / 'index.html'
    return FileResponse(built if built.is_file() else ROOT / 'web' / 'index.html')


@app.get('/')
def home():
    return _front_page()


@app.get('/assets/{asset_path:path}')
def frontend_asset(asset_path: str):
    root = (ROOT / 'frontend' / 'dist' / 'assets').resolve()
    target = (root / asset_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)


@app.get('/new')
@app.get('/jobs/{job_id}')
@app.get('/samples/{case_name}')
def frontend_route():
    return _front_page()
