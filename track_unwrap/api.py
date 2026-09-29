"""Shared local service with persistent work list and editable link count."""
import json
import os
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, StrictInt

from .zones import assist_polygon, build_zones, list_zone_instances, load_zones, suggest_half_turn
from .cracks import (add_manual_crack, clear_cracks, delete_manual_crack,
                     load_cracks, propose_cracks, review_crack, review_cracks,
                     trace_crack)
from .pipeline import Settings, unwrap


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'output' / 'jobs'
pool = ThreadPoolExecutor(max_workers=max(1, min(4, int(os.getenv('TRACK_WORKERS', '1')))), thread_name_prefix='track-worker')
lock = threading.RLock()
jobs: dict[str, dict] = {}
futures: dict[str, Future] = {}
zone_locks: dict[str, threading.Lock] = {}
ARTIFACTS = {'panorama.png', 'panorama.jpg', 'preview.jpg', 'review.jpg', 'panorama_vertical.jpg',
             'quality_report.json', 'provenance.json', 'deepzoom.dzi'}
SAMPLES = {'case1': 'result', 'case2': 'case2'}
TILE_PATTERN = re.compile(r'^[0-9]+_[0-9]+\.jpg$')


class JobDeleted(Exception):
    """Stop a worker after its job has been marked for removal."""


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
    result = data.get('result') or {}
    data['settings'] = data.get('settings') or result.get('settings')
    data['input_count'] = data.get('input_count') or result.get('input_count')
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
        if jobs[job_id].get('status') == 'deleting' and values.get('status') != 'deleting':
            raise JobDeleted()
        jobs[job_id].update(values)
        jobs[job_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
        state = dict(jobs[job_id])
        path = JOBS / job_id / 'job.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    return state


def _deletion_requested(job_id):
    with lock:
        return jobs.get(job_id, {}).get('status') == 'deleting'


def _remove_job_files(job_id):
    root = JOBS.resolve()
    folder = (JOBS / job_id).resolve()
    if folder.parent != root:
        raise ValueError('작업 폴더가 저장 경로 밖에 있습니다.')
    for attempt in range(3):
        try:
            shutil.rmtree(folder)
            break
        except FileNotFoundError:
            break
        except OSError:
            if attempt == 2:
                return False
            time.sleep(0.2)
    with lock:
        jobs.pop(job_id, None)
        futures.pop(job_id, None)
    return True


def _schedule_job(job_id, paths, settings):
    with lock:
        futures[job_id] = pool.submit(run_job, job_id, paths, settings)


def run_job(job_id, paths, settings):
    def progress(message):
        labels = {'Geometry:': '외곽과 피치 검출 중', 'Matched ': '겹치는 제품 위치 확인 중',
                  'Front view:': '외곽·중앙선 기준 정면뷰 보정 중', 'Verified loop:': '한 바퀴 중복 구간 확인 완료',
                  'Rendering:': '원본 합성 중', 'Saved:': '결과 저장 완료'}
        friendly = next((value for prefix, value in labels.items() if message.startswith(prefix)), message)
        set_state(job_id, message=friendly)
    try:
        set_state(job_id, status='processing', message='제품 영역을 확인하고 있습니다.')
        report = unwrap(paths, JOBS / job_id / 'result', settings, progress)
        set_state(job_id, status='complete', message='전개 사진을 생성했습니다.', result=_compact_result(report))
        _sync_quality_report(job_id)
        if not _deletion_requested(job_id):
            try:
                build_zones(JOBS / job_id / 'result')
                set_state(job_id, zones_status='ready')
            except JobDeleted:
                raise
            except Exception as exc:
                set_state(job_id, zones_status='error', zones_message=str(exc))
    except JobDeleted:
        pass
    except Exception as exc:
        try:
            set_state(job_id, status='needs_review', message=str(exc))
        except JobDeleted:
            pass
    finally:
        if _deletion_requested(job_id):
            _remove_job_files(job_id)
        else:
            with lock:
                futures.pop(job_id, None)


def _resume_jobs():
    JOBS.mkdir(parents=True, exist_ok=True)
    with lock:
        jobs.clear()
        futures.clear()
        for path in JOBS.glob('*/job.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
                if state.get('id') != path.parent.name:
                    continue
                state.setdefault('created_at', datetime.fromtimestamp(path.stat().st_ctime, timezone.utc).isoformat())
                jobs[state['id']] = state
            except (ValueError, OSError):
                continue
        deleting = [state['id'] for state in jobs.values() if state.get('status') == 'deleting']
        pending = sorted((state for state in jobs.values() if state['status'] in {'queued', 'processing'}),
                         key=lambda state: (state.get('created_at', ''), state['id']))
    for job_id in deleting:
        _remove_job_files(job_id)
    for state in pending:
        folder = JOBS / state['id'] / 'input'
        paths = sorted((path for path in folder.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})) if folder.is_dir() else []
        if not paths or 'settings' not in state:
            set_state(state['id'], status='needs_review', message='보관된 입력을 찾지 못했습니다. 사진을 다시 제출하세요.')
            continue
        set_state(state['id'], status='queued', message='대기 작업을 복구했습니다.')
        _schedule_job(state['id'], paths, Settings(**state['settings']))


@asynccontextmanager
async def lifespan(_app):
    _resume_jobs()
    yield


app = FastAPI(title='Rubber Track Unwrapping', version='0.3.0', lifespan=lifespan)


class LinkUpdate(BaseModel):
    total_links: StrictInt | None = Field(..., description='Positive count, or null to use image suggestion.')


class ZoneWrite(BaseModel):
    groups: list[dict]
    shapes: list[dict]


class ZoneAssist(BaseModel):
    polygon: list[list[float]]


class ZoneHalfTurn(BaseModel):
    polygon: list[list[float]]
    repeat_pitches: StrictInt


class CrackProposal(BaseModel):
    group_id: str
    sensitivity: str = 'normal'


class CrackReview(BaseModel):
    status: str
    damage_type: str | None = None


class CrackBatchReview(CrackReview):
    candidate_ids: list[str]
    auto_larger: bool = False


class ManualCrack(BaseModel):
    points: list[list[float]]


class GuidedCrackSelection(BaseModel):
    point: list[float] | None = None
    candidate_id: str | None = None
    damage_type: str
    offset_mm: float = Field(ge=0, le=5)
    tolerance: int = Field(ge=15, le=100)


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
    _schedule_job(job_id, paths, settings)
    return {'id': job_id, 'status': 'queued'}


@app.get('/api/jobs')
def list_jobs(limit: int = 50, offset: int = 0):
    if not 1 <= limit <= 100 or offset < 0:
        raise HTTPException(422, '페이지 범위를 확인하세요.')
    with lock:
        ordered = sorted((state for state in jobs.values() if state.get('status') != 'deleting'),
                         key=lambda state: (state.get('created_at', ''), state['id']), reverse=True)
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
    if state.get('status') == 'deleting':
        raise HTTPException(404)
    return _public_state(state)


@app.delete('/api/jobs/{job_id}')
def delete_job(job_id: str):
    _valid_id(job_id)
    with lock:
        state = jobs.get(job_id)
        if state is None:
            raise HTTPException(404)
        if state.get('status') == 'deleting':
            return {'deleted': False, 'pending': True}
        future = futures.get(job_id)
        active = future is not None and not future.done()
        set_state(job_id, status='deleting', message='작업과 파일을 삭제하고 있습니다.')
        cancelled = future.cancel() if future is not None else False
    if not active or cancelled:
        if not _remove_job_files(job_id):
            raise HTTPException(500, '파일 삭제가 완료되지 않았습니다. 서버 재시작 시 다시 시도합니다.')
        return {'deleted': True, 'pending': False}
    return {'deleted': False, 'pending': True}


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


def _zones_dir(job_id):
    state = get_job(job_id)
    if state['status'] != 'complete':
        raise HTTPException(409, '전개 작업이 완료된 뒤 영역을 지정할 수 있습니다.')
    return JOBS / job_id / 'result'


def _zone_lock(job_id):
    with lock:
        return zone_locks.setdefault(job_id, threading.Lock())


@app.get('/api/jobs/{job_id}/zones')
def get_job_zones(job_id: str):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            return load_zones(result_dir) or build_zones(result_dir)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.put('/api/jobs/{job_id}/zones')
def put_job_zones(job_id: str, request: ZoneWrite):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            return build_zones(result_dir, request.groups, request.shapes)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post('/api/jobs/{job_id}/zones/assist')
def assist_job_zone(job_id: str, request: ZoneAssist):
    result_dir = _zones_dir(job_id)
    try:
        return assist_polygon(result_dir, request.polygon)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/jobs/{job_id}/zones/half-turn')
def suggest_job_half_turn(job_id: str, request: ZoneHalfTurn):
    result_dir = _zones_dir(job_id)
    try:
        return suggest_half_turn(result_dir, request.polygon, request.repeat_pitches)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/jobs/{job_id}/zones/instances')
def get_job_zone_instances(job_id: str, group_id: str, section_id: str):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            return list_zone_instances(result_dir, group_id, section_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.get('/api/jobs/{job_id}/cracks')
def get_job_cracks(job_id: str):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        data = load_cracks(result_dir)
        if data is None:
            return {'ready': False, 'stale': False}
        zones = load_zones(result_dir)
        return {**data, 'ready': True, 'stale': not zones or data['zone_created_at'] != zones['created_at']}


@app.delete('/api/jobs/{job_id}/cracks')
def clear_job_cracks(job_id: str):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        clear_cracks(result_dir)
        return {'ready': False, 'stale': False}


@app.post('/api/jobs/{job_id}/cracks/propose')
def propose_job_cracks(job_id: str, request: CrackProposal):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            return {**propose_cracks(result_dir, request.group_id, request.sensitivity), 'ready': True, 'stale': False}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post('/api/jobs/{job_id}/cracks/trace')
def preview_crack_trace(job_id: str, request: GuidedCrackSelection):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            return trace_crack(result_dir, request.point, request.damage_type,
                               request.offset_mm, request.tolerance, request.candidate_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post('/api/jobs/{job_id}/cracks/trace/apply')
def apply_crack_trace(job_id: str, request: GuidedCrackSelection):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            return {**trace_crack(result_dir, request.point, request.damage_type,
                                  request.offset_mm, request.tolerance, request.candidate_id, apply=True),
                    'ready': True, 'stale': False}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.patch('/api/jobs/{job_id}/cracks/{candidate_id}')
def review_job_crack(job_id: str, candidate_id: str, request: CrackReview):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            data = load_cracks(result_dir)
            zones = load_zones(result_dir)
            if data and zones and data['zone_created_at'] != zones['created_at']:
                raise ValueError('영역이 변경되었습니다. 크랙 후보를 다시 생성해 주세요.')
            return {**review_crack(result_dir, candidate_id, request.status, request.damage_type), 'ready': True, 'stale': False}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post('/api/jobs/{job_id}/cracks/review-batch')
def review_job_cracks_batch(job_id: str, request: CrackBatchReview):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            data = load_cracks(result_dir)
            zones = load_zones(result_dir)
            if data and zones and data['zone_created_at'] != zones['created_at']:
                raise ValueError('영역이 변경되었습니다. 크랙 후보를 다시 생성해 주세요.')
            return {**review_cracks(result_dir, request.candidate_ids, request.status,
                                    request.damage_type, request.auto_larger), 'ready': True, 'stale': False}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post('/api/jobs/{job_id}/cracks/manual')
def add_job_manual_crack(job_id: str, request: ManualCrack):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            return {**add_manual_crack(result_dir, request.points), 'ready': True, 'stale': False}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.delete('/api/jobs/{job_id}/cracks/{candidate_id}')
def delete_job_manual_crack(job_id: str, candidate_id: str):
    result_dir = _zones_dir(job_id)
    with _zone_lock(job_id):
        try:
            return {**delete_manual_crack(result_dir, candidate_id), 'ready': True, 'stale': False}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.get('/api/jobs/{job_id}/zones/groups/{group_id}/{kind}.png')
def job_zone_artifact(job_id: str, group_id: str, kind: str):
    result_dir = _zones_dir(job_id)
    if kind not in {'mask', 'overlay', 'authored'} or not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', group_id):
        raise HTTPException(404)
    analysis = load_zones(result_dir)
    if not analysis or group_id not in {group['id'] for group in analysis['groups']}:
        raise HTTPException(404)
    path = result_dir / 'zones' / f'group_{group_id}_{kind}.png'
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get('/api/samples/{case_name}/zones')
def sample_zones(case_name: str):
    if case_name not in SAMPLES:
        raise HTTPException(404)
    result_dir = ROOT / 'output' / SAMPLES[case_name]
    try:
        return load_zones(result_dir) or build_zones(result_dir)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/samples/{case_name}/zones/groups/{group_id}/{kind}.png')
def sample_zone_artifact(case_name: str, group_id: str, kind: str):
    if case_name not in SAMPLES or kind not in {'mask', 'overlay', 'authored'} or not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', group_id):
        raise HTTPException(404)
    result_dir = ROOT / 'output' / SAMPLES[case_name]
    analysis = load_zones(result_dir)
    if not analysis or group_id not in {group['id'] for group in analysis['groups']}:
        raise HTTPException(404)
    path = result_dir / 'zones' / f'group_{group_id}_{kind}.png'
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


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
@app.get('/jobs/{job_id}/analysis')
@app.get('/jobs/{job_id}/cracks')
@app.get('/samples/{case_name}')
def frontend_route():
    return _front_page()
