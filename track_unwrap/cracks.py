"""Reviewable dark-fissure proposals on a human-authored region map.

These are image candidates, not diagnoses. Missing rubber (chips/chunks), depth,
and steel exposure cannot be reliably inferred from this overhead image.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .zones import load_zones


def _paths(result_dir):
    folder = Path(result_dir) / 'cracks'
    return folder, folder / 'review.json'


def load_cracks(result_dir):
    _, path = _paths(result_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _write(result_dir, data):
    folder, path = _paths(result_dir)
    folder.mkdir(exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return data


def _summary(data):
    sections = {item['id']: {'proposed': 0, 'accepted': 0, 'excluded': 0, 'length_mm': 0.}
                for item in data['sections']}
    for item in data['candidates']:
        section = sections[item['section_id']]
        section[{'pending': 'proposed', 'accepted': 'accepted', 'excluded': 'excluded'}[item['status']]] += 1
        if item['status'] == 'accepted':
            section['length_mm'] += item['length_mm']
    for section in sections.values():
        section['length_mm'] = round(section['length_mm'], 1)
    data['summary'] = sections
    data['totals'] = {key: round(sum(value[key] for value in sections.values()), 1) if key == 'length_mm'
                      else sum(value[key] for value in sections.values())
                      for key in ('proposed', 'accepted', 'excluded', 'length_mm')}
    return data


def _pattern_residual(response, anchors, scale_x):
    """Suppress edges that repeat at the same position in a two-pitch motif."""
    h, w = response.shape
    positions = np.rint(np.asarray(anchors) / scale_x).astype(int)
    spans = [(i, max(0, positions[i]), min(w, positions[i+1])) for i in range(len(positions)-1)
             if 0 <= positions[i] < positions[i+1] <= w and positions[i+1]-positions[i] >= 20]
    if len(spans) < 6:
        return response
    templates = {}
    for parity in (0, 1):
        strips = [cv2.resize(response[:, left:right], (128, h), interpolation=cv2.INTER_LINEAR)
                  for index, left, right in spans if index % 2 == parity]
        if len(strips) >= 3:
            templates[parity] = np.median(np.stack(strips), axis=0).astype(np.uint8)
    residual = np.zeros_like(response)
    for index, left, right in spans:
        template = templates.get(index % 2)
        if template is None:
            continue
        baseline = cv2.resize(template, (right-left, h), interpolation=cv2.INTER_LINEAR)
        residual[:, left:right] = cv2.subtract(response[:, left:right], baseline)
    return residual


def propose_cracks(result_dir, group_id, sensitivity='normal'):
    """Generate high-recall, deliberately unclassified crack candidates."""
    if sensitivity not in {'low', 'normal', 'high'}:
        raise ValueError('민감도는 low, normal, high 중 하나여야 합니다.')
    zones = load_zones(result_dir)
    if not zones:
        raise ValueError('먼저 영역을 지정해 주세요.')
    group = next((item for item in zones['groups'] if item['id'] == group_id), None)
    if not group:
        raise ValueError('분류 체계를 찾을 수 없습니다.')
    mask = cv2.imread(str(Path(result_dir) / 'zones' / f'group_{group_id}_mask.png'), cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(str(Path(result_dir) / 'panorama.png'), cv2.IMREAD_GRAYSCALE)
    if mask is None or original is None:
        raise ValueError('전개 사진 또는 영역 마스크를 읽을 수 없습니다.')
    h, w = mask.shape
    gray = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
    # Black-hat finds dark fissures against a locally brighter rubber surface.
    # Two kernel widths tolerate both hairline and wider surface cracks.
    response = np.maximum(
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))))
    scale_x, scale_y = original.shape[1] / w, original.shape[0] / h
    response = _pattern_residual(response, zones['pitch_anchors_x'], scale_x)
    threshold = {'low': 58, 'normal': 43, 'high': 32}[sensitivity]
    binary = np.uint8(response >= threshold)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    ppm = float(zones['nominal_pixels_per_mm'])
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = map(int, stats[label])
        if area < 7 or area > 1400 or max(bw, bh) < 9 or bw > w * .15 or bh > h * .4:
            continue
        roi = labels[y:y+bh, x:x+bw] == label
        yy, xx = np.nonzero(roi)
        xy = np.column_stack((xx, yy)).astype(np.float32)
        covariance = np.cov(xy, rowvar=False)
        eigen = np.linalg.eigvalsh(covariance)
        elongation = float(np.sqrt((eigen[1] + 1) / (eigen[0] + 1)))
        if elongation < 2.0:
            continue
        section_pixels = mask[y:y+bh, x:x+bw][roi]
        section_code = int(np.bincount(section_pixels, minlength=len(group['sections'])+1)[1:].argmax() + 1)
        if not 1 <= section_code <= len(group['sections']):
            continue
        component = np.uint8(roi) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        outline = max(contours, key=cv2.contourArea)
        outline = cv2.approxPolyDP(outline, 1.2, True).reshape(-1, 2)
        if len(outline) < 3:
            continue
        contrast = float(np.mean(response[y:y+bh, x:x+bw][roi]))
        length_px = float(np.sqrt(eigen[1]) * 3.5)
        length_mm = length_px * np.sqrt(scale_x * scale_y) / ppm
        score = round(min(1., (contrast / 105) * min(elongation / 4, 1.)), 3)
        polygon = [[round((x + float(px)) * scale_x, 1), round((y + float(py)) * scale_y, 1)] for px, py in outline]
        candidates.append({'id': uuid4().hex[:12], 'section_id': group['sections'][section_code-1]['id'],
                           'polygon': polygon, 'status': 'pending', 'source': 'automatic',
                           'length_mm': round(length_mm, 1), 'contrast': round(contrast, 1), 'score': score,
                           'bbox': [round(x*scale_x, 1), round(y*scale_y, 1), round(bw*scale_x, 1), round(bh*scale_y, 1)]})
    candidates.sort(key=lambda item: item['score'] * max(1, item['length_mm']) ** .5, reverse=True)
    candidates = candidates[:400]
    data = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'zone_created_at': zones['created_at'],
            'group_id': group_id, 'sensitivity': sensitivity, 'image_size_wh': zones['image_size_wh'],
            'sections': [{'id': section['id'], 'name': section['name'], 'color': section['color']} for section in group['sections']],
            'candidates': candidates}
    return _write(result_dir, _summary(data))


def review_crack(result_dir, candidate_id, status):
    if status not in {'pending', 'accepted', 'excluded'}:
        raise ValueError('검토 상태가 올바르지 않습니다.')
    data = load_cracks(result_dir)
    if not data:
        raise ValueError('먼저 크랙 후보를 생성해 주세요.')
    candidate = next((item for item in data['candidates'] if item['id'] == candidate_id), None)
    if not candidate:
        raise ValueError('크랙 후보를 찾을 수 없습니다.')
    candidate['status'] = status
    return _write(result_dir, _summary(data))


def add_manual_crack(result_dir, points):
    data = load_cracks(result_dir)
    zones = load_zones(result_dir)
    if not data or not zones or data['zone_created_at'] != zones['created_at']:
        raise ValueError('현재 영역에서 크랙 후보를 먼저 생성해 주세요.')
    if not isinstance(points, list) or not 2 <= len(points) <= 100:
        raise ValueError('균열 경로에 점을 2~100개 지정해 주세요.')
    try:
        path = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('균열 경로 좌표를 확인해 주세요.') from exc
    width, height = zones['image_size_wh']
    if path.shape != (len(points), 2) or not np.isfinite(path).all() or np.any(path < 0) or np.any(path[:, 0] >= width) or np.any(path[:, 1] >= height):
        raise ValueError('균열 경로 좌표를 확인해 주세요.')
    length_px = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if length_px < 3:
        raise ValueError('균열 경로가 너무 짧습니다.')
    mask = cv2.imread(str(Path(result_dir) / 'zones' / f"group_{data['group_id']}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError('영역 마스크를 읽을 수 없습니다.')
    h, w = mask.shape
    sx, sy = w / width, h / height
    raster = np.zeros_like(mask)
    work_points = np.rint(path * [sx, sy]).astype(np.int32)
    cv2.polylines(raster, [work_points], False, 255, 3)
    section_pixels = mask[raster > 0]
    code = int(np.bincount(section_pixels, minlength=len(data['sections'])+1)[1:].argmax() + 1)
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1., True).reshape(-1, 2)
    x, y, bw, bh = cv2.boundingRect(contour)
    candidate = {'id': uuid4().hex[:12], 'section_id': data['sections'][code-1]['id'],
                 'polygon': [[round(float(px)/sx, 1), round(float(py)/sy, 1)] for px, py in contour],
                 'status': 'accepted', 'source': 'manual', 'length_mm': round(length_px / zones['nominal_pixels_per_mm'], 1),
                 'contrast': 0., 'score': 1.,
                 'bbox': [round(x/sx, 1), round(y/sy, 1), round(bw/sx, 1), round(bh/sy, 1)]}
    data['candidates'].insert(0, candidate)
    return _write(result_dir, _summary(data))


def delete_manual_crack(result_dir, candidate_id):
    data = load_cracks(result_dir)
    if not data:
        raise ValueError('크랙 후보를 찾을 수 없습니다.')
    candidate = next((item for item in data['candidates'] if item['id'] == candidate_id), None)
    if not candidate or candidate['source'] != 'manual':
        raise ValueError('수동으로 그린 균열만 삭제할 수 있습니다.')
    data['candidates'] = [item for item in data['candidates'] if item['id'] != candidate_id]
    return _write(result_dir, _summary(data))
