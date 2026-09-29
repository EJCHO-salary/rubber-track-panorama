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


DAMAGE_TYPES = {'chunk', 'tear', 'chip_cut'}


def _paths(result_dir):
    folder = Path(result_dir) / 'cracks'
    return folder, folder / 'review.json'


def load_cracks(result_dir):
    _, path = _paths(result_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    ppm = data.get('pixels_per_mm')
    if not ppm:
        zones = load_zones(result_dir)
        ppm = float(zones['nominal_pixels_per_mm']) if zones else 1.
    for candidate in data.get('candidates', []):
        candidate.setdefault('damage_type', None)
        candidate.setdefault('suggested_damage_type', None)
        candidate.setdefault('suggestion_reason', None)
        candidate.setdefault('decision_source', 'manual' if candidate['status'] != 'pending' else None)
        candidate.setdefault('area_mm2', _polygon_area(candidate['polygon']) / ppm ** 2)
    return _summary(data)


def clear_cracks(result_dir):
    """Discard all proposals and review decisions without touching the zone map."""
    _, path = _paths(result_dir)
    path.unlink(missing_ok=True)


def _polygon_area(points):
    if len(points) < 3:
        return 0.
    return float(cv2.contourArea(np.asarray(points, dtype=np.float32)))


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
    data['accepted_by_type'] = {key: sum(item['status'] == 'accepted' and item.get('damage_type') == key
                                      for item in data['candidates']) for key in sorted(DAMAGE_TYPES)}
    data['accepted_by_type']['unclassified'] = sum(item['status'] == 'accepted' and not item.get('damage_type')
                                                   for item in data['candidates'])
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


def _repeated_structure_support(gray, response, anchors, scale_x):
    """Estimate whether darkness recurs at the same location in other pitches.

    The candidate's own pitch is excluded, so one unusual crack cannot vote for
    itself. Two-pitch parity accommodates alternating tread and groove shapes.
    """
    h, w = gray.shape
    positions = np.rint(np.asarray(anchors) / scale_x).astype(int)
    spans = [(index, int(positions[index]), int(positions[index + 1]))
             for index in range(len(positions) - 1)
             if 0 <= positions[index] < positions[index + 1] <= w
             and positions[index + 1] - positions[index] >= 20]
    support = np.zeros((h, w), np.float32)
    for parity in (0, 1):
        strips, locations = [], []
        for index, left, right in spans:
            if index % 2 != parity:
                continue
            darkness = np.uint8((gray[:, left:right] < 125) & (response[:, left:right] > 16))
            normalized = cv2.resize(darkness, (128, h), interpolation=cv2.INTER_NEAREST)
            strips.append(cv2.dilate(normalized, np.ones((5, 9), np.uint8)) > 0)
            locations.append((left, right))
        if len(strips) < 4:
            continue
        stack = np.stack(strips)
        counts = np.sum(stack, axis=0)
        for index, (left, right) in enumerate(locations):
            other_pitches = (counts - stack[index]) / (len(strips) - 1)
            support[:, left:right] = cv2.resize(other_pitches.astype(np.float32), (right-left, h))
    return support


def _periodic_hole_band(gray, anchors, scale_x):
    """Find the central band where pitch anchors repeatedly contain cavities."""
    h, w = gray.shape
    positions = np.rint(np.asarray(anchors) / scale_x).astype(int)
    if len(positions) < 6:
        return None
    pitch = float(np.median(np.diff(positions)))
    radius = max(2, round(pitch * .04))
    midpoints = np.rint((positions[:-1] + positions[1:]) / 2).astype(int)
    anchor_profiles = [np.median(gray[:, x-radius:x+radius], axis=1)
                       for x in positions if radius <= x <= w-radius]
    midpoint_profiles = [np.median(gray[:, x-radius:x+radius], axis=1)
                         for x in midpoints if radius <= x <= w-radius]
    if min(len(anchor_profiles), len(midpoint_profiles)) < 5:
        return None
    contrast = np.median(midpoint_profiles, axis=0) - np.median(anchor_profiles, axis=0)
    contrast = cv2.GaussianBlur(contrast.astype(np.float32)[:, None], (1, 0), 0,
                                sigmaY=max(2, h * .012))[:, 0]
    central = contrast[round(h*.2):round(h*.8)]
    peak = int(np.argmax(central) + round(h*.2))
    if contrast[peak] < 35:
        return None
    limit = max(25, float(contrast[peak]) * .35)
    top, bottom = peak, peak
    while top > 0 and contrast[top-1] > limit:
        top -= 1
    while bottom < h-1 and contrast[bottom+1] > limit:
        bottom += 1
    return (top, bottom) if bottom-top >= h*.08 else None


def _bright_rim_fraction(gray, labels, label, x, y, width, height):
    pad = 5
    left, top = max(0, x-pad), max(0, y-pad)
    right, bottom = min(gray.shape[1], x+width+pad), min(gray.shape[0], y+height+pad)
    component = np.uint8(labels[top:bottom, left:right] == label)
    ring = (cv2.dilate(component, np.ones((5, 5), np.uint8)) > 0) & (component == 0)
    return float(np.mean(gray[top:bottom, left:right][ring] >= 180)) if np.any(ring) else 0.


def _suggest_damage_type(area_mm2, length_mm, elongation, bright_rim):
    """Conservative image-only morphology hint; a user must confirm the mode."""
    width_mm = area_mm2 / max(length_mm, .1)
    if area_mm2 >= 100 and width_mm >= 6 and elongation < 2.5 and bright_rim >= .3:
        return 'chunk', '넓고 불규칙한 결손과 밝은 노출 경계가 함께 보입니다.'
    if area_mm2 >= 10 and width_mm >= 1.5 and bright_rim >= .18:
        return 'chip_cut', '선형 균열보다 폭이 넓고 밝은 절단·뜯김 경계가 보입니다.'
    if length_mm >= 12 and width_mm <= 1.9 and elongation >= 2.2:
        return 'tear', '폭이 좁고 길게 이어진 어두운 균열입니다.'
    return None, None


def _dark_fissure_mask(gray, response, zone_mask, sensitivity):
    """Require a connected near-black core, not merely local edge contrast.

    A per-image low percentile accommodates exposure differences, while a hard
    upper bound prevents chalk marks and ordinary rubber relief from qualifying.
    """
    rubber = gray[zone_mask > 0]
    if rubber.size == 0:
        return np.zeros_like(gray, dtype=np.uint8), 0
    core_limit = int(np.clip(np.percentile(rubber, 1) + 10, 48, 72))
    weak_limit = min({'low': 76, 'normal': 88, 'high': 98}[sensitivity], core_limit + 27)
    contrast_limit = {'low': 45, 'normal': 34, 'high': 25}[sensitivity]
    dark = (gray <= weak_limit) & (response >= contrast_limit) & (zone_mask > 0)
    # A dark line along a bright chalk stroke is often its cast edge, not a fissure.
    chalk_halo = cv2.dilate(np.uint8(gray >= 205), np.ones((5, 5), np.uint8)) > 0
    dark &= ~chalk_halo
    binary = cv2.morphologyEx(np.uint8(dark), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binary, core_limit


def _same_damage_location(first, second):
    ax, ay, aw, ah = first['bbox']
    bx, by, bw, bh = second['bbox']
    overlap = max(0., min(ax + aw, bx + bw) - max(ax, bx)) * max(0., min(ay + ah, by + bh) - max(ay, by))
    smaller = min(aw * ah, bw * bh)
    return smaller > 0 and overlap / smaller >= .5


def _inside_authored(candidate, authored, scale_x, scale_y):
    """Keep a prior review only when its displayed area lies in drawn zones."""
    points = np.asarray(candidate['polygon'], dtype=np.float32)
    if len(points) < 3:
        return False
    contour = np.rint(points / [scale_x, scale_y]).astype(np.int32)
    x, y, width, height = cv2.boundingRect(contour)
    if x < 0 or y < 0 or x + width > authored.shape[1] or y + height > authored.shape[0]:
        return False
    footprint = np.zeros((height, width), np.uint8)
    cv2.fillPoly(footprint, [contour - [x, y]], 1)
    return bool(np.any(footprint) and np.all(authored[y:y+height, x:x+width][footprint > 0] > 0))


def propose_cracks(result_dir, group_id, sensitivity='normal'):
    """Regenerate dark-core candidates while retaining explicit human review."""
    if sensitivity not in {'low', 'normal', 'high'}:
        raise ValueError('민감도는 low, normal, high 중 하나여야 합니다.')
    zones = load_zones(result_dir)
    if not zones:
        raise ValueError('먼저 영역을 지정해 주세요.')
    group = next((item for item in zones['groups'] if item['id'] == group_id), None)
    if not group:
        raise ValueError('분류 체계를 찾을 수 없습니다.')
    previous = load_cracks(result_dir)
    mask = cv2.imread(str(Path(result_dir) / 'zones' / f'group_{group_id}_mask.png'), cv2.IMREAD_GRAYSCALE)
    authored = cv2.imread(str(Path(result_dir) / 'zones' / f'group_{group_id}_authored.png'), cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(str(Path(result_dir) / 'panorama.png'), cv2.IMREAD_GRAYSCALE)
    if mask is None or authored is None or original is None or mask.shape != authored.shape:
        raise ValueError('전개 사진 또는 지정 영역 마스크를 읽을 수 없습니다.')
    h, w = mask.shape
    gray = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
    # Black-hat finds dark fissures against a locally brighter rubber surface.
    # Two kernel widths tolerate both hairline and wider surface cracks.
    response = np.maximum(
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))))
    scale_x, scale_y = original.shape[1] / w, original.shape[0] / h
    repeated_support = _repeated_structure_support(gray, response, zones['pitch_anchors_x'], scale_x)
    hole_band = _periodic_hole_band(gray, zones['pitch_anchors_x'], scale_x)
    anchor_positions = np.asarray(zones['pitch_anchors_x'], dtype=float) / scale_x
    pitch_spacing = float(np.median(np.diff(anchor_positions)))
    response = _pattern_residual(response, zones['pitch_anchors_x'], scale_x)
    # The classification mask fills all gaps with a default section. Only the
    # separately rasterized, user-authored polygons are valid crack territory.
    detection_mask = np.where(authored > 0, mask, 0).astype(np.uint8)
    binary, core_limit = _dark_fissure_mask(gray, response, detection_mask, sensitivity)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    ppm = float(zones['nominal_pixels_per_mm'])
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = map(int, stats[label])
        if area < 7 or area > 1400 or max(bw, bh) < 9 or bw > w * .15 or bh > h * .4:
            continue
        roi = labels[y:y+bh, x:x+bw] == label
        structural_score = float(np.mean(repeated_support[y:y+bh, x:x+bw][roi]))
        if structural_score >= .75:
            continue
        # A horizontal dark lip recurring at the top/bottom of the central
        # sprocket opening is image geometry, not a rubber fissure.
        if hole_band and structural_score >= .55 and bw > bh * 1.8:
            center_x, center_y = x + bw / 2, y + bh / 2
            near_anchor = np.min(np.abs(anchor_positions - center_x)) < pitch_spacing * .32
            near_rim = min(abs(center_y - hole_band[0]), abs(center_y - hole_band[1])) < max(5, h * .025)
            if near_anchor and near_rim:
                continue
        core = np.uint8(roi & (gray[y:y+bh, x:x+bw] <= core_limit))
        core_count, _, core_stats, _ = cv2.connectedComponentsWithStats(core, 8)
        longest_core = max((max(int(sw), int(sh)) for _, _, sw, sh, pixels in core_stats[1:]
                            if pixels >= 4), default=0)
        if longest_core < 5 or np.count_nonzero(core) < max(4, int(area * .06)):
            continue
        yy, xx = np.nonzero(roi)
        xy = np.column_stack((xx, yy)).astype(np.float32)
        covariance = np.cov(xy, rowvar=False)
        eigen = np.linalg.eigvalsh(covariance)
        elongation = float(np.sqrt((eigen[1] + 1) / (eigen[0] + 1)))
        if elongation < 1.7 and not (area >= 35 and np.count_nonzero(core) >= area * .35):
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
        if not _inside_authored({'polygon': polygon}, authored, scale_x, scale_y):
            continue
        area_mm2 = round(_polygon_area(polygon) / ppm ** 2, 2)
        suggestion, reason = _suggest_damage_type(
            area_mm2, length_mm, elongation, _bright_rim_fraction(gray, labels, label, x, y, bw, bh))
        candidates.append({'id': uuid4().hex[:12], 'section_id': group['sections'][section_code-1]['id'],
                           'polygon': polygon, 'status': 'pending', 'source': 'automatic', 'damage_type': None,
                           'suggested_damage_type': suggestion, 'suggestion_reason': reason,
                           'repeated_structure_score': round(structural_score, 2),
                           'decision_source': None,
                           'length_mm': round(length_mm, 1), 'contrast': round(contrast, 1), 'score': score,
                           'area_mm2': area_mm2,
                           'bbox': [round(x*scale_x, 1), round(y*scale_y, 1), round(bw*scale_x, 1), round(bh*scale_y, 1)]})
    candidates.sort(key=lambda item: item['score'] * max(1, item['length_mm']) ** .5, reverse=True)
    candidates = candidates[:400]
    if previous and previous.get('group_id') == group_id and previous.get('zone_created_at') == zones['created_at']:
        reviewed = [item for item in previous['candidates'] if item['status'] != 'pending'
                    and _inside_authored(item, authored, scale_x, scale_y)]
        candidates = reviewed + [item for item in candidates
                                 if not any(_same_damage_location(item, saved) for saved in reviewed)]
    data = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'zone_created_at': zones['created_at'],
            'group_id': group_id, 'sensitivity': sensitivity, 'image_size_wh': zones['image_size_wh'],
            'pixels_per_mm': ppm,
            'sections': [{'id': section['id'], 'name': section['name'], 'color': section['color']} for section in group['sections']],
            'candidates': candidates}
    return _write(result_dir, _summary(data))


def review_crack(result_dir, candidate_id, status, damage_type=None):
    return review_cracks(result_dir, [candidate_id], status, damage_type)


def review_cracks(result_dir, candidate_ids, status, damage_type=None, auto_larger=False):
    if status not in {'pending', 'accepted', 'excluded'}:
        raise ValueError('검토 상태가 올바르지 않습니다.')
    if damage_type is not None and damage_type not in DAMAGE_TYPES:
        raise ValueError('손상 유형이 올바르지 않습니다.')
    if not isinstance(candidate_ids, list) or not candidate_ids or len(candidate_ids) > 400 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('검토할 후보를 1~400개 선택해 주세요.')
    if auto_larger and (status != 'accepted' or damage_type is None):
        raise ValueError('크기 기준 자동 채택에는 손상 유형이 필요합니다.')
    data = load_cracks(result_dir)
    if not data:
        raise ValueError('먼저 크랙 후보를 생성해 주세요.')
    by_id = {item['id']: item for item in data['candidates']}
    if any(candidate_id not in by_id for candidate_id in candidate_ids):
        raise ValueError('크랙 후보를 찾을 수 없습니다.')
    selected = [by_id[candidate_id] for candidate_id in candidate_ids]
    threshold = min(item['area_mm2'] for item in selected)
    if auto_larger and threshold <= 0:
        raise ValueError('선택한 후보의 표시 면적이 없어 크기 기준 자동 채택을 적용할 수 없습니다.')
    for candidate in selected:
        candidate['status'] = status
        if damage_type is not None:
            candidate['damage_type'] = damage_type
        candidate['decision_source'] = None if status == 'pending' else 'manual'
    auto_count = 0
    if auto_larger:
        for candidate in data['candidates']:
            if candidate['id'] in candidate_ids or candidate['source'] != 'automatic' or candidate['status'] != 'pending':
                continue
            if candidate['area_mm2'] >= threshold:
                candidate['status'] = 'accepted'
                candidate['damage_type'] = damage_type
                candidate['decision_source'] = 'auto'
                auto_count += 1
    data['last_review'] = {'selected': len(selected), 'auto_accepted': auto_count}
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
    authored = cv2.imread(str(Path(result_dir) / 'zones' / f"group_{data['group_id']}_authored.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or authored is None or mask.shape != authored.shape:
        raise ValueError('지정 영역 마스크를 읽을 수 없습니다.')
    h, w = mask.shape
    sx, sy = w / width, h / height
    work_points = np.rint(path * [sx, sy]).astype(np.int32)
    centerline = np.zeros_like(mask)
    cv2.polylines(centerline, [work_points], False, 255, 1)
    if np.any((centerline > 0) & (authored == 0)):
        raise ValueError('수동 균열 경로는 지정된 영역 안에 그려 주세요.')
    raster = np.zeros_like(mask)
    cv2.polylines(raster, [work_points], False, 255, 3)
    raster[authored == 0] = 0
    section_pixels = mask[raster > 0]
    code = int(np.bincount(section_pixels, minlength=len(data['sections'])+1)[1:].argmax() + 1)
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1., True).reshape(-1, 2)
    x, y, bw, bh = cv2.boundingRect(contour)
    candidate = {'id': uuid4().hex[:12], 'section_id': data['sections'][code-1]['id'],
                 'polygon': [[round(float(px)/sx, 1), round(float(py)/sy, 1)] for px, py in contour],
                 'status': 'accepted', 'source': 'manual', 'damage_type': None, 'decision_source': 'manual',
                 'length_mm': round(length_px / zones['nominal_pixels_per_mm'], 1),
                 'area_mm2': round(_polygon_area(contour) / (sx * sy * zones['nominal_pixels_per_mm'] ** 2), 2),
                 'contrast': 0., 'score': 1.,
                 'bbox': [round(x/sx, 1), round(y/sy, 1), round(bw/sx, 1), round(bh/sy, 1)]}
    if not _inside_authored(candidate, authored, 1 / sx, 1 / sy):
        raise ValueError('수동 균열 경로는 지정된 영역 안에 그려 주세요.')
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
