"""Pitch-aware, reviewable image-based surface damage analysis.

This is a classical-CV baseline, not a trained defect detector or a structural
integrity assessment. All physical areas are calibrated from the panorama's
nominal pixels/mm and must be reviewed against the source imagery.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .review import save_review


ZONES = ('tread', 'groove', 'sprocket_hole', 'embedded_core')
ZONE_CODES = {name: index + 1 for index, name in enumerate(ZONES)}
MODES = ('chunk', 'tear')


@dataclass
class DamageConfig:
    chunk_min_area_mm2: float = 100.0
    tear_min_length_mm: float = 10.0
    tear_max_width_mm: float = 8.0
    tear_min_aspect: float = 4.0
    residual_sigma: float = 3.5
    residual_floor: float = 16.0
    working_pixels_per_mm: float = 2.0
    include_bright_anomalies: bool = False

    def validate(self):
        if not (1 <= self.chunk_min_area_mm2 <= 10000 and 2 <= self.tear_min_length_mm <= 200
                and .5 <= self.tear_max_width_mm <= 50 and 2 <= self.tear_min_aspect <= 20
                and 2 <= self.residual_sigma <= 10 and 5 <= self.residual_floor <= 80
                and .5 <= self.working_pixels_per_mm <= 4
                and isinstance(self.include_bright_anomalies, bool)):
            raise ValueError('손상 분석 임계값이 허용 범위를 벗어났습니다.')


def _read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'전개 이미지를 읽을 수 없습니다: {path}')
    return image


def _period_template(gray: np.ndarray, pitch_px: int):
    height, width = gray.shape
    count = width // pitch_px
    if count < 3:
        raise ValueError('손상 분석에는 최소 3피치가 보이는 전개 사진이 필요합니다.')
    indexes = np.unique(np.linspace(0, count - 1, min(count, 64), dtype=int))
    tiles = np.stack([gray[:, i * pitch_px:(i + 1) * pitch_px] for i in indexes])
    template = np.median(tiles, axis=0).astype(np.uint8)
    spread = np.median(np.abs(tiles.astype(np.int16) - template), axis=0).astype(np.float32)
    return template, spread, count


def _repeat(tile: np.ndarray, width: int):
    return np.tile(tile, (1, (width + tile.shape[1] - 1) // tile.shape[1]))[:, :width]


def _central_hole_band(template: np.ndarray):
    h = template.shape[0]
    smooth = cv2.GaussianBlur(template.astype(np.float32), (0, 0), 4)
    edge = np.abs(np.gradient(np.mean(smooth, axis=1)))
    top_range = slice(round(.3 * h), round(.5 * h))
    bottom_range = slice(round(.5 * h), round(.7 * h))
    top = top_range.start + int(np.argmax(edge[top_range]))
    bottom = bottom_range.start + int(np.argmax(edge[bottom_range]))
    contrast = float(edge[top] + edge[bottom])
    if bottom - top < .04 * h or contrast < .9:
        return round(.42 * h), round(.58 * h), contrast
    return top, bottom, contrast


def _hole_template(template: np.ndarray, top: int, bottom: int):
    h, p = template.shape
    smooth = cv2.GaussianBlur(template, (0, 0), 2.5)
    middle = smooth[top:bottom]
    threshold = min(float(np.percentile(middle, 28)), float(np.median(middle) - 13))
    dark = (smooth < threshold).astype(np.uint8)
    dark[:top] = 0
    dark[bottom:] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    found = np.zeros_like(dark)
    boxes = []
    for i in range(1, count):
        x, y, w, hh, area = stats[i]
        if area < max(24, .001 * h * p) or w < 3 or hh < 3:
            continue
        if area / (w * hh) < .3 or w > .8 * p or hh > .22 * h:
            continue
        found[labels == i] = 1
        boxes.append((x, y, w, hh))
    if boxes:
        found = cv2.morphologyEx(found, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return found, boxes


def _zone_template(template: np.ndarray):
    h, p = template.shape
    top, bottom, contrast = _central_hole_band(template)
    hole, boxes = _hole_template(template, top, bottom)
    zones = np.full((h, p), ZONE_CODES['tread'], np.uint8)
    upper = max(0, top - round(.04 * h))
    lower = min(h, bottom + round(.04 * h))
    zones[upper:lower] = ZONE_CODES['embedded_core']
    # Pitch-locked dark valleys outside the central band are the groove prior.
    smooth = cv2.GaussianBlur(template, (0, 0), 3.2)
    row_median = np.median(smooth, axis=1, keepdims=True)
    row_spread = np.std(smooth.astype(np.float32), axis=1, keepdims=True)
    valley = ((smooth < row_median - np.maximum(15, .52 * row_spread))).astype(np.uint8)
    valley = cv2.morphologyEx(valley, cv2.MORPH_OPEN, np.ones((5, 3), np.uint8))
    valley[upper:lower] = 0
    # Tiny scattered texture is not a meaningful repeated groove.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(valley, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max(12, .00035 * h * p):
            valley[labels == i] = 0
    zones[valley > 0] = ZONE_CODES['groove']
    zones[hole > 0] = ZONE_CODES['sprocket_hole']
    confidence = 'medium' if boxes and contrast >= .9 else 'low'
    return zones, {'hole_components_per_pitch': len(boxes), 'central_band_fraction': [round(upper / h, 3), round(lower / h, 3)],
                   'confidence': confidence}


def _validate_seeds(seeds, image_width, image_height):
    if not isinstance(seeds, list) or len(seeds) > 100:
        raise ValueError('영역 보정은 최대 100개까지 지정할 수 있습니다.')
    cleaned = []
    for seed in seeds:
        if not isinstance(seed, dict) or seed.get('zone') not in ZONES:
            raise ValueError('영역 이름을 확인하세요.')
        points = seed.get('polygon')
        if not isinstance(points, list) or not 3 <= len(points) <= 100:
            raise ValueError('각 영역은 점 3~100개의 다각형이어야 합니다.')
        polygon = []
        for point in points:
            if (not isinstance(point, (list, tuple)) or len(point) != 2 or
                    not all(isinstance(v, (int, float)) and np.isfinite(v) for v in point)):
                raise ValueError('다각형 좌표를 확인하세요.')
            x, y = float(point[0]), float(point[1])
            if not 0 <= x < image_width or not 0 <= y < image_height:
                raise ValueError('다각형이 전개 사진 밖에 있습니다.')
            polygon.append([round(x, 2), round(y, 2)])
        cleaned.append({'zone': seed['zone'], 'polygon': polygon})
    return cleaned


def _apply_seeds(zone_tile, seeds, original_width, original_height, working_width, working_height):
    pitch = zone_tile.shape[1]
    sx, sy = working_width / original_width, working_height / original_height
    for seed in seeds:
        points = np.array(seed['polygon'], np.float32)
        points[:, 0] *= sx
        points[:, 1] *= sy
        points[:, 0] -= np.floor(np.mean(points[:, 0]) / pitch) * pitch
        polygon = np.round(points).astype(np.int32)
        for offset in (-pitch, 0, pitch):
            shifted = polygon + np.array([offset, 0], np.int32)
            cv2.fillPoly(zone_tile, [shifted], ZONE_CODES[seed['zone']])
    return zone_tile


def _component_candidates(binary, residual, zones, mode, ppm, scale_x, scale_y, config):
    n, labels, stats, centers = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    found = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area < max(5, round(2 * ppm * ppm)):
            continue
        component = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        long_px, short_px = max(rect[1]), min(rect[1])
        long_mm = max(long_px, 1) / ppm
        short_mm = max(short_px, 1) / ppm
        area_mm2 = area / (ppm * ppm)
        aspect = long_mm / max(short_mm, .5)
        looks_tear = (long_mm >= config.tear_min_length_mm and
                      short_mm <= config.tear_max_width_mm and aspect >= config.tear_min_aspect)
        if mode == 'chunk':
            if area_mm2 < config.chunk_min_area_mm2 or looks_tear:
                continue
        elif not looks_tear:
            continue
        local_zone = zones[y:y+h, x:x+w][component > 0]
        counts = np.bincount(local_zone, minlength=5)
        zone = ZONES[int(np.argmax(counts[1:]))]
        score = float(np.median(residual[y:y+h, x:x+w][component > 0]))
        # A contour is only a visual selection aid; exact area uses mask pixels.
        approx = cv2.approxPolyDP(contour, max(1.0, .003 * cv2.arcLength(contour, True)), True)
        polygon = [[round((x + int(pt[0][0])) * scale_x), round((y + int(pt[0][1])) * scale_y)] for pt in approx]
        if len(polygon) < 3:
            polygon = [[round(x * scale_x), round(y * scale_y)], [round((x+w) * scale_x), round(y * scale_y)],
                       [round((x+w) * scale_x), round((y+h) * scale_y)], [round(x * scale_x), round((y+h) * scale_y)]]
        key = f'{mode}:{x}:{y}:{w}:{h}:{area}'
        identifier = hashlib.sha1(key.encode()).hexdigest()[:12]
        found.append({'id': identifier, 'mode': mode, 'zone': zone, 'source': 'auto', 'included': True,
                      'reviewed': False, 'auto_reason': 'local_deviation',
                      'area_mm2': round(area_mm2, 2), 'length_mm': round(long_mm, 2), 'width_mm': round(short_mm, 2),
                      'aspect_ratio': round(aspect, 2), 'residual_score': round(score, 2),
                      'bbox_xywh': [round(x * scale_x), round(y * scale_y), round(w * scale_x), round(h * scale_y)],
                      'polygon': polygon, 'zone_area_mm2': {ZONES[z-1]: round(int(counts[z]) / (ppm*ppm), 2) for z in range(1, 5) if counts[z]}})
    return found


def summarize(analysis):
    zone_area = analysis['zone_areas_mm2']
    active_modes = analysis.get('active_modes', list(MODES))
    selected = [d for d in analysis['candidates'] if d['included'] and d['mode'] in active_modes]
    by_zone = {}
    for zone in ZONES:
        defects = [d for d in selected if d['zone'] == zone]
        area = sum(d['zone_area_mm2'].get(zone, 0) for d in selected)
        by_zone[zone] = {'visible_area_mm2': zone_area[zone], 'damaged_area_mm2': round(area, 2),
                         'damage_percent': round(100 * area / zone_area[zone], 3) if zone_area[zone] else 0,
                         'chunk_count': sum(d['mode'] == 'chunk' for d in defects),
                         'tear_count': sum(d['mode'] == 'tear' for d in defects)}
    total_visible = sum(zone_area.values())
    total_damaged = sum(d['area_mm2'] for d in selected)
    analysis['review_status'] = ('manually_reviewed' if analysis['candidates'] and
                                 all(d.get('reviewed', False) for d in analysis['candidates'])
                                 else 'unverified_auto_candidates')
    analysis['summary'] = {'zones': by_zone, 'total': {'visible_area_mm2': round(total_visible, 2),
                           'damaged_area_mm2': round(total_damaged, 2),
                           'damage_percent': round(100 * total_damaged / total_visible, 3) if total_visible else 0,
                           'chunk_count': sum(d['mode'] == 'chunk' for d in selected),
                           'tear_count': sum(d['mode'] == 'tear' for d in selected)},
                           'excluded_count': sum(not d['included'] for d in analysis['candidates']),
                           'pending_review_count': sum(not d.get('reviewed', False) for d in analysis['candidates'])}
    return analysis


def _atomic_json(path: Path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_analysis(result_dir):
    path = Path(result_dir) / 'damage' / 'analysis.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def save_analysis(result_dir, analysis):
    path = Path(result_dir) / 'damage' / 'analysis.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, summarize(analysis))
    return analysis


def _manual_zone_areas(polygon, zone_mask, width, height, work_ppm):
    points = np.array(polygon, np.float32)
    scale_x, scale_y = zone_mask.shape[1] / width, zone_mask.shape[0] / height
    work_points = np.round(points * [scale_x, scale_y]).astype(np.int32)
    raster = np.zeros(zone_mask.shape, np.uint8)
    cv2.fillPoly(raster, [work_points], 1)
    counts = np.bincount(zone_mask[raster > 0], minlength=5)
    zone = ZONES[int(np.argmax(counts[1:]))]
    portions = {ZONES[z-1]: round(int(counts[z]) / (work_ppm ** 2), 2) for z in range(1, 5) if counts[z]}
    return zone, portions


def analyze_damage(result_dir, zone_seeds=None, config=None):
    result_dir = Path(result_dir)
    report_path = result_dir / 'quality_report.json'
    if not report_path.is_file():
        raise ValueError('전개 품질 보고서가 없습니다.')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    settings = report['settings']
    config = config or DamageConfig()
    config.validate()
    ppm_original = float(settings['pixels_per_mm'])
    if not .5 <= ppm_original <= 8:
        raise ValueError('전개 사진의 해상도 정보를 확인하세요.')
    lossless = result_dir / 'panorama.png'
    image = _read_image(lossless if lossless.is_file() else result_dir / 'panorama.jpg')
    original_h, original_w = image.shape[:2]
    work_ppm = min(ppm_original, config.working_pixels_per_mm)
    work_w = round(original_w * work_ppm / ppm_original)
    work_h = round(original_h * work_ppm / ppm_original)
    work = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), max(.6, .45 * work_ppm))
    pitch = round(float(settings['pitch_mm']) * work_ppm)
    if pitch < 5:
        raise ValueError('피치 해상도가 손상 분석에 부족합니다.')
    template, spread, count = _period_template(gray, pitch)
    zones_tile, zone_detection = _zone_template(template)
    seeds = _validate_seeds(zone_seeds or [], original_w, original_h)
    _apply_seeds(zones_tile, seeds, original_w, original_h, work_w, work_h)
    zones = _repeat(zones_tile, work_w)
    expected = _repeat(template, work_w).astype(np.float32)
    deviation = gray.astype(np.float32) - expected
    # Equalize each pitch's broad illumination before finding local anomalies.
    for start in range(0, work_w, pitch):
        end = min(start + pitch, work_w)
        deviation[:, start:end] -= float(np.median(deviation[:, start:end]))
    spread_full = _repeat(spread, work_w)
    threshold = np.maximum(config.residual_floor, config.residual_sigma * np.maximum(spread_full, 3.0))
    structural_edge = cv2.Canny(template, 32, 90)
    structural_edge = cv2.dilate(structural_edge, np.ones((7, 7), np.uint8))
    threshold += _repeat((structural_edge > 0).astype(np.float32) * 24, work_w)
    dark = np.maximum(-deviation - threshold, 0)
    if config.include_bright_anomalies:
        bright = np.maximum(deviation - threshold * 1.25, 0)
        residual = np.maximum(dark, bright).astype(np.float32)
    else:
        residual = dark.astype(np.float32)
    anomalous = (residual > 0).astype(np.uint8)
    # Suppress edge registration artefacts without hiding the central hole.
    zone_edges = cv2.morphologyEx(zones, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    anomalous[zone_edges > 0] = 0
    anomalous[:, :2] = 0
    anomalous[:, -2:] = 0
    chunk = cv2.morphologyEx(anomalous, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    chunk = cv2.morphologyEx(chunk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    tear = cv2.morphologyEx(anomalous, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    sx, sy = original_w / work_w, original_h / work_h
    chunks = _component_candidates(chunk, residual, zones, 'chunk', work_ppm, sx, sy, config)
    accepted_chunk = np.zeros_like(chunk)
    for item in chunks:
        x, y, w, h = item['bbox_xywh']
        x0, y0 = round(x / sx), round(y / sy)
        x1, y1 = min(work_w, round((x + w) / sx)), min(work_h, round((y + h) / sy))
        accepted_chunk[y0:y1, x0:x1] = 1
    tear[accepted_chunk > 0] = 0
    tears = _component_candidates(tear, residual, zones, 'tear', work_ppm, sx, sy, config)
    candidates = sorted(chunks + tears, key=lambda d: (d['bbox_xywh'][0], d['bbox_xywh'][1], d['mode']))
    # An anomaly recurring at the same within-pitch position is likely design
    # texture or registration error. Keep it inspectable but exclude by default.
    original_pitch = float(settings['pitch_mm']) * ppm_original
    for item in candidates:
        x, y, w, h = item['bbox_xywh']
        phase = (x + w / 2) % original_pitch
        center_y = y + h / 2
        occurrences = set()
        for peer in candidates:
            if peer['mode'] != item['mode']:
                continue
            px, py, pw, ph = peer['bbox_xywh']
            peer_phase = (px + pw / 2) % original_pitch
            phase_distance = abs(phase - peer_phase)
            phase_distance = min(phase_distance, original_pitch - phase_distance)
            if phase_distance <= .08 * original_pitch and abs(center_y - (py + ph / 2)) <= .07 * original_h:
                occurrences.add(int((px + pw / 2) // original_pitch))
        if len(occurrences) >= max(4, round(.1 * count)):
            item['included'] = False
            item['auto_reason'] = 'repeated_design_or_registration_pattern'
    previous = load_analysis(result_dir)
    decisions = {d['id']: d['included'] for d in previous['candidates'] if d.get('reviewed')} if previous else {}
    for item in candidates:
        if item['id'] in decisions:
            item['included'] = decisions[item['id']]
            item['reviewed'] = True
    zone_areas = {name: round(int(np.count_nonzero(zones == code)) / (work_ppm ** 2), 2)
                  for name, code in ZONE_CODES.items()}
    analysis = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'image_size_wh': [original_w, original_h], 'nominal_pixels_per_mm': ppm_original,
                'working_pixels_per_mm': work_ppm, 'pitch_px': float(settings['pitch_mm']) * ppm_original,
                'analyzed_pitch_count': count, 'config': asdict(config), 'zone_seeds': seeds,
                'zone_detection': zone_detection, 'zone_areas_mm2': zone_areas,
                'active_modes': previous.get('active_modes', list(MODES)) if previous else list(MODES),
                'candidates': candidates,
                'review_status': 'unverified_auto_candidates',
                'limitations': ['Image-based surface anomaly estimate; manual review is required.',
                                'Nominal pixel/mm calibration does not replace a physical measurement.',
                                'Lighting, dirt, and registration seams may cause false detections.',
                                'Bright anomalies are excluded by default to avoid chalk false positives; enable them explicitly or add manually.',
                                'Occluded surfaces and subsurface damage cannot be evaluated.']}
    folder = result_dir / 'damage'
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / 'zone_mask.png'), zones)
    if previous:
        for manual in (d for d in previous['candidates'] if d.get('source') == 'manual'):
            manual['zone'], manual['zone_area_mm2'] = _manual_zone_areas(
                manual['polygon'], zones, original_w, original_h, work_ppm)
            manual['area_mm2'] = round(sum(manual['zone_area_mm2'].values()), 2)
            analysis['candidates'].append(manual)
        analysis['candidates'].sort(key=lambda d: (d['bbox_xywh'][0], d['bbox_xywh'][1], d['mode']))
    cv2.imwrite(str(folder / 'anomaly_mask.png'), anomalous * 255)
    preview_scale = min(1, 2400 / work_w)
    preview = cv2.resize(work, (round(work_w * preview_scale), round(work_h * preview_scale)), interpolation=cv2.INTER_AREA)
    colors = {1: (93, 145, 26), 2: (196, 138, 43), 3: (31, 101, 220), 4: (151, 69, 162)}
    zone_preview = cv2.resize(zones, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST)
    tint = np.zeros_like(preview)
    for code, color in colors.items():
        tint[zone_preview == code] = color
    cv2.imwrite(str(folder / 'zone_preview.jpg'), cv2.addWeighted(preview, .7, tint, .3, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    zone_colors = np.zeros_like(work)
    for code, color in colors.items():
        zone_colors[zones == code] = color
    save_review(cv2.addWeighted(work, .72, zone_colors, .28, 0), folder / 'zone_review.jpg')
    marked = work.copy()
    for item in candidates:
        points = np.round(np.array(item['polygon'], np.float32) / [sx, sy]).astype(np.int32)
        cv2.polylines(marked, [points], True, (25, 40, 230) if item['mode'] == 'chunk' else (0, 215, 230), 2)
    save_review(marked, folder / 'candidate_review.jpg')
    return save_analysis(result_dir, analysis)


def add_manual_candidate(result_dir, analysis, mode, polygon):
    if mode not in MODES:
        raise ValueError('손상 모드는 chunk 또는 tear여야 합니다.')
    width, height = analysis['image_size_wh']
    validated = _validate_seeds([{'zone': 'tread', 'polygon': polygon}], width, height)[0]['polygon']
    points = np.array(validated, np.float32)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    if w < 2 or h < 2:
        raise ValueError('손상 영역이 너무 작습니다.')
    ppm = analysis['nominal_pixels_per_mm']
    if cv2.contourArea(points) <= 0:
        raise ValueError('손상 영역의 면적을 확인하세요.')
    zone_mask = cv2.imread(str(Path(result_dir) / 'damage' / 'zone_mask.png'), cv2.IMREAD_GRAYSCALE)
    if zone_mask is None:
        raise ValueError('영역 마스크가 없습니다. 분석을 다시 실행하세요.')
    zone, portions = _manual_zone_areas(validated, zone_mask, width, height, analysis['working_pixels_per_mm'])
    area_mm2 = sum(portions.values())
    if area_mm2 <= 0:
        raise ValueError('손상 영역이 분석 해상도에서 너무 작습니다.')
    rect = cv2.minAreaRect(points)
    length, thickness = max(rect[1]) / ppm, min(rect[1]) / ppm
    item = {'id': uuid4().hex[:12], 'mode': mode, 'zone': zone, 'source': 'manual', 'included': True,
            'reviewed': True, 'auto_reason': None,
            'area_mm2': round(area_mm2, 2), 'length_mm': round(length, 2), 'width_mm': round(thickness, 2),
            'aspect_ratio': round(length / max(thickness, .01), 2), 'residual_score': None,
            'bbox_xywh': [x, y, w, h], 'polygon': validated,
            'zone_area_mm2': portions}
    analysis['candidates'].append(item)
    return save_analysis(result_dir, analysis)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze a completed track panorama.')
    parser.add_argument('--result', type=Path, required=True, help='Folder containing panorama.jpg and quality_report.json')
    args = parser.parse_args()
    result = analyze_damage(args.result)
    print(json.dumps({'zone_detection': result['zone_detection'], 'summary': result['summary'],
                      'candidate_count': len(result['candidates'])}, ensure_ascii=False, indent=2))
