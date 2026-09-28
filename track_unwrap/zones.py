"""User-authored, independent region layers on an unwrapped panorama.

No region or damage label is inferred here. GrabCut only proposes an edge for a
polygon that a person has already drawn; it never writes classification data.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ID = re.compile(r'^[a-zA-Z0-9_-]{1,40}$')
COLOR = re.compile(r'^#[0-9a-fA-F]{6}$')


def _preset():
    sections = [
        {'id': 'unassigned', 'name': '미지정', 'color': '#9ca9a2', 'repeat_mode': 'independent'},
        {'id': 'tread', 'name': '트레드', 'color': '#6f9e62', 'repeat_mode': 'independent'},
        {'id': 'groove', 'name': '골부', 'color': '#d2a44d', 'repeat_mode': 'independent'},
        {'id': 'embedded_core', 'name': '심금 매설 중앙부', 'color': '#a67db0', 'repeat_mode': 'independent'},
        {'id': 'sprocket_hole', 'name': '스프라켓 홀', 'color': '#dc7855', 'repeat_mode': 'independent'},
    ]
    return [{'id': 'geometry', 'name': '형상별 구분', 'default_section_id': 'unassigned', 'sections': sections}]


def _legacy_manual_shapes(result_dir, pitch_px):
    """Preserve only human-drawn polygons from the retired analysis format."""
    path = Path(result_dir) / 'damage' / 'analysis.json'
    if not path.is_file():
        return []
    try:
        seeds = json.loads(path.read_text(encoding='utf-8')).get('zone_seeds', [])
    except (OSError, ValueError):
        return []
    allowed = {'tread', 'groove', 'embedded_core', 'sprocket_hole'}
    migrated = []
    for seed in seeds:
        if not isinstance(seed, dict) or seed.get('zone') not in allowed:
            continue
        polygon = seed.get('polygon')
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        try:
            width = max(float(point[0]) for point in polygon) - min(float(point[0]) for point in polygon)
        except (TypeError, ValueError, IndexError):
            continue
        pitches = max(1, int(np.ceil(width / pitch_px)))
        migrated.append({'id': uuid4().hex[:12], 'group_id': 'geometry', 'section_id': seed['zone'],
                         'polygon': polygon, 'repeat': True, 'repeat_pitches': pitches})
    return migrated


def _metadata(result_dir):
    result_dir = Path(result_dir)
    report = json.loads((result_dir / 'quality_report.json').read_text(encoding='utf-8'))
    settings = report['settings']
    image_path = result_dir / 'panorama.png'
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('전개 이미지를 읽을 수 없습니다.')
    width, height = image.shape[1], image.shape[0]
    nominal_ppm = float(settings.get('pixels_per_mm') or height / float(settings['width_mm']))
    pitch_px = float(settings['pitch_mm']) * nominal_ppm
    if pitch_px <= 0:
        raise ValueError('피치가 올바르지 않습니다.')
    scale = min(1., (2_000_000 / (width * height)) ** .5)
    if scale < 1:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    return image, (width, height), pitch_px, nominal_ppm


def _validate(groups, shapes, width, height, pitch_px, max_pitch_px=None):
    if not isinstance(groups, list) or len(groups) > 16:
        raise ValueError('분류 체계는 최대 16개입니다.')
    if not isinstance(shapes, list) or len(shapes) > 512:
        raise ValueError('형상은 최대 512개입니다.')
    clean_groups, references, used = [], {}, set()
    for group in groups:
        if not isinstance(group, dict) or not ID.fullmatch(str(group.get('id', ''))) or group['id'] in used:
            raise ValueError('분류 체계 ID가 올바르지 않거나 중복되었습니다.')
        used.add(group['id'])
        name = str(group.get('name', '')).strip()
        sections = group.get('sections')
        if not 1 <= len(name) <= 80 or not isinstance(sections, list) or not 1 <= len(sections) <= 32:
            raise ValueError('분류 체계 이름 또는 세부 섹션 수를 확인하세요.')
        seen, clean_sections = set(), []
        for section in sections:
            if not isinstance(section, dict) or not ID.fullmatch(str(section.get('id', ''))) or section['id'] in seen:
                raise ValueError('세부 섹션 ID가 올바르지 않거나 중복되었습니다.')
            seen.add(section['id'])
            label = str(section.get('name', '')).strip()
            color = section.get('color')
            if not 1 <= len(label) <= 80 or not isinstance(color, str) or not COLOR.fullmatch(color):
                raise ValueError('세부 섹션 이름과 색상을 확인하세요.')
            repeat_mode = section.get('repeat_mode', 'independent')
            if repeat_mode not in ('examples', 'independent'):
                raise ValueError('반복 방식이 올바르지 않습니다.')
            clean_sections.append({'id': section['id'], 'name': label, 'color': color.lower(),
                                   'repeat_mode': repeat_mode})
        default = group.get('default_section_id')
        if default not in seen:
            raise ValueError('빈 영역을 채울 기본 섹션을 선택하세요.')
        clean_groups.append({'id': group['id'], 'name': name, 'default_section_id': default, 'sections': clean_sections})
        references[group['id']] = seen
    clean_shapes, shape_ids = [], set()
    for shape in shapes:
        if not isinstance(shape, dict):
            raise ValueError('형상 자료가 올바르지 않습니다.')
        group_id, section_id = shape.get('group_id'), shape.get('section_id')
        if group_id not in references or section_id not in references[group_id]:
            raise ValueError('삭제된 분류 체계나 섹션을 참조하는 형상이 있습니다.')
        shape_id = shape.get('id') or uuid4().hex[:12]
        if not ID.fullmatch(str(shape_id)) or shape_id in shape_ids:
            raise ValueError('형상 ID가 올바르지 않거나 중복되었습니다.')
        shape_ids.add(shape_id)
        points = shape.get('polygon')
        if not isinstance(points, list) or not 3 <= len(points) <= 150:
            raise ValueError('형상은 3~150개의 점으로 그려 주세요.')
        try:
            polygon = [[float(x), float(y)] for x, y in points]
        except (TypeError, ValueError) as exc:
            raise ValueError('형상의 좌표가 올바르지 않습니다.') from exc
        coords = np.asarray(polygon, dtype=np.float64)
        if not np.isfinite(coords).all() or np.any(coords[:, 0] < 0) or np.any(coords[:, 0] > width) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] > height):
            raise ValueError('형상이 사진 경계를 벗어났습니다.')
        if abs(cv2.contourArea(coords.astype(np.float32))) < 4:
            raise ValueError('형상 면적이 너무 작습니다.')
        repeat = shape.get('repeat', True)
        repeat_pitches = shape.get('repeat_pitches', 1)
        if not isinstance(repeat, bool):
            raise ValueError('피치 반복 설정이 올바르지 않습니다.')
        if isinstance(repeat_pitches, bool) or not isinstance(repeat_pitches, int) or not 1 <= repeat_pitches <= 2000:
            raise ValueError('반복 간격은 1~2000피치의 정수로 입력하세요.')
        if repeat and np.ptp(coords[:, 0]) > (max_pitch_px or pitch_px) * repeat_pitches * 1.1:
            raise ValueError('반복 형상은 설정한 반복 간격 안에서 그려 주세요.')
        clean_shapes.append({'id': shape_id, 'group_id': group_id, 'section_id': section_id,
                             'polygon': polygon, 'repeat': repeat, 'repeat_pitches': repeat_pitches})
    return clean_groups, clean_shapes


def _pitch_anchors(image, original_size, pitch_px):
    """Locate the recurring dark center openings in panorama coordinates."""
    width, height = original_size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    working_pitch = pitch_px * w / width
    if working_pitch < 16 or w / working_pitch < 3:
        return np.arange(-pitch_px, width + 2*pitch_px, pitch_px), 'nominal'
    signal = gray[round(h*.44):round(h*.54)].mean(axis=0)
    signal = gaussian_filter1d(signal, sigma=max(2, working_pitch*.05))
    contrast = np.percentile(signal, 90) - np.percentile(signal, 10)
    peaks, _ = find_peaks(-signal, distance=max(3, round(working_pitch*.66)),
                          prominence=max(5, contrast*.15))
    if (len(peaks) < 3 or abs(len(peaks) - w/working_pitch) > max(3, w/working_pitch*.2)
            or np.any(np.diff(peaks) < working_pitch*.6)
            or np.any(np.diff(peaks) > working_pitch*1.5)):
        return np.arange(-pitch_px, width + 2*pitch_px, pitch_px), 'nominal'
    return peaks.astype(float) * width / w, 'image'


def _to_pitch(x, anchors):
    x = np.asarray(x, dtype=float)
    indices = np.arange(len(anchors), dtype=float)
    values = np.interp(x, anchors, indices)
    return np.where(x < anchors[0], (x-anchors[0])/(anchors[1]-anchors[0]),
                    np.where(x > anchors[-1], len(anchors)-1+(x-anchors[-1])/(anchors[-1]-anchors[-2]), values))


def _from_pitch(pitch, anchors):
    pitch = np.asarray(pitch, dtype=float)
    indices = np.arange(len(anchors), dtype=float)
    values = np.interp(pitch, indices, anchors)
    return np.where(pitch < 0, anchors[0]+pitch*(anchors[1]-anchors[0]),
                    np.where(pitch > len(anchors)-1, anchors[-1]+(pitch-len(anchors)+1)*(anchors[-1]-anchors[-2]), values))


def suggest_half_turn(result_dir, polygon, repeat_pitches):
    """Rank 180-degree counterparts in measured pitch space; never save a zone."""
    image, (width, height), pitch_px, _ = _metadata(result_dir)
    if isinstance(repeat_pitches, bool) or not isinstance(repeat_pitches, int) or not 1 <= repeat_pitches <= 8:
        raise ValueError('대칭 제안은 1~8피치 반복 형상에서 사용할 수 있습니다.')
    if not isinstance(polygon, list) or not 3 <= len(polygon) <= 150:
        raise ValueError('먼저 도형을 닫아 주세요.')
    try:
        pts = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('형상 좌표를 확인하세요.') from exc
    if pts.shape != (len(polygon), 2) or not np.isfinite(pts).all() or np.any(pts < 0) or np.any(pts[:, 0] > width) or np.any(pts[:, 1] > height):
        raise ValueError('형상 좌표가 사진을 벗어났습니다.')
    if np.ptp(pts[:, 1]) < height * .035:
        raise ValueError('대칭을 판단하기에 도형 높이가 너무 작습니다.')
    center_y = float(pts[:, 1].mean())
    if abs(center_y - height / 2) < height * .12:
        raise ValueError('중앙 띠의 도형은 반대편 대응을 안정적으로 판단할 수 없습니다.')
    if pts[:, 1].min() < height / 2 < pts[:, 1].max():
        raise ValueError('중앙선을 가로지르는 도형은 180° 대응을 제안할 수 없습니다.')
    anchors, source = _pitch_anchors(image, (width, height), pitch_px)
    period = repeat_pitches
    px_per_pitch = 48
    phase_count = int(np.floor(float(_to_pitch(width, anchors))))
    tile_width = period * px_per_pitch
    tiles = phase_count // period
    if tiles < 3:
        return {'candidates': [], 'pitch_anchor_source': source, 'reason': '비교할 반복 구간이 부족합니다.'}
    # Canonical x coordinates remove stitching drift before assessing symmetry.
    phases = np.arange(tiles * tile_width, dtype=np.float32) / px_per_pitch
    source_x = (_from_pitch(phases, anchors) * image.shape[1] / width).astype(np.float32)
    source_y = np.arange(image.shape[0], dtype=np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canonical = cv2.remap(gray, np.broadcast_to(source_x, (image.shape[0], len(source_x))),
                          np.broadcast_to(source_y[:, None], (image.shape[0], len(source_x))),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sy = image.shape[0] / height
    if center_y > height / 2:
        y0 = max(0, round((height - pts[:, 1].max()) * sy - image.shape[0] * .03))
        y1 = min(image.shape[0] // 2, round((height - pts[:, 1].min()) * sy + image.shape[0] * .03))
    else:
        y0 = max(0, round(pts[:, 1].min() * sy - image.shape[0] * .03))
        y1 = min(image.shape[0] // 2, round(pts[:, 1].max() * sy + image.shape[0] * .03))
    upper = canonical[y0:y1]
    lower = canonical[image.shape[0]-y1:image.shape[0]-y0]
    if upper.shape != lower.shape or min(upper.shape) < 12:
        return {'candidates': [], 'pitch_anchor_source': source, 'reason': '대응 영역이 사진 안에 충분히 보이지 않습니다.'}
    def edge_tiles(band):
        band = band.astype(np.float32)
        dx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(dx, dy)
        return np.median(edge.reshape(edge.shape[0], tiles, tile_width), axis=1)
    top = edge_tiles(upper)[::-1, ::-1]
    bottom = edge_tiles(lower)
    top = (top - top.mean()) / (top.std() + 1e-6)
    bottom = (bottom - bottom.mean()) / (bottom.std() + 1e-6)
    scores = np.asarray([float(np.mean(top * np.roll(bottom, offset, axis=1))) for offset in range(tile_width)])
    if float(np.max(scores)) < .15:
        return {'candidates': [], 'pitch_anchor_source': source,
                'reason': '반대편에서 반복되는 180° 대응 패턴을 충분히 찾지 못했습니다.'}
    candidates = []
    selected_offsets = []
    source_pitch = _to_pitch(pts[:, 0], anchors)
    source_center_pitch = float(source_pitch.mean())
    for offset in np.argsort(scores)[::-1]:
        if any(min(abs(int(offset)-used), tile_width-abs(int(offset)-used)) < px_per_pitch * .18 for used in selected_offsets):
            continue
        # A shift in the folded image implies p_source + p_target = period - shift.
        phase_sum = period - int(offset) / px_per_pitch
        shift_cycles = round((2 * source_center_pitch - phase_sum) / period)
        target_pitch = phase_sum - source_pitch + shift_cycles * period
        target_x = _from_pitch(target_pitch, anchors)
        target_y = height - pts[:, 1]
        if np.any(target_x < 0) or np.any(target_x > width):
            continue
        candidates.append({'polygon': [[round(float(x), 2), round(float(y), 2)] for x, y in zip(target_x, target_y)],
                           'score': round(float(scores[offset]), 3), 'phase_pitches': round(float(phase_sum % period), 3)})
        selected_offsets.append(int(offset))
        if len(candidates) == 3:
            break
    return {'candidates': candidates, 'pitch_anchor_source': source,
            'reason': None if candidates else '사진 안에 표시할 대칭 후보가 없습니다.'}


def _draw(mask, authored, polygon, code, repeat, period_pitches, anchors, original_size, placements=None, phase_pitch=None):
    width, height = original_size
    sx, sy = mask.shape[1] / width, mask.shape[0] / height
    pts = np.asarray(polygon, dtype=np.float64)
    source = int(np.floor(_to_pitch(pts[:, 0].min(), anchors) / period_pitches)) if repeat else 0
    if placements is None:
        placements = (range(int(np.floor(_to_pitch(0, anchors)/period_pitches))-1,
                            int(np.ceil(_to_pitch(width, anchors)/period_pitches))+2) if repeat else [0])
    for index in placements:
        shifted = pts.copy()
        if repeat:
            source_pitch = _to_pitch(pts[:, 0].min(), anchors)
            target_pitch = (source_pitch + (index-source)*period_pitches if phase_pitch is None
                            else phase_pitch + index*period_pitches)
            shifted[:, 0] = _from_pitch(_to_pitch(pts[:, 0], anchors) - source_pitch + target_pitch, anchors)
        if shifted[:, 0].max() < 0 or shifted[:, 0].min() > width:
            continue
        shifted[:, 0] *= sx
        shifted[:, 1] *= sy
        contour = [np.rint(shifted).astype(np.int32)]
        cv2.fillPoly(mask, contour, int(code))
        cv2.fillPoly(authored, contour, 255)


def _exemplar_placements(shapes, period_pitches, anchors, width):
    """Choose one authored polygon for each target period, even if seeds share a period."""
    if not shapes:
        return {}, 0.
    reference = _to_pitch(min(point[0] for point in shapes[0]['polygon']), anchors)
    phase = float(reference % period_pitches)
    return {target: min(shapes, key=lambda shape: (
                abs((target*period_pitches + phase) - _to_pitch(min(point[0] for point in shape['polygon']), anchors)),
                shape['id']))['id']
            for target in range(int(np.floor(_to_pitch(0, anchors)/period_pitches))-1,
                                int(np.ceil(_to_pitch(width, anchors)/period_pitches))+2)}, phase


def load_zones(result_dir):
    path = Path(result_dir) / 'zones' / 'analysis.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 3:
        return None
    for shape in data.get('shapes', []):
        shape.setdefault('repeat_pitches', 1)
    for group in data.get('groups', []):
        for section in group.get('sections', []):
            section.setdefault('repeat_mode', 'independent')
    return data


def build_zones(result_dir, groups=None, shapes=None):
    """Save editable taxonomies and independent, completely filled layer masks."""
    image, size, pitch_px, ppm = _metadata(result_dir)
    initializing = groups is None and shapes is None
    if groups is None:
        groups = _preset()
    if shapes is None:
        shapes = _legacy_manual_shapes(result_dir, pitch_px) if initializing and load_zones(result_dir) is None else []
    anchors, anchor_source = _pitch_anchors(image, size, pitch_px)
    groups, shapes = _validate(groups, shapes, *size, pitch_px, float(np.max(np.diff(anchors))))
    folder = Path(result_dir) / 'zones'
    folder.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    metrics = {}
    for group in groups:
        sections = group['sections']
        codes = {item['id']: index + 1 for index, item in enumerate(sections)}
        default = group['default_section_id']
        mask = np.full((h, w), codes[default], np.uint8)
        authored = np.zeros((h, w), np.uint8)
        modes = {section['id']: section['repeat_mode'] for section in sections}
        exemplar_maps = {}
        for section in sections:
            if section['repeat_mode'] != 'examples':
                continue
            for pitches in {shape['repeat_pitches'] for shape in shapes
                            if shape['group_id'] == group['id'] and shape['section_id'] == section['id'] and shape['repeat']}:
                peers = [shape for shape in shapes if shape['group_id'] == group['id']
                         and shape['section_id'] == section['id'] and shape['repeat'] and shape['repeat_pitches'] == pitches]
                exemplar_maps[(section['id'], pitches)] = _exemplar_placements(peers, pitches, anchors, size[0])
        for shape in shapes:
            if shape['group_id'] == group['id']:
                if shape['repeat'] and modes[shape['section_id']] == 'examples':
                    placements, phase = exemplar_maps[(shape['section_id'], shape['repeat_pitches'])]
                    for target, chosen in placements.items():
                        if chosen == shape['id']:
                            _draw(mask, authored, shape['polygon'], codes[shape['section_id']], True,
                                  shape['repeat_pitches'], anchors, size, [target], phase)
                else:
                    _draw(mask, authored, shape['polygon'], codes[shape['section_id']], shape['repeat'],
                          shape['repeat_pitches'], anchors, size)
        overlay = np.zeros((h, w, 4), np.uint8)
        areas, explicit_areas = {}, {}
        edges = cv2.morphologyEx(authored, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        mm2_per_pixel = (size[0] / w) * (size[1] / h) / (ppm * ppm)
        for section in sections:
            selected = mask == codes[section['id']]
            explicit = selected & (authored > 0)
            rgb = tuple(int(section['color'][i:i+2], 16) for i in (1, 3, 5))
            overlay[selected] = (rgb[2], rgb[1], rgb[0], 40 if section['id'] == default else 108)
            overlay[explicit] = (rgb[2], rgb[1], rgb[0], 132)
            overlay[selected & edges] = (rgb[2], rgb[1], rgb[0], 220)
            areas[section['id']] = round(int(np.count_nonzero(selected)) * mm2_per_pixel, 2)
            explicit_areas[section['id']] = round(int(np.count_nonzero(explicit)) * mm2_per_pixel, 2)
        cv2.imwrite(str(folder / f"group_{group['id']}_mask.png"), mask)
        cv2.imwrite(str(folder / f"group_{group['id']}_authored.png"), authored)
        cv2.imwrite(str(folder / f"group_{group['id']}_overlay.png"), overlay)
        metrics[group['id']] = {'areas_mm2': areas, 'unassigned_pixels': 0, 'coverage_percent': 100.,
                                'explicit_areas_mm2': explicit_areas,
                                'fallback_area_mm2': round(int(np.count_nonzero((mask == codes[default]) & (authored == 0))) * mm2_per_pixel, 2)}
    # Remove files for deleted classification layers after the new document is ready.
    active = {group['id'] for group in groups}
    for path in folder.glob('group_*.png'):
        if not any(path.name.startswith(f'group_{key}_') for key in active):
            path.unlink()
    data = {'version': 3, 'created_at': datetime.now(timezone.utc).isoformat(),
            'image_size_wh': list(size), 'pitch_px': pitch_px,
            'pitch_anchors_x': [round(float(x), 2) for x in anchors], 'pitch_anchor_source': anchor_source,
            'nominal_pixels_per_mm': ppm, 'working_pixels_per_mm': ppm * w / size[0],
            'groups': groups, 'shapes': shapes, 'group_metrics': metrics}
    temporary = folder / 'analysis.json.tmp'
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(folder / 'analysis.json')
    return data


def assist_polygon(result_dir, polygon):
    """Suggest a nearby image edge from a rough human polygon, without saving it."""
    image, (width, height), _, _ = _metadata(result_dir)
    if not isinstance(polygon, list) or not 3 <= len(polygon) <= 150:
        raise ValueError('먼저 세 점 이상으로 대략적인 영역을 그려 주세요.')
    try:
        pts = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('좌표를 확인하세요.') from exc
    if pts.shape != (len(polygon), 2) or not np.isfinite(pts).all():
        raise ValueError('좌표를 확인하세요.')
    sx, sy = image.shape[1] / width, image.shape[0] / height
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    pad = max(12, round(max(x1-x0, y1-y0) * .25))
    left, top = max(0, x0-pad), max(0, y0-pad)
    right, bottom = min(image.shape[1], x1+pad), min(image.shape[0], y1+pad)
    if right-left < 5 or bottom-top < 5 or (right-left)*(bottom-top) > 1_000_000:
        return {'polygon': polygon, 'changed': False, 'confidence': 'low'}
    crop = image[top:bottom, left:right]
    local = np.rint(pts - [left, top]).astype(np.int32)
    seed = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(seed, [local], 1)
    radius = max(2, min(15, round(min(crop.shape[:2]) * .06)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    labels = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    labels[cv2.dilate(seed, kernel) > 0] = cv2.GC_PR_BGD
    labels[seed > 0] = cv2.GC_PR_FGD
    labels[cv2.erode(seed, kernel) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(crop, labels, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
        foreground = np.uint8((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD))
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / max(1, cv2.contourArea(local))
        if not .45 <= area_ratio <= 1.75:
            raise ValueError('불안정한 경계')
        approx = cv2.approxPolyDP(largest, max(1., cv2.arcLength(largest, True)*.004), True)[:, 0, :]
        if len(approx) < 3 or len(approx) > 150:
            raise ValueError('불안정한 경계')
        result = [[round((int(x)+left)/sx, 2), round((int(y)+top)/sy, 2)] for x, y in approx]
        return {'polygon': result, 'changed': True, 'confidence': 'medium'}
    except (cv2.error, ValueError):
        return {'polygon': polygon, 'changed': False, 'confidence': 'low'}
