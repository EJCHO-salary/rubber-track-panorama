"""Image-based damage estimates on a human-authored region map.

The proposed class and area are image measurements, not a depth measurement.
"""
import json
import heapq
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .zones import load_zones


DAMAGE_TYPES = {'chunk', 'tear'}
CHUNK_MIN_AREA_MM2 = 100.  # 1 cm² in calibrated panorama coordinates.
TEAR_MIN_LENGTH_MM = 10.  # Only automatically count tears at least 1 cm long.
AUTO_COMPACT_MIN_AREA_MM2 = 6.  # Smaller marks are not reliable at both panorama resolutions.


def _paths(result_dir):
    folder = Path(result_dir) / 'cracks'
    return folder, folder / 'review.json'


def _automatic_tear_outside_range(item, minimum_mm, maximum_mm):
    """Keep explicit user decisions, but enforce configured automatic tear lengths."""
    return (item.get('source') == 'automatic'
            and item.get('decision_source') != 'manual'
            and item.get('status') != 'excluded'
            and item.get('damage_type') == 'tear'
            and (item.get('length_mm', 0) < minimum_mm
                 or maximum_mm is not None and item.get('length_mm', 0) > maximum_mm))


def load_cracks(result_dir):
    _, path = _paths(result_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    tear_min_mm = data.get('tear_min_length_mm', TEAR_MIN_LENGTH_MM)
    tear_max_mm = data.get('tear_max_length_mm')
    ppm = data.get('pixels_per_mm')
    if not ppm:
        zones = load_zones(result_dir)
        ppm = float(zones['nominal_pixels_per_mm']) if zones else 1.
    legacy = data.get('version', 1) < 2
    for candidate in data.get('candidates', []):
        candidate.setdefault('damage_type', None)
        candidate.setdefault('suggested_damage_type', None)
        candidate.setdefault('suggestion_reason', None)
        candidate.setdefault('decision_source', 'manual' if candidate['status'] != 'pending' else None)
        candidate.setdefault('area_mm2', _polygon_area(candidate['polygon']) / ppm ** 2)
        candidate.setdefault('length_mm', max(candidate['bbox'][2:]) / ppm)
        # Retire the old small-loss class without erasing user decisions.
        old_chip = candidate.get('damage_type') == 'chip_cut' or candidate.get('suggested_damage_type') == 'chip_cut'
        if old_chip:
            previous_type = candidate.get('damage_type')
            candidate['suggested_damage_type'] = 'chunk'
            candidate['suggestion_reason'] = '덩어리형 손상이며 검출 면적이 1 cm² 이상입니다.'
            if candidate['status'] == 'accepted':
                candidate['damage_type'] = previous_type if candidate['decision_source'] == 'manual' and previous_type in DAMAGE_TYPES else 'chunk'
            else:
                candidate['damage_type'] = None
        # Older review files stored proposals and guided selections as pending.
        # Both now participate in the tally without requiring a second click.
        if legacy and candidate['status'] == 'pending' and not old_chip:
            box = candidate['bbox']
            suggestion, reason = (candidate.get('damage_type') or candidate.get('suggested_damage_type'),
                                  candidate.get('suggestion_reason'))
            if suggestion not in DAMAGE_TYPES:
                suggestion, reason = _suggest_damage_type(
                    candidate['area_mm2'], candidate['length_mm'],
                    max(box[2], box[3]) / max(1., min(box[2], box[3])), 0.)
            candidate.update(status='accepted', damage_type=suggestion,
                             suggested_damage_type=suggestion, suggestion_reason=reason,
                             decision_source='manual' if candidate['source'] == 'manual' else 'auto')
    # Unconfirmed automatic compact marks were generated from colour changes,
    # including chalk and lighting. A user-drawn or explicitly reviewed mark
    # is always retained, even when smaller than the automatic size threshold.
    data['candidates'] = [item for item in data.get('candidates', [])
                          if not _automatic_tear_outside_range(item, tear_min_mm, tear_max_mm)
                          and not (item.get('source') == 'automatic'
                                  and item.get('decision_source') != 'manual'
                                  and item.get('status') != 'excluded'
                                  and (item.get('detection_basis') == 'material_change'
                                       or item.get('suggested_damage_type') == 'chunk'
                                       and (item.get('status') == 'pending'
                                            or item.get('area_mm2', 0) < CHUNK_MIN_AREA_MM2)
                                       and item.get('detection_basis') != 'compact_dark_loss'))]
    data['version'] = 2
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
    data['chunk_min_area_mm2'] = CHUNK_MIN_AREA_MM2
    data.setdefault('tear_min_length_mm', TEAR_MIN_LENGTH_MM)
    data.setdefault('tear_max_length_mm', None)
    sections = {item['id']: {'proposed': 0, 'accepted': 0, 'excluded': 0,
                             'length_mm': 0., 'area_mm2': 0.}
                for item in data['sections']}
    for item in data['candidates']:
        section = sections[item['section_id']]
        section[{'pending': 'proposed', 'accepted': 'accepted', 'excluded': 'excluded'}[item['status']]] += 1
        if item['status'] == 'accepted':
            section['length_mm'] += item['length_mm']
            section['area_mm2'] += item.get('area_mm2', 0.)
    for section in sections.values():
        section['length_mm'] = round(section['length_mm'], 1)
        section['area_mm2'] = round(section['area_mm2'], 1)
    data['summary'] = sections
    data['totals'] = {key: round(sum(value[key] for value in sections.values()), 1) if key in {'length_mm', 'area_mm2'}
                      else sum(value[key] for value in sections.values())
                      for key in ('proposed', 'accepted', 'excluded', 'length_mm', 'area_mm2')}
    data['accepted_by_type'] = {key: sum(item['status'] == 'accepted' and item.get('damage_type') == key
                                      for item in data['candidates']) for key in sorted(DAMAGE_TYPES)}
    data['accepted_by_type']['unclassified'] = sum(item['status'] == 'accepted' and not item.get('damage_type')
                                                   for item in data['candidates'])
    data['accepted_area_by_type'] = {
        key: round(sum(item.get('area_mm2', 0.) for item in data['candidates']
                       if item['status'] == 'accepted' and item.get('damage_type') == key), 1)
        for key in sorted(DAMAGE_TYPES)}
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


def _periodic_dark_anomaly(gray, anchors, scale_x):
    """Compare each pitch with the median of its alternating-pitch peers.

    A genuine fissure can sit on a repeated lug or groove edge. Repetition
    alone must not veto it when that particular pitch is substantially darker.
    Per-row brightness correction avoids flagging a whole strip because one
    photograph was lit differently.
    """
    h, w = gray.shape
    positions = np.rint(np.asarray(anchors) / scale_x).astype(int)
    anomaly = np.zeros((h, w), np.float32)
    for parity in (0, 1):
        spans = [(i, int(positions[i]), int(positions[i+1]))
                 for i in range(len(positions)-1)
                 if i % 2 == parity and 0 <= positions[i] < positions[i+1] <= w
                 and positions[i+1]-positions[i] >= 20]
        if len(spans) < 4:
            continue
        strips = np.stack([cv2.resize(gray[:, left:right], (128, h), interpolation=cv2.INTER_LINEAR)
                           for _, left, right in spans]).astype(np.float32)
        baseline = np.median(strips, axis=0)
        for index, (_, left, right) in enumerate(spans):
            difference = baseline - strips[index]
            row_lighting = cv2.GaussianBlur(np.median(difference, axis=1)[:, None].astype(np.float32),
                                            (1, 0), 0, sigmaY=20)
            anomaly[:, left:right] = cv2.resize(difference - row_lighting, (right-left, h),
                                                 interpolation=cv2.INTER_LINEAR)
    return anomaly


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
    """Distinguish thin tears from compact loss; size controls automatic counting."""
    width_mm = area_mm2 / max(length_mm, .1)
    if bright_rim >= .18 and area_mm2 < CHUNK_MIN_AREA_MM2 and elongation < 2.2:
        return 'chunk', '고무 표면과 다른 덩어리형 경계가 보입니다.'
    if length_mm >= 8 and width_mm <= 3.5 and elongation >= 1.4:
        return 'tear', '면적과 관계없이 길고 좁게 이어진 균열 형태입니다.'
    if area_mm2 >= CHUNK_MIN_AREA_MM2:
        return 'chunk', '덩어리형 손상이며 검출 면적이 작업 기준 1 cm² 이상입니다.'
    if elongation >= 2.2 or (length_mm >= 8 and width_mm <= 2.5):
        return 'tear', '길고 좁게 이어진 균열 형태입니다.'
    return 'chunk', '작은 덩어리형 손상입니다.'


def _automatic_candidate(section_id, polygon, bbox, area_mm2, length_mm,
                         contrast, score, suggestion, reason, basis, repeated_score=None):
    """One review schema shared by every automatic detection pass."""
    return {'id': uuid4().hex[:12], 'section_id': section_id,
            'polygon': polygon, 'bbox': bbox, 'area_mm2': area_mm2,
            'length_mm': round(length_mm, 1), 'contrast': round(contrast, 1),
            'score': round(score, 3), 'status': 'accepted', 'source': 'automatic',
            'damage_type': suggestion, 'suggested_damage_type': suggestion,
            'suggestion_reason': reason, 'decision_source': 'auto',
            'detection_basis': basis, 'repeated_structure_score': repeated_score}


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


def _polygon_overlap_fraction(first, second, scale_x, scale_y):
    """Share of a new proposal already covered by a prior proposal.

    A short prior fragment must not hide a longer connected crack that contains
    it. The denominator is the first (new) polygon, not the smaller polygon.
    """
    ax, ay, aw, ah = first['bbox']
    bx, by, bw, bh = second['bbox']
    if ax >= bx+bw or bx >= ax+aw or ay >= by+bh or by >= ay+ah:
        return 0.
    points_a = np.rint(np.asarray(first['polygon']) / [scale_x, scale_y]).astype(np.int32)
    points_b = np.rint(np.asarray(second['polygon']) / [scale_x, scale_y]).astype(np.int32)
    all_points = np.vstack((points_a, points_b))
    x0, y0 = np.min(all_points, axis=0)
    x1, y1 = np.max(all_points, axis=0) + 1
    left = np.zeros((y1-y0, x1-x0), np.uint8)
    right = np.zeros_like(left)
    cv2.fillPoly(left, [points_a - [x0, y0]], 1)
    cv2.fillPoly(right, [points_b - [x0, y0]], 1)
    new_area = int(left.sum())
    return float(np.count_nonzero(left & right) / new_area) if new_area else 0.


def _anomaly_candidates(gray, raw_response, anomaly, mask, authored, group,
                        scale_x, scale_y, pixels_per_mm, core_limit, sensitivity, existing,
                        repeated_support=None):
    """Recover near-black cracks that the repeated-geometry veto would miss."""
    anomaly_limit = {'low': 34, 'normal': 26, 'high': 20}[sensitivity]
    core = ((gray <= min(100, core_limit + 28)) & (anomaly >= anomaly_limit) &
            (raw_response >= {'low': 20, 'normal': 15, 'high': 12}[sensitivity]) &
            (authored > 0))
    chalk_halo = cv2.dilate(np.uint8(gray >= 205), np.ones((5, 5), np.uint8)) > 0
    core &= ~chalk_halo
    binary = cv2.morphologyEx(np.uint8(core), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    binary[authored == 0] = 0
    # The authored mask excludes the openings themselves. Do not erase their
    # entire horizontal band: rubber beside a hole can have a real crack.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    proposals = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area < 8 or area > 1500:
            continue
        region = labels[y:y+height, x:x+width] == label
        # Even a short horizontal lip or a dark patch belongs to the molded
        # pattern when its location repeats across pitches. Lighting can make
        # one copy anomalously dark without turning it into damage.
        if (repeated_support is not None and
                float(np.mean(repeated_support[y:y+height, x:x+width][region])) >= .75):
            continue
        black_count = int(np.count_nonzero(region & (gray[y:y+height, x:x+width] <= core_limit)))
        black_fraction = black_count / area
        if black_count < 4 or black_fraction < .30:
            continue
        mean_anomaly = float(np.mean(anomaly[y:y+height, x:x+width][region]))
        compact_black_damage = black_fraction >= .30 and mean_anomaly >= 38
        if max(width, height) < 7 and not compact_black_damage:
            continue
        yy, xx = np.nonzero(region)
        covariance = np.cov(np.column_stack((xx, yy)).astype(np.float32), rowvar=False)
        eigen = np.linalg.eigvalsh(covariance)
        elongation = float(np.sqrt((eigen[1] + 1) / (eigen[0] + 1)))
        if elongation < 1.8 and area < 30 and not compact_black_damage:
            continue
        contours, _ = cv2.findContours(np.uint8(region), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), .5, True).reshape(-1, 2)
        if len(outline) < 3:
            continue
        polygon = [[round((x+float(px))*scale_x, 1), round((y+float(py))*scale_y, 1)]
                   for px, py in outline]
        candidate = {'polygon': polygon, 'bbox': [round(x*scale_x, 1), round(y*scale_y, 1),
                                                   round(width*scale_x, 1), round(height*scale_y, 1)]}
        if not _inside_authored(candidate, authored, scale_x, scale_y):
            continue
        if any(_polygon_overlap_fraction(candidate, prior, scale_x, scale_y) >= .45
               for prior in existing):
            continue
        codes = mask[y:y+height, x:x+width][region]
        section_code = int(np.bincount(codes, minlength=len(group['sections'])+1)[1:].argmax() + 1)
        if not 1 <= section_code <= len(group['sections']):
            continue
        length_mm = float(np.sqrt(eigen[1]) * 3.5 * np.sqrt(scale_x*scale_y) / pixels_per_mm)
        area_mm2 = round(_polygon_area(polygon) / pixels_per_mm**2, 2)
        contrast = float(np.mean(raw_response[y:y+height, x:x+width][region]))
        score = round(min(.8, (mean_anomaly / 85)
                                * min(1., black_fraction / .5) * min(1., elongation / 4)), 3)
        suggestion, reason = _suggest_damage_type(area_mm2, length_mm, elongation, 0.)
        proposals.append(_automatic_candidate(group['sections'][section_code-1]['id'],
                                              polygon, candidate['bbox'], area_mm2, length_mm,
                                              contrast, score, suggestion, reason,
                                              'pitch_dark_anomaly'))
    return proposals


def _compact_dark_loss_candidates(gray, binary, response, anomaly, repeated,
                                  mask, authored, group, scale_x, scale_y, ppm,
                                  core_limit, existing):
    """Find small isolated cavities by their dark core and local contrast.

    The 1 cm² reference is a size descriptor, not proof that a smaller,
    visibly missing piece of rubber is intact. Compact holes are assessed
    separately from long fissures and repeated molded relief.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    proposals = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if not (16 <= area <= 250 and 4 <= width <= 28 and 4 <= height <= 28
                and max(width, height) / min(width, height) <= 2.7):
            continue
        region = labels[y:y+height, x:x+width] == label
        patch = gray[y:y+height, x:x+width]
        black = int(np.count_nonzero(region & (patch <= core_limit)))
        if black < 5 or black / area < .16:
            continue
        novelty = float(np.mean(anomaly[y:y+height, x:x+width][region]))
        contrast = float(np.mean(response[y:y+height, x:x+width][region]))
        structure = float(np.mean(repeated[y:y+height, x:x+width][region]))
        if novelty < 44 or contrast < 55 or (structure >= .85 and novelty < 62):
            continue
        pad = 4
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(gray.shape[1], x+width+pad), min(gray.shape[0], y+height+pad)
        footprint = np.uint8(labels[y0:y1, x0:x1] == label)
        ring = (cv2.dilate(footprint, np.ones((7, 7), np.uint8)) > 0) & (footprint == 0)
        if not np.any(ring) or (np.median(gray[y0:y1, x0:x1][ring]) -
                                np.median(patch[region])) < 24:
            continue
        contours, _ = cv2.findContours(np.uint8(region), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), .5, True).reshape(-1, 2)
        if len(outline) < 3:
            continue
        polygon = [[round((x + float(px))*scale_x, 1), round((y + float(py))*scale_y, 1)]
                   for px, py in outline]
        bbox = [round(x*scale_x, 1), round(y*scale_y, 1),
                round(width*scale_x, 1), round(height*scale_y, 1)]
        candidate = {'polygon': polygon, 'bbox': bbox}
        if not _inside_authored(candidate, authored, scale_x, scale_y):
            continue
        if any(_polygon_overlap_fraction(candidate, prior, scale_x, scale_y) >= .35
               for prior in existing + proposals):
            continue
        codes = mask[y:y+height, x:x+width][region]
        section_code = int(np.bincount(codes, minlength=len(group['sections'])+1)[1:].argmax()+1)
        if not 1 <= section_code <= len(group['sections']):
            continue
        area_mm2 = round(_polygon_area(polygon)/ppm**2, 2)
        if area_mm2 < AUTO_COMPACT_MIN_AREA_MM2 or area_mm2 >= CHUNK_MIN_AREA_MM2:
            continue
        length_mm = max(width*scale_x, height*scale_y)/ppm
        yy, xx = np.nonzero(region)
        eigen = np.linalg.eigvalsh(np.cov(np.column_stack((xx, yy)).T))
        elongation = float(np.sqrt(eigen[1]/max(eigen[0], .01)))
        damage_type = ('tear' if length_mm >= 8 and elongation >= 2.3
                       and area_mm2/max(length_mm, .1) <= 3.5 else 'chunk')
        reason = ('짙은 중심이 길고 좁게 이어진 국소 균열입니다.' if damage_type == 'tear'
                  else '작지만 국소적으로 깊은 검정 중심과 주변 고무의 결손 경계가 보입니다.')
        score = min(.85, novelty/90 * black/area * 1.5)
        proposals.append(_automatic_candidate(
            group['sections'][section_code-1]['id'], polygon, bbox, area_mm2,
            length_mm, contrast, score, damage_type, reason,
            'compact_dark_loss', round(structure, 2)))
    return proposals


def _is_sprocket_relief(candidate, hole_band, anchors, pitch_spacing, scale_x, scale_y, image_height):
    """Reject the molded horizontal lip and vertical wall around a repeated hole.

    Both can be near-black and locally anomalous, so darkness alone cannot
    distinguish them from a fissure. Keep this check common to every proposal
    pass, including the high-recall anomaly and fine-detail passes.
    """
    if not hole_band or not len(anchors):
        return False
    x, y, width, height = candidate['bbox']
    cx, cy = (x + width / 2) / scale_x, (y + height / 2) / scale_y
    bw, bh = width / scale_x, height / scale_y
    near_anchor = float(np.min(np.abs(anchors - cx))) < pitch_spacing * .32
    if not near_anchor:
        return False
    rim_distance = min(abs(cy - hole_band[0]), abs(cy - hole_band[1]))
    horizontal_lip = bw > bh * 1.8 and rim_distance < max(5, image_height * .025)
    inside_hole_band = hole_band[0] < cy < hole_band[1]
    vertical_wall = (inside_hole_band and bh > bw * 4 and bh >= 5 and
                     float(np.min(np.abs(anchors - cx))) < pitch_spacing * .12)
    return horizontal_lip or vertical_wall


def _black_core_fraction(original_gray, candidate, threshold):
    """Measure real source pixels inside the displayed contour, not its box.

    A low-resolution outline can surround a broad shaded patch even though
    only a thin edge is near-black. Such a polygon must not contribute its
    entire enclosed area to the damage tally.
    """
    polygon = np.rint(np.asarray(candidate['polygon'], np.float32)).astype(np.int32)
    x, y, width, height = cv2.boundingRect(polygon)
    if x < 0 or y < 0 or x + width > original_gray.shape[1] or y + height > original_gray.shape[0]:
        return 0.
    footprint = np.zeros((height, width), np.uint8)
    cv2.fillPoly(footprint, [polygon - [x, y]], 1)
    pixels = original_gray[y:y+height, x:x+width][footprint > 0]
    return float(np.mean(pixels <= threshold)) if pixels.size else 0.


def _persistent_vertical_shadow(original_gray, candidate, pixels_per_mm, pitch_px, core_limit):
    """Reject a fragment of a molded groove that continues above and below it.

    Limit this veto to repeated pitch geometry or the incomplete end pitches;
    a localized vertical fissure must remain detectable.
    """
    x, y, width, height = candidate['bbox']
    if height < max(25, width * 2) or not (
            (candidate.get('repeated_structure_score') or 0) >= .4
            or x < pitch_px or x + width > original_gray.shape[1] - pitch_px):
        return False
    center = round(x + width / 2)
    side = max(round(pixels_per_mm * 7), round(width * .8), 8)
    if center - side < 0 or center + side >= original_gray.shape[1]:
        return False
    span = max(50, round(pixels_per_mm * 30))

    def persistence(top, bottom):
        if bottom - top < min(35, span // 2):
            return 0.
        rows = original_gray[top:bottom]
        center_dark = np.min(rows[:, center-2:center+3], axis=1)
        sides = (rows[:, center-side].astype(np.float32) +
                 rows[:, center+side].astype(np.float32)) / 2
        return float(np.mean((center_dark <= core_limit + 13) & (sides - center_dark >= 15)))

    top, bottom = round(y), round(y + height)
    return (persistence(max(0, top-span), top) >= .5 and
            persistence(bottom, min(original_gray.shape[0], bottom+span)) >= .5)


def _filter_recurrent_anomalies(candidates, anchors, pitch_px, image_height, sensitivity,
                                minimum_score=None):
    """Reject dark relief repeated at the same place in several pitches.

    A single damaged groove can be darker than its pitch template, while a
    recurring groove wall is track geometry. Compare proposal locations in
    pitch coordinates rather than applying a blanket groove exclusion.
    """
    if not candidates or len(anchors) < 3:
        return candidates
    occurrences = {}
    locations = []
    for candidate in candidates:
        x, y, width, height = candidate['bbox']
        center_x, center_y = x + width / 2, y + height / 2
        index = int(np.argmin(np.abs(np.asarray(anchors) - center_x)))
        phase = round((center_x - anchors[index]) / pitch_px / .1)
        row = round(center_y / image_height / .1)
        key = phase, row
        locations.append(key)
        occurrences.setdefault(key, set()).add(index)
    if minimum_score is None:
        minimum_score = {'low': .28, 'normal': .20, 'high': .10}[sensitivity]
    return [candidate for candidate, key in zip(candidates, locations)
            if candidate['score'] >= minimum_score and len(occurrences[key]) < 3]


def _fine_dark_candidates(original, authored, sections, zones, group, existing,
                          sensitivity):
    """Recover narrow near-black fissures lost in the compact zone canvas."""
    _, source_w = authored.shape
    working_w = min(original.shape[1], 6000, round(source_w * 1.75))
    if working_w <= source_w:
        return []
    working_h = round(original.shape[0] * working_w / original.shape[1])
    scale_x, scale_y = original.shape[1] / working_w, original.shape[0] / working_h
    gray = cv2.resize(original, (working_w, working_h), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(authored, (working_w, working_h), interpolation=cv2.INTER_NEAREST)
    section_mask = cv2.resize(sections, (working_w, working_h), interpolation=cv2.INTER_NEAREST)
    response = np.maximum(
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))))
    repeated = _repeated_structure_support(gray, response, zones['pitch_anchors_x'], scale_x)
    black_limit = {'low': 80, 'normal': 90, 'high': 100}[sensitivity]
    contrast_limit = {'low': 33, 'normal': 25, 'high': 18}[sensitivity]
    binary = np.uint8((gray <= black_limit) & (response >= contrast_limit) & (valid > 0))
    halo = cv2.dilate(np.uint8(gray >= 205), np.ones((5, 5), np.uint8)) > 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary[valid == 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    proposals = []
    ppm = float(zones['nominal_pixels_per_mm'])
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area < 8 or area > 300 or max(width, height) < 10:
            continue
        region = labels[y:y+height, x:x+width] == label
        patch = gray[y:y+height, x:x+width]
        near_black = int(np.count_nonzero(region & (patch <= 60)))
        if near_black < 3 or near_black / area < .18:
            continue
        structural_score = float(np.mean(repeated[y:y+height, x:x+width][region]))
        mean_response = float(np.mean(response[y:y+height, x:x+width][region]))
        halo_fraction = float(np.count_nonzero(region & halo[y:y+height, x:x+width]) / area)
        # A chalk stroke is bright, but its white rim can also border a deep
        # tear. Preserve only a convincingly black core in such mixed pixels.
        if halo_fraction > .25 and not (near_black / area >= .35 and mean_response >= 65):
            continue
        # Repeated molded relief is not a new fissure merely because one
        # photograph makes its shadow blacker or sharper than its peers.
        if structural_score >= .75:
            continue
        yy, xx = np.nonzero(region)
        covariance = np.cov(np.column_stack((xx, yy)).astype(np.float32), rowvar=False)
        eigen = np.linalg.eigvalsh(covariance)
        elongation = float(np.sqrt((eigen[1] + 1) / (eigen[0] + 1)))
        if elongation < 1.8:
            continue
        contours, _ = cv2.findContours(np.uint8(region), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), .5, True).reshape(-1, 2)
        if len(outline) < 3:
            continue
        polygon = [[round((x+float(px))*scale_x, 1), round((y+float(py))*scale_y, 1)]
                   for px, py in outline]
        candidate = {'polygon': polygon, 'bbox': [round(x*scale_x, 1), round(y*scale_y, 1),
                                                   round(width*scale_x, 1), round(height*scale_y, 1)]}
        if not _inside_authored(candidate, valid, scale_x, scale_y):
            continue
        if any(_polygon_overlap_fraction(candidate, prior, scale_x, scale_y) >= .45
               for prior in existing):
            continue
        codes = section_mask[y:y+height, x:x+width][region]
        section_code = int(np.bincount(codes, minlength=len(group['sections'])+1)[1:].argmax() + 1)
        if not 1 <= section_code <= len(group['sections']):
            continue
        length_mm = float(np.sqrt(eigen[1]) * 3.5 * np.sqrt(scale_x*scale_y) / ppm)
        area_mm2 = round(_polygon_area(polygon) / ppm**2, 2)
        contrast = mean_response
        score = round(min(.75, (contrast / 120) * min(elongation / 4, 1) *
                          min(1., near_black / area / .35)), 3)
        suggestion, reason = _suggest_damage_type(area_mm2, length_mm, elongation, 0.)
        proposals.append(_automatic_candidate(group['sections'][section_code-1]['id'],
                                              polygon, candidate['bbox'], area_mm2, length_mm,
                                              contrast, score, suggestion, reason,
                                              'fine_dark_core', round(structural_score, 2)))
    return proposals


def _ridge_bridge(gray, response, valid, first, second, max_gap):
    """Trace a short, image-supported path between two dark crack fragments.

    The black core may vanish in a shallow or reflective stretch. A low-cost
    path must still follow a locally dark, high-contrast ridge inside the drawn
    region; proximity alone never joins two components.
    """
    a = np.asarray(first, np.int32)
    b = np.asarray(second, np.int32)
    distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    ia, ib = np.unravel_index(int(np.argmin(distances)), distances.shape)
    start, end = tuple(map(int, a[ia])), tuple(map(int, b[ib]))
    gap = float(np.sqrt(distances[ia, ib]))
    if gap < 2 or gap > max_gap:
        return None
    direction = (np.asarray(end) - start) / gap
    # Prevent a short bridge between adjacent, parallel fissures. Their long
    # axes run across the proposed connection rather than into it.
    if gap >= 7:
        for points in (a, b):
            if len(points) < 5:
                continue
            local = points[np.sum((points - (start if points is a else end)) ** 2, axis=1) <= 15 ** 2]
            if len(local) < 5:
                continue
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(local.T))
            if eigenvalues[1] > eigenvalues[0] * 3 and abs(float(eigenvectors[:, 1] @ direction)) < .42:
                return None
    corridor = min(12, max(5, round(gap * .4)))
    x0 = max(0, min(start[0], end[0]) - corridor - 1)
    y0 = max(0, min(start[1], end[1]) - corridor - 1)
    x1 = min(gray.shape[1], max(start[0], end[0]) + corridor + 2)
    y1 = min(gray.shape[0], max(start[1], end[1]) + corridor + 2)
    patch_gray = gray[y0:y1, x0:x1]
    patch_response = response[y0:y1, x0:x1]
    yy, xx = np.indices(patch_gray.shape)
    cross = np.abs((xx + x0 - start[0]) * direction[1] -
                   (yy + y0 - start[1]) * direction[0])
    allowed = (valid[y0:y1, x0:x1] > 0) & (cross <= corridor)
    start_local = (start[1] - y0, start[0] - x0)
    end_local = (end[1] - y0, end[0] - x0)
    if not allowed[start_local] or not allowed[end_local]:
        return None
    # Dijkstra on a small corridor: strongly prefer a dark relief line, yet
    # allow a brief pale segment when both sides of the fissure continue.
    height, width = patch_gray.shape
    costs = np.full((height, width), np.inf, np.float32)
    costs[start_local] = 0
    previous = np.full((height, width), -1, np.int32)
    queue = [(0., start_local[0], start_local[1])]
    while queue:
        distance, y, x = heapq.heappop(queue)
        if distance > costs[y, x] + 1e-5:
            continue
        if (y, x) == end_local:
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width and allowed[ny, nx]):
                    continue
                contrast = float(patch_response[ny, nx])
                shade = float(patch_gray[ny, nx])
                step = (1.414 if dx and dy else 1.) * (
                    1. + max(0., 25. - contrast) * .11 + max(0., shade - 145.) * .025)
                candidate = distance + step
                if candidate < costs[ny, nx]:
                    costs[ny, nx] = candidate
                    previous[ny, nx] = y * width + x
                    heapq.heappush(queue, (candidate, ny, nx))
    if not np.isfinite(costs[end_local]):
        return None
    path = []
    y, x = end_local
    while (y, x) != start_local:
        path.append((x + x0, y + y0))
        parent = int(previous[y, x])
        if parent < 0:
            return None
        y, x = divmod(parent, width)
    path.append(start)
    path.reverse()
    if len(path) > gap * 1.65 + 3:
        return None
    route = np.asarray(path, np.int32)
    shades = gray[route[:, 1], route[:, 0]]
    ridges = response[route[:, 1], route[:, 0]]
    if (np.mean((ridges >= 8) & (shades <= 185)) < .68 or
            np.mean(ridges >= 16) < .38 or np.median(shades) > 170):
        return None
    return route


def _merge_continuous_candidates(candidates, gray, response, authored,
                                 scale_x, scale_y, pixels_per_mm):
    """Join detected pieces of one fissure without moving across blank rubber."""
    if len(candidates) < 2:
        return candidates
    valid = _repair_narrow_authored_seams(authored, pixels_per_mm / max(scale_x, scale_y))
    outlines = [np.rint(np.asarray(item['polygon']) / [scale_x, scale_y]).astype(np.int32)
                for item in candidates]
    parent = list(range(len(candidates)))
    bridges = []

    def root(index):
        while parent[index] != index:
            index = parent[index]
        return index

    max_gap = min(36, max(7, round(pixels_per_mm * 22 / max(scale_x, scale_y))))
    for i, first in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            second = candidates[j]
            ax, ay, aw, ah = first['bbox']
            bx, by, bw, bh = second['bbox']
            physical_gap = max(0, bx - ax - aw, ax - bx - bw,
                               by - ay - ah, ay - by - bh)
            if physical_gap > max_gap * max(scale_x, scale_y):
                continue
            route = _ridge_bridge(gray, response, valid, outlines[i], outlines[j], max_gap)
            if route is not None:
                parent[root(j)] = root(i)
                bridges.append((i, j, route))
    groups = {}
    for i in range(len(candidates)):
        groups.setdefault(root(i), []).append(i)
    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(candidates[members[0]])
            continue
        all_points = np.vstack([outlines[i] for i in members])
        x0, y0 = np.maximum(np.min(all_points, axis=0) - 3, 0)
        x1, y1 = np.minimum(np.max(all_points, axis=0) + 4, gray.shape[::-1])
        footprint = np.zeros((y1-y0, x1-x0), np.uint8)
        for i in members:
            cv2.fillPoly(footprint, [outlines[i] - [x0, y0]], 1)
        for i, j, route in bridges:
            if root(i) == root(members[0]) and root(j) == root(members[0]):
                cv2.polylines(footprint, [route - [x0, y0]], False, 1, 2)
        footprint[valid[y0:y1, x0:x1] == 0] = 0
        contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            merged.extend(candidates[i] for i in members)
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contours) != 1:
            # A missing bridge or a truly separate crack must not be silently
            # represented by a single polygon that discards smaller pieces.
            merged.extend(candidates[i] for i in members)
            continue
        contour = cv2.approxPolyDP(contour, .8, True).reshape(-1, 2) + [x0, y0]
        polygon = [[round(float(px) * scale_x, 1), round(float(py) * scale_y, 1)]
                   for px, py in contour]
        bx, by, bw, bh = cv2.boundingRect(contour)
        source = max((candidates[i] for i in members), key=lambda item: item['area_mm2'])
        merged.append(dict(source, id=uuid4().hex[:12],
                           polygon=polygon, bbox=[round(bx*scale_x, 1), round(by*scale_y, 1),
                                                  round(bw*scale_x, 1), round(bh*scale_y, 1)],
                           area_mm2=round(_polygon_area(polygon) / pixels_per_mm**2, 2),
                           length_mm=round(max(bw*scale_x, bh*scale_y) / pixels_per_mm, 1),
                           connected_fragments=len(members)))
    return merged


def _directional_tear_extensions(candidates, gray, response, authored, repeated,
                                 scale_x, scale_y, pixels_per_mm, hole_band=None):
    """Continue a black-seeded diagonal tear through locally dark, aligned ridges.

    The seed must be an accepted black-core fissure. A weak extension must have
    the same direction and an image-supported bridge; a broad molded shadow
    cannot become a new tear solely because it is dark.
    """
    weak = np.uint8((gray <= 145) & (response >= 14) & (authored > 0))
    angles = (-60, -45, -30, -15, 15, 30, 45, 60)
    result = []
    for candidate in candidates:
        if (candidate.get('detection_basis') != 'dark_core' or
                candidate.get('damage_type') != 'tear' or
                not 15 <= candidate['length_mm'] <= 80 or
                candidate['area_mm2'] >= CHUNK_MIN_AREA_MM2 or
                (candidate.get('repeated_structure_score') or 0) >= .7):
            result.append(candidate)
            continue
        points = np.asarray(candidate['polygon'], np.float32) / [scale_x, scale_y]
        if len(points) < 4:
            result.append(candidate)
            continue
        eigenvalues, vectors = np.linalg.eigh(np.cov(points.T))
        axis = vectors[:, 1]
        angle = (np.degrees(np.arctan2(axis[1], axis[0])) + 90) % 180 - 90
        if abs(angle) < 14 or abs(angle) > 70 or eigenvalues[1] < eigenvalues[0] * 2:
            result.append(candidate)
            continue
        orientations = [value for value in angles if abs(value-angle) <= 21]
        if not orientations:
            result.append(candidate)
            continue
        bx, by, bw, bh = candidate['bbox']
        if hole_band and by/scale_y < hole_band[1]+10 and (by+bh)/scale_y > hole_band[0]-10:
            result.append(candidate)
            continue
        left = max(0, round(bx/scale_x)-170)
        top = max(0, round(by/scale_y)-170)
        right = min(gray.shape[1], round((bx+bw)/scale_x)+170)
        bottom = min(gray.shape[0], round((by+bh)/scale_y)+170)
        if right-left < 40 or bottom-top < 40:
            result.append(candidate)
            continue
        local = weak[top:bottom, left:right]
        directional = np.zeros_like(local)
        for orientation in orientations:
            kernel = np.zeros((17, 17), np.uint8)
            radians = np.radians(orientation)
            dx, dy = round(np.cos(radians)*8), round(np.sin(radians)*8)
            cv2.line(kernel, (8-dx, 8-dy), (8+dx, 8+dy), 1, 1)
            directional |= cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(directional, 8)
        if count < 2:
            result.append(candidate)
            continue
        original_mask = np.zeros_like(local)
        cv2.fillPoly(original_mask, [np.rint(points-[left, top]).astype(np.int32)], 1)
        overlap = np.bincount(labels[cv2.dilate(original_mask, np.ones((7, 7), np.uint8)) > 0].ravel(),
                              minlength=count)
        overlap[0] = 0
        seed = int(np.argmax(overlap))
        if overlap[seed] < 5 or stats[seed, cv2.CC_STAT_AREA] < 35:
            result.append(candidate)
            continue

        def component_axis(label):
            yy, xx = np.nonzero(labels == label)
            covariance = np.cov(np.column_stack((xx, yy)).T)
            values, vectors_local = np.linalg.eigh(covariance)
            return vectors_local[:, 1], values[1] / max(values[0], .01)

        seed_axis, seed_ratio = component_axis(seed)
        if seed_ratio < 3:
            result.append(candidate)
            continue
        seed_points = cv2.findContours(np.uint8(labels == seed), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1, 2)
        chosen = None
        best_extension = 0.
        for label in range(1, count):
            if label == seed or stats[label, cv2.CC_STAT_AREA] < 25:
                continue
            x, y, width, height, _ = map(int, stats[label])
            gap = max(0, x - stats[seed, 0] - stats[seed, 2],
                      stats[seed, 0] - x - width,
                      y - stats[seed, 1] - stats[seed, 3],
                      stats[seed, 1] - y - height)
            if gap > 65:
                continue
            region = labels[y:y+height, x:x+width] == label
            if float(np.mean(repeated[top+y:top+y+height, left+x:left+x+width][region])) >= .75:
                continue
            other_axis, ratio = component_axis(label)
            if ratio < 3 or abs(float(other_axis @ seed_axis)) < .83:
                continue
            separation = centroids[label] - centroids[seed]
            if abs(float(separation @ seed_axis)) < np.linalg.norm(separation) * .94:
                continue
            extra = abs(float(separation @ seed_axis)) + max(width, height) / 2
            if extra < 30 or extra <= best_extension:
                continue
            other_points = cv2.findContours(np.uint8(labels == label), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1, 2)
            bridge = _ridge_bridge(gray[top:bottom, left:right], response[top:bottom, left:right],
                                   authored[top:bottom, left:right], seed_points, other_points, 70)
            if (bridge is not None and
                    float(np.mean(response[top+bridge[:, 1], left+bridge[:, 0]])) >= 28):
                chosen = label, bridge
                best_extension = extra
        original_left = bx/scale_x-left
        original_right = (bx+bw)/scale_x-left
        seed_left = float(stats[seed, cv2.CC_STAT_LEFT])
        seed_right = seed_left + float(stats[seed, cv2.CC_STAT_WIDTH])
        seed_extension = max(original_left-seed_left, seed_right-original_right)
        if chosen is None and seed_extension < 25:
            result.append(candidate)
            continue
        other, bridge = chosen if chosen is not None else (None, None)
        footprint = original_mask.copy()

        def centerline(label):
            yy, xx = np.nonzero(labels == label)
            # The selected diagonal tears have x as their major axis. A median
            # across each column follows the fissure, not its broad dark halo.
            xs = np.unique(xx)
            return np.asarray([(int(x), int(np.median(yy[xx == x]))) for x in xs], np.int32)

        for label in (seed, other) if other is not None else (seed,):
            line = centerline(label)
            if len(line) >= 2:
                cv2.polylines(footprint, [line], False, 1, 3)
        if bridge is not None:
            cv2.polylines(footprint, [bridge], False, 1, 3)
            # Attach the bridge endpoints to their component centerlines; both
            # connectors lie within the observed dark components.
            for label, endpoint in ((seed, bridge[0]), (other, bridge[-1])):
                line = centerline(label)
                nearest = line[np.argmin(np.sum((line-endpoint)**2, axis=1))]
                cv2.line(footprint, tuple(endpoint), tuple(nearest), 1, 3)
        footprint[authored[top:bottom, left:right] == 0] = 0
        contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            result.append(candidate)
            continue
        contour = cv2.approxPolyDP(contours[0], .8, True).reshape(-1, 2) + [left, top]
        polygon = [[round(float(x)*scale_x, 1), round(float(y)*scale_y, 1)] for x, y in contour]
        x, y, width, height = cv2.boundingRect(contour)
        bbox = [round(x*scale_x, 1), round(y*scale_y, 1),
                round(width*scale_x, 1), round(height*scale_y, 1)]
        if _polygon_area(polygon) <= _polygon_area(candidate['polygon']) * 1.2:
            result.append(candidate)
            continue
        result.append(dict(candidate, polygon=polygon, bbox=bbox,
                           area_mm2=round(_polygon_area(polygon)/pixels_per_mm**2, 2),
                           length_mm=round(max(width*scale_x, height*scale_y)/pixels_per_mm, 1),
                           detection_basis='extended_dark_ridge',
                           suggestion_reason='검은 균열에서 시작해 같은 방향의 어두운 선을 영상 근거로 연결했습니다.'))
    return result


def propose_cracks(result_dir, group_id, sensitivity='normal',
                   tear_min_length_mm=TEAR_MIN_LENGTH_MM, tear_max_length_mm=None):
    """Regenerate dark-core candidates while retaining explicit human review."""
    if sensitivity not in {'low', 'normal', 'high'}:
        raise ValueError('민감도는 low, normal, high 중 하나여야 합니다.')
    if not math.isfinite(tear_min_length_mm) or tear_min_length_mm < 0:
        raise ValueError('티어 길이 하한은 0 mm 이상이어야 합니다.')
    if (tear_max_length_mm is not None
            and (not math.isfinite(tear_max_length_mm)
                 or tear_max_length_mm < tear_min_length_mm)):
        raise ValueError('티어 길이 상한은 하한 이상이어야 합니다.')
    zones = load_zones(result_dir)
    if not zones:
        raise ValueError('먼저 영역을 지정해 주세요.')
    group = next((item for item in zones['groups'] if item['id'] == group_id), None)
    if not group:
        raise ValueError('분류 체계를 찾을 수 없습니다.')
    previous = load_cracks(result_dir)
    mask = cv2.imread(str(Path(result_dir) / 'zones' / f'group_{group_id}_mask.png'), cv2.IMREAD_GRAYSCALE)
    authored = cv2.imread(str(Path(result_dir) / 'zones' / f'group_{group_id}_authored.png'), cv2.IMREAD_GRAYSCALE)
    source_color = cv2.imread(str(Path(result_dir) / 'panorama.png'), cv2.IMREAD_COLOR)
    if mask is None or authored is None or source_color is None or mask.shape != authored.shape:
        raise ValueError('전개 사진 또는 지정 영역 마스크를 읽을 수 없습니다.')
    h, w = mask.shape
    original = cv2.cvtColor(source_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
    # Black-hat finds dark fissures against a locally brighter rubber surface.
    # Two kernel widths tolerate both hairline and wider surface cracks.
    response = np.maximum(
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))))
    raw_response = response
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
        # A molded longitudinal groove can break into several dark fragments
        # under uneven light. Its tall narrow contour still recurs at the
        # same pitch phase; do not report each fragment as a separate tear.
        if structural_score >= .75 or (bh >= 18 and bh >= bw * 2.5 and structural_score >= .45):
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
        outline = cv2.approxPolyDP(outline, .5, True).reshape(-1, 2)
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
        candidates.append(_automatic_candidate(
            group['sections'][section_code-1]['id'], polygon,
            [round(x*scale_x, 1), round(y*scale_y, 1), round(bw*scale_x, 1), round(bh*scale_y, 1)],
            area_mm2, length_mm, contrast, score, suggestion, reason, 'dark_core',
            round(structural_score, 2)))
    candidates.sort(key=lambda item: item['score'] * max(1, item['length_mm']) ** .5, reverse=True)
    candidates = _merge_continuous_candidates(candidates[:400], gray, response, authored,
                                               scale_x, scale_y, ppm)
    for item in candidates:
        box = item['bbox']
        suggestion, reason = _suggest_damage_type(
            item['area_mm2'], item['length_mm'],
            max(box[2], box[3]) / max(1., min(box[2], box[3])), 0.)
        item['damage_type'] = item['suggested_damage_type'] = suggestion
        item['suggestion_reason'] = reason
    candidates = [item for item in candidates
                  if item['damage_type'] != 'chunk' or item['area_mm2'] >= CHUNK_MIN_AREA_MM2]
    anomaly = _periodic_dark_anomaly(gray, zones['pitch_anchors_x'], scale_x)
    anomalous = _anomaly_candidates(gray, raw_response, anomaly, mask, authored, group,
                                    scale_x, scale_y, ppm, core_limit, sensitivity,
                                    candidates, repeated_support)
    candidates.extend(_filter_recurrent_anomalies(
        anomalous, zones['pitch_anchors_x'], zones['pitch_px'], original.shape[0], sensitivity))
    candidates = [item for item in candidates
                  if item['damage_type'] != 'chunk' or item['area_mm2'] >= CHUNK_MIN_AREA_MM2]
    fine = _fine_dark_candidates(original, authored, mask, zones, group,
                                 candidates, sensitivity)
    candidates.extend(fine)
    candidates.extend(_compact_dark_loss_candidates(
        gray, binary, raw_response, anomaly, repeated_support, mask, authored,
        group, scale_x, scale_y, ppm, core_limit, candidates))
    candidates = [item for item in candidates
                  if not _is_sprocket_relief(item, hole_band, anchor_positions,
                                             pitch_spacing, scale_x, scale_y, h)
                  and not (item['suggested_damage_type'] == 'chunk'
                           and item['area_mm2'] < CHUNK_MIN_AREA_MM2
                           and item.get('detection_basis') != 'compact_dark_loss')
                  and (item['area_mm2'] < 10 or
                       _black_core_fraction(original, item, core_limit) >= .20)
                  and not _persistent_vertical_shadow(
                      original, item, ppm, zones['pitch_px'], core_limit)
                  and not (item.get('detection_basis') == 'compact_dark_loss'
                           and hole_band
                           and hole_band[0] <= (item['bbox'][1]+item['bbox'][3]/2)/scale_y <= hole_band[1]
                           and np.min(np.abs(anchor_positions -
                                             (item['bbox'][0]+item['bbox'][2]/2)/scale_x))
                           < pitch_spacing*.25)
                  and min(item['bbox'][0], item['bbox'][1],
                          original.shape[1] - item['bbox'][0] - item['bbox'][2],
                          original.shape[0] - item['bbox'][1] - item['bbox'][3]) >= 2]
    candidates = _directional_tear_extensions(
        candidates, gray, raw_response, authored, repeated_support,
        scale_x, scale_y, ppm, hole_band)
    tears = [item for item in candidates if item['damage_type'] == 'tear']
    candidates = ([item for item in candidates if item['damage_type'] != 'tear'] +
                  _merge_continuous_candidates(tears, gray, raw_response, authored,
                                               scale_x, scale_y, ppm))
    # Join adjacent fissure fragments first, then enforce the physical length
    # threshold on the resulting automatic candidates.
    candidates = [item for item in candidates if not _automatic_tear_outside_range(
        item, tear_min_length_mm, tear_max_length_mm)]
    candidates.sort(key=lambda item: item['score'] * max(1, item['length_mm']) ** .5, reverse=True)
    candidates = candidates[:400]
    if previous and previous.get('group_id') == group_id and previous.get('zone_created_at') == zones['created_at']:
        dismissed = previous.get('dismissed_candidates', []) + [
            {'id': item['id'], 'bbox': item['bbox'], 'polygon': item['polygon']}
            for item in previous['candidates'] if item['status'] == 'excluded']
        preserved = [item for item in previous['candidates']
                     if (item.get('decision_source') == 'manual' and item['status'] == 'accepted'
                         or item.get('source') == 'manual'
                         or item.get('selection_source') in {'guided', 'drawn'})
                     and item['status'] != 'excluded'
                     and (item.get('selection_source') == 'guided'
                          or item.get('selection_source') == 'drawn'
                          or _inside_authored(item, authored, scale_x, scale_y))]
        for item in preserved:
            if not item.get('damage_type'):
                box = item['bbox']
                item['damage_type'] = _suggest_damage_type(
                    item['area_mm2'], item['length_mm'],
                    max(box[2], box[3]) / max(1., min(box[2], box[3])), 0.)[0]
                item['status'] = 'accepted'
                item['decision_source'] = 'auto'
        candidates = preserved + [item for item in candidates
                                  if not any(_same_damage_location(item, saved) for saved in preserved)
                                  and not any(_same_damage_location(item, saved) for saved in dismissed)]
    else:
        dismissed = []
    data = {'version': 2, 'created_at': datetime.now(timezone.utc).isoformat(), 'zone_created_at': zones['created_at'],
            'group_id': group_id, 'sensitivity': sensitivity, 'image_size_wh': zones['image_size_wh'],
            'pixels_per_mm': ppm,
            'tear_min_length_mm': tear_min_length_mm,
            'tear_max_length_mm': tear_max_length_mm,
            'sections': [{'id': section['id'], 'name': section['name'], 'color': section['color']} for section in group['sections']],
            'candidates': candidates, 'dismissed_candidates': dismissed}
    return _write(result_dir, _summary(data))


def _trace_dark_component(gray, valid, seed, tolerance, offset_px, seed_radius_px=18):
    """Select one near-black connected region, then add a measured outline offset."""
    if not 15 <= tolerance <= 100:
        raise ValueError('색상 허용 범위는 15~100이어야 합니다.')
    if gray.shape != valid.shape:
        raise ValueError('사진과 지정 영역 크기가 일치하지 않습니다.')
    sx, sy = seed
    if not (0 <= sx < gray.shape[1] and 0 <= sy < gray.shape[0] and valid[sy, sx]):
        raise ValueError('지정된 영역의 검은 균열을 클릭해 주세요.')
    smooth = cv2.GaussianBlur(gray, (3, 3), .7)
    background = cv2.GaussianBlur(smooth, (0, 0), 12)
    contrast = background.astype(np.int16) - smooth.astype(np.int16)
    radius = int(seed_radius_px)
    x0, y0 = max(0, sx-radius), max(0, sy-radius)
    x1, y1 = min(gray.shape[1], sx+radius+1), min(gray.shape[0], sy+radius+1)
    nearby = valid[y0:y1, x0:x1].astype(bool)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    near_core = (nearby & (smooth[y0:y1, x0:x1] <= 72)
                 & (contrast[y0:y1, x0:x1] >= 17)
                 & ((xx - sx) ** 2 + (yy - sy) ** 2 <= radius ** 2))
    if not np.any(near_core):
        raise ValueError('클릭한 곳 가까이에 검은 핵심부가 없습니다. 사진을 확대해 검은 균열을 다시 클릭해 주세요.')
    yy, xx = np.nonzero(near_core)
    distances = (xx + x0 - sx) ** 2 + (yy + y0 - sy) ** 2
    closest = int(np.argmin(distances))
    sx, sy = int(xx[closest] + x0), int(yy[closest] + y0)
    limit = min(125, max(72, int(smooth[sy, sx]) + tolerance))
    possible = np.uint8((smooth <= limit) & (contrast >= 11) & (valid > 0))
    # A short interruption from glare or compression should not split one tear.
    bridge = 3 if tolerance < 35 else 5 if tolerance < 70 else 7
    possible = cv2.morphologyEx(possible, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge)))
    possible[valid == 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(possible, 8)
    label = int(labels[sy, sx])
    if not 0 < label < count:
        raise ValueError('검은 영역을 연결할 수 없습니다. 허용 범위를 조정해 주세요.')
    x, y, width, height, pixels = map(int, stats[label])
    if pixels < 12:
        raise ValueError('검은 연결 영역이 너무 작습니다. 다른 지점을 클릭해 주세요.')
    if pixels > gray.size * .18:
        raise ValueError('선택이 주변 그림자까지 번졌습니다. 허용 범위를 낮춰 주세요.')
    selected = np.uint8(labels == label)
    if offset_px:
        radius = max(1, int(round(offset_px)))
        selected = cv2.dilate(selected, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1)))
        selected[valid == 0] = 0
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('선택 경계를 만들 수 없습니다.')
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1., True).reshape(-1, 2)
    if len(contour) < 3:
        raise ValueError('선택 경계를 만들 수 없습니다.')
    return contour, selected, (sx, sy)


def _repair_narrow_authored_seams(authored, pixels_per_mm):
    """Join raster gaps below 2 mm between authored regions, without filling large holes."""
    span = max(3, min(31, int(round(pixels_per_mm * 2)) + 1))
    if span % 2 == 0:
        span += 1
    mask = np.uint8(authored > 0)
    across_rows = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, span)))
    across_columns = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1)))
    return cv2.bitwise_or(across_rows, across_columns)


def _manual_near_authored(footprint, authored, pixels_per_mm):
    """Allow a hand-drawn damage outline to cross small region-map gaps.

    Automatic detection still uses the exact authored mask. The manual path
    must mostly cover a drawn region and never stray more than 2.5 mm from it,
    so this does not make large unmarked openings eligible for drawing.
    """
    selected = footprint > 0
    count = int(np.count_nonzero(selected))
    if not count or np.count_nonzero(selected & (authored > 0)) / count < .5:
        return False
    radius = max(1, int(round(2.5 * pixels_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    nearby = cv2.dilate(np.uint8(authored > 0), kernel)
    return not np.any(selected & (nearby == 0))


def trace_crack(result_dir, point, damage_type, offset_mm, tolerance=55, candidate_id=None,
                apply=False, seed_radius_px=18, region=None, accept=False):
    """Preview or persist a Photoshop-like seeded crack selection."""
    if damage_type not in DAMAGE_TYPES:
        raise ValueError('손상 유형이 올바르지 않습니다.')
    if not np.isfinite(offset_mm) or not 0 <= offset_mm <= 5:
        raise ValueError('경계 여유는 0~5 mm로 지정해 주세요.')
    if not 1 <= seed_radius_px <= 320:
        raise ValueError('클릭 허용 거리는 1~320 픽셀이어야 합니다.')
    data, zones = load_cracks(result_dir), load_zones(result_dir)
    if not data or not zones or data['zone_created_at'] != zones['created_at']:
        raise ValueError('현재 영역에서 크랙 후보를 먼저 생성해 주세요.')
    candidate = next((item for item in data['candidates'] if item['id'] == candidate_id), None) if candidate_id else None
    if candidate_id and not candidate:
        raise ValueError('선택한 후보를 찾을 수 없습니다.')
    root = Path(result_dir)
    original = cv2.imread(str(root / 'panorama.png'), cv2.IMREAD_GRAYSCALE)
    authored = cv2.imread(str(root / 'zones' / f"group_{data['group_id']}_authored.png"), cv2.IMREAD_GRAYSCALE)
    sections = cv2.imread(str(root / 'zones' / f"group_{data['group_id']}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if original is None or authored is None or sections is None:
        raise ValueError('전개 사진 또는 지정 영역을 읽을 수 없습니다.')
    image_h, image_w = original.shape
    if region is not None:
        if len(region) != 4 or not all(np.isfinite(value) for value in region):
            raise ValueError('드래그 영역을 확인해 주세요.')
        rx0, ry0, rx1, ry1 = map(float, region)
        if not (0 <= rx0 < rx1 <= image_w and 0 <= ry0 < ry1 <= image_h and
                rx1-rx0 >= 3 and ry1-ry0 >= 3 and (rx1-rx0)*(ry1-ry0) <= 1_000_000):
            raise ValueError('드래그 영역을 사진 안에서 더 작게 지정해 주세요.')
        center_x, center_y = (rx0+rx1)/2, (ry0+ry1)/2
    elif point is not None:
        if len(point) != 2 or not all(np.isfinite(value) for value in point):
            raise ValueError('선택 위치를 확인해 주세요.')
        center_x, center_y = map(float, point)
    elif candidate:
        bx, by, bw, bh = candidate['bbox']
        center_x, center_y = bx + bw/2, by + bh/2
    else:
        raise ValueError('사진에서 검은 균열을 클릭해 주세요.')
    if not (0 <= center_x < image_w and 0 <= center_y < image_h):
        raise ValueError('선택 위치가 사진 밖입니다.')
    ppm = float(zones['nominal_pixels_per_mm'])
    reach = round(ppm * 95)
    left, top = max(0, round(center_x)-reach), max(0, round(center_y)-reach)
    right, bottom = min(image_w, round(center_x)+reach+1), min(image_h, round(center_y)+reach+1)
    if region is not None:
        margin = round(ppm * 10)
        left, top = max(0, min(left, int(rx0)-margin)), max(0, min(top, int(ry0)-margin))
        right, bottom = min(image_w, max(right, int(rx1)+margin)), min(image_h, max(bottom, int(ry1)+margin))
    elif candidate:
        bx, by, bw, bh = candidate['bbox']
        margin = round(ppm * 15)
        left, top = max(0, min(left, int(bx)-margin)), max(0, min(top, int(by)-margin))
        right = min(image_w, max(right, int(bx+bw)+margin))
        bottom = min(image_h, max(bottom, int(by+bh)+margin))
    gray = original[top:bottom, left:right]
    authored_crop = cv2.resize(authored, (image_w, image_h), interpolation=cv2.INTER_NEAREST)[top:bottom, left:right]
    valid = _repair_narrow_authored_seams(authored_crop, ppm)
    # The seed guides a local selection. A dark lug edge must not flood across
    # several pitches just because its shadow touches the chosen fissure.
    growth = np.zeros(gray.shape, np.uint8)
    if region is not None:
        margin = round(ppm * 10)
        gx0, gy0 = max(0, int(rx0)-margin-left), max(0, int(ry0)-margin-top)
        gx1, gy1 = min(gray.shape[1], int(rx1)+margin-left), min(gray.shape[0], int(ry1)+margin-top)
    elif candidate:
        bx, by, bw, bh = candidate['bbox']
        if damage_type == 'tear':
            # A truncated proposal should be allowed to follow the same thin
            # fissure well beyond its old box, without searching far sideways.
            if bh > bw * 1.2:
                margin_x, margin_y = round(ppm * 12), round(ppm * 60)
            elif bw > bh * 1.2:
                margin_x, margin_y = round(ppm * 60), round(ppm * 12)
            else:
                margin_x = margin_y = round(ppm * 40)
        else:
            margin_x = margin_y = round(ppm * 8)
        gx0, gy0 = max(0, int(bx)-margin_x-left), max(0, int(by)-margin_y-top)
        gx1, gy1 = min(gray.shape[1], int(bx+bw)+margin_x-left), min(gray.shape[0], int(by+bh)+margin_y-top)
    else:
        allowance = max(round(ppm * (60 if damage_type == 'tear' else 25)), round(seed_radius_px + offset_mm * ppm))
        gx0, gy0 = max(0, round(center_x)-allowance-left), max(0, round(center_y)-allowance-top)
        gx1, gy1 = min(gray.shape[1], round(center_x)+allowance-left+1), min(gray.shape[0], round(center_y)+allowance-top+1)
    growth[gy0:gy1, gx0:gx1] = 1
    valid = np.uint8((valid > 0) & (growth > 0))
    if region is not None:
        sx0, sy0 = max(0, int(rx0)-left), max(0, int(ry0)-top)
        sx1, sy1 = min(gray.shape[1], int(rx1)-left+1), min(gray.shape[0], int(ry1)-top+1)
        search = np.zeros(gray.shape, bool)
        search[sy0:sy1, sx0:sx1] = True
        search &= valid > 0
        if not np.any(search):
            raise ValueError('드래그 영역이 지정된 분석 영역 밖입니다.')
        contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        evidence = contrast.astype(np.float32) + np.maximum(0, 105-gray.astype(np.float32))
        evidence[~search] = -1
        sy, sx = np.unravel_index(int(np.argmax(evidence)), evidence.shape)
        if evidence[sy, sx] < 40:
            raise ValueError('드래그 영역에서 검은 균열을 찾지 못했습니다. 더 좁게 선택하거나 선 그리기를 사용해 주세요.')
    elif candidate and point is None:
        polygon = np.rint(np.asarray(candidate['polygon']) - [left, top]).astype(np.int32)
        footprint = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(footprint, [polygon], 1)
        admissible = (footprint > 0) & (valid > 0)
        if not np.any(admissible):
            raise ValueError('후보가 지정된 영역 안에 있지 않습니다.')
        local = np.where(admissible, gray, 255)
        sy, sx = np.unravel_index(int(np.argmin(local)), local.shape)
    else:
        sx, sy = round(center_x)-left, round(center_y)-top
    contour, selected, seed = _trace_dark_component(gray, valid, (sx, sy), int(tolerance),
                                                     round(offset_mm*ppm), seed_radius_px)
    contour = contour + [left, top]
    polygon = [[float(x), float(y)] for x, y in contour]
    # A contour can bridge a hole in the authored mask; never silently fill it.
    mask_outline = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask_outline, [contour - [left, top]], 1)
    invalid_count = int(np.count_nonzero((mask_outline > 0) & (valid == 0)))
    if invalid_count > min(20, max(3, round(np.count_nonzero(mask_outline) * .01))):
        raise ValueError('선택 경계가 지정되지 않은 영역을 가로지릅니다. 허용 범위나 여유를 줄여 주세요.')
    area_mm2 = round(_polygon_area(polygon) / ppm ** 2, 2)
    bx, by, bw, bh = cv2.boundingRect(contour.astype(np.int32))
    wide_tear = damage_type == 'tear' and area_mm2 / max(bw, bh) * ppm > 3.5
    section_crop = cv2.resize(sections, (image_w, image_h), interpolation=cv2.INTER_NEAREST)[top:bottom, left:right]
    codes = section_crop[selected > 0]
    code = int(np.bincount(codes, minlength=len(data['sections'])+1)[1:].argmax() + 1)
    result = {'polygon': polygon, 'bbox': [float(bx), float(by), float(bw), float(bh)],
              'area_mm2': area_mm2, 'damage_type': damage_type, 'candidate_id': candidate_id,
              'seed': [float(seed[0]+left), float(seed[1]+top)],
              'warning': '티어로 보기에는 선택 폭이 넓습니다. 그림자 혼입 여부와 손상 유형을 확인해 주세요.'
              if wide_tear else None}
    if not apply:
        return result
    if candidate is None:
        candidate = next((item for item in data['candidates']
                          if _same_damage_location(item, result)
                          and _polygon_overlap_fraction(result, item, 1., 1.) >= .6), None)
    if candidate is None:
        candidate = {'id': uuid4().hex[:12], 'status': 'pending', 'source': 'manual', 'damage_type': None,
                     'decision_source': None, 'score': 1., 'contrast': 0.}
        data['candidates'].insert(0, candidate)
    candidate.update({'section_id': data['sections'][code-1]['id'], 'polygon': polygon,
                      'bbox': result['bbox'], 'area_mm2': area_mm2,
                      'length_mm': round(max(bw, bh) / ppm, 1),
                      'suggested_damage_type': damage_type,
                      'suggestion_reason': '검은 연결 영역을 선택하고 경계 여유를 적용했습니다.',
                      'selection_source': 'guided', 'selection_offset_mm': offset_mm,
                      'selection_tolerance': tolerance})
    if accept:
        candidate['status'] = 'accepted'
        candidate['damage_type'] = damage_type
        candidate['decision_source'] = 'manual'
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


def add_manual_crack(result_dir, points, closed=False, candidate_id=None):
    data = load_cracks(result_dir)
    zones = load_zones(result_dir)
    if not data or not zones or data['zone_created_at'] != zones['created_at']:
        raise ValueError('현재 영역에서 크랙 후보를 먼저 생성해 주세요.')
    if not isinstance(points, list) or not (3 if closed else 2) <= len(points) <= 500:
        raise ValueError('선은 2점 이상, 닫힌 경계는 3점 이상 지정해 주세요.')
    try:
        path = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('균열 경로 좌표를 확인해 주세요.') from exc
    width, height = zones['image_size_wh']
    if path.shape != (len(points), 2) or not np.isfinite(path).all() or np.any(path < 0) or np.any(path[:, 0] >= width) or np.any(path[:, 1] >= height):
        raise ValueError('균열 경로 좌표를 확인해 주세요.')
    length_px = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if closed:
        length_px += float(np.linalg.norm(path[-1] - path[0]))
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
    cv2.polylines(centerline, [work_points], bool(closed), 255, 1)
    working_ppm = zones['nominal_pixels_per_mm'] * min(sx, sy)
    if not _manual_near_authored(centerline, authored, working_ppm):
        raise ValueError('그린 경로가 지정된 영역에서 너무 멉니다. 영역 지도를 확인해 주세요.')
    raster = np.zeros_like(mask)
    if closed:
        cv2.fillPoly(raster, [work_points], 255)
    else:
        cv2.polylines(raster, [work_points], False, 255, 3)
    if not _manual_near_authored(raster, authored, working_ppm):
        raise ValueError('그린 손상 영역이 지정된 영역에서 너무 멉니다. 영역 지도를 확인해 주세요.')
    # Pixels from filled gaps carry the default section, not a user decision.
    section_pixels = mask[(raster > 0) & (authored > 0)]
    code = int(np.bincount(section_pixels, minlength=len(data['sections'])+1)[1:].argmax() + 1)
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1., True).reshape(-1, 2)
    x, y, bw, bh = cv2.boundingRect(contour)
    # Measure the original hand-drawn boundary; the coarse zone raster can skew
    # the reported physical area of a small mark.
    area_mm2 = round(_polygon_area(path) / zones['nominal_pixels_per_mm'] ** 2, 2) if closed else round(
        _polygon_area(contour) / (sx * sy * zones['nominal_pixels_per_mm'] ** 2), 2)
    if closed and area_mm2 < .5:
        raise ValueError('닫힌 경계 안쪽에 손상 면적이 있어야 합니다. 경계를 한 바퀴 둘러 그려 주세요.')
    damage_type = 'chunk' if closed else 'tear'
    candidate = next((item for item in data['candidates'] if item['id'] == candidate_id), None) if candidate_id else None
    if candidate_id and candidate is None:
        raise ValueError('수정할 후보를 찾을 수 없습니다.')
    if candidate is None:
        candidate = {'id': uuid4().hex[:12], 'source': 'manual', 'score': 1., 'contrast': 0.}
        data['candidates'].insert(0, candidate)
    candidate.update({'section_id': data['sections'][code-1]['id'],
                      'polygon': [[round(float(px)/sx, 1), round(float(py)/sy, 1)] for px, py in contour],
                      'status': 'accepted', 'damage_type': damage_type, 'decision_source': 'manual',
                      'suggested_damage_type': damage_type,
                      'suggestion_reason': '사용자가 직접 그린 선입니다.' if not closed else
                      '사용자가 그린 닫힌 경계의 면적으로 분류했습니다.',
                      'selection_source': 'drawn',
                      'length_mm': round(length_px / zones['nominal_pixels_per_mm'], 1),
                      'area_mm2': area_mm2,
                      'bbox': [round(x/sx, 1), round(y/sy, 1), round(bw/sx, 1), round(bh/sy, 1)]})
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


def delete_crack(result_dir, candidate_id):
    """Remove a review candidate and remember the rejection on reproposal."""
    return delete_cracks(result_dir, [candidate_id])


def delete_cracks(result_dir, candidate_ids):
    """Atomically remove selected candidates while retaining reproposal suppression."""
    data = load_cracks(result_dir)
    if not data:
        raise ValueError('크랙 후보를 찾을 수 없습니다.')
    ids = set(candidate_ids)
    if not ids or ids - {item['id'] for item in data['candidates']}:
        raise ValueError('크랙 후보를 찾을 수 없습니다.')
    removed = [item for item in data['candidates'] if item['id'] in ids]
    data.setdefault('dismissed_candidates', []).extend({
        'id': item['id'], 'bbox': item['bbox'], 'polygon': item['polygon']} for item in removed)
    data['candidates'] = [item for item in data['candidates'] if item['id'] not in ids]
    return _write(result_dir, _summary(data))
