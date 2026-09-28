import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from track_unwrap import api
from track_unwrap.damage import (ZONE_CODES, DamageConfig, _component_candidates,
                                 add_manual_candidate, analyze_damage, load_analysis, save_analysis)


def make_panorama(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    ppm, pitch_mm, width_mm, periods = 5, 20, 80, 8
    pitch, height = pitch_mm * ppm, width_mm * ppm
    tile = np.full((height, pitch, 3), 130, np.uint8)
    tile[:175, 9:19] = 77
    tile[225:, 9:19] = 77
    tile[179:221] = 110
    tile[184:215, 38:63] = 30
    image = np.tile(tile, (1, periods, 1))
    # Deliberate damage in different pitches, safely away from the design edges.
    cv2.circle(image, (2 * pitch + 77, 89), 32, (25, 25, 25), -1)
    cv2.line(image, (5 * pitch + 27, 285), (5 * pitch + 27, 390), (20, 20, 20), 8)
    cv2.imwrite(str(folder / 'panorama.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    (folder / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': width_mm, 'pitch_mm': pitch_mm, 'total_links': None,
                     'pixels_per_mm': ppm}}), encoding='utf-8')
    return image


def test_pitch_repeated_zone_seed_and_damage_candidates(tmp_path):
    image = make_panorama(tmp_path)
    seed = {'zone': 'groove', 'polygon': [[125, 300], [145, 300], [145, 320], [125, 320]]}
    analysis = analyze_damage(tmp_path, [seed])
    mask = cv2.imread(str(tmp_path / 'damage' / 'zone_mask.png'), cv2.IMREAD_GRAYSCALE)
    for offset in range(8):
        assert mask[124, 14 + 40 * offset] == ZONE_CODES['groove']
    assert analysis['zone_detection']['hole_components_per_pitch'] >= 1
    assert analysis['summary']['total']['damage_percent'] > 0
    assert any(c['mode'] == 'chunk' and abs(c['bbox_xywh'][0] - 245) < 80 for c in analysis['candidates'])
    assert any(c['mode'] == 'tear' and 480 < c['bbox_xywh'][0] < 560 for c in analysis['candidates'])
    assert analysis['image_size_wh'] == [image.shape[1], image.shape[0]]


def test_auto_hole_is_rectangular_interior_and_core_is_remaining_center_rubber(tmp_path):
    make_panorama(tmp_path)
    analyze_damage(tmp_path)
    mask = cv2.imread(str(tmp_path / 'damage' / 'zone_mask.png'), cv2.IMREAD_GRAYSCALE)
    assert mask[80, 20] == ZONE_CODES['sprocket_hole']
    assert mask[80, 13] == ZONE_CODES['embedded_core']
    assert mask[80, 30] == ZONE_CODES['embedded_core']
    assert mask[70, 20] == ZONE_CODES['embedded_core']
    assert mask[90, 20] == ZONE_CODES['tread']


def test_manual_hole_replaces_auto_hole_across_pitches(tmp_path):
    make_panorama(tmp_path)
    seed = {'zone': 'sprocket_hole', 'polygon': [[125, 185], [145, 185], [145, 213], [125, 213]]}
    analyze_damage(tmp_path, [seed])
    mask = cv2.imread(str(tmp_path / 'damage' / 'zone_mask.png'), cv2.IMREAD_GRAYSCALE)
    assert mask[80, 14] == ZONE_CODES['sprocket_hole']
    assert mask[80, 54] == ZONE_CODES['sprocket_hole']
    assert mask[80, 23] != ZONE_CODES['sprocket_hole']
    assert (tmp_path / 'damage' / 'zone_overlay.png').is_file()


def test_core_and_hole_seeds_replace_both_auto_shapes(tmp_path):
    make_panorama(tmp_path)
    seeds = [
        {'zone': 'embedded_core', 'polygon': [[120, 170], [150, 170], [150, 230], [120, 230]]},
        {'zone': 'sprocket_hole', 'polygon': [[125, 185], [140, 185], [140, 215], [125, 215]]},
    ]
    analyze_damage(tmp_path, seeds)
    mask = cv2.imread(str(tmp_path / 'damage' / 'zone_mask.png'), cv2.IMREAD_GRAYSCALE)
    assert mask[80, 14] == ZONE_CODES['sprocket_hole']
    assert mask[80, 19] == ZONE_CODES['embedded_core']
    assert mask[80, 24] == ZONE_CODES['tread']


def test_manual_decision_and_mode_preserved_on_reanalysis(tmp_path):
    make_panorama(tmp_path)
    analysis = analyze_damage(tmp_path)
    initial = analysis['summary']['total']['damaged_area_mm2']
    target = next(c for c in analysis['candidates'] if c['mode'] == 'chunk')
    target['included'] = False
    target['reviewed'] = True
    analysis['active_modes'] = ['chunk']
    save_analysis(tmp_path, analysis)
    assert load_analysis(tmp_path)['summary']['total']['damaged_area_mm2'] < initial
    manual = add_manual_candidate(tmp_path, load_analysis(tmp_path), 'tear',
                                  [[330, 300], [345, 300], [345, 370], [330, 370]])
    manual_id = next(c['id'] for c in manual['candidates'] if c['source'] == 'manual')
    rerun = analyze_damage(tmp_path)
    assert rerun['active_modes'] == ['chunk']
    assert next(c for c in rerun['candidates'] if c['id'] == target['id'])['included'] is False
    assert any(c['id'] == manual_id for c in rerun['candidates'])


def test_long_narrow_damage_is_tear_even_above_chunk_area():
    mask = np.zeros((200, 100), np.uint8)
    cv2.rectangle(mask, (30, 20), (35, 170), 1, -1)
    residual = np.full(mask.shape, 50, np.float32)
    zones = np.full(mask.shape, ZONE_CODES['tread'], np.uint8)
    config = DamageConfig()
    assert not _component_candidates(mask, residual, zones, 'chunk', 2, 1, 1, config)
    assert len(_component_candidates(mask, residual, zones, 'tear', 2, 1, 1, config)) == 1


def test_damage_api_review_flow(tmp_path, monkeypatch):
    job_id = 'a' * 32
    result = tmp_path / job_id / 'result'
    make_panorama(result)
    monkeypatch.setattr(api, 'JOBS', tmp_path)
    state = {'id': job_id, 'status': 'complete', 'created_at': '2026-01-01T00:00:00+00:00',
             'settings': {'width_mm': 80, 'pitch_mm': 20, 'total_links': None, 'pixels_per_mm': 5}}
    (tmp_path / job_id / 'job.json').write_text(json.dumps(state), encoding='utf-8')
    with TestClient(api.app) as client:
        response = client.post(f'/api/jobs/{job_id}/damage', json={})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['summary']['total']['damage_percent'] > 0
        candidate = next(item for item in data['candidates'] if item['included'])
        response = client.patch(f"/api/jobs/{job_id}/damage/candidates/{candidate['id']}", json={'included': False})
        assert response.status_code == 200
        assert next(c for c in response.json()['candidates'] if c['id'] == candidate['id'])['included'] is False
        assert response.json()['summary']['total']['damaged_area_mm2'] < data['summary']['total']['damaged_area_mm2']
        assert client.patch(f'/api/jobs/{job_id}/damage/modes', json={'active_modes': ['tear']}).status_code == 200
        assert client.post(f'/api/jobs/{job_id}/damage/candidates', json={
            'mode': 'chunk', 'polygon': [[300, 300], [330, 300], [330, 330], [300, 330]]}).status_code == 200
        assert client.get(f'/api/jobs/{job_id}/damage/files/zone_mask.png').status_code == 200
        assert client.get(f'/api/jobs/{job_id}/damage').status_code == 200
