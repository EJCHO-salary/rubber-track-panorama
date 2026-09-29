import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from track_unwrap import api
from track_unwrap.cracks import (_dark_fissure_mask, _repeated_structure_support,
                                 _trace_dark_component,
                                 _suggest_damage_type, add_manual_crack,
                                 delete_manual_crack, load_cracks, propose_cracks,
                                 review_crack, review_cracks, trace_crack)
from track_unwrap.zones import build_zones


def _test_zones(folder, polygon=None):
    taxonomy = [{'id': 'geometry', 'name': '형상', 'default_section_id': 'surface',
                 'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866', 'repeat_mode': 'independent'}]}]
    shapes = [] if polygon is None else [{'id': 'drawn', 'group_id': 'geometry', 'section_id': 'surface',
                                         'repeat': False, 'repeat_pitches': 1, 'polygon': polygon}]
    return build_zones(folder, taxonomy, shapes)


def test_unique_fissure_can_be_reviewed_and_counted_by_section(tmp_path):
    image = np.full((200, 500, 3), 155, np.uint8)
    for x in range(0, 500, 50):
        cv2.rectangle(image, (x+14, 70), (x+25, 130), (45, 45, 45), -1)
    cv2.line(image, (183, 22), (211, 61), (15, 15, 15), 3)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [499, 0], [499, 199], [0, 199]])
    result = propose_cracks(tmp_path, 'geometry', 'high')
    fissures = [item for item in result['candidates'] if item['bbox'][0] <= 195 <= item['bbox'][0]+item['bbox'][2]
                and item['bbox'][1] <= 45 <= item['bbox'][1]+item['bbox'][3]]
    assert fissures, 'unique dark fissure should have a reviewable proposal'
    candidate = fissures[0]
    assert candidate['status'] == 'pending'
    assert candidate['damage_type'] is None
    assert 'suggested_damage_type' in candidate
    reviewed = review_crack(tmp_path, candidate['id'], 'accepted')
    assert reviewed['totals']['accepted'] == 1
    assert reviewed['summary']['surface']['accepted'] == 1
    assert load_cracks(tmp_path)['candidates'][0]['status'] in {'pending', 'accepted'}
    undone = review_crack(tmp_path, candidate['id'], 'excluded')
    assert undone['totals']['accepted'] == 0 and undone['totals']['excluded'] == 1
    manual = add_manual_crack(tmp_path, [[270, 22], [280, 40], [292, 51]])
    assert manual['candidates'][0]['source'] == 'manual'
    assert manual['candidates'][0]['status'] == 'accepted'
    assert manual['totals']['accepted'] == 1
    cleared = delete_manual_crack(tmp_path, manual['candidates'][0]['id'])
    assert cleared['totals']['accepted'] == 0


def test_batch_classification_and_area_rule_preserve_explicit_decisions(tmp_path):
    folder = tmp_path / 'cracks'
    folder.mkdir()
    def candidate(cid, area, status='pending', source='automatic'):
        return {'id': cid, 'section_id': 'surface', 'polygon': [[0, 0], [4, 0], [4, 4]],
                'bbox': [0, 0, 4, 4], 'area_mm2': area, 'length_mm': 4, 'score': .5,
                'status': status, 'source': source}
    review = {'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866'}],
              'pixels_per_mm': 1, 'candidates': [candidate('reference', 10), candidate('bigger', 15),
                candidate('smaller', 9), candidate('ruled_out', 20, 'excluded'),
                candidate('manual', 25, 'accepted', 'manual')]}
    (folder / 'review.json').write_text(json.dumps(review), encoding='utf-8')
    result = review_cracks(tmp_path, ['reference'], 'accepted', 'chunk', True)
    by_id = {item['id']: item for item in result['candidates']}
    assert result['last_review'] == {'selected': 1, 'auto_accepted': 1}
    assert (by_id['reference']['status'], by_id['reference']['damage_type'], by_id['reference']['decision_source']) == ('accepted', 'chunk', 'manual')
    assert (by_id['bigger']['status'], by_id['bigger']['damage_type'], by_id['bigger']['decision_source']) == ('accepted', 'chunk', 'auto')
    assert by_id['smaller']['status'] == 'pending'
    assert by_id['ruled_out']['status'] == 'excluded'
    assert by_id['manual']['status'] == 'accepted' and by_id['manual']['damage_type'] is None
    assert result['accepted_by_type'] == {'chip_cut': 0, 'chunk': 2, 'tear': 0, 'unclassified': 1}
    changed = review_cracks(tmp_path, ['bigger', 'smaller'], 'excluded')
    assert changed['totals']['excluded'] == 3
    assert changed['accepted_by_type']['chunk'] == 1
    assert load_cracks(tmp_path)['candidates'][1]['decision_source'] == 'manual'


def test_dark_core_rejects_bright_relief_and_chalk_edge():
    gray = np.full((80, 100), 145, np.uint8)
    response = np.zeros_like(gray)
    mask = np.ones_like(gray)
    gray[12:45, 15:18] = 25
    response[12:45, 15:18] = 80
    gray[12:45, 50:53] = 105
    response[12:45, 50:53] = 80
    gray[12:45, 75:78] = 25
    gray[12:45, 78:81] = 230
    response[12:45, 75:78] = 80
    binary, core_limit = _dark_fissure_mask(gray, response, mask, 'normal')
    assert core_limit <= 72
    assert binary[20, 16] == 1
    assert binary[20, 51] == 0
    assert binary[20, 76] == 0


def test_repeated_geometry_does_not_suppress_unique_fissure():
    gray = np.full((100, 500), 160, np.uint8)
    response = np.zeros_like(gray)
    for pitch in range(10):
        x = pitch * 50 + 10
        gray[10:60, x:x+4] = 35
        response[10:60, x:x+4] = 80
    gray[65:95, 282:285] = 15
    response[65:95, 282:285] = 80
    support = _repeated_structure_support(gray, response, list(range(0, 501, 50)), 1.)
    assert support[30, 62] >= .75
    assert support[80, 283] < .2


def test_damage_type_hint_requires_geometry_and_exposed_edge():
    assert _suggest_damage_type(18, 30, 4, .05)[0] == 'tear'
    assert _suggest_damage_type(30, 12, 2, .3)[0] == 'chip_cut'
    assert _suggest_damage_type(120, 13, 1.5, .4)[0] == 'chunk'
    assert _suggest_damage_type(120, 13, 1.5, .05)[0] is None


def test_guided_selection_bridges_short_gap_without_neighbor_and_offsets_outline():
    gray = np.full((150, 300), 160, np.uint8)
    cv2.line(gray, (35, 70), (140, 70), 18, 3)
    cv2.line(gray, (147, 70), (250, 70), 18, 3)
    cv2.line(gray, (35, 105), (250, 105), 18, 3)
    valid = np.ones_like(gray)
    contour, selected, _ = _trace_dark_component(gray, valid, (70, 70), 55, 0)
    x, y, width, height = cv2.boundingRect(contour)
    assert x <= 35 and x + width >= 250
    assert y < 70 < y + height and y + height < 95
    assert not selected[105, 70]
    expanded, _, _ = _trace_dark_component(gray, valid, (70, 70), 55, 4)
    assert cv2.contourArea(expanded) > cv2.contourArea(contour)


def test_guided_preview_is_read_only_and_apply_adds_reviewable_candidate(tmp_path, monkeypatch):
    image = np.full((200, 400, 3), 155, np.uint8)
    cv2.line(image, (80, 70), (200, 70), (15, 15, 15), 3)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [399, 0], [399, 199], [0, 199]])
    original = propose_cracks(tmp_path, 'geometry', 'high')
    count = len(original['candidates'])
    monkeypatch.setattr(api, '_zones_dir', lambda _: tmp_path)
    request = {'point': [100, 70], 'damage_type': 'tear', 'offset_mm': .4, 'tolerance': 30}
    with TestClient(api.app) as client:
        preview_response = client.post('/api/jobs/' + 'a'*32 + '/cracks/trace', json=request)
        assert preview_response.status_code == 200
        preview = preview_response.json()
        assert len(load_cracks(tmp_path)['candidates']) == count
        apply_response = client.post('/api/jobs/' + 'a'*32 + '/cracks/trace/apply', json=request)
        assert apply_response.status_code == 200
        applied = apply_response.json()
    assert preview['bbox'][0] <= 100 and preview['area_mm2'] > 0
    assert len(applied['candidates']) == count + 1
    assert applied['candidates'][0]['source'] == 'manual'
    assert applied['candidates'][0]['status'] == 'pending'
    assert applied['candidates'][0]['suggested_damage_type'] == 'tear'
    retained = propose_cracks(tmp_path, 'geometry', 'normal')
    assert applied['candidates'][0]['id'] in {item['id'] for item in retained['candidates']}
    if original['candidates']:
        candidate_id = original['candidates'][0]['id']
        changed = trace_crack(tmp_path, None, 'tear', .4, 30, candidate_id, apply=True)
        assert next(item for item in changed['candidates'] if item['id'] == candidate_id)['selection_source'] == 'guided'
        again = propose_cracks(tmp_path, 'geometry', 'high')
        assert sum(item['id'] == candidate_id for item in again['candidates']) == 1
    with pytest.raises(ValueError, match='검은 핵심부'):
        trace_crack(tmp_path, [100, 195], 'tear', .4, 30)


def test_guided_click_rejects_dark_damage_outside_authored_zone(tmp_path):
    image = np.full((200, 300, 3), 155, np.uint8)
    cv2.line(image, (210, 70), (265, 70), (12, 12, 12), 4)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 3},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [145, 0], [145, 199], [0, 199]])
    propose_cracks(tmp_path, 'geometry', 'normal')
    with pytest.raises(ValueError, match='지정된 영역'):
        trace_crack(tmp_path, [230, 70], 'chip_cut', 1.5, 30)


def test_reproposal_keeps_reviewed_candidate_and_manual_path(tmp_path):
    image = np.full((200, 500, 3), 155, np.uint8)
    cv2.line(image, (183, 22), (211, 61), (15, 15, 15), 3)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [499, 0], [499, 199], [0, 199]])
    first = propose_cracks(tmp_path, 'geometry', 'high')
    assert first['candidates']
    reviewed = review_crack(tmp_path, first['candidates'][0]['id'], 'excluded')
    manual = add_manual_crack(tmp_path, [[270, 22], [280, 40], [292, 51]])
    again = propose_cracks(tmp_path, 'geometry', 'normal')
    assert again['totals']['excluded'] == 1
    assert again['totals']['accepted'] == 1
    assert {item['id'] for item in again['candidates'] if item['status'] != 'pending'} == {
        reviewed['candidates'][0]['id'], manual['candidates'][0]['id']}


def test_clear_all_candidates_then_repropose_from_current_detector(tmp_path, monkeypatch):
    image = np.full((200, 500, 3), 155, np.uint8)
    cv2.line(image, (183, 22), (211, 61), (15, 15, 15), 3)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [499, 0], [499, 199], [0, 199]])
    zone_path = tmp_path / 'zones' / 'analysis.json'
    monkeypatch.setattr(api, '_zones_dir', lambda _: tmp_path)
    job_id = 'a' * 32
    with TestClient(api.app) as client:
        first = client.post(f'/api/jobs/{job_id}/cracks/propose', json={'group_id': 'geometry', 'sensitivity': 'high'}).json()
        assert first['candidates']
        review_crack(tmp_path, first['candidates'][0]['id'], 'excluded')
        add_manual_crack(tmp_path, [[270, 22], [280, 40], [292, 51]])
        assert client.get(f'/api/jobs/{job_id}/cracks').json()['totals']['excluded'] == 1

        cleared = client.delete(f'/api/jobs/{job_id}/cracks')
        assert cleared.status_code == 200 and cleared.json() == {'ready': False, 'stale': False}
        assert client.get(f'/api/jobs/{job_id}/cracks').json() == {'ready': False, 'stale': False}
        assert not (tmp_path / 'cracks' / 'review.json').exists()
        assert zone_path.exists()
        assert client.delete(f'/api/jobs/{job_id}/cracks').status_code == 200

        fresh = client.post(f'/api/jobs/{job_id}/cracks/propose', json={'group_id': 'geometry', 'sensitivity': 'high'}).json()
        assert fresh['candidates']
        assert fresh['totals']['excluded'] == fresh['totals']['accepted'] == 0
        assert all(item['status'] == 'pending' and item['source'] == 'automatic' for item in fresh['candidates'])


def test_only_drawn_area_produces_crack_candidates(tmp_path):
    image = np.full((200, 500, 3), 155, np.uint8)
    cv2.line(image, (183, 22), (211, 61), (15, 15, 15), 3)
    cv2.line(image, (320, 22), (348, 61), (15, 15, 15), 3)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path)
    assert propose_cracks(tmp_path, 'geometry', 'high')['candidates'] == []
    # The fallback classification mask labels the entire image as surface, but
    # the explicitly drawn mask covers only the first dark fissure.
    _test_zones(tmp_path, [[155, 8], [235, 8], [235, 80], [155, 80]])
    result = propose_cracks(tmp_path, 'geometry', 'high')
    assert result['candidates']
    assert all(item['bbox'][0] + item['bbox'][2] < 235 for item in result['candidates'])
    saved = load_cracks(tmp_path)
    obsolete = dict(saved['candidates'][0], id='outside', status='excluded')
    obsolete['polygon'] = [[x + 137, y] for x, y in obsolete['polygon']]
    obsolete['bbox'] = [obsolete['bbox'][0] + 137, *obsolete['bbox'][1:]]
    saved['candidates'].append(obsolete)
    (tmp_path / 'cracks' / 'review.json').write_text(json.dumps(saved), encoding='utf-8')
    regenerated = propose_cracks(tmp_path, 'geometry', 'high')
    assert 'outside' not in {item['id'] for item in regenerated['candidates']}
    with pytest.raises(ValueError, match='지정된 영역'):
        add_manual_crack(tmp_path, [[320, 22], [348, 61]])
