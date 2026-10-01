import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from track_unwrap import api
from track_unwrap.cracks import (_anomaly_candidates, _automatic_candidate, _compact_dark_loss_candidates,
                                 _dark_fissure_mask, _directional_tear_extensions,
                                 _filter_recurrent_anomalies, _fine_dark_candidates,
                                 _black_core_fraction, _is_sprocket_relief,
                                 _persistent_vertical_shadow,
                                 _periodic_dark_anomaly, _polygon_overlap_fraction,
                                 _repeated_structure_support,
                                 _repair_narrow_authored_seams, _ridge_bridge,
                                 _trace_dark_component,
                                 _suggest_damage_type, add_manual_crack,
                                 delete_crack, delete_cracks, delete_manual_crack, load_cracks, propose_cracks,
                                 review_crack, review_cracks, trace_crack)
from track_unwrap.zones import build_zones


def test_sprocket_lip_and_wall_are_not_damage_candidates():
    # Measured geometry from the 450 mm review photo: the first rectangle is
    # its dark horizontal hole lip, the second is a vertical opening wall.
    anchors = np.array([36., 169., 288., 411., 533., 656., 786., 911., 2392.])
    geometry = lambda box: _is_sprocket_relief({'bbox': box}, (275, 361),
                                                anchors, 125., 3.5, 3.5, 643)
    assert geometry([126., 895.8, 199.5, 66.5])
    assert geometry([8344., 1025.3, 17.5, 185.5])
    assert geometry([1006., 1020., 4., 20.])  # Short opening-wall fragment.
    assert geometry([1006., 1032., 7., 73.5])
    assert not geometry([260., 1015., 45., 95.])  # irregular fissure beside the hole


def test_displayed_area_needs_substantial_black_core_support():
    gray = np.full((100, 100), 90, np.uint8)
    gray[20:30, 20:30] = 48
    broad = {'polygon': [[15, 15], [45, 15], [45, 45], [15, 45]]}
    tight = {'polygon': [[20, 20], [30, 20], [30, 30], [20, 30]]}
    assert _black_core_fraction(gray, broad, 72) < .20
    assert _black_core_fraction(gray, tight, 72) > .80


def test_compact_black_losses_survive_below_one_square_centimeter():
    gray = np.full((160, 160), 150, np.uint8)
    binary = np.zeros_like(gray)
    response = np.zeros_like(gray)
    anomaly = np.zeros_like(gray)
    for x, y, size in ((20, 20, 10), (70, 70, 8), (120, 120, 6)):
        gray[y:y+size, x:x+size] = 42
        binary[y:y+size, x:x+size] = 1
        response[y:y+size, x:x+size] = 95
        anomaly[y:y+size, x:x+size] = 65
    repeated = np.full_like(gray, .78, dtype=np.float32)
    mask = np.ones_like(gray)
    authored = np.full_like(gray, 255)
    group = {'sections': [{'id': 'rubber'}]}
    found = _compact_dark_loss_candidates(gray, binary, response, anomaly,
                                          repeated, mask, authored, group,
                                          1., 1., 2.5, 72, [])
    assert sorted(round(c['bbox'][0]) for c in found) == [20, 70]
    assert all(c['damage_type'] == 'chunk' and 6 <= c['area_mm2'] < 100 for c in found)


def test_small_elongated_dark_loss_is_tear_not_chunk():
    gray = np.full((100, 100), 150, np.uint8)
    binary = np.zeros_like(gray)
    cv2.line(binary, (20, 20), (20, 36), 1, 3)
    cv2.line(binary, (20, 20), (26, 20), 1, 3)
    gray[binary > 0] = 35
    response = np.where(binary > 0, 95, 0).astype(np.uint8)
    anomaly = np.where(binary > 0, 65, 0).astype(np.uint8)
    found = _compact_dark_loss_candidates(
        gray, binary, response, anomaly, np.full_like(gray, .5, dtype=np.float32),
        np.ones_like(gray), np.full_like(gray, 255),
        {'sections': [{'id': 'rubber'}]}, 3.5, 3.5, 5., 72, [])
    assert len(found) == 1
    assert found[0]['damage_type'] == 'tear'


def test_diagonal_tear_extends_from_black_seed_into_lighter_continuous_ridge():
    gray = np.full((250, 250), 180, np.uint8)
    response = np.zeros_like(gray)
    cv2.line(gray, (40, 175), (170, 70), 110, 3)
    cv2.line(response, (40, 175), (170, 70), 45, 3)
    core = np.zeros_like(gray)
    cv2.line(core, (140, 95), (170, 70), 1, 3)
    gray[core > 0] = 40
    response[core > 0] = 100
    contour = max(cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea).reshape(-1, 2)
    polygon = contour.tolist()
    candidate = _automatic_candidate('rubber', polygon, [138, 68, 35, 30],
                                     18., 20., 80., .8, 'tear', 'test', 'dark_core', .2)
    found = _directional_tear_extensions([candidate], gray, response,
                                         np.full_like(gray, 255),
                                         np.zeros_like(gray, dtype=np.float32),
                                         1., 1., 2.5)
    assert found[0]['detection_basis'] == 'extended_dark_ridge'
    assert found[0]['bbox'][0] < 60
    assert found[0]['area_mm2'] > candidate['area_mm2']


def test_molded_groove_continuity_is_not_a_localized_tear():
    gray = np.full((500, 300), 140, np.uint8)
    cv2.line(gray, (80, 10), (80, 490), 45, 5)
    groove = {'bbox': [76, 200, 8, 70], 'repeated_structure_score': .55}
    assert _persistent_vertical_shadow(gray, groove, 2.5, 50, 72)
    gray[:180, 76:85] = 140
    gray[290:, 76:85] = 140
    assert not _persistent_vertical_shadow(gray, groove, 2.5, 50, 72)


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
    assert candidate['status'] == 'accepted'
    assert candidate.get('damage_type') in {'tear', 'chunk', None}
    assert candidate['decision_source'] == 'auto'
    reviewed = review_crack(tmp_path, candidate['id'], 'accepted')
    assert reviewed['totals']['accepted'] == result['totals']['accepted']
    assert reviewed['summary']['surface']['accepted'] == result['summary']['surface']['accepted']
    assert load_cracks(tmp_path)['candidates'][0]['status'] == 'accepted'
    undone = review_crack(tmp_path, candidate['id'], 'excluded')
    assert undone['totals']['accepted'] == result['totals']['accepted'] - 1 and undone['totals']['excluded'] == 1
    manual = add_manual_crack(tmp_path, [[270, 22], [280, 40], [292, 51]])
    assert manual['candidates'][0]['source'] == 'manual'
    assert manual['candidates'][0]['status'] == 'accepted'
    assert manual['totals']['accepted'] == result['totals']['accepted']
    cleared = delete_manual_crack(tmp_path, manual['candidates'][0]['id'])
    assert cleared['totals']['accepted'] == result['totals']['accepted'] - 1


def test_batch_classification_and_area_rule_preserve_explicit_decisions(tmp_path):
    folder = tmp_path / 'cracks'
    folder.mkdir()
    def candidate(cid, area, status='pending', source='automatic'):
        return {'id': cid, 'section_id': 'surface', 'polygon': [[0, 0], [4, 0], [4, 4]],
                'bbox': [0, 0, 4, 4], 'area_mm2': area, 'length_mm': 4, 'score': .5,
                'status': status, 'source': source}
    review = {'version': 2, 'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866'}],
              'pixels_per_mm': 1, 'candidates': [candidate('reference', 10), candidate('bigger', 15),
                candidate('smaller', 9), candidate('ruled_out', 20, 'excluded'),
                candidate('manual', 25, 'accepted', 'manual')]}
    (folder / 'review.json').write_text(json.dumps(review), encoding='utf-8')
    result = review_cracks(tmp_path, ['reference'], 'accepted', 'chunk', True)
    by_id = {item['id']: item for item in result['candidates']}
    assert result['last_review'] == {'selected': 1, 'auto_accepted': 1}
    assert (by_id['reference']['status'], by_id['reference']['damage_type'], by_id['reference']['decision_source']) == ('accepted', 'chunk', 'manual')
    assert by_id['bigger']['status'] == 'accepted' and by_id['bigger']['decision_source'] == 'auto'
    assert by_id['smaller']['status'] == 'pending' and by_id['smaller']['decision_source'] is None
    assert by_id['ruled_out']['status'] == 'excluded'
    assert by_id['manual']['status'] == 'accepted' and by_id['manual']['damage_type'] is None
    assert result['accepted_by_type'] == {'chunk': 2, 'tear': 0, 'unclassified': 1}
    assert result['accepted_area_by_type'] == {'chunk': 25, 'tear': 0}
    changed = review_cracks(tmp_path, ['bigger', 'smaller'], 'excluded')
    assert changed['totals']['excluded'] == 3
    assert changed['accepted_by_type']['chunk'] == 1
    assert changed['accepted_area_by_type']['chunk'] == 10
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


def test_pitch_anomaly_recovers_one_deep_crack_on_a_repeated_edge():
    gray = np.full((120, 500), 150, np.uint8)
    for x in range(0, 500, 50):
        cv2.line(gray, (x+12, 20), (x+12, 95), 80, 3)
    cv2.line(gray, (12, 35), (12, 85), 12, 3)
    anchors = list(range(0, 501, 50))
    anomaly = _periodic_dark_anomaly(gray, anchors, 1.)
    assert anomaly[55, 12] > 35
    assert abs(float(anomaly[55, 112])) < 10
    response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    group = {'sections': [{'id': 'rubber'}]}
    candidates = _anomaly_candidates(gray, response, anomaly, np.ones_like(gray),
                                     np.ones_like(gray), group, 1., 1., 2., 72,
                                     'normal', [])
    assert any(c['bbox'][0] <= 12 <= c['bbox'][0]+c['bbox'][2] for c in candidates)
    assert not any(c['bbox'][0] <= 112 <= c['bbox'][0]+c['bbox'][2] for c in candidates)
    outside = np.zeros_like(gray)
    assert _anomaly_candidates(gray, response, anomaly, np.ones_like(gray),
                               outside, group, 1., 1., 2., 72,
                               'normal', []) == []


def test_pitch_anomaly_discards_tall_repeated_groove_wall():
    gray = np.full((120, 120), 150, np.uint8)
    response = np.zeros_like(gray)
    anomaly = np.zeros_like(gray)
    gray[20:100, 30:45] = 20
    response[20:100, 30:45] = 80
    anomaly[20:100, 30:45] = 70
    mask = np.ones_like(gray)
    group = {'sections': [{'id': 'surface'}]}
    args = (gray, response, anomaly, mask, mask, group, 1., 1., 2.5, 72, 'normal', [])
    assert _anomaly_candidates(*args)
    repeated = np.zeros_like(gray, np.float32)
    repeated[20:100, 30:45] = .9
    assert not _anomaly_candidates(*args, repeated_support=repeated)


def test_pitch_anomaly_discards_short_repeated_horizontal_lip():
    gray = np.full((100, 160), 145, np.uint8)
    gray[40:45, 25:90] = 25
    response = np.zeros_like(gray)
    response[40:45, 25:90] = 70
    anomaly = np.zeros_like(gray)
    anomaly[40:45, 25:90] = 65
    mask = np.ones_like(gray, np.uint8)
    group = {'sections': [{'id': 'surface'}]}
    repeated = np.zeros_like(gray, np.float32)
    repeated[40:45, 25:90] = .95
    assert not _anomaly_candidates(gray, response, anomaly, mask, mask, group,
                                   1., 1., 2.5, 72, 'normal', [], repeated)


def test_anomaly_filter_keeps_isolated_crack_and_discards_repeating_relief():
    def candidate(identifier, x, y, score):
        return {'id': identifier, 'bbox': [x, y, 5, 30], 'score': score}
    proposals = [candidate(f'wall-{i}', i*50 + 10, 10, .6) for i in range(4)]
    proposals += [candidate('unique-crack', 210, 70, .44),
                  candidate('weak-noise', 260, 70, .15)]
    kept = _filter_recurrent_anomalies(proposals, list(range(0, 501, 50)),
                                       50, 120, 'normal')
    assert [item['id'] for item in kept] == ['unique-crack']


def test_short_existing_fragment_does_not_suppress_longer_crack():
    short = {'bbox': [10, 10, 4, 10], 'polygon': [[10, 10], [14, 10], [14, 20], [10, 20]]}
    long = {'bbox': [10, 10, 4, 50], 'polygon': [[10, 10], [14, 10], [14, 60], [10, 60]]}
    assert _polygon_overlap_fraction(long, short, 1, 1) < .45
    assert _polygon_overlap_fraction(short, long, 1, 1) >= .95


def test_fine_pass_finds_small_dark_line_only_inside_authored_area():
    original = np.full((180, 400), 150, np.uint8)
    cv2.line(original, (35, 45), (35, 105), 12, 2)
    cv2.line(original, (39, 45), (39, 105), 230, 1)  # Bright edge beside a real black fissure.
    cv2.line(original, (320, 45), (320, 105), 12, 2)
    authored = np.zeros((90, 200), np.uint8)
    authored[:, :100] = 1
    zones = {'pitch_anchors_x': list(range(0, 401, 50)), 'nominal_pixels_per_mm': 2.}
    group = {'sections': [{'id': 'rubber'}]}
    candidates = _fine_dark_candidates(original, authored, np.ones_like(authored),
                                       zones, group, [], 'normal')
    assert any(c['bbox'][0] <= 35 <= c['bbox'][0]+c['bbox'][2] for c in candidates)
    assert all(c['bbox'][0] + c['bbox'][2] < 200 for c in candidates)


def test_damage_type_hint_requires_geometry_and_exposed_edge():
    assert _suggest_damage_type(18, 30, 4, .05)[0] == 'tear'
    assert _suggest_damage_type(327, 254, 7, .05)[0] == 'tear'  # Long fissure over 1 cm².
    assert _suggest_damage_type(30, 12, 2, .3)[0] == 'chunk'
    assert _suggest_damage_type(120, 13, 1.5, .4)[0] == 'chunk'
    assert _suggest_damage_type(120, 13, 1.5, .05)[0] == 'chunk'


def test_automatic_candidate_records_type_before_size_filtering():
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    def proposal(area):
        return _automatic_candidate('surface', polygon, [0, 0, 10, 10], area, 4, 20, .7,
                                    'chunk', 'compact loss', 'dark_core')
    assert proposal(99.99)['status'] == 'accepted' and proposal(99.99)['damage_type'] == 'chunk'
    assert proposal(100)['status'] == 'accepted' and proposal(100)['damage_type'] == 'chunk'


def test_small_automatic_loss_is_removed_and_legacy_manual_decision_survives(tmp_path):
    folder = tmp_path / 'cracks'
    folder.mkdir()
    def candidate(identifier, source):
        return {'id': identifier, 'section_id': 'surface', 'polygon': [[0, 0], [20, 0], [20, 20], [0, 20]],
                'bbox': [0, 0, 20, 20], 'area_mm2': 64, 'length_mm': 8, 'status': 'accepted',
                'source': source, 'damage_type': 'chip_cut', 'suggested_damage_type': 'chip_cut',
                'decision_source': 'auto' if source == 'automatic' else 'manual'}
    override = candidate('tear_override', 'manual')
    override['damage_type'] = 'tear'
    review = {'version': 1, 'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866'}],
              'pixels_per_mm': 2.5, 'candidates': [candidate('auto', 'automatic'), candidate('user', 'manual'), override]}
    (folder / 'review.json').write_text(json.dumps(review), encoding='utf-8')
    migrated = load_cracks(tmp_path)
    assert migrated['version'] == 2
    assert migrated['chunk_min_area_mm2'] == 100
    assert migrated['totals']['accepted'] == 2 and migrated['totals']['proposed'] == 0
    by_id = {item['id']: item for item in migrated['candidates']}
    assert 'auto' not in by_id
    assert by_id['user']['status'] == 'accepted' and by_id['user']['damage_type'] == 'chunk'
    assert by_id['tear_override']['damage_type'] == 'tear'
    assert migrated['accepted_area_by_type'] == {'chunk': 64, 'tear': 64}


def test_verified_small_dark_loss_survives_review_reload(tmp_path):
    folder = tmp_path / 'cracks'
    folder.mkdir()
    candidate = {'id': 'small_pit', 'section_id': 'surface',
                 'polygon': [[0, 0], [10, 0], [10, 10], [0, 10]],
                 'bbox': [0, 0, 10, 10], 'area_mm2': 16, 'length_mm': 4,
                 'status': 'accepted', 'source': 'automatic',
                 'damage_type': 'chunk', 'suggested_damage_type': 'chunk',
                 'decision_source': 'auto', 'detection_basis': 'compact_dark_loss'}
    review = {'version': 2, 'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866'}],
              'pixels_per_mm': 2.5, 'candidates': [candidate]}
    (folder / 'review.json').write_text(json.dumps(review), encoding='utf-8')
    loaded = load_cracks(tmp_path)
    assert loaded['accepted_by_type']['chunk'] == 1
    assert loaded['candidates'][0]['id'] == 'small_pit'


def test_automatic_tears_below_one_centimeter_are_not_counted(tmp_path):
    folder = tmp_path / 'cracks'
    folder.mkdir()
    base = {'section_id': 'surface', 'polygon': [[0, 0], [20, 0], [20, 2], [0, 2]],
            'bbox': [0, 0, 20, 2], 'area_mm2': 2, 'status': 'accepted',
            'source': 'automatic', 'damage_type': 'tear', 'suggested_damage_type': 'tear'}
    review = {'version': 2, 'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866'}],
              'pixels_per_mm': 2.5,
              'candidates': [dict(base, id='short_auto', length_mm=9.9, decision_source='auto'),
                             dict(base, id='at_limit', length_mm=10, decision_source='auto'),
                             dict(base, id='short_reviewed', length_mm=4,
                                  decision_source='manual')]}
    (folder / 'review.json').write_text(json.dumps(review), encoding='utf-8')
    loaded = load_cracks(tmp_path)
    assert loaded['tear_min_length_mm'] == 10
    assert {item['id'] for item in loaded['candidates']} == {'at_limit', 'short_reviewed'}
    assert loaded['accepted_by_type']['tear'] == 2


def test_configured_tear_length_range_keeps_manual_review(tmp_path):
    folder = tmp_path / 'cracks'
    folder.mkdir()
    base = {'section_id': 'surface', 'polygon': [[0, 0], [20, 0], [20, 2], [0, 2]],
            'bbox': [0, 0, 20, 2], 'area_mm2': 2, 'status': 'accepted',
            'source': 'automatic', 'damage_type': 'tear', 'suggested_damage_type': 'tear'}
    review = {'version': 2, 'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866'}],
              'pixels_per_mm': 2.5, 'tear_min_length_mm': 5, 'tear_max_length_mm': 20,
              'candidates': [dict(base, id='too_short', length_mm=4.9, decision_source='auto'),
                             dict(base, id='lower_limit', length_mm=5, decision_source='auto'),
                             dict(base, id='upper_limit', length_mm=20, decision_source='auto'),
                             dict(base, id='too_long', length_mm=20.1, decision_source='auto'),
                             dict(base, id='long_reviewed', length_mm=21, decision_source='manual')]}
    (folder / 'review.json').write_text(json.dumps(review), encoding='utf-8')
    loaded = load_cracks(tmp_path)
    assert (loaded['tear_min_length_mm'], loaded['tear_max_length_mm']) == (5, 20)
    assert {item['id'] for item in loaded['candidates']} == {
        'lower_limit', 'upper_limit', 'long_reviewed'}
    assert loaded['accepted_by_type']['tear'] == 3


def test_tear_length_range_validation_is_applied_before_detection(tmp_path):
    with pytest.raises(ValueError, match='하한'):
        propose_cracks(tmp_path, 'geometry', tear_min_length_mm=-1)
    with pytest.raises(ValueError, match='상한'):
        propose_cracks(tmp_path, 'geometry', tear_min_length_mm=10, tear_max_length_mm=9.9)


def test_chalk_colour_change_does_not_produce_automatic_chunk(tmp_path):
    color = np.full((120, 500, 3), 155, np.uint8)
    cv2.putText(color, '10', (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (235, 235, 235), 3)
    cv2.imwrite(str(tmp_path / 'panorama.png'), color)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 48, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [499, 0], [499, 119], [0, 119]])
    result = propose_cracks(tmp_path, 'geometry')
    assert not any(item['damage_type'] == 'chunk' for item in result['candidates'])
    assert not any(item['detection_basis'] == 'material_change' for item in result['candidates'])
    assert result['totals']['proposed'] == 0


def test_legacy_material_only_proposals_are_retired_without_losing_manual_override(tmp_path):
    folder = tmp_path / 'cracks'
    folder.mkdir()
    candidate = {'section_id': 'surface', 'polygon': [[0, 0], [20, 0], [20, 20], [0, 20]],
                 'bbox': [0, 0, 20, 20], 'area_mm2': 400, 'length_mm': 20,
                 'source': 'automatic', 'status': 'accepted', 'damage_type': 'chunk',
                 'suggested_damage_type': 'chunk', 'detection_basis': 'material_change'}
    review = {'version': 2, 'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866'}],
              'pixels_per_mm': 1, 'candidates': [dict(candidate, id='auto', decision_source='auto'),
                                               dict(candidate, id='edited', decision_source='manual')]}
    (folder / 'review.json').write_text(json.dumps(review), encoding='utf-8')
    result = load_cracks(tmp_path)
    assert [item['id'] for item in result['candidates']] == ['edited']
    assert result['accepted_area_by_type']['chunk'] == 400


def test_drawn_line_and_closed_area_classify_and_refine_in_place(tmp_path, monkeypatch):
    image = np.full((400, 400, 3), 155, np.uint8)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 160, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [399, 0], [399, 399], [0, 399]])
    propose_cracks(tmp_path, 'geometry')
    line = add_manual_crack(tmp_path, [[40, 40], [50, 55], [70, 85]])
    assert line['candidates'][0]['damage_type'] == 'tear'
    small = add_manual_crack(tmp_path, [[105, 105], [125, 105], [125, 125], [105, 125]], closed=True)
    assert small['candidates'][0]['damage_type'] == 'chunk'
    assert small['candidates'][0]['area_mm2'] == 64
    identifier = small['candidates'][0]['id']
    large = add_manual_crack(tmp_path, [[105, 105], [145, 105], [145, 145], [105, 145]],
                             closed=True, candidate_id=identifier)
    assert len(large['candidates']) == 2
    assert next(item for item in large['candidates'] if item['id'] == identifier)['damage_type'] == 'chunk'
    assert large['accepted_by_type']['chunk'] == 1 and large['accepted_by_type']['tear'] == 1
    assert large['summary']['surface']['area_mm2'] == large['totals']['area_mm2']
    assert large['totals']['area_mm2'] >= next(item for item in large['candidates'] if item['id'] == identifier)['area_mm2']
    monkeypatch.setattr(api, '_zones_dir', lambda _: tmp_path)
    with TestClient(api.app) as client:
        response = client.post('/api/jobs/' + 'a'*32 + '/cracks/manual', json={
            'points': [[105, 105], [145, 105], [145, 145], [105, 145]],
            'closed': True, 'candidate_id': identifier})
    assert response.status_code == 200
    assert len(response.json()['candidates']) == 2
    assert response.json()['accepted_by_type']['chunk'] == 1
    with pytest.raises(ValueError, match='손상 면적'):
        add_manual_crack(tmp_path, [[210, 100], [220, 100], [230, 100]], closed=True)


def test_manual_damage_crosses_small_unlabeled_seam_but_not_large_opening(tmp_path):
    image = np.full((200, 200, 3), 155, np.uint8)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [199, 0], [199, 199], [0, 199]])
    propose_cracks(tmp_path, 'geometry')
    authored_path = tmp_path / 'zones' / 'group_geometry_authored.png'
    authored = cv2.imread(str(authored_path), cv2.IMREAD_GRAYSCALE)
    authored[20:110, 55:58] = 0  # A narrow gap between drawn region polygons.
    authored[30:120, 110:170] = 0  # A genuinely unmarked opening.
    cv2.imwrite(str(authored_path), authored)
    line = add_manual_crack(tmp_path, [[40, 45], [70, 75]])
    assert line['candidates'][0]['damage_type'] == 'tear'
    closed = add_manual_crack(tmp_path,
                              [[40, 50], [73, 50], [73, 85], [40, 85]], closed=True)
    assert closed['candidates'][0]['status'] == 'accepted'
    with pytest.raises(ValueError, match='지정된 영역에서 너무 멉니다'):
        add_manual_crack(tmp_path, [[125, 40], [145, 90]])


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
    with pytest.raises(ValueError, match='검은 핵심부'):
        _trace_dark_component(gray, valid, (10, 70), 55, 0)
    _, _, snapped = _trace_dark_component(gray, valid, (10, 70), 55, 0, 35)
    assert snapped[0] >= 33


def test_shallow_offset_fissure_bridges_but_parallel_or_blank_regions_do_not():
    gray = np.full((90, 135), 155, np.uint8)
    response = np.zeros_like(gray)
    valid = np.ones_like(gray)
    cv2.line(gray, (15, 35), (50, 35), 18, 3)
    cv2.line(gray, (72, 39), (110, 39), 18, 3)
    cv2.line(gray, (50, 35), (72, 39), 125, 2)
    cv2.line(response, (50, 35), (72, 39), 20, 2)
    first = np.asarray([(x, 35) for x in range(15, 51)], np.int32)
    second = np.asarray([(x, 39) for x in range(72, 111)], np.int32)
    route = _ridge_bridge(gray, response, valid, first, second, 30)
    assert route is not None
    assert route[0, 0] == 50 and route[-1, 0] == 72
    assert np.max(route[:, 1]) <= 41
    response[:] = 0
    assert _ridge_bridge(gray, response, valid, first, second, 30) is None
    cv2.line(response, (50, 35), (72, 39), 20, 2)
    valid[:, 61:64] = 0
    assert _ridge_bridge(gray, response, valid, first, second, 30) is None

    # A textured strip between adjacent parallel cracks is not continuation.
    parallel = np.asarray([(55, y) for y in range(10, 61)], np.int32)
    upright = np.asarray([(40, y) for y in range(10, 61)], np.int32)
    assert _ridge_bridge(gray, response, np.ones_like(valid), upright, parallel, 30) is None


def test_guided_selection_crosses_thin_zone_seam_but_not_large_hole():
    gray = np.full((130, 120), 155, np.uint8)
    cv2.line(gray, (35, 12), (35, 115), 15, 4)
    authored = np.ones_like(gray)
    authored[61:68, :] = 0
    authored[10:42, 70:105] = 0
    _, before, _ = _trace_dark_component(gray, authored, (35, 25), 20, 0)
    assert not before[95, 35]
    repaired = _repair_narrow_authored_seams(authored, 5)
    assert repaired[64, 35] and not repaired[25, 85]
    _, after, _ = _trace_dark_component(gray, repaired, (35, 25), 20, 0)
    assert after[95, 35]
    assert not after[25, 85]


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
        trace_crack(tmp_path, [230, 70], 'chunk', 1.5, 30)


def test_guided_wide_tear_is_previewed_with_warning_instead_of_blocked(tmp_path):
    image = np.full((200, 300, 3), 155, np.uint8)
    cv2.rectangle(image, (110, 60), (135, 110), (15, 15, 15), -1)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [299, 0], [299, 199], [0, 199]])
    propose_cracks(tmp_path, 'geometry', 'normal')
    with pytest.raises(ValueError, match='검은 핵심부'):
        trace_crack(tmp_path, [75, 85], 'tear', .4, 30)
    preview = trace_crack(tmp_path, [75, 85], 'tear', .4, 30, seed_radius_px=50)
    assert preview['seed'][0] >= 110
    assert preview['area_mm2'] > 0
    assert '선택 폭이 넓습니다' in preview['warning']


def test_guided_refinement_can_extend_beyond_old_candidate_box(tmp_path):
    image = np.full((320, 320, 3), 155, np.uint8)
    cv2.line(image, (100, 20), (100, 300), (15, 15, 15), 4)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [319, 0], [319, 319], [0, 319]])
    review = propose_cracks(tmp_path, 'geometry', 'normal')
    review['candidates'] = [{'id': 'truncated', 'status': 'pending', 'source': 'automatic',
                             'section_id': 'surface', 'score': 1., 'contrast': 100.,
                             'bbox': [94, 20, 13, 46],
                             'polygon': [[94, 20], [106, 20], [106, 65], [94, 65]]}]
    (tmp_path / 'cracks' / 'review.json').write_text(json.dumps(review), encoding='utf-8')
    preview = trace_crack(tmp_path, None, 'tear', 0, 20, candidate_id='truncated')
    assert preview['bbox'][1] + preview['bbox'][3] >= 295


def test_dragged_dark_region_can_be_extracted_and_accepted(tmp_path):
    image = np.full((220, 420, 3), 150, np.uint8)
    cv2.line(image, (75, 35), (75, 170), (10, 10, 10), 4)
    cv2.line(image, (325, 35), (325, 170), (10, 10, 10), 4)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [210, 0], [210, 219], [0, 219]])
    propose_cracks(tmp_path, 'geometry', 'normal')
    preview = trace_crack(tmp_path, None, 'tear', 0, 30, region=[55, 25, 95, 190])
    assert preview['bbox'][0] < 75 < preview['bbox'][0] + preview['bbox'][2]
    assert preview['bbox'][0] + preview['bbox'][2] < 210
    saved = trace_crack(tmp_path, None, 'tear', 0, 30, region=[55, 25, 95, 190], apply=True, accept=True)
    assert saved['totals']['accepted'] == 1
    assert saved['candidates'][0]['damage_type'] == 'tear'
    with pytest.raises(ValueError, match='지정된 분석 영역 밖'):
        trace_crack(tmp_path, None, 'tear', 0, 30, region=[300, 25, 350, 190])


def test_deleted_automatic_candidate_stays_removed_after_reproposal(tmp_path):
    image = np.full((180, 400, 3), 150, np.uint8)
    cv2.line(image, (75, 35), (75, 145), (10, 10, 10), 4)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [399, 0], [399, 179], [0, 179]])
    first = propose_cracks(tmp_path, 'geometry', 'high')
    target = first['candidates'][0]
    deleted = delete_crack(tmp_path, target['id'])
    assert target['id'] not in {item['id'] for item in deleted['candidates']}
    again = propose_cracks(tmp_path, 'geometry', 'high')
    assert not any(abs(item['bbox'][0] - target['bbox'][0]) < 10 for item in again['candidates'])


def test_batch_delete_is_atomic_and_remembers_each_candidate(tmp_path):
    image = np.full((180, 400, 3), 150, np.uint8)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    _test_zones(tmp_path, [[0, 0], [399, 0], [399, 179], [0, 179]])
    propose_cracks(tmp_path, 'geometry', 'low')
    add_manual_crack(tmp_path, [[50, 30], [51, 80], [52, 120]])
    second = add_manual_crack(tmp_path, [[250, 30], [251, 80], [252, 120]])
    ids = [item['id'] for item in second['candidates']]
    assert len(ids) == 2
    with pytest.raises(ValueError, match='찾을 수 없습니다'):
        delete_cracks(tmp_path, [ids[0], 'missing'])
    assert {item['id'] for item in load_cracks(tmp_path)['candidates']} == set(ids)
    deleted = delete_cracks(tmp_path, ids)
    assert deleted['candidates'] == []
    assert {item['id'] for item in deleted['dismissed_candidates']} == set(ids)


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
    assert again['totals']['excluded'] == 0
    assert again['totals']['accepted'] >= 1
    assert reviewed['candidates'][0]['id'] not in {item['id'] for item in again['candidates']}
    assert manual['candidates'][0]['id'] in {item['id'] for item in again['candidates']}


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

        fresh = client.post(f'/api/jobs/{job_id}/cracks/propose', json={
            'group_id': 'geometry', 'sensitivity': 'high',
            'tear_min_length_mm': 5, 'tear_max_length_mm': 100}).json()
        assert fresh['candidates']
        assert (fresh['tear_min_length_mm'], fresh['tear_max_length_mm']) == (5, 100)
        assert fresh['totals']['excluded'] == 0 and fresh['totals']['accepted'] > 0
        assert all(item['status'] == 'accepted' and item['damage_type'] for item in fresh['candidates'])
        assert all(5 <= item['length_mm'] <= 100 for item in fresh['candidates']
                   if item['source'] == 'automatic' and item['damage_type'] == 'tear')


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
