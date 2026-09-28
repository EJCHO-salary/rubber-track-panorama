import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from track_unwrap import api
from track_unwrap.zones import assist_polygon, build_zones, load_zones


def make_panorama(folder):
    folder.mkdir(parents=True, exist_ok=True)
    image = np.full((200, 400, 3), 125, np.uint8)
    for offset in range(0, 400, 50):
        cv2.rectangle(image, (offset + 12, 72), (offset + 32, 128), (45, 45, 45), -1)
    cv2.imwrite(str(folder / 'panorama.png'), image)
    (folder / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')


def groups():
    return [
        {'id': 'geometry', 'name': '형상별', 'default_section_id': 'unknown',
         'sections': [{'id': 'unknown', 'name': '미지정', 'color': '#aaaaaa'},
                      {'id': 'groove', 'name': '골부', 'color': '#ff0000'}]},
        {'id': 'process', 'name': '제조공법별', 'default_section_id': 'b',
         'sections': [{'id': 'b', 'name': 'B 파트', 'color': '#aaaaaa'},
                      {'id': 'a', 'name': 'A 파트', 'color': '#00ff00'}]},
    ]


def test_independent_layers_repeat_and_one_off(tmp_path):
    make_panorama(tmp_path)
    shapes = [
        {'id': 'repeat', 'group_id': 'geometry', 'section_id': 'groove', 'repeat': True,
         'repeat_pitches': 1,
         'polygon': [[4, 15], [10, 15], [10, 55], [4, 55]]},
        {'id': 'one', 'group_id': 'process', 'section_id': 'a', 'repeat': False,
         'repeat_pitches': 1,
         'polygon': [[2, 15], [90, 15], [90, 55], [2, 55]]},
    ]
    result = build_zones(tmp_path, groups(), shapes)
    geometry = cv2.imread(str(tmp_path / 'zones' / 'group_geometry_mask.png'), 0)
    process = cv2.imread(str(tmp_path / 'zones' / 'group_process_mask.png'), 0)
    assert result['version'] == 3
    assert result['group_metrics']['geometry']['unassigned_pixels'] == 0
    assert result['group_metrics']['process']['unassigned_pixels'] == 0
    assert geometry[30, 7] == geometry[30, 57] == geometry[30, 357] == 2
    assert geometry[30, 20] == 1
    assert process[30, 7] == process[30, 57] == 2
    assert process[30, 357] == 1
    assert result['group_metrics']['geometry']['areas_mm2']['groove'] > 0
    assert result['group_metrics']['process']['areas_mm2']['a'] > 0


def test_two_pitch_phase_and_multiple_shapes_in_one_section(tmp_path):
    make_panorama(tmp_path)
    shapes = [
        {'id': 'first', 'group_id': 'geometry', 'section_id': 'groove', 'repeat': True,
         'repeat_pitches': 2, 'polygon': [[4, 15], [10, 15], [10, 55], [4, 55]]},
        {'id': 'second', 'group_id': 'geometry', 'section_id': 'groove', 'repeat': True,
         'repeat_pitches': 2, 'polygon': [[61, 15], [67, 15], [67, 55], [61, 55]]},
    ]
    result = build_zones(tmp_path, groups()[:1], shapes)
    mask = cv2.imread(str(tmp_path / 'zones' / 'group_geometry_mask.png'), 0)
    assert mask[30, 7] == mask[30, 107] == mask[30, 307] == 2
    assert mask[30, 64] == mask[30, 164] == mask[30, 364] == 2
    assert mask[30, 57] == mask[30, 157] == 1
    assert [shape['repeat_pitches'] for shape in result['shapes']] == [2, 2]


def test_taxonomy_can_be_renamed_and_deleted_and_bad_refs_rejected(tmp_path):
    make_panorama(tmp_path)
    edited = groups()
    edited[0]['name'] = '사용자 지정 표면'
    edited[0]['sections'][1]['name'] = '움푹한 곳'
    assert build_zones(tmp_path, edited, [])['groups'][0]['sections'][1]['name'] == '움푹한 곳'
    assert len(build_zones(tmp_path, [edited[1]], [])['groups']) == 1
    assert not (tmp_path / 'zones' / 'group_geometry_mask.png').exists()
    with pytest.raises(ValueError, match='참조'):
        build_zones(tmp_path, [edited[1]], [{'group_id': 'geometry', 'section_id': 'groove',
                                              'polygon': [[1, 1], [5, 1], [5, 5]], 'repeat': False}])
    assert build_zones(tmp_path, [], [])['groups'] == []


def test_repeat_width_guard_and_edge_assist_does_not_save(tmp_path):
    make_panorama(tmp_path)
    initial = build_zones(tmp_path)
    bad = {'group_id': 'geometry', 'section_id': 'groove', 'repeat': True,
           'polygon': [[1, 1], [90, 1], [90, 25], [1, 25]]}
    with pytest.raises(ValueError, match='반복 간격'):
        build_zones(tmp_path, initial['groups'], [bad])
    result = assist_polygon(tmp_path, [[7, 68], [36, 68], [36, 132], [7, 132]])
    assert len(result['polygon']) >= 3
    assert result['confidence'] in {'low', 'medium'}
    assert load_zones(tmp_path) == initial


def test_migrate_only_legacy_manual_seeds(tmp_path):
    make_panorama(tmp_path)
    legacy = tmp_path / 'damage'
    legacy.mkdir()
    (legacy / 'analysis.json').write_text(json.dumps({'zone_seeds': [
        {'zone': 'groove', 'polygon': [[5, 10], [58, 10], [58, 40], [5, 40]]},
        {'zone': 'not_a_zone', 'polygon': [[1, 1], [8, 1], [8, 8]]},
    ], 'candidates': [{'mode': 'chunk'}]}), encoding='utf-8')
    result = build_zones(tmp_path)
    assert len(result['shapes']) == 1
    assert result['shapes'][0]['repeat_pitches'] == 2


def test_api_classification_contract_and_no_auto_damage(tmp_path, monkeypatch):
    job_id = 'a' * 32
    result = tmp_path / job_id / 'result'
    make_panorama(result)
    (tmp_path / job_id / 'job.json').write_text(json.dumps({
        'id': job_id, 'status': 'complete', 'created_at': '2026-01-01T00:00:00+00:00',
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    monkeypatch.setattr(api, 'JOBS', tmp_path)
    with TestClient(api.app) as client:
        initial = client.get(f'/api/jobs/{job_id}/zones')
        assert initial.status_code == 200
        assert initial.json()['shapes'] == []
        saved = client.put(f'/api/jobs/{job_id}/zones', json={'groups': groups(), 'shapes': []})
        assert saved.status_code == 200, saved.text
        assert len(saved.json()['groups']) == 2
        assert client.get(f'/api/jobs/{job_id}/zones/groups/process/overlay.png').status_code == 200
        assert client.get(f'/api/jobs/{job_id}/zones/groups/unknown/overlay.png').status_code == 404
        assert client.get(f'/api/jobs/{job_id}/damage').status_code == 404
