import json

import cv2
import numpy as np

from track_unwrap.cracks import add_manual_crack, delete_manual_crack, load_cracks, propose_cracks, review_crack, review_cracks
from track_unwrap.zones import build_zones


def test_unique_fissure_can_be_reviewed_and_counted_by_section(tmp_path):
    image = np.full((200, 500, 3), 155, np.uint8)
    for x in range(0, 500, 50):
        cv2.rectangle(image, (x+14, 70), (x+25, 130), (45, 45, 45), -1)
    cv2.line(image, (183, 22), (211, 61), (15, 15, 15), 3)
    cv2.imwrite(str(tmp_path / 'panorama.png'), image)
    (tmp_path / 'quality_report.json').write_text(json.dumps({
        'settings': {'width_mm': 80, 'pitch_mm': 20, 'pixels_per_mm': 2.5},
    }), encoding='utf-8')
    taxonomy = [{'id': 'geometry', 'name': '형상', 'default_section_id': 'surface',
                 'sections': [{'id': 'surface', 'name': '표면', 'color': '#668866', 'repeat_mode': 'independent'}]}]
    build_zones(tmp_path, taxonomy, [])
    result = propose_cracks(tmp_path, 'geometry', 'high')
    fissures = [item for item in result['candidates'] if item['bbox'][0] <= 195 <= item['bbox'][0]+item['bbox'][2]
                and item['bbox'][1] <= 45 <= item['bbox'][1]+item['bbox'][3]]
    assert fissures, 'unique dark fissure should have a reviewable proposal'
    candidate = fissures[0]
    assert candidate['status'] == 'pending'
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
