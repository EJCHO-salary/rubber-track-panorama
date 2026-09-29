import json

import cv2
import numpy as np

from track_unwrap.cracks import add_manual_crack, delete_manual_crack, load_cracks, propose_cracks, review_crack
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
