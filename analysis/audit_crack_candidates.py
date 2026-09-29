"""Visual development audit of crack candidates against a small hand-reviewed set.

This is a deliberately selected feasibility sample, not a population accuracy
estimate. Run with the original local panorama; no photo is stored in Git.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def nearest(candidates, center, radius=45):
    x, y = center
    matches = []
    for candidate in candidates:
        bx, by, width, height = candidate['bbox']
        distance = float(np.hypot(bx + width / 2 - x, by + height / 2 - y))
        if distance <= radius:
            matches.append((distance, candidate))
    return min(matches, key=lambda item: item[0])[1] if matches else None


def panel(image, label, candidate, side):
    center_x, center_y = label['center']
    width, height = 450, 320
    left = max(0, min(image.shape[1] - width, center_x - width // 2))
    top = max(0, min(image.shape[0] - height, center_y - height // 2))
    crop = image[top:top + height, left:left + width].copy()
    if candidate:
        polygon = np.rint(np.asarray(candidate['polygon']) - [left, top]).astype(np.int32)
        cv2.polylines(crop, [polygon], True, (40, 210, 250) if side == 'before' else (65, 225, 85), 2, cv2.LINE_AA)
    canvas = np.full((350, width, 3), 30, np.uint8)
    canvas[:height] = crop
    outcome = 'DETECTED' if candidate else 'NONE'
    text = f"{side.upper()} | {label['kind'].upper()} | {outcome} | x={center_x}"
    cv2.putText(canvas, text, (7, 340), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def audit(result_dir, before_path, after_path, labels_path, output_dir):
    image = cv2.imread(str(result_dir / 'panorama.png'))
    if image is None:
        raise ValueError('원본 전개 사진을 읽을 수 없습니다.')
    before = json.loads(before_path.read_text(encoding='utf-8'))['candidates']
    after = json.loads(after_path.read_text(encoding='utf-8'))['candidates']
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    positives = [item for item in labels if item['kind'] == 'damage']
    negatives = [item for item in labels if item['kind'] == 'structure']
    typed = [item for item in positives if item.get('type')]
    summary = {
        'note': '의도적으로 고른 개발용 시각 검토 사례이며 전체 사진의 정밀도·재현율은 아닙니다.',
        'before_candidates': len(before), 'after_candidates': len(after),
        'damage_retained': sum(nearest(after, item['center']) is not None for item in positives),
        'damage_total': len(positives),
        'structure_removed': sum(nearest(after, item['center']) is None for item in negatives),
        'structure_total': len(negatives),
        'suggested_type_matches': sum((match := nearest(after, item['center'])) is not None
                                      and match.get('suggested_damage_type') == item['type'] for item in typed),
        'typed_examples': len(typed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'feasibility.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    examples = [item for item in labels if item.get('show')]
    montage = np.vstack([np.hstack([panel(image, item, nearest(before, item['center']), 'before'),
                                    panel(image, item, nearest(after, item['center']), 'after')])
                         for item in examples])
    cv2.imwrite(str(output_dir / 'feasibility.jpg'), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('before_review', type=Path)
    parser.add_argument('after_review', type=Path)
    parser.add_argument('--labels', type=Path, default=Path(__file__).with_name('crack_audit_labels.json'))
    parser.add_argument('--output', type=Path, default=Path('output/crack_audit'))
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.result_dir, arguments.before_review, arguments.after_review,
                           arguments.labels, arguments.output), ensure_ascii=False, indent=2))
