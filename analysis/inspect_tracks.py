"""Read-only image measurements for the five reference track photographs.

This is a diagnostic, not a production stitcher. ROI masks are deliberately
hand-selected for this sample. No source photograph or synthesized image is written.
Run: python analysis/inspect_tracks.py
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
cv2.setRNGSeed(42)
cv2.setNumThreads(4)


def features(gray, region):
    h, w = gray.shape
    mask = np.zeros_like(gray)
    if region == "tread":
        mask[int(h * .015):int(h * .985), int(w * .12):int(w * .90)] = 255
    else:
        mask[:, :int(w * .055)] = 255
        mask[:, int(w * .975):] = 255
    return cv2.SIFT_create(nfeatures=12000, contrastThreshold=.025).detectAndCompute(gray, mask)


def compare(f1, f2):
    k1, d1 = f1
    k2, d2 = f2
    result = {"keypoints": [len(k1), len(k2)]}
    if d1 is None or d2 is None or min(len(d1), len(d2)) < 2:
        return result
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio(a, b):
        return {(m.queryIdx, m.trainIdx) for m, n in matcher.knnMatch(a, b, k=2)
                if m.distance < .72 * n.distance}
    forward = ratio(d1, d2)
    backward = ratio(d2, d1)
    matches = sorted((a, b) for a, b in forward if (b, a) in backward)
    result["mutual_ratio_matches"] = len(matches)
    if len(matches) < 8:
        return result
    src = np.float32([k1[a].pt for a, _ in matches])
    dst = np.float32([k2[b].pt for _, b in matches])
    for model in ["affine", "homography"]:
        if model == "affine":
            matrix, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                                  ransacReprojThreshold=3,
                                                  maxIters=10000, confidence=.999)
            if matrix is not None:
                matrix = np.vstack([matrix, [0, 0, 1]])
        else:
            matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3,
                                                 maxIters=10000, confidence=.999)
        if matrix is None or inliers is None:
            continue
        use = inliers.ravel().astype(bool)
        pred = cv2.perspectiveTransform(src[:, None, :], matrix)[:, 0, :]
        errors = np.linalg.norm(pred - dst, axis=1)
        bounds1 = np.stack([src[use].min(axis=0), src[use].max(axis=0)])
        bounds2 = np.stack([dst[use].min(axis=0), dst[use].max(axis=0)])
        result[model] = {
            "matrix_source_to_target": matrix.tolist(),
            "inliers": int(use.sum()),
            "inlier_ratio": round(float(use.mean()), 4),
            "fit_error_median_px": round(float(np.median(errors[use])), 3),
            "fit_error_p95_px": round(float(np.percentile(errors[use], 95)), 3),
            "median_displacement_xy_px": np.median(dst[use] - src[use], axis=0).round(2).tolist(),
            "source_inlier_bounds_xy": bounds1.round(2).tolist(),
            "target_inlier_bounds_xy": bounds2.round(2).tolist(),
            "inlier_points_source": src[use].round(2).tolist(),
            "inlier_points_target": dst[use].round(2).tolist(),
            "inliers_x_bins_0_480_720_1200": np.histogram(src[use, 0], bins=[0, 480, 720, 1200])[0].tolist(),
            "enough_inliers_for_diagnostic_candidate": bool(use.sum() >= 20),
        }
    return result


def main():
    items, cache = [], {}
    for path in sorted((ROOT / "ref_img").glob("*.jpg")):
        with Image.open(path) as raw:
            orientation = raw.getexif().get(274)
            rgb = np.array(ImageOps.exif_transpose(raw).convert("RGB"))
            stored_size = list(raw.size)
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (round(w * 1600 / h), 1600), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        cache[path.name] = {region: features(gray, region) for region in ["tread", "background"]}
        items.append({"file": path.name, "stored_size_wh": stored_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "oriented_size_wh": [w, h], "exif_orientation": orientation,
                      "analysis_size_wh": [gray.shape[1], gray.shape[0]]})
    pairs = []
    for i, first in enumerate(items):
        for second in items[i + 1:]:
            pair = {"source": first["file"], "target": second["file"]}
            for region in ["tread", "background"]:
                pair[region] = compare(cache[first["file"]][region], cache[second["file"]][region])
            pairs.append(pair)
            compact = {"pair": [pair["source"], pair["target"]]}
            for region in ["tread", "background"]:
                compact[region] = {k: v for k, v in pair[region].get("homography", {}).items()
                                   if k in ["inliers", "inlier_ratio", "fit_error_median_px",
                                            "fit_error_p95_px", "median_displacement_xy_px"]}
            print(json.dumps(compact), flush=True)
    report = {"purpose": "Diagnostic only; inlier fit residual is not independent metric accuracy.",
              "opencv_version": cv2.__version__, "width_mm": 450, "pitch_mm": 86,
              "total_pitch_count": None, "ransac_threshold_analysis_px": 3,
              "ratio_threshold": .72, "images": items, "pairs": pairs}
    OUT.mkdir(exist_ok=True)
    (OUT / "measurements.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
