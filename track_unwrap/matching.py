import cv2
import numpy as np
from .geometry import UnwrapError


def feature_pairs(first, second):
    sift = cv2.SIFT_create(nfeatures=16000, contrastThreshold=.018)
    values = []
    for im in (first, second):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        mask[:, :int(gray.shape[1]*.025)] = 0
        mask[:, int(gray.shape[1]*.975):] = 0
        values.append(sift.detectAndCompute(gray, mask))
    (ka, da), (kb, db) = values
    if da is None or db is None:
        return np.empty((0, 2)), np.empty((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    def match(a, b):
        return {(m.queryIdx, m.trainIdx) for m, n in bf.knnMatch(a,b,k=2) if m.distance < .74*n.distance}
    ab, ba = match(da, db), match(db, da)
    good = sorted((a,b) for a,b in ab if (b,a) in ba)
    return (np.float32([ka[a].pt for a,b in good]).reshape(-1,2),
            np.float32([kb[b].pt for a,b in good]).reshape(-1,2))


def estimate_pitch_shift(first, second, start_a, start_b, pitch_px):
    a,b = feature_pairs(first, second)
    if len(a) < 15:
        raise UnwrapError("Insufficient distinct surface correspondences between adjacent images.")
    plausible = np.abs(a[:,0]-b[:,0]) < first.shape[1]*.055
    delta = (a[:,1]-b[:,1])/pitch_px + start_a-start_b
    integers = np.rint(delta).astype(int)
    labels, counts = np.unique(integers[plausible], return_counts=True)
    if not len(labels):
        raise UnwrapError("No geometrically plausible correspondence.")
    shift = int(labels[np.argmax(counts)])
    use = plausible & (np.abs(delta-shift) < .20)
    count = int(use.sum())
    alternatives = sorted([int(c) for k,c in zip(labels,counts) if k != shift], reverse=True)
    if count < 20 or (alternatives and alternatives[0] > count*.7):
        raise UnwrapError("Pitch identity is ambiguous; overlapping surface needs review.")
    result = {"pitch_shift": shift, "matches": len(a), "accepted_matches": count,
              "median_phase_error_pitch": float(np.median(delta[use]-shift)),
              "x_span_fraction": float(np.ptp(a[use,0])/first.shape[1]),
              "source_points": a[use].tolist(), "target_points": b[use].tolist()}
    return shift, result
