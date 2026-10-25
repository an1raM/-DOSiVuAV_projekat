import os
from collections import deque

import cv2 as cv
import numpy as np

from undistort import undistort
from thresholded_binary import threshold, color_mask
from lane_detection import find_lane_pixels, fit_polynomial

INPUT_VIDEO = 'test_videos/project_video03.mp4'
OUTPUT_VIDEO = 'docs/project_video03_video_out.mp4'
CALIBRATION = False
HISTORY = 8
MAX_LOST = 15
LANE_WIDTH_M = 3.7
LANE_LENGTH_M = 30


SRC_FRAC_BY_SIZE = {
    (960, 540): np.float32([[0.44, 0.65], [0.57, 0.65], [0.90, 1.0], [0.14, 1.0]]),
    (1280, 720): np.float32([[0.44, 0.65], [0.57, 0.65], [0.86, 0.97], [0.18, 0.97]]),
}
DEFAULT_SRC_FRAC = SRC_FRAC_BY_SIZE[(1280, 720)]
DST_FRAC = np.float32([[0.25, 0.0], [0.75, 0.0], [0.75, 1.0], [0.25, 1.0]])
DST_FRAC = np.float32([[0.25, 0.0], [0.75, 0.0], [0.75, 1.0], [0.25, 1.0]])


class Geometry:
    """Perspective matrices and meter conversion for one frame size."""
    def __init__(self, w, h):
        src = SRC_FRAC_BY_SIZE.get((w, h), DEFAULT_SRC_FRAC) * np.float32([w, h])
        dst = DST_FRAC * np.float32([w, h])
        self.m = cv.getPerspectiveTransform(src, dst)
        self.m_inv = cv.getPerspectiveTransform(dst, src)
        self.xm = LANE_WIDTH_M / (0.5 * w)
        self.ym = LANE_LENGTH_M / h
        self.size = (w, h)

    def warp(self, img):
        return cv.warpPerspective(img, self.m, self.size, flags=cv.INTER_LINEAR)


def search_around_poly(binary, left_fit, right_fit, margin):
    """Pixels within +-margin of the previous polynomials."""
    ys, xs = (binary > 0).nonzero()
    lm = np.abs(xs - np.polyval(left_fit, ys)) < margin
    rm = np.abs(xs - np.polyval(right_fit, ys)) < margin
    return xs[lm], ys[lm], xs[rm], ys[rm]


def is_plausible(left_fit, right_fit, w, h):
    """Lane width must be a sensible fraction of the image and the lines roughly parallel."""
    ys = np.array([0, h // 2, h - 1])
    widths = (np.polyval(right_fit, ys) - np.polyval(left_fit, ys)) / w
    return bool(np.all((widths > 0.3) & (widths < 0.7)) and widths.max() - widths.min() < 0.12)


def radius_from_fit(fit, y_eval, geo):
    """Radius of curvature in meters from a pixel polynomial."""
    a = fit[0] * geo.xm / geo.ym ** 2
    b = fit[1] * geo.xm / geo.ym
    y = y_eval * geo.ym
    return (1 + (2 * a * y + b) ** 2) ** 1.5 / max(abs(2 * a), 1e-9)


class LaneTracker:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.min_pixels = max(60, int(0.0002 * w * h))
        self.margin = int(0.08 * w)
        self.left_hist, self.right_hist = deque(maxlen=HISTORY), deque(maxlen=HISTORY)
        self.lost = 0

    def update(self, binary):
        found = False
        if self.left_hist:
            lx, ly, rx, ry = search_around_poly(binary, self.left_hist[-1],
                                                self.right_hist[-1], self.margin)
            if len(lx) < self.min_pixels or len(rx) < self.min_pixels:
                lx, ly, rx, ry, _ = find_lane_pixels(binary, margin=self.margin)
        else:
            lx, ly, rx, ry, _ = find_lane_pixels(binary, margin=self.margin)

        if len(lx) >= self.min_pixels and len(rx) >= self.min_pixels:
            lf, rf = fit_polynomial(lx, ly), fit_polynomial(rx, ry)
            if is_plausible(lf, rf, self.w, self.h):
                self.left_hist.append(lf)
                self.right_hist.append(rf)
                found = True

        self.lost = 0 if found else self.lost + 1
        if self.lost > MAX_LOST:
            for d in (self.left_hist, self.right_hist):
                d.clear()

        if not self.left_hist:
            return None
        return (np.mean(self.left_hist, axis=0), np.mean(self.right_hist, axis=0))


def draw_lines(img, geo, fits):
    """Paint the real lane markings red: only the paint pixels that lie near the fitted curves."""
    w, h = geo.size
    band = np.zeros((h, w), np.uint8)
    ploty = np.arange(h)
    thickness = int(0.05 * w)
    for fit in fits:
        pts = np.stack([np.polyval(fit, ploty), ploty], axis=1).astype(np.int32)
        cv.polylines(band, [pts], False, 255, thickness)
    band = cv.warpPerspective(band, geo.m_inv, (w, h))

    paint = cv.bitwise_and(color_mask(img), band)
    paint = cv.dilate(paint, np.ones((2, 2), np.uint8))
    result = img.copy()
    result[paint > 0] = (0, 0, 255)
    return result


def process_frame(frame, geo, tracker):
    img = undistort(frame) if CALIBRATION else frame
    binary = geo.warp(threshold(img))
    state = tracker.update(binary)
    if state is None:
        return img
    left_fit, right_fit = state

    result = draw_lines(img, geo, (left_fit, right_fit))
    h, w = binary.shape
    radius = (radius_from_fit(left_fit, h - 1, geo) + radius_from_fit(right_fit, h - 1, geo)) / 2
    center = (np.polyval(left_fit, h - 1) + np.polyval(right_fit, h - 1)) / 2
    offset = (w / 2 - center) * geo.xm
    side = 'left' if offset < 0 else 'right'

    texts = [f'Radius of curvature: {radius:.0f} m',
             f'Vehicle is {abs(offset):.2f} m {side} of center']
    for i, t in enumerate(texts):
        cv.putText(result, t, (30, 40 + 35 * i), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return result


cap = cv.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f'Cannot open input video: {os.path.abspath(INPUT_VIDEO)}')
fps = cap.get(cv.CAP_PROP_FPS) or 25
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
writer = cv.VideoWriter(OUTPUT_VIDEO, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

geo, tracker = Geometry(w, h), LaneTracker(w, h)
frames = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = process_frame(frame, geo, tracker)
        writer.write(result)
        frames += 1
        cv.imshow('Lane detection', result)
        if cv.waitKey(1) & 0xFF == 27:
            break
finally:
    print('Frames written:', frames)
    cap.release()
    writer.release()
    cv.destroyAllWindows()