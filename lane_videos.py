import os
from collections import deque

import cv2 as cv
import numpy as np

from undistort import undistort
from thresholded_binary import threshold
from perspective import warp
from lane_detection import (find_lane_pixels, fit_polynomial, draw_lane,
                            XM_PER_PIX, YM_PER_PIX)

INPUT_VIDEO = 'test_videos/challenge03.mp4'
OUTPUT_VIDEO = 'docs/challenge_out2.mp4'
HISTORY = 8
MIN_PIXELS = 300
MAX_LOST = 10


def search_around_poly(binary_warped, left_fit, right_fit, margin=80):
    """Take the pixels that lie within +-margin of the previous polynomials."""
    nonzero_y, nonzero_x = (binary_warped > 0).nonzero()
    left_mask = np.abs(nonzero_x - np.polyval(left_fit, nonzero_y)) < margin
    right_mask = np.abs(nonzero_x - np.polyval(right_fit, nonzero_y)) < margin
    return (nonzero_x[left_mask], nonzero_y[left_mask],
            nonzero_x[right_mask], nonzero_y[right_mask])


def radius_from_fit(fit, y_eval):
    """Radius of curvature in meters, converting the pixel polynomial to meters."""
    a = fit[0] * XM_PER_PIX / YM_PER_PIX ** 2
    b = fit[1] * XM_PER_PIX / YM_PER_PIX
    y = y_eval * YM_PER_PIX
    return (1 + (2 * a * y + b) ** 2) ** 1.5 / max(abs(2 * a), 1e-9)


def is_plausible(left_fit, right_fit, h):
    """Lane width must be realistic and the two lines roughly parallel."""
    ys = np.array([0, h // 2, h - 1])
    widths = (np.polyval(right_fit, ys) - np.polyval(left_fit, ys)) * XM_PER_PIX
    return bool(np.all((widths > 2.5) & (widths < 5.0)) and widths.max() - widths.min() < 1.0)


class LaneTracker:
    def __init__(self):
        self.left_hist = deque(maxlen=HISTORY)
        self.right_hist = deque(maxlen=HISTORY)
        self.lost = 0

    def update(self, binary_warped):
        h = binary_warped.shape[0]
        found = False

        if self.left_hist:
            lx, ly, rx, ry = search_around_poly(binary_warped, self.left_hist[-1], self.right_hist[-1])
            if len(lx) < MIN_PIXELS or len(rx) < MIN_PIXELS:
                lx, ly, rx, ry, _ = find_lane_pixels(binary_warped)
        else:
            lx, ly, rx, ry, _ = find_lane_pixels(binary_warped)

        if len(lx) >= MIN_PIXELS and len(rx) >= MIN_PIXELS:
            left_fit, right_fit = fit_polynomial(lx, ly), fit_polynomial(rx, ry)
            if is_plausible(left_fit, right_fit, h):
                self.left_hist.append(left_fit)
                self.right_hist.append(right_fit)
                found = True

        self.lost = 0 if found else self.lost + 1
        if self.lost > MAX_LOST:
            self.left_hist.clear()
            self.right_hist.clear()

        if not self.left_hist:
            return None, None
        return np.mean(self.left_hist, axis=0), np.mean(self.right_hist, axis=0)


def process_frame(frame, tracker):
    undist = undistort(frame)
    binary_warped = warp(threshold(undist))
    h, w = binary_warped.shape
    left_fit, right_fit = tracker.update(binary_warped)
    if left_fit is None:
        return undist

    result = draw_lane(undist, binary_warped, left_fit, right_fit)
    radius = (radius_from_fit(left_fit, h - 1) + radius_from_fit(right_fit, h - 1)) / 2
    center = (np.polyval(left_fit, h - 1) + np.polyval(right_fit, h - 1)) / 2
    offset = (w / 2 - center) * XM_PER_PIX
    side = 'left' if offset < 0 else 'right'
    cv.putText(result, f'Radius of curvature: {radius:.0f} m', (30, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(result, f'Vehicle is {abs(offset):.2f} m {side} of center', (30, 90),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return result


cap = cv.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f'Cannot open input video: {os.path.abspath(INPUT_VIDEO)}')
fps = cap.get(cv.CAP_PROP_FPS) or 25
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
writer = cv.VideoWriter(OUTPUT_VIDEO, cv.VideoWriter_fourcc(*'mp4v'), fps, size)

tracker = LaneTracker()
frames = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = process_frame(frame, tracker)
    writer.write(result)
    frames += 1
    cv.imshow('Lane detection', result)
    if cv.waitKey(1) & 0xFF == 27:
        break

print('Frames written:', frames)
cap.release()
writer.release()
cv.destroyAllWindows()