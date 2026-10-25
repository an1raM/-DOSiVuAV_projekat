import os
import cv2 as cv
import numpy as np

INPUT_VIDEO = 'test_videos/project_video02.mp4'
OUTPUT_VIDEO = 'docs/hough_video_out.mp4'
SMOOTHING = 0.8


def region_of_interest(edges):
    """Keep only the trapezoid in front of the car."""
    h, w = edges.shape
    polygon = np.array([[(int(0.05 * w), h), (int(0.45 * w), int(0.58 * h)),
                         (int(0.55 * w), int(0.58 * h)), (int(0.95 * w), h)]], np.int32)
    mask = np.zeros_like(edges)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(edges, mask)


def split_segments(lines, min_slope=0.4):
    """Split Hough segments into left and right line points by slope sign."""
    left, right = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < min_slope:
            continue
        (left if slope < 0 else right).append(((x1, y1), (x2, y2)))
    return left, right


def fit_line(segments):
    """Fit x = a*y + b through all segment endpoints."""
    if not segments:
        return None
    pts = np.array(segments).reshape(-1, 2)
    return np.polyfit(pts[:, 1], pts[:, 0], 1)


def process_frame(frame, state):
    h, w = frame.shape[:2]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 100, 170, apertureSize=3)
    edges = region_of_interest(edges)

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                           minLineLength=30, maxLineGap=100)
    left, right = ([], []) if lines is None else split_segments(lines)

    y_bottom, y_top = h, int(0.6 * h)
    overlay = np.zeros_like(frame)
    for name, segs in (('left', left), ('right', right)):
        fit = fit_line(segs)
        if fit is not None:
            prev = state.get(name)
            state[name] = fit if prev is None else SMOOTHING * prev + (1 - SMOOTHING) * fit
        if state.get(name) is None:
            continue
        a, b = state[name]
        p1 = (int(a * y_bottom + b), y_bottom)
        p2 = (int(a * y_top + b), y_top)
        cv.line(overlay, p1, p2, (0, 0, 255), 10)

    return cv.addWeighted(frame, 1, overlay, 0.8, 0)


cap = cv.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f'Cannot open input video: {os.path.abspath(INPUT_VIDEO)}')

fps = cap.get(cv.CAP_PROP_FPS) or 25
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
writer = cv.VideoWriter(OUTPUT_VIDEO, cv.VideoWriter_fourcc(*'mp4v'), fps, size)
if not writer.isOpened():
    raise SystemExit('Cannot open the output video for writing')

state = {}
frames = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = process_frame(frame, state)
    writer.write(result)
    frames += 1
    cv.imshow('Lane lines', result)
    if cv.waitKey(1) & 0xFF == 27:
        break

print('Frames written:', frames)
cap.release()
writer.release()
cv.destroyAllWindows()