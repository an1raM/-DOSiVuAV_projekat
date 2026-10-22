import numpy as np
import cv2 as cv

from undistort import undistort
from thresholded_binary import threshold
from perspective import warp, M_INV, DST

XM_PER_PIX = 3.7 / (DST[1][0] - DST[0][0])
YM_PER_PIX = 30 / 720


def find_lane_pixels(binary_warped, n_windows=9, margin=100, min_pix=50):
    """Find lane pixels with a histogram + sliding window search."""
    binary = (binary_warped > 0).astype(np.uint8)
    h, w = binary.shape

    histogram = np.sum(binary[h // 2:, :], axis=0)
    midpoint = w // 2
    left_current = np.argmax(histogram[:midpoint])
    right_current = np.argmax(histogram[midpoint:]) + midpoint

    nonzero_y, nonzero_x = binary.nonzero()
    window_height = h // n_windows

    left_inds, right_inds = [], []
    out_img = np.zeros((h, w, 3), np.uint8)

    for i in range(n_windows):
        y_low = h - (i + 1) * window_height
        y_high = h - i * window_height

        xl_low, xl_high = left_current - margin, left_current + margin
        xr_low, xr_high = right_current - margin, right_current + margin

        in_y = (nonzero_y >= y_low) & (nonzero_y < y_high)
        good_left = (in_y & (nonzero_x >= xl_low) & (nonzero_x < xl_high)).nonzero()[0]
        good_right = (in_y & (nonzero_x >= xr_low) & (nonzero_x < xr_high)).nonzero()[0]

        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left) > min_pix:
            left_current = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > min_pix:
            right_current = int(np.mean(nonzero_x[good_right]))

    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)

    leftx, lefty = nonzero_x[left_inds], nonzero_y[left_inds]
    rightx, righty = nonzero_x[right_inds], nonzero_y[right_inds]

    out_img[lefty, leftx] = (0, 0, 255)
    out_img[righty, rightx] = (255, 0, 0)

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(x, y):
    """Fit x = A*y^2 + B*y + C."""
    return np.polyfit(y, x, 2)


def curvature_radius(leftx, lefty, rightx, righty, y_eval):
    """Radius of curvature in meters at y_eval (pixels), averaged over both lines."""
    left_fit = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)
    y = y_eval * YM_PER_PIX

    left_r = (1 + (2 * left_fit[0] * y + left_fit[1]) ** 2) ** 1.5 / abs(2 * left_fit[0])
    right_r = (1 + (2 * right_fit[0] * y + right_fit[1]) ** 2) ** 1.5 / abs(2 * right_fit[0])
    return (left_r + right_r) / 2


def vehicle_offset(left_fit, right_fit, img_width, y_eval):
    """Offset of the vehicle from the lane center in meters (negative = left of center)."""
    left_x = np.polyval(left_fit, y_eval)
    right_x = np.polyval(right_fit, y_eval)
    lane_center = (left_x + right_x) / 2
    return (img_width / 2 - lane_center) * XM_PER_PIX


def draw_lane(undist, binary_warped, left_fit, right_fit):
    """Draw the lane area on the warped view and project it back onto the original image."""
    h, w = binary_warped.shape
    ploty = np.linspace(0, h - 1, h)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    color_warp = np.zeros((h, w, 3), np.uint8)
    pts_left = np.transpose(np.vstack([left_fitx, ploty]))
    pts_right = np.flipud(np.transpose(np.vstack([right_fitx, ploty])))
    pts = np.vstack((pts_left, pts_right)).astype(np.int32)
    cv.fillPoly(color_warp, [pts], (0, 255, 0))

    unwarped = cv.warpPerspective(color_warp, M_INV, (w, h))
    return cv.addWeighted(undist, 1, unwarped, 0.3, 0)


def process_image(img):
    """Full pipeline for one BGR image. Returns the result and the sliding window debug image."""
    undist = undistort(img)
    binary_warped = warp(threshold(undist))
    h, w = binary_warped.shape

    leftx, lefty, rightx, righty, debug = find_lane_pixels(binary_warped)

    if len(leftx) == 0 or len(rightx) == 0:
        return undist, debug   # no lane found, return the undistorted image unchanged

    left_fit = fit_polynomial(leftx, lefty)
    right_fit = fit_polynomial(rightx, righty)

    ploty = np.linspace(0, h - 1, h)
    for fit, name in ((left_fit, 'f_left'), (right_fit, 'f_right')):
        pts = np.stack([np.polyval(fit, ploty), ploty], axis=1).astype(np.int32)
        cv.polylines(debug, [pts], False, (0, 255, 255), 3)

    for i, (fit, name) in enumerate(((left_fit, 'f_left'), (right_fit, 'f_right'))):
        text = f'{name}(y) = {fit[0]:.2e}*y^2 + {fit[1]:.2e}*y + {fit[2]:.1f}'
        cv.putText(debug, text, (30, 40 + 35 * i), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    radius = curvature_radius(leftx, lefty, rightx, righty, h - 1)
    offset = vehicle_offset(left_fit, right_fit, w, h - 1)

    result = draw_lane(undist, binary_warped, left_fit, right_fit)
    side = 'left' if offset < 0 else 'right'
    cv.putText(result, f'Radius of curvature: {radius:.0f} m', (30, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(result, f'Vehicle is {abs(offset):.2f} m {side} of center', (30, 90),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return result, debug


if __name__ == '__main__':
    img = cv.imread('test_images/test1.jpg')
    result, debug = process_image(img)

    cv.imwrite('docs/color_fit_lines.png', debug)
    cv.imwrite('docs/example_output.png', result)

    cv.imshow('Sliding windows', debug)
    cv.imshow('Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()