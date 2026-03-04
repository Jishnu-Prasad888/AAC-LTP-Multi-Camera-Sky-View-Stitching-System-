import cv2
import numpy as np

# === SETTINGS ===
video1_path = "1.mp4"   # LEFT video
video2_path = "2.mp4"   # RIGHT video
output_path = "merged_output.mp4"

resize_height = 720
max_features = 5000
BLEND_BAND = 200        # feathering width in pixels
# =================


def resize_to_height(frame, height):
    h, w = frame.shape[:2]
    if h == height:
        return frame
    scale = height / h
    return cv2.resize(frame, (int(w * scale), height))


def compute_homography(img_left, img_right, max_features=5000):
    """
    Returns H such that: warped_right = warpPerspective(img_right, H, canvas_size)
    aligns img_right onto img_left's coordinate system.
    """
    gray_l = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp_l, des_l = orb.detectAndCompute(gray_l, None)
    kp_r, des_r = orb.detectAndCompute(gray_r, None)

    if des_l is None or des_r is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(des_l, des_r, k=2)

    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    print(f"  Good matches after ratio test: {len(good)}")
    if len(good) < 10:
        return None

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, 5.0)
    print(f"  RANSAC inliers: {mask.sum() if mask is not None else 0}")
    return H


def get_canvas_params(H, w_l, h_l, w_r, h_r):
    """
    Given homography H (maps right→left coords), compute:
    - canvas size (width, height)
    - x/y offset to shift left image so everything stays positive
    """
    corners_r = np.float32([[0,0],[w_r,0],[w_r,h_r],[0,h_r]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_r, H)

    all_x = np.concatenate([[0, w_l], warped_corners[:,0,0]])
    all_y = np.concatenate([[0, h_l], warped_corners[:,0,1]])

    x_min = min(0, float(all_x.min()))
    y_min = min(0, float(all_y.min()))
    x_max = float(all_x.max())
    y_max = float(all_y.max())

    offset_x = int(-x_min)
    offset_y = int(-y_min)
    canvas_w = int(x_max - x_min) + 1
    canvas_h = int(y_max - y_min) + 1

    return canvas_w, canvas_h, offset_x, offset_y


def build_seam_mask(canvas_l, canvas_r, blend_band):
    """
    Build per-pixel alpha (0=use left, 1=use right) with feathered seam.
    """
    h, w = canvas_l.shape[:2]
    has_l = (canvas_l.max(axis=2) > 5).astype(np.float32)
    has_r = (canvas_r.max(axis=2) > 5).astype(np.float32)
    overlap = has_l * has_r

    overlap_cols = np.where(overlap.max(axis=0) > 0)[0]

    alpha = np.zeros((h, w), dtype=np.float32)

    if len(overlap_cols) == 0:
        # No overlap — just paste side by side
        alpha[has_r > 0] = 1.0
        return alpha

    seam_start = int(overlap_cols.min())
    seam_end   = int(overlap_cols.max())
    seam_w     = max(seam_end - seam_start, 1)

    # Feather over the overlap region (clamped to blend_band)
    actual_band = min(blend_band, seam_w)
    feather_start = seam_end - actual_band

    for col in range(w):
        if col <= feather_start:
            alpha[:, col] = 0.0       # fully left
        elif col <= seam_end:
            t = (col - feather_start) / actual_band
            # smooth step
            t = t * t * (3 - 2 * t)
            alpha[:, col] = t
        else:
            alpha[:, col] = 1.0       # fully right

    # Where only one source exists, use that source
    alpha = np.where((has_l > 0) & (has_r < 0.5), 0.0, alpha)
    alpha = np.where((has_r > 0) & (has_l < 0.5), 1.0, alpha)
    # Where neither exists, 0
    alpha = np.where((has_l < 0.5) & (has_r < 0.5), 0.0, alpha)

    return alpha


def crop_black_borders(frame, tolerance=5):
    """
    Crop any all-black rows/cols from the frame edges.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return frame
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return frame[r0:r1+1, c0:c1+1]


# ── Open videos ───────────────────────────────────────────────────────────────
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)
fps  = cap1.get(cv2.CAP_PROP_FPS) or 30.0

ret1, raw1 = cap1.read()
ret2, raw2 = cap2.read()
if not ret1 or not ret2:
    raise RuntimeError("Could not read frames from one or both videos.")

f1 = resize_to_height(raw1, resize_height)
f2 = resize_to_height(raw2, resize_height)
h, w1 = f1.shape[:2]
_,  w2 = f2.shape[:2]

# ── Compute homography once ───────────────────────────────────────────────────
print("Computing homography...")
H = compute_homography(f1, f2, max_features)

if H is None:
    print("WARNING: Homography failed — falling back to simple side-by-side.")
    canvas_w = w1 + w2
    canvas_h = h
    offset_x = 0
    offset_y = 0
    H_final = None
else:
    canvas_w, canvas_h, offset_x, offset_y = get_canvas_params(H, w1, h, w2, h)
    # Adjust H to account for the canvas offset
    T = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float64)
    H_final = T @ H
    print(f"Canvas size: {canvas_w} x {canvas_h}, offset: ({offset_x}, {offset_y})")


def stitch(frame_l, frame_r):
    fl = resize_to_height(frame_l, resize_height)
    fr = resize_to_height(frame_r, resize_height)

    # Build left canvas
    cv_l = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    hl, wl = fl.shape[:2]
    ey = min(offset_y + hl, canvas_h)
    ex = min(offset_x + wl, canvas_w)
    cv_l[offset_y:ey, offset_x:ex] = fl[:ey-offset_y, :ex-offset_x]

    # Build right canvas (warped)
    cv_r = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    if H_final is not None:
        cv2.warpPerspective(fr, H_final, (canvas_w, canvas_h), dst=cv_r,
                            flags=cv2.INTER_LINEAR)
    else:
        hr, wr = fr.shape[:2]
        cv_r[:hr, w1:w1+wr] = fr

    # Blend
    alpha = build_seam_mask(cv_l, cv_r, BLEND_BAND)
    a3 = alpha[:, :, np.newaxis].astype(np.float32)
    blended = (cv_l.astype(np.float32) * (1 - a3) +
               cv_r.astype(np.float32) * a3)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── First frame — also used to find crop bounds ───────────────────────────────
print("Stitching first frame to determine crop region...")
first = stitch(raw1, raw2)

# Compute crop bounds from first frame and reuse for all frames
gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
mask = gray > 5
rows = np.any(mask, axis=1)
cols = np.any(mask, axis=0)
r0 = int(np.where(rows)[0][0])  if rows.any() else 0
r1 = int(np.where(rows)[0][-1]) if rows.any() else canvas_h - 1
c0 = int(np.where(cols)[0][0])  if cols.any() else 0
c1 = int(np.where(cols)[0][-1]) if cols.any() else canvas_w - 1

crop_h = r1 - r0 + 1
crop_w = c1 - c0 + 1
print(f"Crop region: x={c0}:{c1}, y={r0}:{r1}  →  {crop_w} x {crop_h}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
out.write(first[r0:r1+1, c0:c1+1])

# ── Process all remaining frames ──────────────────────────────────────────────
frame_count = 1
while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        break
    merged = stitch(f1, f2)
    out.write(merged[r0:r1+1, c0:c1+1])
    frame_count += 1
    if frame_count % 60 == 0:
        print(f"  {frame_count} frames done...")

cap1.release()
cap2.release()
out.release()
print(f"\nDone! {frame_count} frames → {output_path}  ({crop_w}x{crop_h})")