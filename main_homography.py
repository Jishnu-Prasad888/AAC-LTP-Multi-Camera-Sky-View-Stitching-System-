import cv2
import numpy as np

# === SETTINGS ===
video1_path = "1.mp4"
video2_path = "2.mp4"
output_path = "merged_output.mp4"

resize_height = 720       # Both videos resized to this height
blend_band = 150          # Width of the feathered blend zone at the seam
max_features = 3000       # ORB features to detect for alignment
# =================


def resize_to_height(frame, height):
    h, w = frame.shape[:2]
    scale = height / h
    return cv2.resize(frame, (int(w * scale), height))


def compute_homography(ref_frame, warp_frame, max_features=3000):
    """
    Compute homography that maps warp_frame onto ref_frame's plane
    using ORB feature matching.
    """
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(ref_frame, None)
    kp2, des2 = orb.detectAndCompute(warp_frame, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m_n in raw_matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 10:
        print(f"Warning: only {len(good)} good matches found. Falling back to simple stitch.")
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    print(f"Homography computed using {mask.sum()} inliers out of {len(good)} matches.")
    return H


def create_blend_mask(shape, left_start, right_end, band):
    """
    Creates a float32 alpha mask for feathered blending.
    Fully opaque outside the band, gradient through the seam.
    """
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(band):
        alpha = i / band
        col = left_start + i
        if 0 <= col < w:
            mask[:, col] = alpha
    mask[:, :left_start] = 0.0
    return mask


def stitch_frames(frame1, frame2, H, output_w, output_h, blend_band):
    """
    Warp frame2 using H, place both onto a wide canvas, feather-blend the seam.
    """
    canvas_shape = (output_h, output_w, 3)

    # Place frame1 on left of canvas
    canvas1 = np.zeros(canvas_shape, dtype=np.float32)
    h1, w1 = frame1.shape[:2]
    canvas1[:h1, :w1] = frame1.astype(np.float32)

    # Warp frame2 into canvas1's coordinate space
    canvas2 = np.zeros(canvas_shape, dtype=np.float32)
    if H is not None:
        warped2 = cv2.warpPerspective(frame2.astype(np.float32), H,
                                      (output_w, output_h))
        canvas2 = warped2
    else:
        # Fallback: just paste frame2 to the right with simple overlap
        h2, w2 = frame2.shape[:2]
        x_offset = output_w - w2
        canvas2[:h2, x_offset:x_offset + w2] = frame2.astype(np.float32)

    # Build masks
    # Find where each canvas has content (non-zero)
    mask1 = (canvas1.sum(axis=2) > 0).astype(np.float32)
    mask2 = (canvas2.sum(axis=2) > 0).astype(np.float32)
    overlap = mask1 * mask2

    if overlap.sum() == 0:
        # No overlap: just add
        result = np.clip(canvas1 + canvas2, 0, 255)
        return result.astype(np.uint8)

    # Find overlap column range
    overlap_cols = np.where(overlap.max(axis=0) > 0)[0]
    if len(overlap_cols) == 0:
        result = np.clip(canvas1 + canvas2, 0, 255)
        return result.astype(np.uint8)

    seam_start = int(overlap_cols.min())
    seam_end = int(overlap_cols.max())
    seam_width = seam_end - seam_start + 1

    # Build smooth alpha that goes 0→1 across the overlap region
    alpha = np.zeros((output_h, output_w), dtype=np.float32)
    for col in range(seam_start, seam_end + 1):
        t = (col - seam_start) / max(seam_width - 1, 1)
        alpha[:, col] = t
    # Left of seam: fully canvas1, right of seam: fully canvas2
    alpha[:, :seam_start] = 0.0
    alpha[:, seam_end + 1:] = 1.0
    # Where mask2 has no content, use canvas1
    alpha = alpha * mask2 + (1.0 - mask2) * 0.0
    # Where mask1 has no content, use canvas2
    no_mask1 = (1.0 - mask1)
    alpha = np.where(no_mask1 > 0, 1.0, alpha)

    alpha3 = np.dstack([alpha] * 3)
    blended = canvas1 * (1.0 - alpha3) + canvas2 * alpha3
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── Open videos ──────────────────────────────────────────────────────────────
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

fps = cap1.get(cv2.CAP_PROP_FPS) or 30.0

ret1, frame1_raw = cap1.read()
ret2, frame2_raw = cap2.read()
if not ret1 or not ret2:
    raise RuntimeError("Could not read from one or both videos.")

frame1 = resize_to_height(frame1_raw, resize_height)
frame2 = resize_to_height(frame2_raw, resize_height)

h, w1 = frame1.shape[:2]
_, w2 = frame2.shape[:2]

# ── Compute homography from first frames ────────────────────────────────────
print("Computing homography from first frames (this is reused for all frames)...")
H = compute_homography(frame1, frame2, max_features)

# Estimate output canvas size
# Warp the four corners of frame2 to see where they land
if H is not None:
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h], [0, h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners2, H)
    all_x = np.concatenate([[0, w1], warped_corners[:, 0, 0]])
    all_y = np.concatenate([[0, h],  warped_corners[:, 0, 1]])
    x_min, x_max = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
    y_min, y_max = int(np.floor(all_y.min())), int(np.ceil(all_y.max()))
    # Shift H if canvas starts below 0
    if x_min < 0 or y_min < 0:
        tx = max(0, -x_min)
        ty = max(0, -y_min)
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        H = T @ H
        # Also shift frame1's position
        frame1_offset_x = tx
        frame1_offset_y = ty
    else:
        frame1_offset_x = 0
        frame1_offset_y = 0
    output_w = max(x_max + frame1_offset_x, w1 + frame1_offset_x) + abs(x_min)
    output_h = max(y_max + frame1_offset_y, h + frame1_offset_y)
    output_w = max(output_w, w1 + w2 - blend_band)  # sanity floor
else:
    output_w = w1 + w2 - blend_band
    output_h = resize_height
    frame1_offset_x = 0
    frame1_offset_y = 0

output_w = int(output_w)
output_h = int(min(output_h, resize_height * 2))  # cap height

print(f"Output canvas: {output_w} x {output_h}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))


def process_frame_pair(f1, f2):
    f1 = resize_to_height(f1, resize_height)
    f2 = resize_to_height(f2, resize_height)

    # Place f1 at its offset on canvas
    canvas_shape = (output_h, output_w, 3)
    canvas1 = np.zeros(canvas_shape, dtype=np.float32)
    fh, fw = f1.shape[:2]
    ey = min(frame1_offset_y + fh, output_h)
    ex = min(frame1_offset_x + fw, output_w)
    canvas1[frame1_offset_y:ey, frame1_offset_x:ex] = \
        f1[:ey - frame1_offset_y, :ex - frame1_offset_x].astype(np.float32)

    # Warp f2
    canvas2 = np.zeros(canvas_shape, dtype=np.float32)
    if H is not None:
        warped2 = cv2.warpPerspective(f2.astype(np.float32), H,
                                      (output_w, output_h))
        canvas2 = warped2
    else:
        h2, w2_ = f2.shape[:2]
        xo = output_w - w2_
        canvas2[:h2, xo:xo + w2_] = f2.astype(np.float32)

    # Blend
    mask1 = (canvas1.sum(axis=2) > 0).astype(np.float32)
    mask2 = (canvas2.sum(axis=2) > 0).astype(np.float32)

    overlap_cols = np.where((mask1 * mask2).max(axis=0) > 0)[0]
    if len(overlap_cols) == 0:
        result = np.clip(canvas1 + canvas2, 0, 255)
        return result.astype(np.uint8)

    seam_start = int(overlap_cols.min())
    seam_end = int(overlap_cols.max())
    seam_width = seam_end - seam_start + 1

    alpha = np.zeros((output_h, output_w), dtype=np.float32)
    cols = np.arange(seam_start, seam_end + 1)
    t = (cols - seam_start) / max(seam_width - 1, 1)
    alpha[:, seam_start:seam_end + 1] = t[np.newaxis, :]
    alpha[:, seam_end + 1:] = 1.0
    alpha = np.where(mask1 < 0.5, 1.0, alpha)
    alpha = alpha * mask2
    alpha = np.where((mask1 < 0.5) & (mask2 > 0.5), 1.0, alpha)
    alpha = np.where((mask1 > 0.5) & (mask2 < 0.5), 0.0, alpha)

    alpha3 = np.dstack([alpha] * 3)
    blended = canvas1 * (1.0 - alpha3) + canvas2 * alpha3
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── Write first frame ────────────────────────────────────────────────────────
merged = process_frame_pair(frame1_raw, frame2_raw)
out.write(merged)

# ── Process remaining frames ─────────────────────────────────────────────────
frame_count = 1
while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        break
    merged = process_frame_pair(f1, f2)
    out.write(merged)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"  Processed {frame_count} frames...")

cap1.release()
cap2.release()
out.release()
print(f"\nDone! {frame_count} frames written to: {output_path}")