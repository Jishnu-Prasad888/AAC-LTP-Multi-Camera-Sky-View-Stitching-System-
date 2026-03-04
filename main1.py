import cv2
import numpy as np
import subprocess
import os

# === SETTINGS ===
video1_path = "1.mp4"   # LEFT camera
video2_path = "2.mp4"   # RIGHT camera
output_path = "merged_output.mp4"
temp_path   = "merged_raw.mp4"   # intermediate file, deleted after re-encode

RESIZE_HEIGHT = 720
BLEND_BAND    = 300     # feather width in pixels — increase if seam is visible
MAX_FEATURES  = 8000    # ORB features per frame
CALIB_FRAMES  = 10      # number of frames used to compute robust homography
# =================


# ─── Helpers ──────────────────────────────────────────────────────────────────

def resize_to_height(frame, height):
    h, w = frame.shape[:2]
    if h == height:
        return frame
    return cv2.resize(frame, (int(w * height / h), height))


def match_features(img_a, img_b, max_features):
    """ORB + Lowe ratio test. Returns matched point pairs (pts_a, pts_b)."""
    orb  = cv2.ORB_create(max_features)
    ga   = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb   = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    kpa, desa = orb.detectAndCompute(ga, None)
    kpb, desb = orb.detectAndCompute(gb, None)
    if desa is None or desb is None or len(kpa) < 4 or len(kpb) < 4:
        return None, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(desa, desb, k=2)
    good = [m for m, n in raw if m.distance < 0.72 * n.distance]
    if len(good) < 10:
        return None, None
    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def compute_robust_homography(cap1, cap2, calib_frames, max_features):
    """
    Sample `calib_frames` evenly across the video, collect all match points,
    compute a single homography from the full combined point set.
    Cameras are fixed so one H fits all frames.
    """
    total = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT),
                    cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    step  = max(1, total // calib_frames)

    all_pts1, all_pts2 = [], []

    for i in range(calib_frames):
        pos = min(i * step, total - 1)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, pos)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, pos)
        r1, f1 = cap1.read()
        r2, f2 = cap2.read()
        if not r1 or not r2:
            continue
        f1 = resize_to_height(f1, RESIZE_HEIGHT)
        f2 = resize_to_height(f2, RESIZE_HEIGHT)
        pts1, pts2 = match_features(f1, f2, max_features)
        if pts1 is not None:
            all_pts1.append(pts1)
            all_pts2.append(pts2)
        print(f"  Calibration frame {i+1}/{calib_frames} — "
              f"{len(pts1) if pts1 is not None else 0} matches")

    if not all_pts1:
        raise RuntimeError("Could not find any feature matches across calibration frames.")

    pts1_all = np.vstack(all_pts1)
    pts2_all = np.vstack(all_pts2)

    H, mask = cv2.findHomography(pts2_all, pts1_all, cv2.RANSAC, 4.0,
                                  maxIters=5000, confidence=0.9995)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  Homography computed from {len(pts1_all)} points — {inliers} inliers")
    if H is None:
        raise RuntimeError("findHomography failed.")
    return H


def compute_canvas_and_offset(H, w1, h1, w2, h2):
    """
    Project corners of right image through H to find where they land,
    then size the canvas to hold both images with no clipping.
    Returns (canvas_w, canvas_h, offset_x, offset_y, H_shifted)
    where H_shifted already incorporates the canvas offset translation.
    """
    corners2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    wc = cv2.perspectiveTransform(corners2, H)

    all_x = np.concatenate([[0, w1], wc[:,0,0]])
    all_y = np.concatenate([[0, h1], wc[:,0,1]])

    xmin, xmax = float(all_x.min()), float(all_x.max())
    ymin, ymax = float(all_y.min()), float(all_y.max())

    ox = int(max(0, -xmin))   # shift everything right if right-image lands left of 0
    oy = int(max(0, -ymin))
    cw = int(xmax - xmin) + 1
    ch = int(ymax - ymin) + 1

    T = np.array([[1,0,ox],[0,1,oy],[0,0,1]], dtype=np.float64)
    return cw, ch, ox, oy, T @ H


def compute_crop_bounds(canvas_w, canvas_h, ox, oy, w1, h1,
                        H_shifted, w2, h2):
    """
    Find the largest rectangle that contains valid pixels from BOTH cameras,
    with no black from either side — i.e. the intersection of both warped images.
    Falls back to tight crop (union minus black) if intersection is too small.
    """
    # Left image occupies a simple rectangle on canvas
    l_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    l_mask[oy:oy+h1, ox:ox+w1] = 255

    # Right image: warp a white image through H_shifted
    white2 = np.ones((h2, w2), dtype=np.uint8) * 255
    r_mask = cv2.warpPerspective(white2, H_shifted, (canvas_w, canvas_h))
    r_mask = (r_mask > 128).astype(np.uint8) * 255

    intersection = cv2.bitwise_and(l_mask, r_mask)
    union        = cv2.bitwise_or (l_mask, r_mask)

    def tight_bounds(m):
        rows = np.any(m > 0, axis=1)
        cols = np.any(m > 0, axis=0)
        r0 = int(np.where(rows)[0][0]);  r1 = int(np.where(rows)[0][-1])
        c0 = int(np.where(cols)[0][0]);  c1 = int(np.where(cols)[0][-1])
        return r0, r1, c0, c1

    # Try intersection crop first (cleanest — no black at all)
    if intersection.max() > 0:
        ir0, ir1, ic0, ic1 = tight_bounds(intersection)
        inter_area = (ir1-ir0) * (ic1-ic0)
        # Use intersection only if it's at least 40% of the union area
        ur0, ur1, uc0, uc1 = tight_bounds(union)
        union_area  = (ur1-ur0) * (uc1-uc0)
        if inter_area >= 0.40 * union_area:
            print(f"  Using intersection crop: {ic1-ic0+1}x{ir1-ir0+1}")
            return ir0, ir1, ic0, ic1

    # Fall back: crop union (removes outer black, may have thin black wedges)
    ur0, ur1, uc0, uc1 = tight_bounds(union)
    print(f"  Using union crop: {uc1-uc0+1}x{ur1-ur0+1}")
    return ur0, ur1, uc0, uc1


def build_blend_weights(canvas_h, canvas_w, ox, oy, w1, h1,
                        H_shifted, w2, h2, blend_band):
    """
    Per-pixel alpha map: 0 = use left, 1 = use right.
    Feathered with smoothstep over the seam, anchored to the overlap end.
    """
    # Distance-transform based weights for smooth blending
    l_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    l_mask[oy:oy+h1, ox:ox+w1] = 255
    r_mask_f = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    white2 = np.ones((h2, w2), dtype=np.uint8) * 255
    r_warped = cv2.warpPerspective(white2, H_shifted, (canvas_w, canvas_h))
    r_mask_f = (r_warped > 128).astype(np.uint8) * 255

    has_l = (l_mask > 0).astype(np.float32)
    has_r = (r_mask_f > 0).astype(np.float32)
    overlap = has_l * has_r

    # Find overlap columns
    ov_cols = np.where(overlap.max(axis=0) > 0)[0]
    alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    if len(ov_cols) == 0:
        alpha[has_r > 0] = 1.0
    else:
        seam_end   = int(ov_cols.max())
        actual_band = min(blend_band, len(ov_cols))
        feather_start = seam_end - actual_band

        cols = np.arange(canvas_w, dtype=np.float32)
        t = np.clip((cols - feather_start) / max(actual_band, 1), 0.0, 1.0)
        # smoothstep
        t = t * t * (3 - 2 * t)
        alpha[:] = t[np.newaxis, :]

        # Outside-only regions
        alpha = np.where((has_l > 0) & (has_r < 0.5), 0.0, alpha)
        alpha = np.where((has_r > 0) & (has_l < 0.5), 1.0, alpha)
        alpha = np.where((has_l < 0.5) & (has_r < 0.5), 0.0, alpha)

    return alpha[:, :, np.newaxis].astype(np.float32)  # (H,W,1)


# ─── Main ─────────────────────────────────────────────────────────────────────

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)
fps  = cap1.get(cv2.CAP_PROP_FPS) or 30.0

# Read one frame to get dimensions
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, tmp1 = cap1.read(); tmp1 = resize_to_height(tmp1, RESIZE_HEIGHT)
_, tmp2 = cap2.read(); tmp2 = resize_to_height(tmp2, RESIZE_HEIGHT)
h1, w1 = tmp1.shape[:2]
h2, w2 = tmp2.shape[:2]

# ── Step 1: robust homography from multiple calibration frames ────────────────
print("\n[1/4] Computing robust homography from calibration frames...")
H_raw = compute_robust_homography(cap1, cap2, CALIB_FRAMES, MAX_FEATURES)

# ── Step 2: canvas geometry ───────────────────────────────────────────────────
print("\n[2/4] Computing canvas layout...")
canvas_w, canvas_h, ox, oy, H_shifted = compute_canvas_and_offset(
    H_raw, w1, h1, w2, h2)
print(f"  Full canvas: {canvas_w} x {canvas_h}  (left offset: {ox},{oy})")

# ── Step 3: crop bounds & blend weights (computed once) ───────────────────────
print("\n[3/4] Computing crop bounds and blend weights...")
r0, r1, c0, c1 = compute_crop_bounds(
    canvas_w, canvas_h, ox, oy, h1, w1, H_shifted, w2, h2)
out_w = c1 - c0 + 1
out_h = r1 - r0 + 1

alpha3 = build_blend_weights(
    canvas_h, canvas_w, ox, oy, w1, h1, H_shifted, w2, h2, BLEND_BAND)
# Pre-crop the alpha map too
alpha3_crop = alpha3[r0:r1+1, c0:c1+1]

print(f"  Output size: {out_w} x {out_h}")

# ── Step 4: process all frames ────────────────────────────────────────────────
print(f"\n[4/4] Stitching video → {temp_path} (raw)")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(temp_path, fourcc, fps, (out_w, out_h))

cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        break

    f1 = resize_to_height(f1, RESIZE_HEIGHT)
    f2 = resize_to_height(f2, RESIZE_HEIGHT)

    # Place left image on canvas
    cv_l = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ey = min(oy + h1, canvas_h); ex = min(ox + w1, canvas_w)
    cv_l[oy:ey, ox:ex] = f1[:ey-oy, :ex-ox]

    # Warp right image onto canvas
    cv_r = cv2.warpPerspective(f2, H_shifted, (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR)

    # Blend and crop
    blended = (cv_l.astype(np.float32) * (1 - alpha3_crop) +
               cv_r[r0:r1+1, c0:c1+1].astype(np.float32) * alpha3_crop)
    writer.write(np.clip(blended, 0, 255).astype(np.uint8))

    frame_count += 1
    if frame_count % 60 == 0:
        print(f"  {frame_count} frames processed...")

cap1.release()
cap2.release()
writer.release()
print(f"\n{frame_count} frames written. Re-encoding to H.264 for browser compatibility...")
subprocess.run([
    "ffmpeg", "-y", "-i", temp_path,
    "-vcodec", "libx264",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    output_path
], check=True)
os.remove(temp_path)
print(f"Done! {frame_count} frames → {output_path}  [{out_w}x{out_h} @ {fps:.1f}fps]")