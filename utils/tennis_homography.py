"""
Tennis Court Homography Overlay
================================
Uses keypoints predicted by your ML model to:
  1. Compute a homography between the camera frame and a canonical court.
  2. Project court lines back onto the original video (bird's-eye warp).
  3. Render a mini-map overlay with player/ball positions.

Usage
-----
    python tennis_homography.py --input match.mp4 --output result.mp4

Plug your model in at the `predict_keypoints()` stub below.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  1.  CANONICAL COURT TEMPLATE
#      Real-world dimensions (metres) → pixel mini-map
# ═══════════════════════════════════════════════════════════════

SCALE  = 50      # pixels per metre
MARGIN = 35      # blank border around court (px)

# Standard ITF dimensions
COURT_W         = 10.97   # doubles width
COURT_H         = 23.77   # full length
SINGLES_W       = 8.23
SERVICE_BOX_H   = 6.40
NET_Y           = COURT_H / 2


def _m2p(x_m: float, y_m: float,
         scale: float = SCALE, margin: int = MARGIN) -> tuple[int, int]:
    """Convert metres → pixel (col, row)."""
    return (int(x_m * scale) + margin,
            int(y_m * scale) + margin)


def build_court_template(scale: int = SCALE,
                         margin: int = MARGIN
                         ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Build a top-down court image and return the 14 canonical keypoints.

    Keypoint order — MUST match your model's output order:
        0  top-left   doubles baseline corner
        1  top-right  doubles baseline corner
        2  bottom-right doubles baseline corner
        3  bottom-left  doubles baseline corner
        4  top-left   singles / baseline
        5  top-right  singles / baseline
        6  bottom-right singles / baseline
        7  bottom-left  singles / baseline
        8  top-left   service-box / singles sideline
        9  top-right  service-box / singles sideline
        10 bottom-right service-box / singles sideline
        11 bottom-left  service-box / singles sideline
        12 left  net post
        13 right net post

    Returns
    -------
    court_img  : BGR image (H×W×3)
    kp_world   : float32 array of shape (14, 2) — pixel coords on court_img
    (W_px, H_px): size of court_img
    """
    W_px = int(COURT_W * scale) + 2 * margin
    H_px = int(COURT_H * scale) + 2 * margin

    img = np.zeros((H_px, W_px, 3), dtype=np.uint8)
    img[:] = (34, 107, 58)   # ITF green

    def mp(x, y):
        return _m2p(x, y, scale, margin)

    WHITE = (255, 255, 255)
    T = 2

    so = (COURT_W - SINGLES_W) / 2   # singles sideline offset

    # Compute all key pixel coords
    dtl = mp(0,           0)
    dtr = mp(COURT_W,     0)
    dbr = mp(COURT_W,     COURT_H)
    dbl = mp(0,           COURT_H)
    stl = mp(so,          0)
    str_ = mp(so + SINGLES_W, 0)
    sbr = mp(so + SINGLES_W, COURT_H)
    sbl = mp(so,          COURT_H)
    net_l  = mp(0,        NET_Y)
    net_r  = mp(COURT_W,  NET_Y)
    svtl = mp(so,          SERVICE_BOX_H)
    svtr = mp(so + SINGLES_W, SERVICE_BOX_H)
    svbr = mp(so + SINGLES_W, COURT_H - SERVICE_BOX_H)
    svbl = mp(so,          COURT_H - SERVICE_BOX_H)
    svc_mid_t = mp(COURT_W / 2, SERVICE_BOX_H)
    svc_mid_b = mp(COURT_W / 2, COURT_H - SERVICE_BOX_H)

    # Draw lines
    cv2.rectangle(img, dtl, dbr, WHITE, T)          # doubles outline
    cv2.line(img, stl,  sbl,  WHITE, T)             # singles left
    cv2.line(img, str_, sbr,  WHITE, T)             # singles right
    cv2.line(img, net_l, net_r, WHITE, T + 1)       # net (thicker)
    cv2.line(img, svtl, svtr, WHITE, T)             # top service line
    cv2.line(img, svbl, svbr, WHITE, T)             # bottom service line
    cv2.line(img, svc_mid_t, svc_mid_b, WHITE, T)  # centre service line

    # Center marks on baselines
    cx = int(COURT_W / 2 * scale) + margin
    cv2.line(img, (cx, margin), (cx, margin + 8), WHITE, T)
    cv2.line(img, (cx, H_px - margin - 8), (cx, H_px - margin), WHITE, T)

    kp_world = np.array([
        dtl,   # 0
        dtr,   # 1
        dbr,   # 2
        dbl,   # 3
        stl,   # 4
        str_,  # 5
        sbr,   # 6
        sbl,   # 7
        svtl,  # 8
        svtr,  # 9
        svbr,  # 10
        svbl,  # 11
        net_l, # 12
        net_r, # 13
    ], dtype=np.float32)

    return img, kp_world, (W_px, H_px)


# ═══════════════════════════════════════════════════════════════
#  2.  MODEL INTERFACE  ← plug your model in here
# ═══════════════════════════════════════════════════════════════

def predict_keypoints(frame: np.ndarray, model) -> np.ndarray | None:
    """
    Run your ML model on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray  — BGR frame from cv2.VideoCapture
    model : any         — your model object (torch, tf, onnx, …)

    Returns
    -------
    np.float32 array of shape (N, 2), pixel (x, y) per keypoint,
    in the same order as kp_world above.
    Return None if the model is not confident enough.
    """
    # ── REPLACE the body below with your inference call ──────
    #
    # Example for a PyTorch model that returns normalised coords:
    #
    #   import torch, torchvision.transforms as T
    #   h, w = frame.shape[:2]
    #   inp = T.ToTensor()(frame).unsqueeze(0)
    #   with torch.no_grad():
    #       pred = model(inp)[0].cpu().numpy()   # shape (14, 2) in [0, 1]
    #   pred[:, 0] *= w
    #   pred[:, 1] *= h
    #   conf = model.confidence(inp)
    #   return pred.astype(np.float32) if conf > 0.5 else None
    #
    raise NotImplementedError("Replace predict_keypoints() with your model call.")


# ═══════════════════════════════════════════════════════════════
#  3.  HOMOGRAPHY HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    min_points: int = 4,
    ransac_thresh: float = 5.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Compute the homography H that maps src_pts (image keypoints)
    to dst_pts (world/template keypoints).

    Uses RANSAC to discard outlier keypoints automatically.

    Returns (H, mask) or (None, None) when estimation fails.
    """
    if src_pts is None or len(src_pts) < min_points:
        return None, None

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    return H, mask


def warp_court_onto_frame(
    court_template: np.ndarray,
    H_world_to_frame: np.ndarray,
    frame: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Warp the flat court template BACK onto the camera frame using
    the *inverse* homography so the lines align with perspective.

    Parameters
    ----------
    H_world_to_frame : inverse of the frame→world H
    alpha            : blend strength (0=invisible, 1=fully opaque)
    """
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(court_template, H_world_to_frame, (w, h))

    # Only blend where the template has non-black content
    mask = (warped.sum(axis=2) > 0).astype(np.float32)[:, :, np.newaxis]
    blended = (frame * (1 - alpha * mask) +
               warped * alpha * mask).astype(np.uint8)
    return blended


# ─── Mini-map ───────────────────────────────────────────────────

def draw_minimap(
    frame: np.ndarray,
    court_template: np.ndarray,
    kp_frame: np.ndarray,
    H: np.ndarray,
    extra_points: list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
    position: str = "bottom-right",
    size_frac: float = 0.22,
) -> np.ndarray:
    """
    Paste a mini top-down court map in a corner.

    Parameters
    ----------
    extra_points : list of (point_array, BGR_color) — e.g. player/ball positions
                   Each point_array has shape (1, 2) in frame pixel coords.
    position     : 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left'
    size_frac    : fraction of frame width occupied by the mini-map
    """
    fh, fw = frame.shape[:2]
    th, tw = court_template.shape[:2]

    target_w = max(80, int(fw * size_frac))
    target_h = int(th * target_w / tw)
    mini = cv2.resize(court_template.copy(), (target_w, target_h))

    sx, sy = target_w / tw, target_h / th

    # Project keypoints onto mini-map
    if H is not None and kp_frame is not None:
        proj = cv2.perspectiveTransform(
            kp_frame.reshape(-1, 1, 2), H).reshape(-1, 2)
        for px, py in proj:
            mx, my = int(px * sx), int(py * sy)
            if 0 <= mx < target_w and 0 <= my < target_h:
                cv2.circle(mini, (mx, my), 3, (0, 255, 255), -1)

    # Optional extra points (players, ball, …)
    if extra_points and H is not None:
        for pts, color in extra_points:
            proj_extra = cv2.perspectiveTransform(
                pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            for px, py in proj_extra:
                mx, my = int(px * sx), int(py * sy)
                if 0 <= mx < target_w and 0 <= my < target_h:
                    cv2.circle(mini, (mx, my), 5, color, -1)
                    cv2.circle(mini, (mx, my), 5, (255, 255, 255), 1)

    # Paste with a semi-transparent background
    pad = 10
    if position == "bottom-right":
        x0, y0 = fw - target_w - pad, fh - target_h - pad
    elif position == "bottom-left":
        x0, y0 = pad, fh - target_h - pad
    elif position == "top-right":
        x0, y0 = fw - target_w - pad, pad
    else:   # top-left
        x0, y0 = pad, pad

    roi = frame[y0:y0+target_h, x0:x0+target_w]
    frame[y0:y0+target_h, x0:x0+target_w] = cv2.addWeighted(
        roi, 0.25, mini, 0.75, 0)
    cv2.rectangle(frame,
                  (x0 - 2, y0 - 2),
                  (x0 + target_w + 2, y0 + target_h + 2),
                  (255, 255, 255), 1)
    return frame


# ─── Debug overlays ─────────────────────────────────────────────

def draw_keypoints_on_frame(
    frame: np.ndarray,
    kp_frame: np.ndarray,
    mask: np.ndarray | None = None,
    color_inlier=(0, 255, 0),
    color_outlier=(0, 80, 200),
) -> np.ndarray:
    """
    Draw the raw model keypoints on the camera frame.
    Inliers (accepted by RANSAC) are green; outliers are orange.
    """
    for i, (x, y) in enumerate(kp_frame.astype(int)):
        is_inlier = mask is None or (mask[i][0] == 1)
        col = color_inlier if is_inlier else color_outlier
        cv2.circle(frame, (x, y), 5, col, -1)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
        cv2.putText(frame, str(i), (x + 7, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return frame


def draw_projected_court_lines(
    frame: np.ndarray,
    H_inv: np.ndarray,
    court_template: np.ndarray,
) -> np.ndarray:
    """
    Alternative to warp_court_onto_frame: project individual line segments
    directly for a cleaner, AR-style overlay (no texture bleed).
    """
    h, w = frame.shape[:2]

    # Sample white pixels from the template and project each line
    # by warping the whole template — same result, simpler code.
    warped = cv2.warpPerspective(court_template, H_inv, (w, h))
    mask = warped[:, :, 1] > 100   # green channel (white lines are bright)
    frame[mask] = cv2.addWeighted(
        frame, 0.4, warped, 0.6, 0)[mask]
    return frame


# ═══════════════════════════════════════════════════════════════
#  4.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_video(
    input_path: str,
    output_path: str,
    model,
    *,
    draw_lines: bool = True,
    draw_kp: bool = True,
    draw_mini: bool = True,
    line_alpha: float = 0.40,
    minimap_pos: str = "bottom-right",
    fallback_frames: int = 30,      # reuse last good H for this many frames
) -> None:
    """
    Read every frame, predict keypoints, compute H, write annotated output.

    Parameters
    ----------
    model           : your ML model object
    draw_lines      : project court lines onto the frame
    draw_kp         : show keypoint dots on the frame
    draw_mini       : show the mini-map overlay
    line_alpha      : opacity of projected court lines (0–1)
    minimap_pos     : corner for the mini-map
    fallback_frames : max frames to carry forward the last good H
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Cannot create output: {output_path}")

    court_template, kp_world, _ = build_court_template()

    H_cache       = None   # last valid homography
    frames_no_H   = 0      # consecutive frames without a fresh H
    frame_idx     = 0

    print(f"Processing {total} frames at {fps:.1f} fps …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── (a) model inference ──────────────────────────────
        kp_frame = None
        try:
            kp_frame = predict_keypoints(frame, model)
        except Exception as exc:
            print(f"  [frame {frame_idx}] model error: {exc}")

        # ── (b) homography ───────────────────────────────────
        H, mask = compute_homography(kp_frame, kp_world)

        if H is not None:
            H_cache     = H
            frames_no_H = 0
        else:
            frames_no_H += 1
            if frames_no_H <= fallback_frames:
                H = H_cache          # carry forward
            else:
                H = None             # too stale, give up

        # ── (c) overlays ─────────────────────────────────────
        if H is not None:

            if draw_lines:
                try:
                    H_inv = np.linalg.inv(H)
                    frame = warp_court_onto_frame(
                        court_template, H_inv, frame, alpha=line_alpha)
                except np.linalg.LinAlgError:
                    pass   # singular matrix — skip this frame

            if draw_kp and kp_frame is not None:
                frame = draw_keypoints_on_frame(frame, kp_frame, mask)

            if draw_mini:
                frame = draw_minimap(
                    frame, court_template,
                    kp_frame, H,
                    position=minimap_pos,
                )

        # ── (d) status HUD ───────────────────────────────────
        if H is not None:
            status_txt = "TRACKING"
            status_col = (0, 220, 0)
        elif H_cache is not None:
            status_txt = f"FALLBACK ({frames_no_H}f)"
            status_col = (0, 165, 255)
        else:
            status_txt = "LOST"
            status_col = (0, 0, 220)

        cv2.putText(frame,
                    f"{frame_idx + 1}/{total}  [{status_txt}]",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame,
                    f"{frame_idx + 1}/{total}  [{status_txt}]",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, status_col, 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total}  ({pct:.1f}%)")

    cap.release()
    out.release()
    print(f"\n✓ Done — saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════
#  5.  BONUS: SINGLE-FRAME QUICK TEST
#      Run this to verify your keypoints are correct before
#      processing the full video.
# ═══════════════════════════════════════════════════════════════

def test_single_frame(
    image_path: str,
    kp_frame_list: list[tuple[float, float]],
    output_path: str = "test_output.jpg",
) -> None:
    """
    Manually supply keypoints (e.g. clicked by hand) to verify
    the homography on a single image before running the full pipeline.

    Example
    -------
    test_single_frame(
        "frame.jpg",
        kp_frame_list=[
            (120, 85),   # kp 0 — top-left doubles corner in frame
            (1145, 82),  # kp 1 — top-right
            (1160, 700), # kp 2 — bottom-right
            (108, 703),  # kp 3 — bottom-left
            # add remaining points as needed …
        ],
    )
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    court_template, kp_world, _ = build_court_template()
    kp_frame = np.array(kp_frame_list, dtype=np.float32)

    # Use only the points you supplied
    n = min(len(kp_frame), len(kp_world))
    H, mask = compute_homography(kp_frame[:n], kp_world[:n])

    if H is None:
        print("⚠  Homography failed — check that your points are in the right order.")
        return

    # Project lines
    H_inv = np.linalg.inv(H)
    result = warp_court_onto_frame(court_template, H_inv, img, alpha=0.45)
    result = draw_keypoints_on_frame(result, kp_frame, mask)
    result = draw_minimap(result, court_template, kp_frame, H)

    cv2.imwrite(output_path, result)
    print(f"✓ Test image saved to: {output_path}")

    # Reprojection error (sanity check)
    proj = cv2.perspectiveTransform(
        kp_frame[:n].reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(proj - kp_world[:n], axis=1)
    print(f"  Reprojection errors (px) — mean: {errors.mean():.2f}, "
          f"max: {errors.max():.2f}")


# ═══════════════════════════════════════════════════════════════
#  6.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Tennis court homography overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  required=True,  help="Input video path")
    p.add_argument("--output", required=True,  help="Output video path")
    p.add_argument("--alpha",  type=float, default=0.40,
                   help="Line overlay opacity (0–1)")
    p.add_argument("--minimap-pos", default="bottom-right",
                   choices=["bottom-right", "bottom-left",
                            "top-right", "top-left"])
    p.add_argument("--fallback-frames", type=int, default=30,
                   help="Carry last good H for this many frames on failure")
    p.add_argument("--no-lines",  action="store_true")
    p.add_argument("--no-kp",     action="store_true")
    p.add_argument("--no-mini",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── Load YOUR model here ─────────────────────────────────
    # from your_package import CourtKeypointModel
    # model = CourtKeypointModel.from_pretrained("weights.pth")
    model = None   # ← replace this

    process_video(
        input_path     = args.input,
        output_path    = args.output,
        model          = model,
        draw_lines     = not args.no_lines,
        draw_kp        = not args.no_kp,
        draw_mini      = not args.no_mini,
        line_alpha     = args.alpha,
        minimap_pos    = args.minimap_pos,
        fallback_frames= args.fallback_frames,
    )
