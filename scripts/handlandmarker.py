#  hand tracking code with mediapipe

import os

import cv2  # Video capture and image processing
import mediapipe as mp  # Hand detection and tracking
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request  # For downloading model file
import numpy as np

MODEL_PATH = 'hand_landmarker.task'
# Check if hand model file exists, if not download it
if not os.path.exists(MODEL_PATH):
    print("Downloading hand detection model (one-time setup)...")
    MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# MediaPipe detects 21 points per hand. We only need the 5 finger tips.
FINGER_TIPS = {
    'Thumb': 4,
    'Index': 8,
    'Middle': 12,
    'Ring': 16,
    'Pinky': 20
}

# Configure MediaPipe HandLandmarker (new API for v0.10+)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,  # Detect up to 1 hand
    min_hand_detection_confidence=0.8,  # Stricter - reduce false detections
    min_hand_presence_confidence=0.8,   # Stricter - reduce ghost hands
    min_tracking_confidence=0.8         # Stricter - more stable tracking
)

# Create hand detector
detector = vision.HandLandmarker.create_from_options(options)


# ============================================================================
# Hand helper functions
# ============================================================================

def is_pinching(landmarks, threshold: float = 0.04) -> bool:
    """Return True if thumb tip and index tip are close enough to count as a pinch."""
    thumb = landmarks[FINGER_TIPS['Thumb']]
    index = landmarks[FINGER_TIPS['Index']]
    dist = np.sqrt(
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2 +
        (thumb.z - index.z) ** 2
    )
    return dist < threshold


def get_pinch_point(landmarks, frame_w, frame_h):
    """Return pixel (x, y) of the midpoint between thumb tip and index tip."""
    thumb = landmarks[FINGER_TIPS['Thumb']]
    index = landmarks[FINGER_TIPS['Index']]
    px = int((thumb.x + index.x) / 2 * frame_w)
    py = int((thumb.y + index.y) / 2 * frame_h)
    return px, py


def draw_finger_tips(frame, landmarks, frame_w, frame_h, pinching):
    """Draw thumb tip, index tip, and a connecting line."""
    color = (0, 0, 255) if pinching else (255, 0, 0)
    for name in ('Thumb', 'Index'):
        lm = landmarks[FINGER_TIPS[name]]
        cx, cy = int(lm.x * frame_w), int(lm.y * frame_h)
        cv2.circle(frame, (cx, cy), 10, color, -1)
    thumb = landmarks[FINGER_TIPS['Thumb']]
    index = landmarks[FINGER_TIPS['Index']]
    pt1 = (int(thumb.x * frame_w), int(thumb.y * frame_h))
    pt2 = (int(index.x * frame_w), int(index.y * frame_h))
    cv2.line(frame, pt1, pt2, (0, 0, 255) if pinching else (255, 255, 255), 2)


# ============================================================================
# Paper geometry helpers
# ============================================================================

def get_paper_rect(frame_w, frame_h, size=280):
    """Return (x1, y1, x2, y2) for a square paper centered in the frame."""
    x1 = (frame_w - size) // 2
    y1 = (frame_h - size) // 2
    return x1, y1, x1 + size, y1 + size


def point_on_paper(pt, rect):
    """Return True if pixel point (x, y) lies inside the paper rect."""
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def get_fold_edge(pt, rect):
    """
    Decide which half of the paper was grabbed.
    Returns 'top', 'bottom', 'left', or 'right' depending on which
    quadrant of the paper the pinch point lands in.
    """
    x, y = pt
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x - cx, y - cy
    if abs(dy) >= abs(dx):
        return 'top' if dy < 0 else 'bottom'
    return 'left' if dx < 0 else 'right'


def compute_fold_progress(grab_pt, current_pt, rect, edge):
    """
    Map pinch-drag distance to a fold progress value in [0.0, 1.0].
    0.0 = paper is flat, 1.0 = paper is fully folded over.
    The drag direction that increases progress depends on the fold edge:
      top    → drag downward
      bottom → drag upward
      left   → drag rightward
      right  → drag leftward
    """
    x1, y1, x2, y2 = rect
    if edge == 'top':
        drag     = current_pt[1] - grab_pt[1]
        max_drag = max(y2 - grab_pt[1], 1)
    elif edge == 'bottom':
        drag     = grab_pt[1] - current_pt[1]
        max_drag = max(grab_pt[1] - y1, 1)
    elif edge == 'left':
        drag     = current_pt[0] - grab_pt[0]
        max_drag = max(x2 - grab_pt[0], 1)
    else:  # right
        drag     = grab_pt[0] - current_pt[0]
        max_drag = max(grab_pt[0] - x1, 1)
    return float(np.clip(drag / max_drag, 0.0, 1.0))


# ============================================================================
# Paper drawing functions
# ============================================================================

def draw_flat_paper(frame, rect, texture=None):
    """
    Draw the paper as a flat square.
    If a texture image is provided it is blended in; otherwise a solid
    light-lavender colour is used.
    """
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    if texture is not None:
        cv2.addWeighted(texture, 0.85, roi, 0.15, 0, roi)
    else:
        paper_layer = np.full_like(roi, (230, 230, 255))
        cv2.addWeighted(paper_layer, 0.85, roi, 0.15, 0, roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 180), 2)


def draw_folded_paper(frame, rect, edge, progress, texture=None):
    """
    Draw the paper mid-fold, preserving the paper.png texture on both halves.

    The half indicated by `edge` rotates over the centre fold line:
      progress = 0.0  → paper is flat
      progress = 0.5  → flap is perpendicular (foreshortened to a thin strip)
      progress = 1.0  → flap is fully reflected onto the opposite half

    If `texture` is supplied (a BGR image sized to the paper rect), it is
    perspective-warped onto both the static half and the animated flap.
    The back face is a horizontally-flipped, slightly darkened copy.
    """
    x1, y1, x2, y2 = rect
    pw, ph = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    BORDER = (140, 140, 180)
    CREASE = (170, 170, 195)
    taper  = int(min(pw, ph) * 0.08 * np.sin(np.pi * progress))

    def blit_half(frame_x1, frame_y1, frame_x2, frame_y2,
                  tex_x1, tex_y1, tex_x2, tex_y2):
        """Directly copy a rectangular slice of texture onto a frame region."""
        region = texture[tex_y1:tex_y2, tex_x1:tex_x2]
        frame[frame_y1:frame_y2, frame_x1:frame_x2] = region
        cv2.rectangle(frame, (frame_x1, frame_y1), (frame_x2, frame_y2), BORDER, 2)

    def warp_flap(tex_x1, tex_y1, tex_x2, tex_y2, dst_pts, is_back):
        """Perspective-warp a texture slice onto an arbitrary quad in the frame."""
        tw, th = tex_x2 - tex_x1, tex_y2 - tex_y1
        region = texture[tex_y1:tex_y2, tex_x1:tex_x2].copy()
        if is_back:
            region = cv2.flip(region, 1)          # mirror the back face
            region = (region * 0.75).astype(np.uint8)  # slightly darker
        src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
        dst = np.float32(dst_pts)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(region, M, (frame.shape[1], frame.shape[0]))
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(dst_pts, dtype=np.int32)], 255)
        frame[mask > 0] = warped[mask > 0]
        cv2.polylines(frame, [np.array(dst_pts, dtype=np.int32)], True, BORDER, 2)

    def filled_quad(pts, color):
        arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(frame, [arr], color)
        cv2.polylines(frame, [arr], True, BORDER, 2)

    if texture is not None:
        if edge == 'top':
            flap_h  = cy - y1
            top_y   = int(y1 + 2 * flap_h * progress)
            is_back = progress > 0.5
            blit_half(x1, cy, x2, y2,  0, flap_h, pw, ph)
            warp_flap(0, 0, pw, flap_h,
                      [[x1 + taper, top_y], [x2 - taper, top_y], [x2, cy], [x1, cy]],
                      is_back)
            cv2.line(frame, (x1, cy), (x2, cy), CREASE, 3)

        elif edge == 'bottom':
            flap_h  = y2 - cy
            bot_y   = int(y2 - 2 * flap_h * progress)
            is_back = progress > 0.5
            blit_half(x1, y1, x2, cy,  0, 0, pw, ph - flap_h)
            warp_flap(0, ph - flap_h, pw, ph,
                      [[x1, cy], [x2, cy], [x2 - taper, bot_y], [x1 + taper, bot_y]],
                      is_back)
            cv2.line(frame, (x1, cy), (x2, cy), CREASE, 3)

        elif edge == 'left':
            flap_w  = cx - x1
            left_x  = int(x1 + 2 * flap_w * progress)
            is_back = progress > 0.5
            blit_half(cx, y1, x2, y2,  flap_w, 0, pw, ph)
            warp_flap(0, 0, flap_w, ph,
                      [[left_x, y1 + taper], [cx, y1], [cx, y2], [left_x, y2 - taper]],
                      is_back)
            cv2.line(frame, (cx, y1), (cx, y2), CREASE, 3)

        else:  # right
            flap_w  = x2 - cx
            right_x = int(x2 - 2 * flap_w * progress)
            is_back = progress > 0.5
            blit_half(x1, y1, cx, y2,  0, 0, pw - flap_w, ph)
            warp_flap(pw - flap_w, 0, pw, ph,
                      [[cx, y1], [right_x, y1 + taper], [right_x, y2 - taper], [cx, y2]],
                      is_back)
            cv2.line(frame, (cx, y1), (cx, y2), CREASE, 3)

    else:
        # Fallback: solid colour rendering when no texture is available
        PAPER_COLOR = (230, 230, 255)
        BACK_COLOR  = (195, 205, 225)
        if edge == 'top':
            flap_h = cy - y1
            filled_quad([(x1, cy), (x2, cy), (x2, y2), (x1, y2)], PAPER_COLOR)
            top_y = int(y1 + 2 * flap_h * progress)
            filled_quad([(x1 + taper, top_y), (x2 - taper, top_y), (x2, cy), (x1, cy)],
                        BACK_COLOR if progress > 0.5 else PAPER_COLOR)
            cv2.line(frame, (x1, cy), (x2, cy), CREASE, 3)
        elif edge == 'bottom':
            flap_h = y2 - cy
            filled_quad([(x1, y1), (x2, y1), (x2, cy), (x1, cy)], PAPER_COLOR)
            bot_y = int(y2 - 2 * flap_h * progress)
            filled_quad([(x1, cy), (x2, cy), (x2 - taper, bot_y), (x1 + taper, bot_y)],
                        BACK_COLOR if progress > 0.5 else PAPER_COLOR)
            cv2.line(frame, (x1, cy), (x2, cy), CREASE, 3)
        elif edge == 'left':
            flap_w = cx - x1
            filled_quad([(cx, y1), (x2, y1), (x2, y2), (cx, y2)], PAPER_COLOR)
            left_x = int(x1 + 2 * flap_w * progress)
            filled_quad([(left_x, y1 + taper), (cx, y1), (cx, y2), (left_x, y2 - taper)],
                        BACK_COLOR if progress > 0.5 else PAPER_COLOR)
            cv2.line(frame, (cx, y1), (cx, y2), CREASE, 3)
        else:  # right
            flap_w = x2 - cx
            filled_quad([(x1, y1), (cx, y1), (cx, y2), (x1, y2)], PAPER_COLOR)
            right_x = int(x2 - 2 * flap_w * progress)
            filled_quad([(cx, y1), (right_x, y1 + taper), (right_x, y2 - taper), (cx, y2)],
                        BACK_COLOR if progress > 0.5 else PAPER_COLOR)
            cv2.line(frame, (cx, y1), (cx, y2), CREASE, 3)


# ============================================================================
# Camera & paper setup
# ============================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera. Check permissions.")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("Failed to read frame")
    exit(1)
frame_h, frame_w, _ = frame.shape

paper_rect = get_paper_rect(frame_w, frame_h)

# Load paper.png as a texture and resize it to the paper square
_paper_src   = cv2.imread('paper.png', cv2.IMREAD_COLOR)
paper_texture = None
if _paper_src is not None:
    x1, y1, x2, y2 = paper_rect
    paper_texture = cv2.resize(_paper_src, (x2 - x1, y2 - y1))

# ============================================================================
# Fold state
# ============================================================================
fold_edge      = None   # 'top' | 'bottom' | 'left' | 'right' | None
fold_progress  = 0.0    # 0.0 (flat) → 1.0 (fully folded)
fold_committed = False  # True once the user releases a completed fold
grab_pt        = None   # pixel coords where the pinch started on the paper
was_pinching   = False  # pinch state from the previous frame

# ============================================================================
# Main loop
# ============================================================================
print("Camera open.  Pinch on the paper to fold it.  Press 'r' to reset, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    # ── Draw paper ──────────────────────────────────────────────────────────
    if fold_edge is None:
        draw_flat_paper(frame, paper_rect, paper_texture)
    else:
        draw_folded_paper(frame, paper_rect, fold_edge, fold_progress, paper_texture)

    # ── Hand & pinch logic ──────────────────────────────────────────────────
    pinching = False
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        pinching  = is_pinching(landmarks)
        pinch_pt  = get_pinch_point(landmarks, frame_w, frame_h)

        if pinching:
            if not was_pinching:
                # Pinch just started — begin a fold if on paper and not committed
                if point_on_paper(pinch_pt, paper_rect) and not fold_committed:
                    grab_pt       = pinch_pt
                    fold_edge     = get_fold_edge(pinch_pt, paper_rect)
                    fold_progress = 0.0
            else:
                # Pinch held — update live fold progress
                if grab_pt is not None and fold_edge is not None and not fold_committed:
                    fold_progress = compute_fold_progress(
                        grab_pt, pinch_pt, paper_rect, fold_edge
                    )
        else:
            if was_pinching:
                if fold_edge is not None and fold_progress > 0.3:
                    # Released past 30 % → commit the fold
                    fold_committed = True
                    fold_progress  = 1.0
                else:
                    # Released too early → snap back flat
                    fold_edge      = None
                    fold_progress  = 0.0
                    fold_committed = False
            grab_pt = None

        draw_finger_tips(frame, landmarks, frame_w, frame_h, pinching)
        if pinching:
            cv2.circle(frame, pinch_pt, 7, (0, 200, 255), -1)  # pinch midpoint

    was_pinching = pinching

    # ── HUD ─────────────────────────────────────────────────────────────────
    hud = "PINCHING" if pinching else "Not pinching"
    cv2.putText(frame, hud, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255) if pinching else (0, 255, 0), 3)
    cv2.putText(frame, "Pinch paper to fold  |  'r' reset  |  'q' quit",
                (10, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Origami", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        fold_edge      = None
        fold_progress  = 0.0
        fold_committed = False
        grab_pt        = None

cap.release()
cv2.destroyAllWindows()
