# =============================================================================
# AR Origami — Square Paper
# =============================================================================
# This program opens your webcam and draws a virtual square piece of paper
# on screen.  You can fold it by pinching your thumb and index finger together
# on the paper and dragging, just like folding real paper.
#
# How the overall flow works:
#   1. Open the camera
#   2. Every frame: detect your hand using MediaPipe
#   3. Check if you are pinching on the paper
#   4. If yes, animate a fold based on how far you drag
#   5. Draw the folded paper with the paper.png texture on top of the camera feed
#   6. Show the result in a window
# =============================================================================

import os

import cv2                          # OpenCV: reads camera frames and draws on them
import mediapipe as mp              # Google's MediaPipe: detects hands and finger positions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request               # Built-in Python module to download files from the internet
import numpy as np                  # NumPy: fast math on arrays (used here for distance and transforms)

# -----------------------------------------------------------------------------
# Step 1: Make sure the hand-detection model file exists on disk.
# MediaPipe needs a pre-trained `.task` file to recognise hands.
# The first time you run this, it will be downloaded automatically (~9 MB).
# -----------------------------------------------------------------------------
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading hand detection model (one-time setup)...")
    MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)   # downloads the file and saves it locally
    print("Model downloaded successfully!")

# -----------------------------------------------------------------------------
# Step 2: MediaPipe tracks 21 numbered points (landmarks) on the hand.
# We only care about the fingertip numbers so we can look them up by name.
# Numbers come from the MediaPipe hand landmark diagram:
#   4 = thumb tip,  8 = index tip,  12 = middle tip,  16 = ring tip,  20 = pinky tip
# -----------------------------------------------------------------------------
FINGER_TIPS = {
    'Thumb':  4,
    'Index':  8,
    'Middle': 12,
    'Ring':   16,
    'Pinky':  20
}

# -----------------------------------------------------------------------------
# Step 3: Configure and create the hand detector.
# BaseOptions tells MediaPipe where the model file lives.
# HandLandmarkerOptions sets how strict the detector should be.
#   num_hands=1          → only track one hand (keeps things simple)
#   confidence values    → 0.8 means "80% sure before calling it a hand"
#                          Higher = fewer false detections, but may miss quick moves
# -----------------------------------------------------------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8
)
detector = vision.HandLandmarker.create_from_options(options)


# =============================================================================
# HAND HELPER FUNCTIONS
# These functions take the raw MediaPipe landmark data and answer useful
# questions like: "Is the user pinching?" and "Where exactly is the pinch?"
# =============================================================================

def is_pinching(landmarks, threshold: float = 0.04) -> bool:
    """
    Decides whether the user is currently pinching (thumb tip and index tip
    are very close together).

    MediaPipe gives us x, y, z coordinates in *normalised* values between 0 and 1
    (not raw pixels), so 0.04 means "4% of the frame width/height" — roughly
    the size of a fingertip on screen.

    We compute the straight-line (Euclidean) distance in 3D space between
    the two finger tips and check if it is smaller than the threshold.
    """
    thumb = landmarks[FINGER_TIPS['Thumb']]   # landmark #4: the very tip of the thumb
    index = landmarks[FINGER_TIPS['Index']]   # landmark #8: the very tip of the index finger

    # Euclidean distance formula in 3D: sqrt( (x2-x1)² + (y2-y1)² + (z2-z1)² )
    dist = np.sqrt(
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2 +
        (thumb.z - index.z) ** 2
    )
    return dist < threshold   # True = pinching, False = not pinching


def get_pinch_point(landmarks, frame_w, frame_h):
    """
    Returns the pixel position (x, y) of the midpoint between the thumb tip
    and index tip.  This is the "grab point" — where on screen the pinch is.

    MediaPipe gives normalised coords (0–1), so we multiply by the frame
    width/height to get actual pixel numbers.
    """
    thumb = landmarks[FINGER_TIPS['Thumb']]
    index = landmarks[FINGER_TIPS['Index']]
    # Average of the two tips = midpoint between them
    px = int((thumb.x + index.x) / 2 * frame_w)
    py = int((thumb.y + index.y) / 2 * frame_h)
    return px, py


def draw_finger_tips(frame, landmarks, frame_w, frame_h, pinching):
    """
    Draws a small filled circle on the thumb tip and index tip so the user
    can see which fingers are being tracked.
    Also draws a line connecting the two tips.
    Color is RED when pinching, BLUE when not pinching.
    """
    # Choose colour: BGR format — (Blue, Green, Red)
    color = (0, 0, 255) if pinching else (255, 0, 0)  # red or blue

    for name in ('Thumb', 'Index'):
        lm = landmarks[FINGER_TIPS[name]]
        # Convert normalised coords → pixel coords
        cx, cy = int(lm.x * frame_w), int(lm.y * frame_h)
        cv2.circle(frame, (cx, cy), 10, color, -1)   # -1 fills the circle solid

    # Draw a line between the two tips
    thumb = landmarks[FINGER_TIPS['Thumb']]
    index = landmarks[FINGER_TIPS['Index']]
    pt1 = (int(thumb.x * frame_w), int(thumb.y * frame_h))
    pt2 = (int(index.x * frame_w), int(index.y * frame_h))
    line_color = (0, 0, 255) if pinching else (255, 255, 255)  # red or white
    cv2.line(frame, pt1, pt2, line_color, 2)


# =============================================================================
# PAPER GEOMETRY HELPERS
# These functions define where the paper lives on screen and handle all the
# maths needed to map a pinch position to a fold action.
# =============================================================================

def get_paper_rect(frame_w, frame_h, size=400, offset_x=500):
    """
    Calculates and returns the screen position of the paper as a rectangle.
    The rectangle is defined by its top-left (x1, y1) and bottom-right (x2, y2) corners.

    By default the paper is centred on screen, then shifted 500 px to the right
    via offset_x so it sits nicely beside the hand.
    """
    x1 = (frame_w - size) // 2 + offset_x   # centre horizontally, then shift right
    y1 = (frame_h - size) // 2              # centre vertically
    return x1, y1, x1 + size, y1 + size


def point_on_paper(pt, rect):
    """
    Checks whether a pixel point `pt` (x, y) is inside the paper rectangle.
    Used to decide if the user is actually grabbing the paper or touching empty space.
    Returns True if the point is inside (inclusive of the edges).
    """
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def get_fold_edge(pt, rect):
    """
    Given a pinch point on the paper, decides which quadrant of the paper
    was grabbed — and therefore which half should fold.

    Imagine dividing the paper into 4 quadrants with a + in the centre.
    The function compares how far the pinch is from the centre horizontally
    vs vertically to pick the dominant direction.

    Returns one of: 'top', 'bottom', 'left', 'right'
    """
    x, y = pt
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2   # centre point of the paper

    # How far is the pinch from the centre in each direction?
    dx = x - cx   # positive = right of centre, negative = left of centre
    dy = y - cy   # positive = below centre, negative = above centre

    # Whichever displacement is larger (in absolute value) wins.
    # abs() removes the sign so we can compare magnitudes.
    if abs(dy) >= abs(dx):
        return 'top' if dy < 0 else 'bottom'   # pinch is more up/down than left/right
    return 'left' if dx < 0 else 'right'        # pinch is more left/right than up/down


def compute_fold_progress(grab_pt, current_pt, rect, edge):
    """
    While the user is holding a pinch and dragging, this function calculates
    HOW FAR the fold has progressed as a number between 0.0 and 1.0:
        0.0 = paper is completely flat (just started dragging)
        0.5 = paper flap is at 90 degrees (sticking straight up)
        1.0 = paper flap has folded all the way over to the other side

    The "drag" is measured in the direction that makes sense for the fold:
        'top' fold  → drag DOWN  (we want the top flap to fall toward the bottom)
        'bottom'    → drag UP
        'left'      → drag RIGHT
        'right'     → drag LEFT

    max_drag is the maximum distance the finger CAN travel before the fold is 100%.
    We divide drag by max_drag to get a 0–1 fraction.
    np.clip ensures the result never goes below 0 or above 1.
    """
    x1, y1, x2, y2 = rect
    if edge == 'top':
        drag     = current_pt[1] - grab_pt[1]       # how many pixels have we dragged down?
        max_drag = max(y2 - grab_pt[1], 1)           # maximum possible downward drag
    elif edge == 'bottom':
        drag     = grab_pt[1] - current_pt[1]        # dragged up
        max_drag = max(grab_pt[1] - y1, 1)
    elif edge == 'left':
        drag     = current_pt[0] - grab_pt[0]        # dragged right
        max_drag = max(x2 - grab_pt[0], 1)
    else:  # right
        drag     = grab_pt[0] - current_pt[0]        # dragged left
        max_drag = max(grab_pt[0] - x1, 1)
    return float(np.clip(drag / max_drag, 0.0, 1.0))


# =============================================================================
# PAPER DRAWING FUNCTIONS
# Everything here is about making the paper look right on screen —
# flat, mid-fold, or fully folded — with the paper.png texture.
# =============================================================================

def draw_flat_paper(frame, rect, texture=None, texture_crop=None):
    """
    Draws the paper in its completely flat (unfolded) state.

    The paper is drawn by blending the paper texture image on top of the
    camera feed inside the paper rectangle.  `addWeighted` mixes two images:
        result = texture * 0.85  +  camera_feed * 0.15
    So the paper is 85% opaque — you can barely see the camera through it.

    `texture_crop` is used when the paper has already been folded one or more
    times.  After a fold, `active_rect` (the visible paper area) is smaller,
    and `texture_crop` stores which part of paper.png to show inside it.
    Without this, the texture would always show the full image even on a
    half-sized paper, making it look stretched.
    """
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]   # roi = region of interest: the pixel slice where the paper goes

    if texture is not None and texture_crop is not None:
        # Multi-fold case: show only the correct crop of the texture
        tx1, ty1, tx2, ty2 = texture_crop
        region = texture[ty1:ty2, tx1:tx2]    # slice the texture image
        cv2.addWeighted(region, 0.85, roi, 0.15, 0, roi)
    elif texture is not None:
        # First fold (full paper): show the whole texture
        cv2.addWeighted(texture, 0.85, roi, 0.15, 0, roi)
    else:
        # No texture file found: fall back to a solid lavender rectangle
        paper_layer = np.full_like(roi, (230, 230, 255))   # fill with lavender colour
        cv2.addWeighted(paper_layer, 0.85, roi, 0.15, 0, roi)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 180), 2)   # draw the border


def draw_folded_paper(frame, rect, texture_crop, edge, progress, texture=None):
    """
    Draws the paper WHILE it is being folded — animating the fold in real time.

    HOW THE FOLD ANIMATION WORKS:
    -  The paper is split into two halves along the centre line.
    -  The stationary half is drawn as a flat rectangle (blit_half).
    -  The moving half (called the "flap") is drawn as a trapezoid that
       rotates over the fold line as `progress` goes from 0 → 1.
    -  At progress=0.5 the flap is perpendicular to the screen (edge-on),
       so it looks like a thin strip — this is the "foreshortening" effect.
    -  At progress=1.0 the flap has rotated 180° and lies flat on the
       opposite half (like real paper after folding).

    TAPER (foreshortening illusion):
    -  Real paper, when rotating away from you, appears to get narrower at
       the edges.  We fake this with `taper` — a few pixels we shave off
       each side of the flap.  np.sin(np.pi * progress) peaks at 0.5 (90°)
       and is 0 at both 0 and 1 — matching the real geometry perfectly.

    BACK FACE:
    -  When progress > 0.5 the underside of the paper is facing us.
       We show a horizontally mirrored, slightly darker version of the texture.

    texture_crop tracks which part of paper.png maps onto the current
    active_rect, so the texture always lines up correctly no matter how many
    folds have already been committed.
    """
    x1, y1, x2, y2 = rect
    pw, ph = x2 - x1, y2 - y1                        # paper width and height in pixels
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2         # centre point of the paper
    tcrop_x1, tcrop_y1, tcrop_x2, tcrop_y2 = texture_crop

    BORDER = (140, 140, 180)   # colour for paper edges
    CREASE = (170, 170, 195)   # colour for the fold crease line

    # Foreshortening taper — peaks at progress=0.5, zero at 0 and 1
    taper  = int(min(pw, ph) * 0.08 * np.sin(np.pi * progress))

    # ─── Inner helper: blit_half ────────────────────────────────────────────
    def blit_half(frame_x1, frame_y1, frame_x2, frame_y2,
                  rel_tx1, rel_ty1, rel_tx2, rel_ty2):
        """
        Copies a rectangular slice of the texture directly onto the frame.
        Used for the stationary half of the paper (the half that doesn't move).
        `rel_tx*` coords are relative to texture_crop's top-left corner.
        """
        region = texture[tcrop_y1 + rel_ty1 : tcrop_y1 + rel_ty2,
                         tcrop_x1 + rel_tx1 : tcrop_x1 + rel_tx2]
        frame[frame_y1:frame_y2, frame_x1:frame_x2] = region
        cv2.rectangle(frame, (frame_x1, frame_y1), (frame_x2, frame_y2), BORDER, 2)

    # ─── Inner helper: warp_flap ─────────────────────────────────────────────
    def warp_flap(rel_tx1, rel_ty1, rel_tx2, rel_ty2, dst_pts, is_back):
        """
        Perspective-warps a texture slice onto an arbitrary quadrilateral (4-point shape).
        This is what makes the folding flap look 3D.

        Steps:
          1. Extract the correct region from the texture image.
          2. If it's the back face, mirror it horizontally and darken it.
          3. Compute a perspective transform matrix M that maps the
             rectangular texture region onto the trapezoid (dst_pts).
          4. Apply the transform to the whole frame size (warpPerspective
             needs the full canvas — we mask out only the flap area after).
          5. Create a binary mask for the flap shape and composite the
             warped texture onto the frame only inside that mask.
        """
        tw, th = rel_tx2 - rel_tx1, rel_ty2 - rel_ty1
        region = texture[tcrop_y1 + rel_ty1 : tcrop_y1 + rel_ty2,
                         tcrop_x1 + rel_tx1 : tcrop_x1 + rel_tx2].copy()
        if is_back:
            region = cv2.flip(region, 1)                         # mirror left-right
            region = (region * 0.75).astype(np.uint8)            # multiply every pixel by 0.75 → 25% darker

        # src: the four corners of the rectangular texture region
        src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
        # dst: the four corners of the trapezoid on screen
        dst = np.float32(dst_pts)

        # getPerspectiveTransform solves for the 3×3 matrix M that maps src → dst
        M = cv2.getPerspectiveTransform(src, dst)
        # warpPerspective applies M to every pixel of `region`, stretching it to fit dst
        warped = cv2.warpPerspective(region, M, (frame.shape[1], frame.shape[0]))

        # Build a white mask matching the shape of the flap polygon
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(dst_pts, dtype=np.int32)], 255)

        # Write warped pixels onto the frame only where the mask is white
        frame[mask > 0] = warped[mask > 0]
        cv2.polylines(frame, [np.array(dst_pts, dtype=np.int32)], True, BORDER, 2)

    # ─── Inner helper: filled_quad (fallback, no texture) ────────────────────
    def filled_quad(pts, color):
        """Draws a filled solid-colour polygon — used when no texture file was found."""
        arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(frame, [arr], color)
        cv2.polylines(frame, [arr], True, BORDER, 2)

    # ─── Fold rendering per edge ─────────────────────────────────────────────
    # Each branch below handles one fold direction.
    # The logic for all four is the same — only which axis moves changes.

    if texture is not None:
        if edge == 'top':
            flap_h  = cy - y1          # height of the TOP half (this half folds)
            # As progress goes 0→1, top_y travels from y1 → cy → y2
            # At 0: top_y = y1 (flat).  At 0.5: top_y = cy (perpendicular).  At 1: top_y = y2 (fully over)
            top_y   = int(y1 + 2 * flap_h * progress)
            is_back = progress > 0.5   # show back face once we pass 90°
            blit_half(x1, cy, x2, y2,  0, flap_h, pw, ph)    # draw stationary bottom half
            warp_flap(0, 0, pw, flap_h,                        # warp top half as moving flap
                      [[x1 + taper, top_y], [x2 - taper, top_y], [x2, cy], [x1, cy]],
                      is_back)
            cv2.line(frame, (x1, cy), (x2, cy), CREASE, 3)    # draw the fold crease

        elif edge == 'bottom':
            flap_h  = y2 - cy
            bot_y   = int(y2 - 2 * flap_h * progress)  # bottom edge travels y2 → cy → y1
            is_back = progress > 0.5
            blit_half(x1, y1, x2, cy,  0, 0, pw, ph - flap_h)  # stationary top half
            warp_flap(0, ph - flap_h, pw, ph,
                      [[x1, cy], [x2, cy], [x2 - taper, bot_y], [x1 + taper, bot_y]],
                      is_back)
            cv2.line(frame, (x1, cy), (x2, cy), CREASE, 3)

        elif edge == 'left':
            flap_w  = cx - x1
            left_x  = int(x1 + 2 * flap_w * progress)  # left edge travels x1 → cx → x2
            is_back = progress > 0.5
            blit_half(cx, y1, x2, y2,  flap_w, 0, pw, ph)  # stationary right half
            warp_flap(0, 0, flap_w, ph,
                      [[left_x, y1 + taper], [cx, y1], [cx, y2], [left_x, y2 - taper]],
                      is_back)
            cv2.line(frame, (cx, y1), (cx, y2), CREASE, 3)

        else:  # right
            flap_w  = x2 - cx
            right_x = int(x2 - 2 * flap_w * progress)  # right edge travels x2 → cx → x1
            is_back = progress > 0.5
            blit_half(x1, y1, cx, y2,  0, 0, pw - flap_w, ph)  # stationary left half
            warp_flap(pw - flap_w, 0, pw, ph,
                      [[cx, y1], [right_x, y1 + taper], [right_x, y2 - taper], [cx, y2]],
                      is_back)
            cv2.line(frame, (cx, y1), (cx, y2), CREASE, 3)

    else:
        # ── Fallback: no texture — draw solid lavender rectangles instead ──
        PAPER_COLOR = (230, 230, 255)
        BACK_COLOR  = (195, 205, 225)   # slightly darker for the back side
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


# =============================================================================
# CAMERA & PAPER SETUP  (runs once at startup)
# =============================================================================

# Open the default webcam (index 0 = built-in camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera. Check permissions.")
    exit(1)

# Read one frame just to find out the camera resolution
ret, frame = cap.read()
if not ret:
    print("Failed to read frame")
    exit(1)
frame_h, frame_w, _ = frame.shape   # height, width, channels (we ignore channels)

# Compute the paper's screen position
paper_rect = get_paper_rect(frame_w, frame_h)

# Load paper.png and resize it to exactly match the paper square on screen.
# This pre-resized image is called `paper_texture` and re-used every frame.
_paper_src    = cv2.imread('paper.png', cv2.IMREAD_COLOR)
paper_texture = None
if _paper_src is not None:
    x1, y1, x2, y2 = paper_rect
    paper_texture = cv2.resize(_paper_src, (x2 - x1, y2 - y1))

# =============================================================================
# FOLD STATE  (these variables track what the paper looks like right now)
# =============================================================================

# Every time the user commits a fold, the fold direction is stored here.
fold_history = []

# `active_rect` is the paper area that is still visible and interactive.
# After each fold it shrinks to the half that remained.
# Example: after a 'top' fold, the top half is gone → active_rect becomes the bottom half.
active_rect = paper_rect

# `texture_rect` tells us which part of paper_texture to show inside active_rect.
# Initially it covers the whole texture (0, 0, width, height).
_ar_x1, _ar_y1, _ar_x2, _ar_y2 = paper_rect
texture_rect = (0, 0, _ar_x2 - _ar_x1, _ar_y2 - _ar_y1)

fold_edge     = None   # which edge is currently being folded ('top'/'bottom'/'left'/'right'), or None
fold_progress = 0.0    # how far the current fold has gone (0.0 = flat, 1.0 = fully folded)
grab_pt       = None   # the pixel position where the pinch started
was_pinching  = False  # was the user pinching in the previous frame? (used to detect pinch START)

# =============================================================================
# MAIN LOOP  —  runs every frame (~30 times per second)
# =============================================================================
print("Camera open.  Pinch on the paper to fold it.  Press 'c' to clear, 'q' to quit.")

while True:
    # ── 1. Grab a new frame from the webcam ──────────────────────────────────
    ret, frame = cap.read()
    if not ret:   # if reading failed (camera disconnected etc.), stop
        break

    # Flip the image horizontally so it acts like a mirror.
    # Without this, moving your right hand moves the on-screen hand to the LEFT.
    frame = cv2.flip(frame, 1)

    # ── 2. Prepare the frame for MediaPipe ───────────────────────────────────
    # OpenCV uses BGR colour order; MediaPipe expects RGB.  We convert here.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Wrap the numpy array in a MediaPipe Image object
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # Run hand detection — this fills `result` with hand landmark positions
    result    = detector.detect(mp_image)

    # ── 3. Draw the paper on top of the camera feed ──────────────────────────
    # If no fold is in progress, draw the paper flat.
    # If a fold is in progress, draw it mid-animation using fold_edge and fold_progress.
    if fold_edge is None:
        draw_flat_paper(frame, active_rect, paper_texture, texture_rect)
    else:
        draw_folded_paper(frame, active_rect, texture_rect, fold_edge, fold_progress, paper_texture)

    # ── 4. Hand & pinch interaction logic ────────────────────────────────────
    pinching = False
    if result.hand_landmarks:   # only runs if MediaPipe found a hand
        landmarks = result.hand_landmarks[0]   # get the first (and only) hand's landmarks
        pinching  = is_pinching(landmarks)
        pinch_pt  = get_pinch_point(landmarks, frame_w, frame_h)

        if pinching:
            if not was_pinching:
                # ── Pinch JUST STARTED (transition: open → pinch) ────────────
                # Only start a fold if the pinch is on the paper and no fold is already running
                if point_on_paper(pinch_pt, active_rect) and fold_edge is None:
                    grab_pt       = pinch_pt               # remember where the grab started
                    fold_edge     = get_fold_edge(pinch_pt, active_rect)   # which edge to fold
                    fold_progress = 0.0                    # start at 0% folded
            else:
                # ── Pinch is HELD (was pinching last frame too) ───────────────
                # Update fold_progress based on how far the finger has dragged
                if grab_pt is not None and fold_edge is not None:
                    fold_progress = compute_fold_progress(
                        grab_pt, pinch_pt, active_rect, fold_edge
                    )
        else:
            # ── Pinch JUST RELEASED (transition: pinch → open) ───────────────
            if was_pinching and fold_edge is not None:
                if fold_progress > 0.3:
                    # ── Fold committed (dragged far enough) ──────────────────
                    # Shrink active_rect to the half that remains after folding.
                    # Also shrink texture_rect so the texture crop stays aligned.
                    ax1, ay1, ax2, ay2 = active_rect
                    acx, acy = (ax1 + ax2) // 2, (ay1 + ay2) // 2   # centre of current paper
                    tx1, ty1, tx2, ty2 = texture_rect
                    tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2   # centre of texture crop

                    if fold_edge == 'top':
                        # Top half folded away → what remains is the bottom half
                        active_rect  = (ax1, acy, ax2, ay2)
                        texture_rect = (tx1, tcy, tx2, ty2)
                    elif fold_edge == 'bottom':
                        active_rect  = (ax1, ay1, ax2, acy)
                        texture_rect = (tx1, ty1, tx2, tcy)
                    elif fold_edge == 'left':
                        active_rect  = (acx, ay1, ax2, ay2)
                        texture_rect = (tcx, ty1, tx2, ty2)
                    else:  # right
                        active_rect  = (ax1, ay1, acx, ay2)
                        texture_rect = (tx1, ty1, tcx, ty2)

                    fold_history.append(fold_edge)   # log this fold
                    fold_edge     = None              # clear the live fold
                    fold_progress = 0.0
                else:
                    # ── Fold cancelled (released too early, < 30% dragged) ───
                    fold_edge     = None
                    fold_progress = 0.0
            grab_pt = None   # clear grab point whenever pinch is released

        # Draw the tracked fingertips and pinch midpoint on screen
        draw_finger_tips(frame, landmarks, frame_w, frame_h, pinching)
        if pinching:
            # Yellow-orange dot at the midpoint between thumb and index — the grab point
            cv2.circle(frame, pinch_pt, 7, (0, 200, 255), -1)

    # Remember the current pinch state so next frame we can detect transitions
    was_pinching = pinching

    # ── 5. Draw HUD (heads-up display) text ──────────────────────────────────
    hud = "PINCHING" if pinching else "Not pinching"
    cv2.putText(frame, hud, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255) if pinching else (0, 255, 0), 3)
    cv2.putText(frame, f"Folds: {len(fold_history)}  |  Pinch=fold  'c'=clear  'r'=reset  'q'=quit",
                (10, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── 6. Show the finished frame ────────────────────────────────────────────
    cv2.imshow("Origami", frame)

    # ── 7. Check keyboard input (waitKey waits 1 ms, returns the key pressed) ─
    key = cv2.waitKey(1) & 0xFF   # & 0xFF masks to 8 bits (needed on some systems)
    if key == ord('q'):
        break   # quit the loop
    elif key == ord('c'):
        # Clear / unfold: restore everything to the starting state
        fold_history  = []
        active_rect   = paper_rect
        _ar_x1, _ar_y1, _ar_x2, _ar_y2 = paper_rect
        texture_rect  = (0, 0, _ar_x2 - _ar_x1, _ar_y2 - _ar_y1)
        fold_edge     = None
        fold_progress = 0.0
        grab_pt       = None
    elif key == ord('r'):
        # Same as 'c' — both reset the paper
        fold_history  = []
        active_rect   = paper_rect
        _ar_x1, _ar_y1, _ar_x2, _ar_y2 = paper_rect
        texture_rect  = (0, 0, _ar_x2 - _ar_x1, _ar_y2 - _ar_y1)
        fold_edge     = None
        fold_progress = 0.0
        grab_pt       = None

# =============================================================================
# CLEANUP  —  always runs after the loop ends (whether the user pressed 'q'
#             or the window was closed)
# =============================================================================
cap.release()            # release the camera so other apps can use it
cv2.destroyAllWindows()  # close all OpenCV windows

