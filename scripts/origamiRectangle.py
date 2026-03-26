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
# maths needed to map a pinch position to a free-form fold.
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


def rect_to_poly(rect):
    """
    Converts a rectangle (x1,y1,x2,y2) into an ordered list of four corner
    vertices as float32 numpy points — the format all the geometry functions use.
    Order: top-left, top-right, bottom-right, bottom-left (clockwise).
    """
    x1, y1, x2, y2 = rect
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def point_in_poly(pt, poly):
    """
    Returns True if pixel point `pt` is inside the (possibly non-rectangular)
    paper polygon `poly`.  Uses OpenCV's pointPolygonTest which returns a
    positive number for points inside, 0 on the boundary, negative outside.
    """
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def signed_dist_to_line(pts, origin, normal):
    """
    For every point in array `pts` (shape N×2), computes its signed perpendicular
    distance to a line.

    The line is defined by:
      - `origin` : any point that lies on the line (the fold start point)
      - `normal` : a unit vector perpendicular to the line (the drag direction)

    Signed distance = dot(point - origin, normal)
      Positive → point is on the same side as the normal vector (the "folding" side)
      Negative → point is on the opposite side (the "stationary" side)
      Zero     → point lies exactly on the fold line

    This is used to split the paper polygon into two halves.
    """
    return (pts - origin) @ normal   # matrix multiply: each row dotted with normal


def reflect_points(pts, origin, normal):
    """
    Reflects an array of points across a line defined by `origin` and `normal`.

    How reflection across a line works:
      reflected = point - 2 * dot(point - origin, normal) * normal

    In plain English: project the point onto the normal direction, then move it
    twice that distance backward — so it ends up on the mirror side.
    This is the core of free-form origami: the flap vertices get reflected to
    find where they will land when fully folded over.
    """
    dists = signed_dist_to_line(pts, origin, normal)   # perpendicular distances
    # dists[:, None] reshapes (N,) → (N,1) so we can multiply by normal (shape 2,)
    return pts - 2.0 * dists[:, None] * normal


def clip_poly_to_halfplane(poly, origin, normal, keep_positive):
    """
    Clips a convex polygon to one side of a line (a "half-plane clip").
    This is the Sutherland–Hodgman algorithm applied to one clipping edge.

    The line divides space into two half-planes:
      positive side  →  signed distance > 0  (the folding flap)
      negative side  →  signed distance ≤ 0  (the stationary half)

    For each edge of the polygon, we check whether the endpoints are on the
    desired side and either keep them or add the intersection with the fold line.

    Returns the clipped polygon as a numpy array of vertices, or None if the
    polygon is entirely on the wrong side (nothing to draw).
    """
    if len(poly) < 3:
        return None

    sign = 1.0 if keep_positive else -1.0
    dists   = signed_dist_to_line(poly, origin, normal) * sign
    output  = []

    n = len(poly)
    for i in range(n):
        cur_pt, cur_d = poly[i], dists[i]
        prv_pt, prv_d = poly[(i - 1) % n], dists[(i - 1) % n]

        if cur_d >= 0:               # current vertex is on the KEEP side
            if prv_d < 0:            # previous was on the DISCARD side → add intersection
                t = prv_d / (prv_d - cur_d)          # interpolation parameter (0–1)
                output.append(prv_pt + t * (cur_pt - prv_pt))
            output.append(cur_pt)
        elif prv_d >= 0:             # current is on DISCARD, previous was on KEEP → add intersection
            t = prv_d / (prv_d - cur_d)
            output.append(prv_pt + t * (cur_pt - prv_pt))

    if len(output) < 3:
        return None
    return np.array(output, dtype=np.float32)


def compute_fold_progress(drag_vec, max_drag):
    """
    Converts the current drag vector into a 0.0–1.0 fold progress value.

    `drag_vec`  = current_pt - grab_pt  (a 2D vector in pixel space)
    `max_drag`  = the maximum useful drag distance (half the paper width/height)

    We project the drag vector onto the fold normal (which IS the drag direction
    at the moment the pinch started) to get a scalar distance, then divide by
    max_drag.  np.clip keeps it in [0, 1].
    """
    progress = float(np.linalg.norm(drag_vec)) / max(max_drag, 1)
    return float(np.clip(progress, 0.0, 1.0))


# =============================================================================
# PAPER DRAWING FUNCTIONS
# Everything here is about making the paper look right on screen —
# flat, mid-fold, or fully folded — with the paper.png texture.
# =============================================================================

# Size of the paper texture image (set once after loading, used for UV mapping)
PAPER_TEX_SIZE = (400, 400)   # (width, height) — updated at startup


def uv_to_tex(pts_screen, paper_poly_screen, paper_tex_size):
    """
    Given points in screen space and the current paper polygon, returns the
    corresponding pixel coordinates inside paper_texture.

    This is 'UV mapping' — mapping 2D screen positions onto a 2D texture.
    We use a perspective warp: fit a warp from the paper polygon→texture rect,
    then apply it to the query points. That way the texture always aligns with
    the (possibly folded) polygon shape, not just the original rectangle.

    Returns a numpy array of (tx, ty) pixel coordinates.
    """
    tw, th = paper_tex_size
    tex_corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    # Use only the first 4 vertices of the polygon for the perspective warp
    src = np.float32(paper_poly_screen[:4])
    M = cv2.getPerspectiveTransform(src, tex_corners)
    # perspectiveTransform expects shape (N, 1, 2)
    pts_h = np.array(pts_screen, dtype=np.float32).reshape(-1, 1, 2)
    tex_pts = cv2.perspectiveTransform(pts_h, M).reshape(-1, 2)
    return tex_pts


def draw_poly_with_texture(frame, poly, texture, paper_poly_orig):
    """
    Draws a filled polygon `poly` on the frame, textured with `texture`.

    Steps:
      1. Compute where each vertex of `poly` maps in texture space (UV mapping).
      2. Build a perspective warp from the bounding rect of `poly` in texture
         space to `poly`'s position on screen.
      3. Warp the texture and mask-composite it onto the frame.

    `paper_poly_orig` is the original 4-corner rectangle of the paper at its
    current active state — used as the UV reference frame.
    """
    if len(poly) < 3:
        return

    # Get bounding box of the polygon in screen space — used as intermediate canvas
    xmin = int(np.floor(poly[:, 0].min()))
    ymin = int(np.floor(poly[:, 1].min()))
    xmax = int(np.ceil(poly[:, 0].max()))
    ymax = int(np.ceil(poly[:, 1].max()))
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax = min(xmax, frame.shape[1] - 1)
    ymax = min(ymax, frame.shape[0] - 1)
    if xmax <= xmin or ymax <= ymin:
        return

    # For each corner of the texture region corresponding to the polygon, find
    # the UV coords using the original paper polygon as reference
    tex_pts = uv_to_tex(poly, paper_poly_orig, texture.shape[1::-1])

    # Clamp texture coordinates to texture dimensions
    tw, th = texture.shape[1], texture.shape[0]
    tex_pts[:, 0] = np.clip(tex_pts[:, 0], 0, tw - 1)
    tex_pts[:, 1] = np.clip(tex_pts[:, 1], 0, th - 1)

    # Build perspective warp: texture polygon → screen polygon
    # We take up to 4 points for getPerspectiveTransform
    n = min(len(poly), 4)
    src_pts = np.float32(tex_pts[:n])
    dst_pts = np.float32(poly[:n])
    if n < 4:
        # For triangles, pad to 4 by duplicating last point (degenerate warp)
        src_pts = np.vstack([src_pts, src_pts[-1:]])
        dst_pts = np.vstack([dst_pts, dst_pts[-1:]])
    M = cv2.getPerspectiveTransform(src_pts[:4], dst_pts[:4])
    warped = cv2.warpPerspective(texture, M, (frame.shape[1], frame.shape[0]))

    # Mask: only draw inside the polygon
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    frame[mask > 0] = warped[mask > 0]

    # Draw border
    cv2.polylines(frame, [poly.astype(np.int32)], True, (140, 140, 180), 2)


def draw_flat_paper(frame, poly, texture, paper_poly_ref):
    """
    Draws the paper in its flat (unfolded) state.

    `poly`           — the current paper shape (numpy array of vertices).
    `paper_poly_ref` — the 4-corner reference polygon used for UV mapping.

    The paper texture is blended 85% on top of the camera feed.
    """
    if texture is not None:
        draw_poly_with_texture(frame, poly, texture, paper_poly_ref)
    else:
        # Fallback: solid lavender fill
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        overlay = frame.copy()
        overlay[mask > 0] = (230, 230, 255)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.polylines(frame, [poly.astype(np.int32)], True, (140, 140, 180), 2)


def draw_folded_paper(frame, stationary_poly, flap_poly, fold_origin, fold_normal,
                      progress, texture, paper_poly_ref):
    """
    Draws the paper WHILE it is being folded — animating the fold in real time.

    HOW FREE-FORM FOLDING WORKS:
    ─────────────────────────────
    The paper is divided into two pieces by the fold line:
      • stationary_poly  — the half that does NOT move (drawn flat).
      • flap_poly        — the half the user is dragging (drawn as a 3D rotating flap).

    The flap animates by LINEARLY INTERPOLATING each vertex between its current
    position (progress=0, flat) and its REFLECTED position across the fold line
    (progress=1, fully folded over).

    At progress=0.5 the flap is edge-on (a line) — the foreshortening taper
    squeezes it to look like it's pointing straight at the camera.

    TAPER (foreshortening):
    ───────────────────────
    We also squeeze the flap slightly perpendicular to the fold line, by nudging
    each vertex toward the fold line.  The squeeze peaks at progress=0.5 (90°)
    and is 0 at both ends — just like real paper geometry.
    taper_factor = sin(π * progress)   ← 0 at 0 and 1, peaks at 0.5

    BACK FACE:
    ──────────
    Once progress > 0.5 the underside is visible.  We darken the texture
    slightly and mirror it horizontally so the print reads backwards
    (just like real paper from the back).
    """
    CREASE = (170, 170, 195)
    BORDER = (140, 140, 180)

    # ── 1. Draw the stationary half ────────────────────────────────────────
    if stationary_poly is not None and len(stationary_poly) >= 3:
        draw_flat_paper(frame, stationary_poly, texture, paper_poly_ref)

    if flap_poly is None or len(flap_poly) < 3:
        return

    # ── 2. Compute reflected (fully-folded) positions of the flap vertices ─
    flap_reflected = reflect_points(flap_poly, fold_origin, fold_normal)

    # ── 3. Interpolate between flat (progress=0) and reflected (progress=1) ─
    # At progress=0: animated_poly == flap_poly (flat)
    # At progress=1: animated_poly == flap_reflected (fully folded over)
    animated_poly = (1.0 - progress) * flap_poly + progress * flap_reflected

    # ── 4. Apply foreshortening taper ──────────────────────────────────────
    # Compute each vertex's distance from the fold line, then squeeze it
    # toward the fold line proportionally.
    taper_factor = np.sin(np.pi * progress) * 0.08  # peaks at 90°, max 8% squeeze
    dists = signed_dist_to_line(animated_poly, fold_origin, fold_normal)
    animated_poly = animated_poly - taper_factor * dists[:, None] * fold_normal

    # ── 5. Draw the flap with the correct texture face ────────────────────
    is_back = progress > 0.5

    if texture is not None:
        # Build texture for the flap: warp from paper_poly_ref → flap_poly (not animated)
        # so the UV mapping follows the ORIGINAL flat position of that patch.
        n = min(len(flap_poly), 4)
        tex_pts = uv_to_tex(flap_poly[:n], paper_poly_ref, texture.shape[1::-1])
        tw, th  = texture.shape[1], texture.shape[0]
        tex_pts[:, 0] = np.clip(tex_pts[:, 0], 0, tw - 1)
        tex_pts[:, 1] = np.clip(tex_pts[:, 1], 0, th - 1)

        src_pts = np.float32(tex_pts)
        dst_pts = np.float32(animated_poly[:n])
        if n < 4:
            src_pts = np.vstack([src_pts, src_pts[-1:]])
            dst_pts = np.vstack([dst_pts, dst_pts[-1:]])

        # Prepare the texture patch (back face = mirror + darken)
        tex_patch = texture.copy()
        if is_back:
            tex_patch = cv2.flip(tex_patch, 1)              # mirror left-right
            tex_patch = (tex_patch * 0.75).astype(np.uint8) # 25% darker

        M = cv2.getPerspectiveTransform(src_pts[:4], dst_pts[:4])
        warped = cv2.warpPerspective(tex_patch, M, (frame.shape[1], frame.shape[0]))

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [animated_poly.astype(np.int32)], 255)
        frame[mask > 0] = warped[mask > 0]
        cv2.polylines(frame, [animated_poly.astype(np.int32)], True, BORDER, 2)

    else:
        # Fallback: solid colour
        color = (195, 205, 225) if is_back else (230, 230, 255)
        cv2.fillPoly(frame, [animated_poly.astype(np.int32)], color)
        cv2.polylines(frame, [animated_poly.astype(np.int32)], True, BORDER, 2)

    # ── 6. Draw the fold crease line across the full paper ─────────────────
    # Project fold_origin onto the paper edges to find crease endpoints.
    # We draw the crease as a line perpendicular to fold_normal, passing
    # through fold_origin, clipped to the paper bounding box.
    fold_tangent = np.array([-fold_normal[1], fold_normal[0]])  # perpendicular to normal
    crease_len = 600   # long enough to cross any paper size
    pt1 = (fold_origin + fold_tangent * crease_len).astype(int)
    pt2 = (fold_origin - fold_tangent * crease_len).astype(int)
    cv2.line(frame, tuple(pt1), tuple(pt2), CREASE, 2)


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

# Compute the paper's screen position (initial rectangle)
paper_rect = get_paper_rect(frame_w, frame_h)

# Load paper.png and resize it to exactly match the paper square on screen.
_paper_src    = cv2.imread('paper.png', cv2.IMREAD_COLOR)
paper_texture = None
if _paper_src is not None:
    x1, y1, x2, y2 = paper_rect
    paper_texture = cv2.resize(_paper_src, (x2 - x1, y2 - y1))
    PAPER_TEX_SIZE = (x2 - x1, y2 - y1)

# =============================================================================
# FOLD STATE  (these variables track what the paper looks like right now)
# =============================================================================

# `paper_poly` is the current shape of the visible paper as an ordered polygon.
# It starts as a rectangle and becomes an arbitrary polygon after free-form folds.
paper_poly = rect_to_poly(paper_rect)

# `paper_poly_ref` is the 4-corner reference used for UV texture mapping.
# We keep this as the last committed polygon's bounding quad.
paper_poly_ref = paper_poly.copy()

# `fold_history` records each committed fold as a dict with origin + normal,
# so folds could theoretically be undone or replayed in the future.
fold_history = []

# Live fold state (cleared after each fold is committed or cancelled)
fold_origin   = None   # pixel point where the pinch started (numpy float32 array)
fold_normal   = None   # unit vector = direction of drag (perpendicular to crease)
fold_progress = 0.0    # 0.0 = flat, 1.0 = fully folded
fold_max_drag = 1.0    # estimated maximum useful drag distance (pixels)
flap_poly     = None   # the polygon piece that is currently folding
stat_poly     = None   # the polygon piece that stays stationary

grab_pt      = None    # screen pixel where the pinch began (Python tuple)
was_pinching = False   # previous-frame pinch state (for transition detection)

# =============================================================================
# MAIN LOOP  —  runs every frame (~30 times per second)
# =============================================================================
print("Camera open.  Pinch anywhere on the paper and drag to fold at any angle.")
print("Press 'c' or 'r' to reset, 'q' to quit.")

while True:
    # ── 1. Grab a new frame from the webcam ──────────────────────────────────
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame so it behaves like a mirror (left↔right)
    frame = cv2.flip(frame, 1)

    # ── 2. Prepare the frame for MediaPipe ───────────────────────────────────
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    # ── 3. Draw the paper on the camera feed ─────────────────────────────────
    if fold_origin is None:
        # No fold in progress: draw the paper flat
        draw_flat_paper(frame, paper_poly, paper_texture, paper_poly_ref)
    else:
        # Fold in progress: animate it
        draw_folded_paper(frame, stat_poly, flap_poly,
                          fold_origin, fold_normal,
                          fold_progress, paper_texture, paper_poly_ref)

    # ── 4. Hand & pinch interaction logic ────────────────────────────────────
    pinching = False
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        pinching  = is_pinching(landmarks)
        pinch_pt  = get_pinch_point(landmarks, frame_w, frame_h)

        if pinching:
            if not was_pinching:
                # ── Pinch JUST STARTED ────────────────────────────────────────
                # Check that the pinch is actually on the paper polygon
                if point_in_poly(pinch_pt, paper_poly) and fold_origin is None:
                    grab_pt = np.array(pinch_pt, dtype=np.float32)
                    # Don't compute the fold line yet — we need the drag vector,
                    # which requires the finger to move at least a few pixels first.
                    fold_origin = grab_pt.copy()
                    fold_normal = None      # computed once drag is big enough
                    fold_progress = 0.0
                    flap_poly  = None
                    stat_poly  = paper_poly.copy()

            else:
                # ── Pinch HELD — update the fold ─────────────────────────────
                if grab_pt is not None and fold_origin is not None:
                    current = np.array(pinch_pt, dtype=np.float32)
                    drag_vec = current - grab_pt

                    drag_len = np.linalg.norm(drag_vec)
                    if drag_len > 8:  # ignore tiny jitter (< 8 px)
                        # The drag direction IS the fold normal (direction the flap swings)
                        new_normal = drag_vec / drag_len

                        # Recompute the polygon split whenever the drag vector changes.
                        # This lets the user rotate their wrist for a diagonal fold!
                        if fold_normal is None or not np.allclose(new_normal, fold_normal, atol=0.05):
                            fold_normal = new_normal

                            # Split paper_poly into the flap side (same side as drag)
                            # and the stationary side (opposite side).
                            # The fold line passes through fold_origin, perpendicular to fold_normal.
                            flap_poly = clip_poly_to_halfplane(
                                paper_poly, fold_origin, fold_normal, keep_positive=True)
                            stat_poly = clip_poly_to_halfplane(
                                paper_poly, fold_origin, fold_normal, keep_positive=False)

                        # Estimate max_drag as the farthest point of the flap from the fold line
                        if flap_poly is not None:
                            dists = signed_dist_to_line(flap_poly, fold_origin, fold_normal)
                            fold_max_drag = max(dists.max(), 1.0)

                        fold_progress = compute_fold_progress(drag_vec, fold_max_drag)

        else:
            # ── Pinch JUST RELEASED ───────────────────────────────────────────
            if was_pinching and fold_origin is not None and fold_normal is not None:
                if fold_progress > 0.3 and flap_poly is not None:
                    # ── Fold committed ────────────────────────────────────────
                    # The new paper shape = stationary half (the flap vanished
                    # behind it, reflected onto the other side).
                    # We update paper_poly to only the stationary part, which is
                    # already clipped correctly by clip_poly_to_halfplane.
                    if stat_poly is not None and len(stat_poly) >= 3:
                        paper_poly = stat_poly.copy()
                        # Keep a 4-corner ref for UV by using the bounding box
                        xmin, ymin = paper_poly.min(axis=0)
                        xmax, ymax = paper_poly.max(axis=0)
                        paper_poly_ref = np.float32(
                            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                        )
                    fold_history.append({
                        'origin': fold_origin.copy(),
                        'normal': fold_normal.copy(),
                    })
                # Reset live fold state regardless of commit/cancel
                fold_origin   = None
                fold_normal   = None
                fold_progress = 0.0
                flap_poly     = None
                stat_poly     = None
            grab_pt = None

        # Draw fingertip indicators and pinch midpoint
        draw_finger_tips(frame, landmarks, frame_w, frame_h, pinching)
        if pinching:
            cv2.circle(frame, pinch_pt, 7, (0, 200, 255), -1)  # orange dot at pinch

    was_pinching = pinching

    # ── 5. Draw HUD ──────────────────────────────────────────────────────────
    hud = "PINCHING" if pinching else "Not pinching"
    cv2.putText(frame, hud, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255) if pinching else (0, 255, 0), 3)
    cv2.putText(frame,
                f"Folds: {len(fold_history)}  |  Pinch+drag=fold (any angle)  'c'=clear  'q'=quit",
                (10, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── 6. Show the finished frame ────────────────────────────────────────────
    cv2.imshow("Origami", frame)

    # ── 7. Keyboard input ────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('c'), ord('r')):
        # Reset paper to its original flat state
        paper_poly     = rect_to_poly(paper_rect)
        paper_poly_ref = paper_poly.copy()
        fold_history   = []
        fold_origin    = None
        fold_normal    = None
        fold_progress  = 0.0
        flap_poly      = None
        stat_poly      = None
        grab_pt        = None

# =============================================================================
# CLEANUP  —  always runs after the loop ends
# =============================================================================
cap.release()
cv2.destroyAllWindows()

