# =============================================================================
# origami_backend.py  —  Guppy Folds FastAPI backend
# =============================================================================
# Browser sends webcam frames over WebSocket → we run MediaPipe + origami
# logic → we send back the processed JPEG frame → browser displays it.
# =============================================================================

import base64, json, os, urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# -----------------------------------------------------------------------------
# Paths — relative to this file so they work from any working directory
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent   # origami/ root
MODEL_PATH = str(BASE_DIR / 'hand_landmarker.task')
PAPER_PNG  = str(BASE_DIR / 'paper.png')

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model…")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
        'hand_landmarker/float16/1/hand_landmarker.task', MODEL_PATH)
    print("Done.")

# ---------------------------------------------------------------------------
# MediaPipe detector  (one global instance shared across all connections)
# ---------------------------------------------------------------------------
_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.8,
        min_tracking_confidence=0.8,
    ))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)

# ---------------------------------------------------------------------------
# Pure geometry helpers (no side-effects)
# ---------------------------------------------------------------------------
def _is_pinching(lm, thr=0.04):
    t, i = lm[4], lm[8]
    return ((t.x-i.x)**2 + (t.y-i.y)**2 + (t.z-i.z)**2) ** 0.5 < thr

def _pinch_pt(lm, fw, fh):
    t, i = lm[4], lm[8]
    return int((t.x+i.x)/2*fw), int((t.y+i.y)/2*fh)

def _paper_rect(fw, fh, size=280):
    x1 = (fw - size) // 2
    y1 = (fh - size) // 2
    return x1, y1, x1+size, y1+size

def _fold_edge(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    cx, cy = (x1+x2)/2, (y1+y2)/2
    dx, dy = x-cx, y-cy
    if abs(dy) >= abs(dx):
        return 'top' if dy < 0 else 'bottom'
    return 'left' if dx < 0 else 'right'

def _fold_progress(gpt, cpt, rect, edge):
    x1, y1, x2, y2 = rect
    if   edge == 'top':    d, m = cpt[1]-gpt[1], max(y2-gpt[1], 1)
    elif edge == 'bottom': d, m = gpt[1]-cpt[1], max(gpt[1]-y1, 1)
    elif edge == 'left':   d, m = cpt[0]-gpt[0], max(x2-gpt[0], 1)
    else:                  d, m = gpt[0]-cpt[0], max(gpt[0]-x1, 1)
    return float(np.clip(d/m, 0, 1))

# ---------------------------------------------------------------------------
# Paper drawing helpers
# ---------------------------------------------------------------------------
def _draw_flat(frame, rect, tex, tc):
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    if tex is not None and tc is not None:
        p1, q1, p2, q2 = tc
        cv2.addWeighted(tex[q1:q2, p1:p2], 0.85, roi, 0.15, 0, roi)
    elif tex is not None:
        cv2.addWeighted(tex, 0.85, roi, 0.15, 0, roi)
    else:
        cv2.addWeighted(np.full_like(roi, (230,230,255)), 0.85, roi, 0.15, 0, roi)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (140,140,180), 2)

def _draw_folded(frame, rect, tc, edge, progress, tex):
    x1, y1, x2, y2 = rect
    pw, ph = x2-x1, y2-y1
    cx, cy = (x1+x2)//2, (y1+y2)//2
    tcx1, tcy1, tcx2, tcy2 = tc
    BORDER, CREASE = (140,140,180), (170,170,195)
    taper = int(min(pw,ph) * 0.08 * np.sin(np.pi * progress))

    def blit(fx1, fy1, fx2, fy2, rtx1, rty1, rtx2, rty2):
        frame[fy1:fy2, fx1:fx2] = tex[tcy1+rty1:tcy1+rty2, tcx1+rtx1:tcx1+rtx2]
        cv2.rectangle(frame, (fx1,fy1), (fx2,fy2), BORDER, 2)

    def warp(rtx1, rty1, rtx2, rty2, dst, ib):
        tw, th = rtx2-rtx1, rty2-rty1
        r = tex[tcy1+rty1:tcy1+rty2, tcx1+rtx1:tcx1+rtx2].copy()
        if ib:
            r = cv2.flip(r, 1)
            r = (r * 0.75).astype(np.uint8)
        M = cv2.getPerspectiveTransform(
            np.float32([[0,0],[tw,0],[tw,th],[0,th]]), np.float32(dst))
        w = cv2.warpPerspective(r, M, (frame.shape[1], frame.shape[0]))
        mk = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mk, [np.int32(dst)], 255)
        frame[mk>0] = w[mk>0]
        cv2.polylines(frame, [np.int32(dst)], True, BORDER, 2)

    def quad(pts, col):
        cv2.fillPoly(frame, [np.int32(pts)], col)
        cv2.polylines(frame, [np.int32(pts)], True, BORDER, 2)

    if tex is not None:
        if edge == 'top':
            fh_ = cy-y1; ty = int(y1+2*fh_*progress); ib = progress>0.5
            blit(x1,cy,x2,y2, 0,fh_,pw,ph)
            warp(0,0,pw,fh_, [[x1+taper,ty],[x2-taper,ty],[x2,cy],[x1,cy]], ib)
            cv2.line(frame, (x1,cy), (x2,cy), CREASE, 3)
        elif edge == 'bottom':
            fh_ = y2-cy; by = int(y2-2*fh_*progress); ib = progress>0.5
            blit(x1,y1,x2,cy, 0,0,pw,ph-fh_)
            warp(0,ph-fh_,pw,ph, [[x1,cy],[x2,cy],[x2-taper,by],[x1+taper,by]], ib)
            cv2.line(frame, (x1,cy), (x2,cy), CREASE, 3)
        elif edge == 'left':
            fw_ = cx-x1; lx = int(x1+2*fw_*progress); ib = progress>0.5
            blit(cx,y1,x2,y2, fw_,0,pw,ph)
            warp(0,0,fw_,ph, [[lx,y1+taper],[cx,y1],[cx,y2],[lx,y2-taper]], ib)
            cv2.line(frame, (cx,y1), (cx,y2), CREASE, 3)
        else:
            fw_ = x2-cx; rx = int(x2-2*fw_*progress); ib = progress>0.5
            blit(x1,y1,cx,y2, 0,0,pw-fw_,ph)
            warp(pw-fw_,0,pw,ph, [[cx,y1],[rx,y1+taper],[rx,y2-taper],[cx,y2]], ib)
            cv2.line(frame, (cx,y1), (cx,y2), CREASE, 3)
    else:
        PC, BC = (230,230,255), (195,205,225)
        if edge == 'top':
            fh_ = cy-y1; ty = int(y1+2*fh_*progress)
            quad([(x1,cy),(x2,cy),(x2,y2),(x1,y2)], PC)
            quad([(x1+taper,ty),(x2-taper,ty),(x2,cy),(x1,cy)], BC if progress>0.5 else PC)
            cv2.line(frame, (x1,cy), (x2,cy), CREASE, 3)
        elif edge == 'bottom':
            fh_ = y2-cy; by = int(y2-2*fh_*progress)
            quad([(x1,y1),(x2,y1),(x2,cy),(x1,cy)], PC)
            quad([(x1,cy),(x2,cy),(x2-taper,by),(x1+taper,by)], BC if progress>0.5 else PC)
            cv2.line(frame, (x1,cy), (x2,cy), CREASE, 3)
        elif edge == 'left':
            fw_ = cx-x1; lx = int(x1+2*fw_*progress)
            quad([(cx,y1),(x2,y1),(x2,y2),(cx,y2)], PC)
            quad([(lx,y1+taper),(cx,y1),(cx,y2),(lx,y2-taper)], BC if progress>0.5 else PC)
            cv2.line(frame, (cx,y1), (cx,y2), CREASE, 3)
        else:
            fw_ = x2-cx; rx = int(x2-2*fw_*progress)
            quad([(x1,y1),(cx,y1),(cx,y2),(x1,y2)], PC)
            quad([(cx,y1),(rx,y1+taper),(rx,y2-taper),(cx,y2)], BC if progress>0.5 else PC)
            cv2.line(frame, (cx,y1), (cx,y2), CREASE, 3)

# ---------------------------------------------------------------------------
# WebSocket endpoint — one connection per browser tab, holds all fold state
# ---------------------------------------------------------------------------
@app.websocket('/ws')
async def ws_origami(ws: WebSocket):
    await ws.accept()
    # All fold state lives here — isolated per connection
    s = dict(pr=None, ar=None, tr=None, tex=None,
             edge=None, prog=0.0, gpt=None, was=False, hist=[])
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ── keyboard command (c=clear, r=reset) ──────────────────────────
            if msg['type'] == 'cmd':
                if msg.get('key') in ('c', 'r') and s['pr']:
                    x1, y1, x2, y2 = s['pr']
                    s.update(ar=s['pr'], tr=(0,0,x2-x1,y2-y1),
                             edge=None, prog=0.0, gpt=None, was=False, hist=[])
                continue

            # ── decode incoming JPEG frame ────────────────────────────────────
            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(msg['data']), np.uint8),
                cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)          # mirror so it feels like a selfie-cam
            fh, fw = frame.shape[:2]

            # ── init state on first frame ─────────────────────────────────────
            if s['pr'] is None:
                pr = _paper_rect(fw, fh)
                s['pr'] = pr
                s['ar'] = pr
                x1, y1, x2, y2 = pr
                s['tr'] = (0, 0, x2-x1, y2-y1)
                src = cv2.imread(PAPER_PNG, cv2.IMREAD_COLOR)
                if src is not None:
                    s['tex'] = cv2.resize(src, (x2-x1, y2-y1))

            # ── hand detection (run on clean frame BEFORE paper overlay) ─────
            # Running MediaPipe after drawing the paper would hide the hand
            # underneath the paper texture, causing missed detections.
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            pinching = False
            lm_data  = None
            pp       = None

            if result.hand_landmarks:
                lm_data  = result.hand_landmarks[0]
                pinching = _is_pinching(lm_data)
                pp       = _pinch_pt(lm_data, fw, fh)

            # ── draw paper ───────────────────────────────────────────────────
            if s['edge'] is None:
                _draw_flat(frame, s['ar'], s['tex'], s['tr'])
            else:
                _draw_folded(frame, s['ar'], s['tr'], s['edge'], s['prog'], s['tex'])

            # ── draw finger tip dots on top of paper ─────────────────────────
            if lm_data is not None:
                col = (0,0,255) if pinching else (255,0,0)
                for idx in (4, 8):
                    l = lm_data[idx]
                    cv2.circle(frame, (int(l.x*fw), int(l.y*fh)), 10, col, -1)

                if pinching:
                    if not s['was']:
                        # pinch just started — begin a fold if on paper
                        ax1, ay1, ax2, ay2 = s['ar']
                        if ax1 <= pp[0] <= ax2 and ay1 <= pp[1] <= ay2 and s['edge'] is None:
                            s['gpt']  = pp
                            s['edge'] = _fold_edge(pp, s['ar'])
                            s['prog'] = 0.0
                    elif s['gpt'] and s['edge']:
                        # pinch held — update progress
                        s['prog'] = _fold_progress(s['gpt'], pp, s['ar'], s['edge'])
                else:
                    if s['was'] and s['edge']:
                        if s['prog'] > 0.3:
                            # commit fold — shrink active_rect and texture_rect to remaining half
                            ax1, ay1, ax2, ay2 = s['ar']
                            acx, acy = (ax1+ax2)//2, (ay1+ay2)//2
                            tx1, ty1, tx2, ty2 = s['tr']
                            tcx, tcy = (tx1+tx2)//2, (ty1+ty2)//2
                            e = s['edge']
                            if   e == 'top':    s['ar']=(ax1,acy,ax2,ay2);  s['tr']=(tx1,tcy,tx2,ty2)
                            elif e == 'bottom': s['ar']=(ax1,ay1,ax2,acy);  s['tr']=(tx1,ty1,tx2,tcy)
                            elif e == 'left':   s['ar']=(acx,ay1,ax2,ay2);  s['tr']=(tcx,ty1,tx2,ty2)
                            else:               s['ar']=(ax1,ay1,acx,ay2);  s['tr']=(tx1,ty1,tcx,ty2)
                            s['hist'].append(e)
                        s['edge'] = None
                        s['prog'] = 0.0
                    s['gpt'] = None

            s['was'] = pinching

            # ── encode and send processed frame back to browser ──────────────
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            await ws.send_text(json.dumps({
                'type':  'frame',
                'data':  base64.b64encode(buf).decode(),
                'folds': len(s['hist']),
            }))

    except WebSocketDisconnect:
        pass    # client closed the tab — nothing to clean up


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)

