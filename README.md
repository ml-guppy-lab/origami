# 🦢 AR Origami — Fold Paper with Your Bare Hands

> A real-time augmented reality origami simulator powered by **MediaPipe Hand Landmarker** and **OpenCV**.  
> No controllers. No buttons. Just your hands and a pinch. ✨

---

## 📸 Screenshots

| Flat Paper | Edge Fold | Corner Dog-ear |
|:---:|:---:|:---:|
| ![screenshot 1](screenshots/screenshot1.png) | ![screenshot 2](screenshots/screenshot2.png) | ![screenshot 3](screenshots/screenshot3.png) |

*(Add your screenshots to a `screenshots/` folder)*

---

## 🎮 How to Play

| Gesture | Action |
|---|---|
| 👌 **Pinch** on the centre of an edge | Fold that half over |
| 👌 **Pinch** near a corner | Make a diagonal dog-ear fold |
| **Drag** while pinching | Control how far the fold goes |
| **Release** past 30 % | Fold commits and stays |
| **Release** early | Snaps back flat |
| ⌨️ **`c`** | Clear / unfold everything |
| ⌨️ **`r`** | Reset paper |
| ⌨️ **`q`** | Quit |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ml-guppy-lab/origami.git
cd origami

# 2. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install -r requirements.txt

# 4. Run!
.venv/bin/python3 scripts/origamiSquare.py      # square paper
.venv/bin/python3 scripts/origamiRectangle.py   # vertical rectangle paper
```

> **Note:** The hand landmark model (`hand_landmarker.task`) is downloaded automatically on first run.

---

## 🗂 Project Structure

```
origami/
├── scripts/
│   ├── origamiSquare.py       # Square paper with edge + corner folds
│   ├── origamiRectangle.py    # Vertical rectangle variant
│   └── handlandmarker.py      # Original prototype
├── paper.png                  # Paper texture
├── hand_landmarker.task        # MediaPipe model (auto-downloaded)
└── requirements.txt
```

---

## 🛠 Tech Stack

- **[MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)** — 21-point hand tracking
- **OpenCV** — video capture, image processing, fold rendering
- **NumPy** — geometry & perspective transforms
- **Python 3.12**

---

## ✨ Features

- 📄 Real paper texture preserved through every fold
- 🔀 Unlimited sequential edge folds (top / bottom / left / right)
- 📐 Diagonal dog-ear corner folds
- 🎥 Live fold animation with foreshortening taper
- 🔙 Back face rendering (mirrored, slightly darker)
- ♻️ One-key reset

---

## 📲 Follow Along

I post all my project demos on Instagram — come say hi! 👋

[![Instagram](https://img.shields.io/badge/@TheMLGuppy-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/themlguppy/)

---

## 📄 License

[MIT](LICENSE)
