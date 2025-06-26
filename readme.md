# 🏃‍♂️ Player Tracking using YOLO, Feature Embeddings, and OCR

This project performs player detection and tracking in sports videos using:

- **YOLOv11** for person detection
- **Deep CNN embeddings** and **HSV color histograms** for appearance-based tracking
- **OCR (EasyOCR)** for reading jersey numbers
- **Cosine similarity** for player matching over frames

---

## 📦 Requirements

Install required dependencies:

```bash
pip install -U ultralytics easyocr opencv-python-headless torch torchvision scikit-learn
apt-get update && apt-get install -y libgl1-mesa-glx
```

```bash
├── player_tracking.py            # Main tracking script
├── best.pt                       # YOLOv11 model trained to detect players (Link : "https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view")
├── 15sec_input_720p.mp4               # Input sports video
├── tracked_players_output.mp4    # Output video with tracked players
└── README.md                     # Project documentation
```

## ▶️ How to Run

1. **Update the paths in the `main()` function** inside `player_tracking.py`:

```python
model_path = "path/to/your/best.pt"
video_path = "path/to/your/input_video.mp4"
output_path = "tracked_players_output.mp4"
```

2. **Run the script:**

```bash
python player_tracking.py
```

## 🧠 How It Works

### 🚀 Pipeline Overview

1. **Detection**
   - Uses YOLOv11 to detect players in each video frame.

2. **Feature Extraction**
   - Extracts two types of features from each detected player:
     - **Color Histogram** (HSV)
     - **Deep Features** using a custom CNN

3. **Jersey Number OCR**
   - Uses EasyOCR to read 1–2 digit jersey numbers from the player patch.

4. **Tracking**
   - Matches new detections to existing players using:
     - Cosine similarity of features
     - Position proximity
     - Bonus score if jersey number matches

5. **Visualization**
   - Draws bounding boxes with player ID and jersey number.
   - Outputs processed frames to a new video file.

---

## 🔍 Matching Thresholds & Criteria

| Criterion               | Description                                              |
|------------------------|----------------------------------------------------------|
| **Detection Confidence** | YOLO detections must have confidence > `0.4`            |
| **Minimum Size**        | Detections smaller than 30×50 pixels are ignored         |
| **Matching Threshold**  | Cosine similarity must exceed `0.65` to consider match   |
| **Position Proximity**  | Players must be within 250px of previous position        |
| **Jersey Bonus**        | +0.15 added to similarity score if jersey number matches |

---

## ✅ Pros

- Combines **visual features** and **textual OCR** for robust tracking
- No need for complex motion prediction algorithms
- Works across multiple sports (e.g., football, basketball)
- Compatible with custom-trained or official YOLOv11 models

---

## ⚠️ Limitations

- **OCR Inaccuracy**  
  - OCR may fail under motion blur, occlusion, or low resolution

- **Appearance Confusion**  
  - Players with similar kits and nearby positions might get misidentified
  - A crowd of Players seems confusing to the identification and thus misidentification


## 📷 Screenshots