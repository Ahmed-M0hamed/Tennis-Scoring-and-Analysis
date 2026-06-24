# 🚀 Tennis Scoring & Analysis System 🎾

An end-to-end **AI-powered tennis analysis system** capable of automatically analyzing tennis matches, tracking players and the ball, detecting game events, and generating scoring insights.

---

## 🎥 Demo

> The output video can be found in the `preds/` folder.

### Demo Video


<!-- <video src="preds/output.mp4" controls width="100%"></video> -->
https://github.com/Ahmed-M0hamed/Tennis-Scoring-and-Analysis/blob/main/preds/output.mp4


If the video preview is not supported by GitHub, you can download and watch it directly from the `preds` folder.

---

## Models Checkpoints 
https://drive.google.com/drive/folders/1X5CKHkweX5iXwSeX03U3k1Lpcs8QXXm1?usp=sharing

## 🔍 Project Highlights

### 📊 Dataset & Modeling

- Collected and used annotated datasets from **Roboflow**.
- Trained **three different YOLO models** using **Ultralytics** for:
  - 🎾 Tennis ball detection
  - 🏃 Player and racket detection
  - 🏟️ Tennis court keypoint detection

---

### 🎯 Detection Filtering

Developed custom Python pipelines to filter detections and reliably identify:

- The two players
- Their rackets
- The tennis ball
- Court keypoints

All detections are visualized in real time using **OpenCV**.

---

### 🔧 Handling Missing Detections

To improve tracking robustness:

- Applied **interpolation techniques** to recover missing ball detections.
- Used **Savitzky–Golay smoothing** to reduce noise and obtain smoother trajectories.

---

### 🏸 Ball Bounce & Racket Hit Detection

To accurately detect game events, multiple approaches were combined:

- Analyzed horizontal and vertical changes in ball trajectories.
- Used signal analysis techniques to detect peaks and valleys.
- Leveraged **Pandas-based filtering logic** to distinguish between:
  - Ball bounces on the court
  - Racket hits

---

### 🏟️ Homography & Court Analytics

Built multiple homography-based court representations using **OpenCV**:

- A **mini-court view** to transform the broadcast perspective.
- Ball and player footprint mapping on the court.
- Player movement **heatmaps** for match analysis.

---

### 📈 Player Analytics

Generated real-time player statistics including:

- Distance covered
- Player speed
- Real-time player information cards

---

### 🏆 Automated Scoring System

Implemented rule-based scoring logic to:

- Detect whether a serve lands inside the correct service box.
- Determine whether subsequent shots are **IN** or **OUT**.
- Display automatic line calls during rallies.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- Roboflow
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn

---

## 🧠 Key Concepts

- Computer Vision
- Object Detection
- Multi-Object Tracking
- Homography Transformation
- Sports Analytics
- Signal Processing
- Event Detection

---

## 📂 Repository Structure

```text
├── models/
├── datasets/
├── preds/           # Output videos and predictions
├── notebooks/
├── utils/
├── main.py
└── README.md
```

---

## 📌 Future Improvements

- Player identification and tracking across the entire match.
- Shot classification (forehand, backhand, serve, volley).
- Automatic scoreboard generation.
- Rally statistics and advanced match analytics.

---

## ⭐ If you found this project interesting, consider giving it a star!
