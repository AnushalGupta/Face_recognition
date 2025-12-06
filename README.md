# Face_Recognition: Lightweight AI Access Control

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![AI Engine](https://img.shields.io/badge/InsightFace-Buffalo__S-green?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Edge-orange?style=for-the-badge)

**Face_Recognition** is a high-performance, real-time access control system optimized for low-end hardware, turnstiles, offices, and CCTV setups.

It uses a **Motion-Triggered Architecture** and the **Buffalo_S** Edge AI model to deliver **high FPS face recognition** on inexpensive devices such as Intel NUCs and Raspberry Pi 4, without requiring server-grade GPUs.

---

## 🚀 Key Features

- **⚡ Motion-Activated AI**  
  Inference sleeps at near 0% CPU until movement is detected.

- **📱 Edge Optimized Model**  
  Uses `buffalo_s` with dynamic downscaling for **30+ FPS** processing.

- **🖥️ Modern GUI Dashboard**  
  Beautiful dark UI built using **CustomTkinter**, featuring sidebar navigation, indicators, and live camera feed.

- **📝 One-Click User Registration**  
  Add new users instantly through the GUI while the camera keeps running.

- **📊 Automated Logging**  
  Every access attempt (Name, Timestamp, Confidence, Status) is saved to `access_log.csv`.

---

## 🛠️ Requirements

This project is built specifically for **Python 3.12**.

> ⚠️ **Important:**  
> This project requires **`numpy < 2.0`**.  
> Numpy 2.0+ breaks compatibility with current ONNX + InsightFace builds.

---

## 📦 Dependencies (`core_requirements.txt`)

```text
customtkinter==5.2.2
opencv-python==4.10.0.84
onnxruntime==1.18.0
pillow==10.3.0
scikit-learn==1.5.0
numpy<2.0
```

---

## 📥 Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Face_Recognition.git
cd Face_Recognition
```

---

### 2️⃣ Install InsightFace (Windows Users – Critical!)

Installing InsightFace through pip on Windows requires **6GB+ C++ Build Tools**, which most users do not have.

To avoid this:

1. Download the pre-compiled wheel:  
   **insightface-0.7.3-cp312-cp312-win_amd64.whl**
2. Place it inside your project folder.
3. Install it manually:

```bash
pip install insightface-0.7.3-cp312-cp312-win_amd64.whl
```

---

### 3️⃣ Install Remaining Dependencies

```bash
pip install -r core_requirements.txt
```

---

## ▶️ Usage

### Start the System

```bash
python main_app.py
```

---

### 📌 Status Indicators

| Indicator | Meaning |
|----------|----------|
| **STATUS: Standby** | No motion detected → AI sleeping |
| **STATUS: Scanning…** | Motion detected → AI active |
| **ACCESS: [Name]** | User recognized → Access granted |

---

### 📝 Register a New User

1. Click **“Add New User”** in the sidebar.  
2. Enter the user’s name.  
3. The system captures a snapshot from the live feed and saves it automatically.

---

### 📊 Access Logs

Check:

```
access_log.csv
```

This includes timestamps, names, confidence scores, and access decisions.

---

## ⚙️ Configuration (Top of `main_app.py`)

```python
FRAME_SKIP = 5           # Increase to reduce CPU usage
RESIZE_FACTOR = 0.5      # Lower values = Faster inference
MOTION_THRESHOLD = 1000  # Sensitivity for motion detection
```

---

## 🧪 Experimental & Advanced Features

### 👁️ Liveness Detection (Anti-Spoofing)

Prevents unlocking with a printed photo or mobile screen.

To enable:

1. Install MediaPipe:  
   ```bash
   pip install mediapipe
   ```
2. Enable the `check_liveness()` function in **SecurityEngine**.

Uses **EAR (Eye Aspect Ratio)** detection to ensure blinking.

---

## 📦 Packaging as a Portable `.exe`

You can create a standalone executable:

```bash
pyi-makespec --onefile --windowed --name="Face_Recognition" --add-data "faces;faces" --collect-all insightface main_app.py
pyinstaller Face_Recognition.spec
```

---

## 🤝 Contributing

Contributions are welcome!  
Fork the repo → push changes → open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**.  
See the `LICENSE` file for details.
