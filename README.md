# Robotic Foosball – Autonomous Foosball Opponent

This repository contains the full implementation of an **AI-controlled robotic foosball opponent**, integrating real-time computer vision, embedded controls, physics-based game logic, and custom mechanical actuation.

The goal: **detect a fast-moving ball, predict its path, and actuate robotic rods to intercept and shoot — all autonomously.**

---

## 🏗 System Overview
The project is built around four coordinated subsystems:

1. **Networking & Infrastructure**
2. **Computer Vision**
3. **Motor Control & Game Logic**
4. **Mechanical / Electrical Design**

These run across multiple devices (Jetson TX2 + Raspberry Pi 5) with strict latency and real-time constraints.'

### Architecture Diagram

<img width="514" height="366" alt="image" src="https://github.com/user-attachments/assets/d7d4823d-ddf0-40ea-a64b-5f4248f155a2" />

---

## 📡 Networking & Infrastructure
**Purpose:** stream high-resolution video in real time and isolate dependencies.

- **Camera:** Raspberry Pi Camera Module v3 → higher bandwidth, lower latency, better optics than USB
- **Compute:** NVIDIA Jetson TX2 for GPU inference
- **Streaming:** GStreamer pipelines over **UDP** to minimize buffering
- **Optimizations:**
  - Hardware H.264 decoding on Jetson
  - Static IP addressing
  - <1 ms IPC using **ZeroMQ**
- **Containerization:** Python 3.6 (CUDA) and Python 3.8 (motor control) split via **Docker**

Why this architecture?
- Jetson requires outdated CUDA/Python libraries
- Motor drivers require newer Python
- Real-time constraints required low overhead and ruled out heavier IPC method

---

## 🎯 Computer Vision (Ball Detection & Localization)
**Purpose:** locate the foosball ball fast enough for real-time actuation.

Pipeline:
1. Split 2304×1296 frames into **16×16 = 256 tiles**
2. **MobileNetV3-Small classifier** finds the tile containing the ball
3. A second **MobileNetV3-Small localizer** outputs the (x, y) inside that tile

Why this approach?
- Shrinks search space and allows for parrellelizing the task (avoids scanning the entire image)
- Smaller input -> Faster inference
- MobileNetV3-Small (~2.5M params) runs ~15 ms inference

Dataset:
- ~6,000 labeled frames from 4 minutes of gameplay
- Custom-built OpenCV labelling GUI
- Pixel & spatial data augmentation to diversify dataset, and help the model handle variance in noise and lighting
- Classifier achieved **99% AUC**
- Localization MSE ≈ **11.2 px**

This balances **speed and accuracy under hardware limits**.

---

## 🧠 Game Logic & Trajectory
**Purpose:** convert ball predictions into lateral positioning and kicks.

Steps:
1. Receive two sequential (x, y) coords → compute velocity
2. Determine zone-based intercept direction
3. Divide the table into **3 vertical defense zones**
4. Trigger multiple rods (multithreaded defense) based on the ball's predicted intercept
5. Subdivide vertically into **3 player zones per rod**
6. Move the rod laterally to match ball position
7. Issue a “kick” once aligned

Why this works:
- No full physics simulation required
- Fast and deterministic
- Handles most gameplay without extra sensors
- Inference happens fast enough to correct for non-linear ball movement

Known gaps:
- Zero-velocity cases not handled
- Ball directly under rod is not treated as “shootable”
- Fixed-timing servos limit precision

---

## ⚙️ Mechanical & Electrical Design
**Purpose:** physically move rods quickly across the table.

Hardware:
- 3 motorized opponent robotic rods 
- **Rack-and-pinion** for lateral travel
- **Gearbox flipper** for rapid kick rotation
- PLA-printed parts sized around printer constraints
- Bearings to reduce friction
- FT90R + SG90 servos, controlled over **PCA9685 (I²C)**

Why rack-and-pinion?
- Compact
- Easy gearing
- Easily 3D printable

Mechanical refinements:
- Added stabilizer linkage to stop wobble
- Rebuilt joints after breakage
- Converted circular male/female joints → square alignment pegs

---

## 🚧 Next Steps & Recommendations
### Computer Vision
- Replace TX2 with modern GPU
- Remove network streaming by attaching camera directly
- Export to **TensorRT**, prune/quantize models
- Collect more data for edge-case lighting and fast motion

### Game Logic
- Handle stationary ball logic
- Implement defensive memory/strategy
- Support multi-rod predictive movement

### Mechanical
- Replace continuous-rotation SG90s → **steppers or high-speed PWM servos**
- Add limit switches for calibration
- Replace PVC frame with rigid aluminum or steel
- Increase torque & durability

### System Architecture
- Remove Python version split
- Collapse into a single high-performance environment
