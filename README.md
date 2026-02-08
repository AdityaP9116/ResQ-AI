# ResQ-AI: Autonomous Disaster Response Drone

**ResQ-AI** is an intelligent "Structural Scout" agent designed for Urban Search and Rescue (USAR). It leverages computer vision and AI to identify hazards, assess structural integrity, and map safe paths for rescue teams in real-time.

## 📅 Project Timeline & Status

### ✅ **Phase 1: Situational Awareness (Completed)**
*   **Goal**: Detect hazards (`Collapsed Building`, `Fire`, `Flood`, `Traffic Incident`) in aerial imagery.
*   **Model**: YOLOv8-Small (trained on AIDER dataset).
*   **Status**:
    *   [x] Dataset Setup & Validation (AIDER)
    *   [x] Training Pipeline Implementation (`train_yolo.py`)
    *   [x] Cloud Training Workflow (Google Colab due to local GPU incompatibility)
    *   [x] Model Export to TensorRT (`export_trt.py`)
    *   [x] Inference Validation on RTX 5070 (`test_inference.py`)

### 🚧 **Phase 2: Structural Segmentation (Upcoming)**
*   **Goal**: Pixel-level segmentation of rubble, roads, and safe paths.
*   **Model**: YOLOv8-Seg or SegFormer.
*   **Dataset**: RescueNet.

### 🔮 **Phase 3: Reasoning & Insight (Future)**
*   **Goal**: VQA for complex queries ("Is this wall safe?").
*   **Model**: Vision-Language Model (VLM).

---

## 📂 Repository Structure

*   `train_yolo.py`: Main training script (configured for GPU, falls back to CPU/Colab if needed).
*   `export_trt.py`: Converts trained `.pt` models to optimized TensorRT `.engine` format.
*   `test_inference.py`: Validates the TensorRT engine with real inference.
*   `run_on_colab.ipynb`: Jupyter Notebook for training heavily on free cloud GPUs.
*   `data_colab.yaml`: Dataset configuration for Colab.
*   `Datasets/`: Contains the AIDER dataset (git-ignored or tracked via DVC/LFS usually, here kept local).

## 🚀 How to Run

### 1. Training (Google Colab Recommended)
Due to PyTorch binary incompatibility with RTX 50-series (Blackwell), training is currently recommended on Colab:
1.  Upload `run_on_colab.ipynb` to Google Colab.
2.  Upload `dataset.zip` (zipped `Datasets` folder).
3.  Run all cells.
4.  Download `best.pt` weights.

### 2. Export & Inference (Local - RTX 5070)
Once you have the `best.pt` weights:
```bash
# Export to TensorRT Engine
python export_trt.py

# Run Inference Test
python test_inference.py
```
