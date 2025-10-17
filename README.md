# KITTI 3D Object Detection Pipeline

This repository provides a full LiDAR + RGB processing pipeline for the KITTI dataset, including 2D detection loading, 3D projection, frustum culling, clustering, and 3D bounding box estimation.

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone git@github.com:Cgcollins91/ENPM818Z_FALL_2025_RWA_1.git
cd ENPM818Z_FALL_2025_RWA_1/
```

### 2. Create Conda Environment

This project includes an `environment.yml` for reproducible setup.

```bash
conda env create -f environment.yml
conda activate carla
```

---

## Dataset Setup

Download the KITTI **object detection dataset**:

* Images (`image_2/`)
* Velodyne LiDAR scans (`velodyne/`)
* Calibration files (`calib/`)
* Labels (`label_2/`)

Set your dataset path like this:

```
ENPM818Z_FALL_2025_RWA_1/
├── training/
    ├── image_2/
    ├── velodyne/
    ├── calib/
    └── label_2/
```
---

## Run Pipeline

This runs the full pipeline for a specific KITTI frame using only one line:

```bash
python pipeline.py --idx 000123
```

Where:

* `--idx` = KITTI frame index (zero-padded to 6 digits(0-200))
---

## Project Structure

```
ENPM818Z_FALL_2025_RWA_1/
├── pipeline.py                 # Entry script – runs full pipeline
├── starter.py                  # KITTI loaders for image + lidar
├── detector.py                 # KITTI label loading
├── environment.yml             # Conda setup
└── README.md                   # Documentation
```

---

## Major Script Descriptions

| Script        | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `pipeline.py`     | Orchestrates loading, projection, frustum filtering, visualization |
| `starter.py`  | Contains `load_kitti_image()` and `load_kitti_lidar_scan()`        |
| `detector.py` | Loads KITTI 2D label files from `label_2`                         |
---

## Output

Running the pipeline produces:

* RGB + LiDAR projection
* Depth visualization
* 2D detection overlays
* 3D LiDAR clusters
* Axis-aligned / oriented bounding boxes

---

## Dependencies

* Python 3.9+
* NumPy
* OpenCV
* Open3D
* Matplotlib
---
