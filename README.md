# Emo-Face-Rec-System

## Overview
**Emo-Face-Rec-System** is a robust, real-time computer vision system designed for simultaneous **Face Recognition**, **Emotion Analysis**, and **Image Quality Assessment** across multiple video streams.

Built with a producer-consumer multiprocessing architecture, it efficiently handles high-throughput video data from RTSP streams, performing face detection, recognition, and emotion analysis.

## Key Features
*   **Multi-Stream Support**: Concurrent processing of multiple RTSP video feeds using Python `multiprocessing`.
*   **Real-Time Face Recognition**: Utilizes **DeepFace** (ArcFace backbone) for face identification.
*   **Emotion Analysis**:
    *   Powered by **ResEmoteNet** (Bridging Accuracy and Loss Reduction in Facial Emotion Recognition).
    *   Trained on a custom **AffectNet** dataset (37,303 images).
    *   Analyzes batches of face crops to determine emotional trends over time.
*   **Quality Assessment**: Integrated **CR-FIQA** (Face Image Quality Assessment) to filter and select high-quality face samples.
*   **Non-Blocking Capture**: Custom threaded video capture implementation to minimize latency.
*   **Dynamic Person Management**: Automatic assignment of new faces to "Unknown" IDs with clustering-like logic.

## Architecture
The system follows a modular **Producer-Consumer** pattern:

1.  **Stream Producers (`multi_stream.py`)**:
    *   Connect to RTSP cameras.
    *   Detect faces using **MTCNN**.
    *   Push face crops to a shared file system queue.
2.  **Main Consumer (`main.py`)**:
    *   Monitors the file system for new face crops.
    *   Performs Face Recognition (ArcFace).
    *   Assigns faces to known Persons or creates new IDs.
    *   Manages "Outlier" detection to clean up noisy clusters.
3.  **Emotion Engine (`emo_rec.py`)**:
    *   Periodically scans person-specific folders.
    *   Runs the **ResEmoteNet** model on batches of faces.
    *   Logs detailed emotion probabilities to CSV.
4.  **Quality Module (`Quality Assessment/`)**:
    *   Runs independently to score and filter saved face images based on quality (CR-FIQA).

## Installation

### Prerequisites
*   Windows OS (tested on Windows 11)
*   Anaconda or Miniconda
*   NVIDIA GPU (Recommended for CUDA acceleration)

### Environment Setup
The system relies on specific Conda environments. (See `run_both_scripts.bat` for activation logic).

#### Option 1: Using Conda (Recommended)
This method ensures all system dependencies are handled automatically.

```bash
conda env create -f environment.yml
conda activate deepface_emotion_env
```

#### Option 2: Using Pip
If you prefer a manual setup or are not using Conda:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** The `requirements.txt` includes CPU versions of PyTorch by default. If you have an NVIDIA GPU, please install the CUDA-enabled version of PyTorch manually *after* running the above command.

### Quality Assessment Environment Setup
The Quality Assessment module requires a separate environment with specific dependencies.

#### Option 1: Using Conda (Recommended)
```bash
cd "Quality Assessment"
conda env create -f environment.yml
conda activate FIQ_gpu_env
```

#### Option 2: Using Pip
```bash
cd "Quality Assessment"
# Create a virtual environment (optional but recommended)
python -m venv venv_quality
# Windows:
venv_quality\Scripts\activate
# Linux/Mac:
source venv_quality/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Pretrained Models

This project relies on two pretrained models: an **Emotion Recognition model (ResEmoteNet)** and a **Face Image Quality Assessment model (CR-FIQA)**.

### Emotion Recognition Model (ResEmoteNet)
The emotion recognition module uses a pretrained **ResEmoteNet** model trained on a custom subset of the AffectNet dataset.

Due to file size limitations, the model weights are **not stored in this repository**.

**Download links:**
- **best_model.pth**: [Google Drive – best_model.pth](https://drive.google.com/file/d/1rp3t7fmjASaeLcwauCrJZhzW6eAzT1vn/view?usp=drive_link)
- **finalized_model.sav**: [Google Drive – finalized_model.sav](https://drive.google.com/file/d/1fArws_LxbN1GWU1NY4pSTXmtFkuhIueJ/view?usp=drive_link)

After downloading, place both files in the main project directory:

```
Emo-Face-Rec-System/
├── best_model.pth
└── finalized_model.sav
```

These files are required for running the emotion recognition pipeline (`emo_rec.py`).

---

### Face Image Quality Assessment Model (CR-FIQA)
The quality assessment module is based on **CR-FIQA** (CVPR 2023).

The pretrained CR-FIQA models are **not redistributed** in this repository. Please follow the **official CR-FIQA repository** for model download and installation instructions:

https://github.com/fdbtrs/CR-FIQA

After completing the CR-FIQA setup, ensure the pretrained models are located at the following paths:

```
Quality Assessment/
├── insightface/model/insightface-0000.params
└── CR-FIQAL/181952backbone.pth
```

These paths are required by the quality assessment scripts (e.g. `fixed_quality_score.py`, `run_quality.py`).

## Usage

### Running the System
The easiest way to start the entire system is using the provided batch script:

```bat
run_both_scripts.bat
```

This script will:
1.  Activate the necessary Conda environments.
2.  Launch `run_main.py` (Main Face & Emotion System).
3.  Launch `run_quality.py` (Quality Assessment Module).

### Configuration
*   **RTSP Streams**: Edit `main.py` (variable `streams`) to add or remove camera or stream URLs.
*   **Person Naming**: Edit `emo_rec.py` (dictionary `people_name`) to map Person IDs (e.g., `person_0`) to real names for CSV reports.

## Directory Structure
*   `main.py`: Core logic and orchestration.
*   `multi_stream.py`: Multiprocessing stream handling.
*   `emo_rec.py`: Emotion recognition logic.
*   `stream_handler.py`: Threaded video capture class.
*   `Quality Assessment/`: Sub-module for face quality scoring.
*   `Persons_Faces/`: Storage for recognized face crops (auto-generated).

## Acknowledgments

This project utilizes the following models and libraries:

### Emotion Recognition Model - ResEmoteNet
This project uses the ResEmoteNet model for emotion recognition:

```bibtex
@misc{nguyen2024resemotenet,
      title={ResEmoteNet: Bridging Accuracy and Loss Reduction in Facial Emotion Recognition},
      author={Thien Nguyen Phu and Nhu Tai Do and Huy Hoang Nguyen and Hyung-Jeong Yang and Guee-Sang Lee and Soo-Hyung Kim},
      year={2024},
      eprint={2409.10545},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.10545}
}
```

### Face Recognition - DeepFace
This project uses the DeepFace library for face recognition:

```bibtex
@article{serengil2024lightface,
  title     = {A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules},
  author    = {Serengil, Sefik and Ozpinar, Alper},
  journal   = {Journal of Information Technologies},
  volume    = {17},
  number    = {2},
  pages     = {95-107},
  year      = {2024},
  doi       = {10.17671/gazibtd.1399077},
  url       = {https://dergipark.org.tr/en/pub/gazibtd/issue/84331/1399077},
  publisher = {Gazi University}
}
```

```bibtex
@inproceedings{serengil2021hyperextended,
  title     = {HyperExtended LightFace: A Facial Attribute Analysis Framework},
  author    = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle = {2021 International Conference on Engineering and Emerging Technologies (ICEET)},
  pages     = {1-4},
  year      = {2021},
  organization={IEEE}
}
```

```bibtex
@inproceedings{serengil2020lightface,
  title     = {LightFace: A Hybrid Deep Face Recognition Framework},
  author    = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle = {2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages     = {23-27},
  year      = {2020},
  doi       = {10.1109/ASYU50717.2020.9259802},
  organization={IEEE}
}
```

### Quality Assessment - CR-FIQA
This project uses the CR-FIQA model for face image quality assessment:

```bibtex
@InProceedings{Boutros_2023_CVPR,
    author    = {Boutros, Fadi and Fang, Meiling and Klemt, Marcel and Fu, Biying and Damer, Naser},
    title     = {CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {5836-5845}
}
```

