Absolutely. Here is the **final README.md in clear, professional English**, ready to copy directly into your GitHub repository.

````markdown
# Head and Neck Organ Segmentation using Hybrid U-Net and Transformer

<p align="center">
  <b>AI-Based Multi-Organ Segmentation of Head and Neck CT Scans</b>
</p>

<p align="center">
  A deep learning-based system for automatic segmentation and visualization of Head and Neck Organs-at-Risk (OARs) from CT scans using a Hybrid U-Net and Transformer architecture.
</p>

---

## 📌 Overview

**Head and Neck Organ Segmentation** is a deep learning-based medical image segmentation project designed to automatically identify and segment multiple **Organs-at-Risk (OARs)** from Head and Neck CT scans.

The proposed approach combines a **U-Net-based Convolutional Neural Network (CNN)** with a **Transformer Encoder**. The CNN encoder extracts hierarchical spatial features from CT images, while the Transformer captures long-range contextual relationships within the learned feature representation. The decoder then reconstructs the segmentation map using skip connections.

The project also provides an interactive web application that allows users to:

- Upload CT scans in NRRD format
- Navigate through CT slices
- Perform AI-based multi-organ segmentation
- Visualize segmentation masks
- View organ boundaries and overlays
- Analyze detected organs
- View segmentation performance metrics
- Download segmentation results
- Generate segmentation reports

---

## 🎯 Objectives

The main objectives of this project are:

- Develop an automated multi-organ segmentation system for Head and Neck CT scans.
- Combine U-Net and Transformer architectures for spatial and contextual feature learning.
- Segment multiple Head and Neck Organs-at-Risk (OARs).
- Build an interactive web application for CT scan analysis.
- Visualize predicted segmentation masks and organ boundaries.
- Evaluate the model using standard segmentation metrics.
- Provide organ-wise segmentation analysis.
- Provide downloadable segmentation results and reports.

---

## 🧠 Proposed Method

The proposed system uses a **Hybrid U-Net + Transformer** architecture.

The U-Net component performs hierarchical spatial feature extraction and reconstruction, while the Transformer module is integrated into the bottleneck to capture long-range dependencies and global contextual information.

### Overall Pipeline

```text
                         CT Scan
                            │
                            ▼
                   Data Preprocessing
                            │
                            ▼
                  Volume Normalization
                            │
                            ▼
                    Slice Extraction
                            │
                            ▼
                    Resize to 256×256
                            │
                            ▼
                   ┌────────────────┐
                   │  CNN Encoder   │
                   └────────────────┘
                            │
                            ▼
                       Bottleneck
                            │
                            ▼
                   ┌────────────────┐
                   │   Transformer  │
                   │     Encoder    │
                   └────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  CNN Decoder   │
                   │ + Skip Connections
                   └────────────────┘
                            │
                            ▼
                    Segmentation Head
                            │
                            ▼
                     31-Class Output
                            │
                            ▼
                  Multi-Organ Segmentation
                            │
                            ▼
                 Visualization & Analysis
````

---

# 🏗️ Model Architecture

The Hybrid U-Net + Transformer model consists of an encoder, bottleneck, Transformer module, and decoder.

## Encoder

The encoder extracts hierarchical features from the input CT slice.

The channel progression is:

```text
Input
1 Channel
   ↓
64 Channels
   ↓
128 Channels
   ↓
256 Channels
   ↓
512 Channels
```

The encoder progressively learns higher-level representations while reducing the spatial representation.

---

## Bottleneck

At the deepest part of the network, the feature representation is expanded:

```text
512 → 1024
```

A `1×1` convolution projects the representation:

```text
1024 → 512
```

This 512-dimensional representation is passed to the Transformer.

---

## Transformer Encoder

The Transformer module captures long-range relationships between spatial features.

### Transformer Configuration

| Parameter           |             Value |
| ------------------- | ----------------: |
| Embedding Dimension |               512 |
| Attention Heads     |                 8 |
| Transformer Layers  |                 2 |
| Activation          |              GELU |
| Batch First         |              True |
| Normalization       | Pre-Normalization |

After Transformer processing, the representation is projected back:

```text
512 → 1024
```

and passed to the decoder.

---

## Decoder

The decoder reconstructs the spatial resolution of the segmentation map.

It uses:

* Transposed convolutions
* Skip connections
* Decoder blocks

Skip connections transfer important spatial information from the encoder to the decoder.

The final segmentation head produces:

```text
64 → 31
```

Therefore, the model generates a **31-class segmentation output**.

---

# 🧬 Segmentation Classes

The model performs multi-class segmentation with:

**30 Organs-at-Risk + 1 Background = 31 Classes**

| Class ID | Organ           |
| -------: | --------------- |
|        0 | Background      |
|        1 | A_Carotid_L     |
|        2 | A_Carotid_R     |
|        3 | Arytenoid       |
|        4 | Bone_Mandible   |
|        5 | Brainstem       |
|        6 | BuccalMucosa    |
|        7 | Cavity_Oral     |
|        8 | Cochlea_L       |
|        9 | Cochlea_R       |
|       10 | Cricopharyngeus |
|       11 | Esophagus_S     |
|       12 | Eye_AL          |
|       13 | Eye_AR          |
|       14 | Eye_PL          |
|       15 | Eye_PR          |
|       16 | Glnd_Lacrimal_L |
|       17 | Glnd_Lacrimal_R |
|       18 | Glnd_Submand_L  |
|       19 | Glnd_Submand_R  |
|       20 | Glnd_Thyroid    |
|       21 | Glottis         |
|       22 | Larynx_SG       |
|       23 | Lips            |
|       24 | OpticChiasm     |
|       25 | OpticNrv_L      |
|       26 | OpticNrv_R      |
|       27 | Parotid_L       |
|       28 | Parotid_R       |
|       29 | Pituitary       |
|       30 | SpinalCord      |

---

# 🏥 Dataset

The project uses the **HaN-Seg (Head and Neck Segmentation)** dataset.

The dataset contains:

* 42 3D CT cases
* 30 Organs-at-Risk
* CT volumes
* Corresponding segmentation annotations

The dataset is divided at the **case level** to prevent slices from the same case from appearing across different dataset splits.

### Dataset Split

| Dataset    | Number of Cases |
| ---------- | --------------: |
| Training   |              29 |
| Validation |               6 |
| Testing    |               7 |
| **Total**  |          **42** |

A fixed random seed is used for reproducible dataset splitting.

---

# ⚙️ Data Preprocessing

The CT volumes undergo preprocessing before being provided to the model.

### Preprocessing Pipeline

```text
Raw CT Volume
      ↓
Dataset Validation
      ↓
Resampling
      ↓
Image Resizing
      ↓
Volume Normalization
      ↓
Slice Extraction
      ↓
256 × 256 CT Slice
      ↓
Model Input
```

### Target Voxel Spacing

```text
1.0 × 1.0 × 2.0 mm
```

### Model Input Size

Each CT slice is resized to:

```text
256 × 256 pixels
```

The model receives an input tensor of:

```text
[Batch, Channel, Height, Width]

[1, 1, 256, 256]
```

---

# 🔬 Inference Pipeline

The complete inference workflow is:

```text
CT NRRD Upload
      ↓
Read 3D CT Volume
      ↓
Volume Normalization
      ↓
Select CT Slice
      ↓
Resize to 256 × 256
      ↓
Convert to PyTorch Tensor
      ↓
Hybrid U-Net + Transformer
      ↓
31-Class Prediction
      ↓
Segmentation Mask
      ↓
Overlay + Boundaries
      ↓
Organ Detection
      ↓
Organ-Wise Analysis
      ↓
Performance Metrics
      ↓
Download / Report
```

---

# 📊 Evaluation Metrics

The model is evaluated using standard semantic segmentation metrics.

### Dice Similarity Coefficient

Measures the overlap between the predicted segmentation and the ground-truth segmentation.

### Intersection over Union (IoU)

Measures the intersection between the prediction and ground truth relative to their union.

### Precision

Measures the proportion of predicted positive pixels that are actually correct.

### Recall

Measures the proportion of actual positive pixels correctly detected by the model.

### Pixel Accuracy

Measures the percentage of correctly classified pixels.

---

# 📈 Model Performance

The trained Hybrid U-Net + Transformer model was evaluated on:

**1,040 test slices**

### Overall Test Results

| Metric         |     Result |
| -------------- | ---------: |
| Pixel Accuracy | **99.79%** |
| Mean Dice      | **26.90%** |
| Mean IoU       | **19.59%** |
| Mean Precision | **30.69%** |
| Mean Recall    | **29.86%** |

> **Note:** Pixel accuracy can be strongly influenced by the large background region in medical image segmentation. Therefore, Dice and IoU are also important when evaluating actual segmentation overlap.

---

## 🧬 Selected Organ Results

| Organ         | Dice Score |
| ------------- | ---------: |
| Bone_Mandible | **81.90%** |
| Cavity_Oral   | **78.42%** |
| SpinalCord    | **69.38%** |
| Parotid_R     | **65.41%** |
| Brainstem     | **62.46%** |
| Parotid_L     | **61.28%** |

Performance varies across organs because different structures have different sizes, shapes, and visual characteristics.

---

# 💻 Web Application

The project includes an interactive web application for CT scan segmentation and visualization.

## Key Features

### 1. CT Scan Upload

Users can upload CT volumes in **NRRD format**.

### 2. CT Slice Viewer

Users can navigate through the slices of the uploaded CT volume.

### 3. AI Segmentation

The trained Hybrid U-Net + Transformer model performs multi-organ segmentation.

### 4. Segmentation Visualization

The application displays:

* Original CT
* Segmentation Mask
* Overlay
* Organ Boundaries

### 5. Multi-Organ Detection

The application provides information about detected organs, including:

* Organ name
* Class ID
* Location
* Pixel area
* Segmentation boundary

### 6. Performance Dashboard

The application displays:

* Dice
* IoU
* Precision
* Recall
* Pixel Accuracy

### 7. Organ-Wise Analysis

The application provides analysis of detected organs and their segmented areas.

### 8. Result Downloads

Users can download generated segmentation results and analysis data.

### 9. Automatic Reporting

The application provides segmentation information and analysis in a report format.

### 10. Slice Navigation

Users can navigate through the CT volume using the slice navigation controls.

---

# 🖼️ Application Output

The application provides three primary visualization views:

```text
┌───────────────────┐
│    Original CT    │
└───────────────────┘

          +

┌───────────────────┐
│ Segmentation Mask │
└───────────────────┘

          +

┌───────────────────┐
│ Overlay +         │
│ Organ Boundaries  │
└───────────────────┘
```

The overlay combines the predicted segmentation with the original CT slice to make the location of segmented organs easier to visualize.

---

# 🧪 Example Test Case

An example CT volume used during application testing:

```text
CT Volume:
1024 × 1024 × 136

Selected Slice:
68 / 136

Detected Organs:
9
```

### Example Slice-Level Metrics

```text
Dice       : 20.97%
IoU        : 13.81%
Precision  : 16.59%
Recall     : 52.51%
Accuracy   : 96.15%
```

> These values represent an individual selected slice and should not be confused with the overall test-set results.

---

# 🧪 Testing

The application was tested for the following functionalities:

| Test Case                    | Result   |
| ---------------------------- | -------- |
| CT NRRD Upload               | ✅ Passed |
| CT Volume Validation         | ✅ Passed |
| Slice Selection              | ✅ Passed |
| Model Inference              | ✅ Passed |
| Multi-Organ Segmentation     | ✅ Passed |
| Segmentation Mask Generation | ✅ Passed |
| Overlay Visualization        | ✅ Passed |
| Organ Detection              | ✅ Passed |
| Performance Evaluation       | ✅ Passed |
| Result Download              | ✅ Passed |
| Report Generation            | ✅ Passed |

---

# 🛠️ Technology Stack

### Programming Languages

* Python
* JavaScript
* HTML5
* CSS3

### Deep Learning

* PyTorch
* Torchvision
* NumPy

### Medical Image Processing

* SimpleITK
* NRRD

### Machine Learning

* Scikit-learn

### Backend

* Flask
* Flask-CORS
* Gunicorn

### Frontend

* HTML5
* CSS3
* JavaScript

### Version Control and Deployment

* Git
* GitHub
* Render

---

# 📁 Project Structure

```text
Head-and-Neck-Organ-Segmentation/
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── model/
│   └── hybrid_unet_transformer.py
│
├── multi_organ_dataset.py
├── train_multiorgan.py
├── train_multiorgan_resume.py
├── evaluate_multiorgan.py
├── evaluation_metrics.json
│
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/sravanig905-hue/Head-and-Neck-Organ-Segmentation.git
```

Navigate to the project directory:

```bash
cd Head-and-Neck-Organ-Segmentation
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Backend

Start the Flask backend:

```bash
python backend/app.py
```

The backend runs locally at:

```text
http://127.0.0.1:5000
```

---

# 🌐 Run the Frontend

Open another terminal from the project root:

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Open the application in your browser:

```text
http://127.0.0.1:5500
```

---

# ☁️ Deployment

The application is designed for separate frontend and backend deployment.

## Backend

The Flask backend can be deployed as a **Render Web Service**.

### Build Command

```text
pip install -r requirements.txt
```

### Start Command

```text
gunicorn backend.app:app
```

The backend requires access to the trained model during deployment.

## Frontend

The frontend can be deployed as a **Render Static Site** using:

```text
frontend/
```

as the publish directory.

The frontend should be configured to communicate with the deployed backend URL.

---

# 📦 Model

The project uses a trained:

**Hybrid U-Net + Transformer**

model for 31-class multi-organ segmentation.

The model consists of:

* CNN encoder
* Transformer bottleneck
* CNN decoder
* Skip connections
* Multi-class segmentation head

The trained model file is large and is therefore handled separately from the normal source-code files.

---

# 🔮 Future Enhancements

Potential future improvements include:

* Improved segmentation of small OARs
* Longer training schedules
* Advanced data augmentation
* Improved Transformer and attention mechanisms
* 3D volumetric segmentation
* GPU-based inference
* Larger-scale validation
* Clinical validation
* Improved visualization and reporting
* Integration with medical imaging platforms

---

# 👨‍💻 Project Information

### Project Title

**Head and Neck Organ Segmentation for CT-Scans using Hybrid U-Net and Transformer-based Deep Learning**

### Domain

**Artificial Intelligence | Deep Learning | Medical Image Processing**

### Application

**Automatic Multi-Organ Segmentation of Head and Neck CT Scans**

### Model

**Hybrid U-Net + Transformer**

### Dataset

**HaN-Seg**

### Number of Classes

**31 (30 OARs + Background)**

---

# 📜 License

This project is developed for academic and research purposes.


