# Large Language Model-Based Uncertainty-Adjusted Label Extraction for AI Model Development in Upper Extremity Radiography

This repository contains simplified code used in the paper "" The code is provided primarily to demonstrate the label extraction methodology and CNN model architecture used in the study.

## Overview

The repository includes:
- **Automatic label extraction** using GPT-4o (`generate_labels_secure.py`)
- **Data preprocessing** scripts (`Data_prep_clavicle.py` and `Data_prep_elbow_thumb.py`) 
- **CNN training and testing** scripts (`clav_training_testing.py` and `elbow_thumb_training_testing.py`)
- **Mock label generation** for testing without API access (`generate_labels_mock.py`)

## Prerequisites

- Python 3.8+
- Install required packages: `pip install -r requirements.txt`
- Azure OpenAI API access (for actual label generation)

## Directory Structure

Your project should be organized as follows:

```
project_root/
├── Templates/              # JSON templates for label extraction
│   ├── clavicle.json
│   ├── elbow.json
│   └── thumb.json
├── Images/                 # Medical images
│   ├── 1.png              # For clavicle (single view)
│   ├── 2.png
│   ├── 1/                 # For elbow/thumb (dual view)
│   │   ├── ap.png
│   │   └── lat.png
│   └── 2/
│       ├── ap.png
│       └── lat.png                
├── Output                # Labels and other results will be stored here
└── Mock_reports_*.csv      # Input report files as csv
```

## Input Data Format

### Report CSV Files
Your input CSV files should contain medical reports with the following columns:
- `ID`: Patient identifier (string/number)
- `Report`: Medical report text (string)


### Image Files
- **Clavicle**: Single PNG images named by patient ID (e.g., `1.png`, `2.png`)
- **Elbow/Thumb**: Dual view images in patient folders:
  - `patient_id/ap.png` (anterior-posterior view)
  - `patient_id/lat.png` (lateral view)

## Usage Workflow

### 1. Label Generation
First, generate labels from medical reports:

**With Azure OpenAI API:**
```bash
# Set up environment variables first
export AZURE_OPENAI_API_KEY="your_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint"
python generate_labels_secure.py
```

### 2. Data Preparation
Process the generated labels and create training datasets:

```bash
# For clavicle data
python Data_prep_clavicle.py

# For elbow/thumb data  
python Data_prep_elbow_thumb.py
```

### 3. Model Training
Train CNN models on the prepared datasets:

```bash
# For clavicle classification
python clav_training_testing.py --batch_size 4 --num_epochs 25

# For elbow/thumb classification
python elbow_thumb_training_testing.py --batch_size 4 --num_epochs 25
```

## Model Architecture

- **Clavicle**: Single-view ResNet-50 for multi-label classification
- **Elbow/Thumb**: Dual-view ResNet-50 with feature concatenation for multi-label classification

## Important Notes

- This is simplified code for demonstration purposes

- Real medical images and reports are required for actual clinical applications

## Citation

If you use this code in your research, please cite:

""

