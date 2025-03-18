# RSNA 2024 Lumbar Spine Dataset

## Dataset Structure

The dataset is organized in the following structure:

```
/scratch/bcwg/Lumberspine/
├── dataset/
│   ├── train_images/
│   │   └── [patient_id]/
│   │       └── [study_id]/
│   │           └── [1-N].dcm
│   ├── test_images/
│   │   └── [patient_id]/
│   │       └── [study_id]/
│   │           └── [1-N].dcm
│   ├── train.csv
│   ├── test_series_descriptions.csv
│   ├── train_series_descriptions.csv
│   └── sample_submission.csv
└── train/
    ├── checkpoints/
    ├── logs/
    └── processed_data/
        ├── images/
        │   └── [patient_id]/
        │       └── [study_id]/
        │           └── [1-N].png
        ├── dataset.json
        └── train_dataset.csv
```

## File Descriptions

### CSV Files
- `train.csv`: Labels for the training set, including study IDs and condition severity levels
- `test_series_descriptions.csv`: Descriptions of test set MRI series
- `train_series_descriptions.csv`: Descriptions of training set MRI series
- `sample_submission.csv`: Example submission format with probability predictions

### Image Data
- `train_images/`: Training set DICOM images
- `test_images/`: Test set DICOM images
- Each image is organized by:
  - `patient_id/`: Unique identifier for each patient
  - `study_id/`: Identifier for each MRI study
  - `[1-N].dcm`: Numbered DICOM image files within each study

### Processed Data
After running the preprocessing script, the following will be generated in the `train/processed_data/` directory:
- `images/`: Converted PNG images from DICOM files
- `dataset.json`: Dataset metadata and image-text pairs
- `train_dataset.csv`: Training dataset information with image paths and labels

## Conditions and Labels

### Spine Levels
- L1-L2
- L2-L3
- L3-L4
- L4-L5
- L5-S1

### Conditions
1. Spinal Canal Stenosis
2. Neural Foraminal Narrowing (Left/Right)
3. Subarticular Stenosis (Left/Right)

### Severity Levels
- Normal/Mild
- Moderate
- Severe

## MRI Series Types
- Sagittal T1
- Axial T2
- Sagittal T2/STIR

## Training Directory Structure
The `/scratch/bcwg/Lumberspine/train/` directory contains:
- `checkpoints/`: Model checkpoints during training
- `logs/`: Training logs and metrics
- `processed_data/`: Preprocessed images and dataset files

## Usage
1. First, run the preprocessing script:
   ```bash
   python prepare_data.py
   ```

2. The preprocessed data will be saved in the `train/processed_data/` directory

3. Training can be started using:
   ```bash
   python train.py
   ``` 