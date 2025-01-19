# MOJ Project

## Project Overview

The MOJ project is designed for processing, tagging, and training machine learning models on legal verdicts. This document provides an overview of the project's structure, data organization, and the steps required to run the project.

---

## Directory Structure

### 1. *Data*

All project data is located in the resources/data directory, organized as follows:

- *Source*: Raw .docx verdicts are stored in resources/data/source.
- *Tagging*: Contains manual tagging, feature sentences, and other related files at resources/data/tagging.
- *Training*: Data splits for training, testing, and evaluation are in resources/data/training.

### 2. *Configurations*

Configuration files for various stages of the pipeline are located in resources/configs:

- *Predict*: Configuration files for prediction tasks are in resources/configs/predict.
- *Preprocessing, Training, etc.*: Other configuration files are located in relevant subdirectories within resources/configs.

### 3. *Results*

Processed verdicts and other outputs are stored in:

- results/db: Final verdict database after the full processing pipeline.

---

## Workflow

### Main Pipeline

The primary scripts for running the project are located in src/flows. Follow these steps to use and train the model:

1. *Train Sentence Classification*Train a model for sentence classification:src/flows/train/train_sentence_cls.py
2. *Predict Sentences*Run inference on sentences:src/flows/inference/predict_sentences.py
3. *Feature Extraction*Extract features from the data:src/flows/features/feature_extraction.py
4. *Check Similarity*
   Train a similarity model for cases:
   src/flows/train/train_similarity_case.py

### Additional Scripts

- *View Prompts*:Prompts used in feature extraction are located at:src/scripts/features/feature_extraction/prompts
- *Combine Features*:
  Combine extracted features into a complete vector:
  src/flows/add_features/add_features.py

---

## Running the Project

### Prerequisites

Ensure you have the required dependencies installed. Refer to the requirements.txt file (if available) or set up your environment as per the project's setup guidelines.

### Steps to Execute

1. Navigate to the specific flow directory in src/flows.
2. Run the respective Python script based on your task:

   - Training: train/train_sentence_cls.py
   - Prediction: inference/predict_sentences.py
   - Feature extraction: features/feature_extraction.py
   - Similarity: train/train_similarity_case.py
3. Modify the necessary configuration files in resources/configs as needed for your task.
4. Results will be saved in the results/db directory.

---

## Notes

- For detailed explanations of each step, refer to the in-script comments and the configuration files.
- Update the configurations before running scripts to match your data and processing requirements.

---

Feel free to contact the project maintainer for further assistance.
