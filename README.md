# SA2 - Object Detection with YOLOv26: Bike vs. Motor Classification

## Project Overview
This repository contains the model training, evaluation, and diagnostic analysis for the **SA2 - Video-Based Data Gathering for Machine Learning Research** activity. The objective of this project is to deploy the YOLOv26 architecture to accurately detect and classify two specific vehicle types: **Bikes** and **Motors**. 

Beyond basic training, this project critically analyzes the model's performance through confusion matrices and statistical metrics (mAP50, Precision, Recall, F1 Score) to determine its readiness for real-world deployment.

## Dataset Preparation & Balancing
The original dataset suffered from a severe class imbalance (approx. 500 Bikes to 3,100 Motors). To resolve this and meet the assignment requirements, the dataset was systematically balanced:
* **Undersampling:** The majority class was filtered down to create a balanced base set.
* **Oversampling (Augmentation):** A 4x augmentation strategy was applied to the base set to hit the required dataset size.
* **Final Split:** * **Training Set:** 1,200 images
  * **Validation Set:** 83 pure, un-augmented images to ensure valid, real-world evaluation metrics.

## Methodology & Hyperparameter Setups
Three separate models were initialized and trained to compare how different hyperparameter configurations affect YOLOv26's learning efficiency and accuracy. 

| Model | Epochs | Image Size | Optimizer | Batch Size | Learning Rate (lr0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | 25 | 640 | AdamW | 4 | 0.01 |
| **Model 2** | 30 | 640 | SGD | 20 | 0.001 |
| **Model 3** | 40 | 640 | Auto | -1 (Auto) | 0.0001 |

## Performance Metrics & Evaluation
After validation, the models achieved the following statistical results:

| Metric | Model 1 (AdamW \| 25 epochs) | Model 2 (SGD \| 30 epochs) | Model 3 (Auto \| 40 epochs) |
| :--- | :--- | :--- | :--- |
| **True Positives** <br>*(Diagonal)* | **180** <br>(82 Bike, 98 Motor) | **198** <br>(89 Bike, 109 Motor) | **211** <br>(94 Bike, 117 Motor) |
| **False Positives** <br>*(Background misclassified as object)* | **120** <br>(54 Bike, 66 Motor) | **94** <br>(22 Bike, 72 Motor) | **84** <br>(28 Bike, 56 Motor) |
| **False Negatives** <br>*(Missed detections)* | **86** <br>(30 Bike, 56 Motor) | **70** <br>(22 Bike, 48 Motor) | **59** <br>(19 Bike, 40 Motor) |

### Diagnostic Outputs
*Please refer to the `assets/` folder in this repository for the exported Confusion Matrices and real-time loss graphs.*

A full diagnostic discussion—detailing the True Positives, background errors (False Positives), and an analysis of which classes the models struggled with the most—can be found in the attached PDF report.

## Repository Structure
* `SA2_YOLOv8_NMS_Free_Detection_final.ipynb` - The primary Google Colab notebook containing all initialization, training loops, and validation testing.
* `assets/` - Directory containing screenshot outputs, final epoch text files, and confusion matrices.
* `SA2_Model_Discussion.pdf` - The final comprehensive report and diagnostic analysis.

## How to Run
1. Clone this repository.
2. Open the `.ipynb` file in Google Colab.
3. Ensure runtime hardware acceleration is set to **T4 GPU**.
4. Mount your Google Drive and point the `data.yaml` path to your exported Roboflow dataset.
5. Execute the cells sequentially.
