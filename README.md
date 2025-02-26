# Breast Cancer Classification using Deep Learning

## Project Overview
This project focuses on the classification of breast cancer images into benign and malignant categories using machine learning and deep learning techniques. The study includes data preprocessing, exploratory data analysis, and the development of multiple classification models, including traditional machine learning models and convolutional neural networks (CNNs).

## Directory Structure
```
breast_cancer_classification/
│── data/
│   ├── X.npy  # Image dataset
│   ├── Y.npy  # Corresponding labels
│── breast_cancer_classification.ipynb  # Jupyter Notebook containing the implementation
│── README.md  # Project documentation
```

## Dataset
The dataset consists of images stored in `X.npy` and their corresponding labels in `Y.npy`. The dataset is balanced, ensuring fair model evaluation.

## Implementation Steps
### 1. Data Exploration
- Class distribution analysis
- Visualizing sample images
- Pixel intensity distribution analysis
- Basic statistical analysis

### 2. Data Preprocessing
- Normalization of image pixel values
- Splitting dataset into training, validation, and test sets (70%-20%-10%)

### 3. Model Development
#### Traditional Machine Learning Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- k-Nearest Neighbors (k-NN)

#### Deep Learning Models
- Basic CNN Model
- Enhanced CNN Model with data augmentation and batch normalization
- Further Enhanced CNN Model with L2 regularization

### 4. Model Evaluation
- Accuracy, precision, recall, and F1-score
- Confusion Matrix
- Receiver Operating Characteristic (ROC) Curve
- Precision-Recall Curve

## Results
| Model | Accuracy |
|--------|----------|
| Logistic Regression | 69% |
| SVM | 75% |
| Random Forest | 75% |
| k-NN | 75% |
| Basic CNN | 75.64% |
| Enhanced CNN | 68.55% |
| Further Enhanced CNN | **77.64%** |

## Model Comparison Plot
![Model Comparison Plot](/Users/arkamandol/Downloads/Breast_cancer.png)

## Observations
- The CNN models performed better than traditional ML models, confirming their effectiveness in medical image classification.
- Further enhancements with L2 regularization improved CNN performance.
- Data augmentation did not always lead to higher accuracy and required careful tuning.

## Future Improvements
- Hyperparameter tuning using Grid Search or Random Search.
- Exploring advanced CNN architectures such as ResNet or Inception.
- Using ensemble methods to combine multiple models for improved performance.

## Contact Information
- **Email**: [arkamandol56@gmail.com](mailto:arkamandol56@gmail.com)
- **GitHub**: [arkamandol5](https://github.com/arkamandol5)
- **LinkedIn**: [Arka Mandol](https://www.linkedin.com/in/arka-mandol-0b249716a/)


