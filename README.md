# Classification Models – AIML Practical

## 📌 Project Overview

This project implements and compares four supervised machine learning classification algorithms:

- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naïve Bayes

The models are trained and evaluated using the **Breast Cancer Wisconsin dataset**. Their performance is compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- Log Loss

The purpose of this practical is to understand different classification algorithms and identify their suitability for different classification tasks.

---

## 🎯 Objectives

- Understand the concept of supervised classification.
- Implement SVM, Decision Tree, KNN, and Naïve Bayes.
- Prepare training and testing datasets.
- Train and test different classification models.
- Evaluate models using standard performance metrics.
- Compare the performance of the four algorithms.
- Identify the most suitable algorithm based on the evaluation results.

---

## 📊 Dataset

### Breast Cancer Wisconsin Dataset

The **Breast Cancer Wisconsin dataset** is a binary classification dataset available through Scikit-learn.

The dataset contains:

- **569 observations**
- **30 numerical features**
- **2 target classes**

The two classes are:

- Malignant
- Benign

The objective is to classify a tumor as malignant or benign based on the given diagnostic features.

---

## 🤖 Machine Learning Algorithms

### 1. Support Vector Machine (SVM)

SVM finds an optimal decision boundary that separates different classes while maximizing the margin between them.

**Advantages:**
- Effective for high-dimensional data.
- Performs well for complex classification problems.
- Can provide high classification accuracy.

**Limitations:**
- Requires feature scaling.
- Requires parameter tuning.
- Can be slower for very large datasets.

---

### 2. Decision Tree

Decision Tree uses a tree-like structure to make classification decisions based on input features.

**Advantages:**
- Easy to understand.
- Easy to visualize.
- Does not require feature scaling.
- Provides interpretable decision rules.

**Limitations:**
- Can overfit the training data.
- Large trees can become complex.

---

### 3. K-Nearest Neighbors (KNN)

KNN classifies a new data point based on the majority class among its nearest neighbors.

In this project, **K = 5** is used.

**Advantages:**
- Simple to understand.
- Easy to implement.
- Useful when similarity between observations is meaningful.

**Limitations:**
- Sensitive to feature scaling.
- Sensitive to the choice of K.
- Prediction can be slower for large datasets.

---

### 4. Naïve Bayes

Naïve Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that the features are conditionally independent given the class.

**Advantages:**
- Fast and simple.
- Memory efficient.
- Useful for high-dimensional classification problems.

**Limitations:**
- The feature independence assumption may not always be valid.

---

## 📈 Evaluation Metrics

The following metrics are used to evaluate the classification models.

### Accuracy

Measures the proportion of correctly classified observations.

### Precision

Measures how many predicted positive observations are actually positive.

### Recall

Measures how many actual positive observations are correctly identified.

### F1-Score

Provides a balance between precision and recall.

### Log Loss

Measures the quality of predicted probabilities.

For log loss, **lower values indicate better performance**.

---

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Scikit-learn | Machine learning algorithms and evaluation |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| VS Code / Google Colab / Jupyter Notebook | Development and execution |

---

## 📁 Project Structure

```text
Classification-Models/
│
├── classification_models.py
├── README.md
├── requirements.txt
└── screenshots/
    ├── output.png
    └── results.png
```

---

## 🔧 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/Classification-Models.git
```

### Step 2: Open the project folder

```bash
cd Classification-Models
```

### Step 3: Install the required libraries

```bash
pip install -r requirements.txt
```

---

## 📦 Required Libraries

The project uses the following Python libraries:

```text
pandas
numpy
scikit-learn
```

---

## ▶️ How to Run

Run the Python program using:

```bash
python classification_models.py
```

The program will:

1. Load the Breast Cancer Wisconsin dataset.
2. Split the dataset into training and testing sets.
3. Create the four classification models.
4. Train each model.
5. Generate predictions.
6. Calculate accuracy, precision, recall, F1-score, and log loss.
7. Display the comparative performance table.
8. Identify the best model based on F1-score.

---

## 📊 Results

The models were evaluated using an **80:20 train-test split**.

| Model | Accuracy | Precision | Recall | F1-Score | Log Loss |
|---|---:|---:|---:|---:|---:|
| SVM | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.0860 |
| Decision Tree | 0.9123 | 0.9559 | 0.9028 | 0.9286 | 3.1617 |
| KNN | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.4073 |
| Naïve Bayes | 0.9386 | 0.9452 | 0.9583 | 0.9517 | 0.3713 |

### Best Performing Model

Based on the F1-score, **SVM performed the best** among the four models.

SVM achieved:

- **Accuracy:** 98.25%
- **Precision:** 98.61%
- **Recall:** 98.61%
- **F1-Score:** 98.61%
- **Log Loss:** 0.0860

---

## 🔍 Comparative Analysis

| Algorithm | Main Strength | Suitable For |
|---|---|---|
| SVM | High performance and effective decision boundary | High-dimensional and complex classification |
| Decision Tree | Easy interpretation | Rule-based decision making |
| KNN | Similarity-based classification | Small datasets and pattern recognition |
| Naïve Bayes | Fast probabilistic classification | Text and high-dimensional classification |

The results show that **SVM performed best for this particular dataset and configuration**.

However, there is no universally best classification algorithm. The selection of an algorithm depends on the dataset, performance requirements, interpretability, computational resources, and application requirements.


## 📝 Conclusion

Four classification algorithms—SVM, Decision Tree, KNN, and Naïve Bayes—were successfully implemented and evaluated using the Breast Cancer Wisconsin dataset.

The models were compared using accuracy, precision, recall, F1-score, and log loss.

Among the tested models, **SVM achieved the best overall performance**, with the highest accuracy and F1-score and the lowest log loss.

This practical demonstrates the importance of comparing multiple machine learning algorithms before selecting a model for a particular classification problem.

---



## 📌 Note

This project is developed as part of the **TY B.Tech. Artificial Intelligence and Machine Learning Practical Assignment**.
