import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

# Step 1: Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 3: Create classification models
models = {

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            probability=True,
            random_state=42
        )
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    "Naive Bayes": GaussianNB()
}

# Step 4: Train and evaluate models
results = []

for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Predict class labels
    y_pred = model.predict(X_test)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    loss = log_loss(
        y_test,
        y_prob
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Log Loss": loss
    })

# Step 5: Create result table
results_df = pd.DataFrame(results)

# Display results
print("\nClassification Model Comparison")
print(
    results_df.round(4).to_string(
        index=False
    )
)

# Step 6: Find the best model
best_index = results_df[
    "F1-Score"
].idxmax()

best_model = results_df.loc[
    best_index,
    "Model"
]

print(
    "\nBest Model based on F1-Score:",
    best_model
)
