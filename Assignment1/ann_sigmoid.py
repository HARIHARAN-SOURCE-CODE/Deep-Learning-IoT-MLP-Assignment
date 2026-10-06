#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
DEEP LEARNING & IoT ASSIGNMENT
ANN-1: SIGMOID ACTIVATION
======================================================================

Architecture:

    Input Features : 3
                     1. Temperature
                     2. Humidity
                     3. Motion

    Hidden Layer   : 16 neurons

    Activation     : Sigmoid

    Output         : Binary classification
                     0 = Normal
                     1 = Abnormal

Optimizer:
    Adam

Learning Rate:
    0.001

Batch Size:
    32

Epochs:
    100

Data Split:
    64% Training
    16% Validation
    20% Testing

Training:
    Epoch-by-epoch using partial_fit()

Dataset:
    Iot_Sensor_data.csv

======================================================================
"""

# ======================================================================
# 1. IMPORT LIBRARIES
# ======================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ======================================================================
# 2. SETTINGS
# ======================================================================

RANDOM_STATE = 42

# Dataset
DATA_PATH = (
    r"E:\HW\Deep-Learning\Assignments_Submission"
    r"\data\Iot_Sensor_data.csv"
)

# Result folder
RESULT_FOLDER = (
    r"E:\HW\Deep-Learning\Assignments_Submission"
    r"\Result\ANN1_Sigmoid_Results"
)

# Create result folder if it does not exist
os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

# ANN settings
HIDDEN_NEURONS = 16
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100

# Data split
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


# ======================================================================
# 3. LOAD DATASET
# ======================================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ======================================================================
# 4. CHECK REQUIRED COLUMNS
# ======================================================================

required_columns = [
    "temperature",
    "humidity",
    "motion",
    "label"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"was not found in the dataset."
        )

print("\nAll required columns are available.")


# ======================================================================
# 5. PREPARE INPUT AND OUTPUT
# ======================================================================

X = df[
    [
        "temperature",
        "humidity",
        "motion"
    ]
].values

y = df["label"].values

print("\nInput features:")
print("1. Temperature")
print("2. Humidity")
print("3. Motion")

print("\nOutput:")
print("0 = Normal")
print("1 = Abnormal")

print("\nTotal samples:", len(X))


# ======================================================================
# 6. CLASS DISTRIBUTION
# ======================================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)

print(class_counts)

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    ["Normal (0)", "Abnormal (1)"],
    [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
)

plt.title(
    "IoT Sensor Dataset - Class Distribution"
)

plt.xlabel("Class")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "01_Class_Distribution.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 7. FEATURE DISTRIBUTION PLOTS
# ======================================================================

features = [
    "temperature",
    "humidity",
    "motion"
]

for i, feature in enumerate(
    features,
    start=2
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        df[feature],
        bins=30,
        edgecolor="black"
    )

    plt.title(
        f"{feature.capitalize()} Distribution"
    )

    plt.xlabel(
        feature.capitalize()
    )

    plt.ylabel(
        "Frequency"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_FOLDER,
            f"{i:02d}_{feature}_Distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ======================================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# ======================================================================

print("\n" + "=" * 70)
print("DATA SPLITTING")
print("=" * 70)

# ---------------------------------------------------------------
# First split:
# 80% temporary data
# 20% final test data
# ---------------------------------------------------------------

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ---------------------------------------------------------------
# Second split:
# 80% of temporary = 64% total training
# 20% of temporary = 16% total validation
# ---------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(
    "\nTraining samples   :",
    len(X_train)
)

print(
    "Validation samples :",
    len(X_val)
)

print(
    "Testing samples    :",
    len(X_test)
)

print("\nApproximate split:")

print("Training   = 64%")
print("Validation = 16%")
print("Testing    = 20%")


# ======================================================================
# 9. STANDARDIZATION
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()

# IMPORTANT:
# Fit scaler ONLY on training data

X_train_scaled = scaler.fit_transform(
    X_train
)

# Transform validation and test data

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "\nStandardScaler fitted using training data only."
)

print(
    "\nTraining data shape   :",
    X_train_scaled.shape
)

print(
    "Validation data shape :",
    X_val_scaled.shape
)

print(
    "Testing data shape    :",
    X_test_scaled.shape
)


# ======================================================================
# 10. CREATE ANN-1 SIGMOID MODEL
# ======================================================================

print("\n" + "=" * 70)
print("CREATING ANN-1 SIGMOID MODEL")
print("=" * 70)

model = MLPClassifier(
    hidden_layer_sizes=(HIDDEN_NEURONS,),
    activation="logistic",
    solver="adam",
    learning_rate_init=LEARNING_RATE,
    batch_size=BATCH_SIZE,

    # One epoch for each partial_fit() call
    max_iter=1,

    random_state=RANDOM_STATE,

    # We manually handle validation
    early_stopping=False,

    shuffle=True
)

print("\nModel architecture:")

print(
    "Input Features : 3 "
    "(Temperature, Humidity, Motion)"
)

print(
    "Hidden Layer   : 16 neurons"
)

print(
    "Activation     : Sigmoid"
)

print(
    "Output         : Binary "
    "(Normal / Abnormal)"
)

print(
    "Optimizer      : Adam"
)

print(
    "Learning Rate  :",
    LEARNING_RATE
)

print(
    "Batch Size     :",
    BATCH_SIZE
)

print(
    "Epochs         :",
    EPOCHS
)


# ======================================================================
# 11. TRAIN MODEL EPOCH BY EPOCH
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING ANN-1 - SIGMOID")
print("=" * 70)

epoch_losses = []
epoch_accuracies = []
epoch_f1_scores = []

classes = np.array([0, 1])

start_time = time.time()

for epoch in range(EPOCHS):

    # --------------------------------------------------------------
    # Train for ONE epoch
    # --------------------------------------------------------------

    model.partial_fit(
        X_train_scaled,
        y_train,
        classes=classes
    )

    # --------------------------------------------------------------
    # Training loss
    # --------------------------------------------------------------

    current_loss = model.loss_

    # --------------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------------

    validation_pred = model.predict(
        X_val_scaled
    )

    validation_accuracy = accuracy_score(
        y_val,
        validation_pred
    )

    validation_f1 = f1_score(
        y_val,
        validation_pred,
        zero_division=0
    )

    # --------------------------------------------------------------
    # Store results
    # --------------------------------------------------------------

    epoch_losses.append(
        current_loss
    )

    epoch_accuracies.append(
        validation_accuracy
    )

    epoch_f1_scores.append(
        validation_f1
    )

    # --------------------------------------------------------------
    # Print epoch information
    # --------------------------------------------------------------

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Loss: {current_loss:.6f} | "
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f} | "
        f"Validation F1: "
        f"{validation_f1:.4f}"
    )


training_time = (
    time.time() - start_time
)

training_epochs = len(
    epoch_losses
)

print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    "\nTotal Epochs:",
    training_epochs
)

print(
    "Final Training Loss:",
    round(
        epoch_losses[-1],
        6
    )
)

print(
    "Final Validation Accuracy:",
    round(
        epoch_accuracies[-1],
        4
    )
)

print(
    "Final Validation F1:",
    round(
        epoch_f1_scores[-1],
        4
    )
)

print(
    "Training Time:",
    round(
        training_time,
        2
    ),
    "seconds"
)


# ======================================================================
# 12. FINAL TEST PREDICTION
# ======================================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

y_pred = model.predict(
    X_test_scaled
)

y_pred_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# ======================================================================
# 13. CALCULATE FINAL METRICS
# ======================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ======================================================================
# 14. PRINT FINAL RESULTS
# ======================================================================

print("\nFinal Test Results")
print("-" * 50)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)

print(
    f"Epochs    : {training_epochs}"
)


# ======================================================================
# 15. CLASSIFICATION REPORT
# ======================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Normal",
        "Abnormal"
    ],
    zero_division=0
)

print(report)


# ======================================================================
# 16. CONFUSION MATRIX
# ======================================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "ANN-1 Sigmoid - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    [
        "Normal",
        "Abnormal"
    ]
)

plt.yticks(
    [0, 1],
    [
        "Normal",
        "Abnormal"
    ]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "06_Sigmoid_Confusion_Matrix.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 17. TRAINING LOSS CURVE
# ======================================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        training_epochs + 1
    ),
    epoch_losses,
    linewidth=2
)

plt.title(
    "ANN-1 Sigmoid - Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.xticks(
    range(
        1,
        training_epochs + 1,
        5
    )
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "07_Sigmoid_Training_Loss.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 18. VALIDATION ACCURACY CURVE
# ======================================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        training_epochs + 1
    ),
    epoch_accuracies,
    linewidth=2
)

plt.title(
    "ANN-1 Sigmoid - Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")

plt.ylim(
    0,
    1.05
)

plt.xticks(
    range(
        1,
        training_epochs + 1,
        5
    )
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "08_Sigmoid_Validation_Accuracy.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 19. VALIDATION F1-SCORE CURVE
# ======================================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        training_epochs + 1
    ),
    epoch_f1_scores,
    linewidth=2
)

plt.title(
    "ANN-1 Sigmoid - Validation F1-Score"
)

plt.xlabel("Epoch")
plt.ylabel("F1-Score")

plt.ylim(
    0,
    1.05
)

plt.xticks(
    range(
        1,
        training_epochs + 1,
        5
    )
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "09_Sigmoid_Validation_F1.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 20. SIGMOID ACTIVATION FUNCTION CURVE
# ======================================================================

x_sigmoid = np.linspace(
    -10,
    10,
    500
)

y_sigmoid = 1 / (
    1 + np.exp(-x_sigmoid)
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    x_sigmoid,
    y_sigmoid,
    linewidth=2
)

plt.title(
    "Sigmoid Activation Function"
)

plt.xlabel("Input")
plt.ylabel("Sigmoid Output")

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "10_Sigmoid_Activation_Function.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 21. FINAL F1-SCORE
# ======================================================================

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    ["Sigmoid"],
    [f1]
)

plt.title(
    "ANN-1 Sigmoid - Final Test F1-Score"
)

plt.xlabel(
    "Activation Function"
)

plt.ylabel(
    "F1-Score"
)

plt.ylim(
    0,
    1.05
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "11_Sigmoid_Final_F1.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 22. ACTUAL VS PREDICTED
# ======================================================================

plt.figure(
    figsize=(10, 5)
)

sample_indices = np.arange(
    len(y_test)
)

plt.scatter(
    sample_indices,
    y_test,
    label="Actual",
    alpha=0.7
)

plt.scatter(
    sample_indices,
    y_pred,
    label="Predicted",
    marker="x",
    alpha=0.7
)

plt.title(
    "ANN-1 Sigmoid - Actual vs Predicted"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Class"
)

plt.yticks(
    [0, 1],
    [
        "Normal",
        "Abnormal"
    ]
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "12_Sigmoid_Actual_vs_Predicted.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 23. PREDICTION PROBABILITY
# ======================================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    sample_indices,
    y_pred_probability,
    linewidth=1.5
)

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1.5
)

plt.title(
    "ANN-1 Sigmoid - Abnormal Prediction Probability"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Probability of Abnormal"
)

plt.ylim(
    -0.05,
    1.05
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "13_Sigmoid_Prediction_Probability.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 24. TEMPERATURE VS HUMIDITY
# ======================================================================

plt.figure(
    figsize=(8, 6)
)

normal_mask = (
    y == 0
)

abnormal_mask = (
    y == 1
)

plt.scatter(
    df.loc[
        normal_mask,
        "temperature"
    ],
    df.loc[
        normal_mask,
        "humidity"
    ],
    label="Normal",
    alpha=0.6
)

plt.scatter(
    df.loc[
        abnormal_mask,
        "temperature"
    ],
    df.loc[
        abnormal_mask,
        "humidity"
    ],
    label="Abnormal",
    alpha=0.6
)

plt.title(
    "Temperature vs Humidity"
)

plt.xlabel("Temperature")
plt.ylabel("Humidity")

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "14_Temperature_vs_Humidity.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 25. METRICS BAR CHART
# ======================================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    metric_names,
    metric_values
)

plt.title(
    "ANN-1 Sigmoid - Final Test Metrics"
)

plt.xlabel("Metric")
plt.ylabel("Score")

plt.ylim(
    0,
    1.05
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "15_Sigmoid_Metrics.png"
    ),
    dpi=300
)

plt.close()


# ======================================================================
# 26. FIRST 30 TEST PREDICTIONS
# ======================================================================

print("\n" + "=" * 70)
print("FIRST 30 TEST PREDICTIONS")
print("=" * 70)

prediction_table = pd.DataFrame({

    "Sample": np.arange(
        1,
        min(
            30,
            len(y_test)
        ) + 1
    ),

    "Actual": y_test[:30],

    "Predicted": y_pred[:30],

    "Abnormal_Probability":
        y_pred_probability[:30]
})

print(
    prediction_table.to_string(
        index=False
    )
)


# ======================================================================
# 27. SAVE PREDICTION RESULTS TO CSV
# ======================================================================

prediction_results = pd.DataFrame({

    "Actual_Label":
        y_test,

    "Predicted_Label":
        y_pred,

    "Abnormal_Probability":
        y_pred_probability
})

prediction_csv_path = os.path.join(
    RESULT_FOLDER,
    "Sigmoid_Test_Predictions.csv"
)

prediction_results.to_csv(
    prediction_csv_path,
    index=False
)

print(
    "\nPrediction results saved to:"
)

print(
    prediction_csv_path
)


# ======================================================================
# 28. SAVE FINAL METRICS TO CSV
# ======================================================================

metrics_results = pd.DataFrame({

    "Model": [
        "ANN-1 Sigmoid"
    ],

    "Activation": [
        "Sigmoid"
    ],

    "Input_Features": [
        3
    ],

    "Hidden_Neurons": [
        HIDDEN_NEURONS
    ],

    "Output_Type": [
        "Binary (Normal / Abnormal)"
    ],

    "Batch_Size": [
        BATCH_SIZE
    ],

    "Learning_Rate": [
        LEARNING_RATE
    ],

    "Training_Epochs": [
        training_epochs
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1_Score": [
        f1
    ],

    "Training_Time_Seconds": [
        training_time
    ]
})

metrics_csv_path = os.path.join(
    RESULT_FOLDER,
    "Sigmoid_Final_Metrics.csv"
)

metrics_results.to_csv(
    metrics_csv_path,
    index=False
)

print(
    "\nFinal metrics saved to:"
)

print(
    metrics_csv_path
)


# ======================================================================
# 29. SAVE MODEL USING JOBLIB
# ======================================================================

model_path = os.path.join(
    RESULT_FOLDER,
    "ANN1_Sigmoid_Model.joblib"
)

scaler_path = os.path.join(
    RESULT_FOLDER,
    "ANN1_Sigmoid_Scaler.joblib"
)

joblib.dump(
    model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

print("\n" + "=" * 70)
print("MODEL SAVING")
print("=" * 70)

print(
    "\nANN model saved to:"
)

print(
    model_path
)

print(
    "\nScaler saved to:"
)

print(
    scaler_path
)


# ======================================================================
# 30. FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("ANN-1 SIGMOID - FINAL SUMMARY")
print("=" * 70)

print(
    "Architecture       : "
    f"3 Input Features -> "
    f"{HIDDEN_NEURONS} Hidden Neurons -> "
    "Binary Output"
)

print(
    "Input Features     : "
    "Temperature, Humidity, Motion"
)

print(
    "Activation         : Sigmoid"
)

print(
    "Output             : "
    "0 = Normal, 1 = Abnormal"
)

print(
    f"Batch Size         : {BATCH_SIZE}"
)

print(
    f"Learning Rate      : {LEARNING_RATE}"
)

print(
    f"Training Epochs    : {training_epochs}"
)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Precision          : {precision:.4f}"
)

print(
    f"Recall             : {recall:.4f}"
)

print(
    f"F1-Score           : {f1:.4f}"
)

print(
    f"Training Time      : "
    f"{training_time:.2f} seconds"
)

print(
    "\nAll results saved in:"
)

print(
    RESULT_FOLDER
)

print(
    "\nTraining and evaluation completed successfully."
)

print("=" * 70)