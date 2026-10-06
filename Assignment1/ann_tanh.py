#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
DEEP LEARNING & IoT ASSIGNMENT
ANN-2: TANH ACTIVATION
======================================================================

Architecture:

    Input Layer  : 3 features
                   1. Temperature
                   2. Humidity
                   3. Motion

    Hidden Layer : 16 neurons

    Output Layer : 1 neuron

Input Features:
    1. Temperature
    2. Humidity
    3. Motion

Output:
    0 = Normal
    1 = Abnormal

Activation:
    Tanh

Optimizer:
    Adam

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

import joblib


# ======================================================================
# 2. SETTINGS
# ======================================================================

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Dataset path
# ----------------------------------------------------------------------

DATA_PATH = r"E:\HW\Deep-Learning\Assignments_Submission\data\Iot_Sensor_data.csv"


# ----------------------------------------------------------------------
# Tanh result folder
# ----------------------------------------------------------------------

RESULT_FOLDER = (
    r"E:\HW\Deep-Learning\Assignments_Submission\Result\ANN_Tanh_Result"
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ----------------------------------------------------------------------
# ANN settings
# ----------------------------------------------------------------------

HIDDEN_NEURONS = 16

BATCH_SIZE = 32

LEARNING_RATE = 0.001

EPOCHS = 100


# ----------------------------------------------------------------------
# Data split
# ----------------------------------------------------------------------

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

print(
    "Dataset shape:",
    df.shape
)

print("\nFirst 5 rows:")

print(
    df.head()
)

print("\nColumn names:")

print(
    df.columns.tolist()
)


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
            f"Required column '{column}' was not found in the dataset."
        )

print(
    "\nAll required columns are available."
)


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


print("\n" + "=" * 70)
print("INPUT FEATURES AND OUTPUT")
print("=" * 70)

print("\nInput features:")
print("1. Temperature")
print("2. Humidity")
print("3. Motion")

print("\nOutput:")
print("0 = Normal")
print("1 = Abnormal")

print(
    "\nTotal samples:",
    len(X)
)

print(
    "Number of input features:",
    X.shape[1]
)


# ======================================================================
# 6. CLASS DISTRIBUTION
# ======================================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = pd.Series(
    y
).value_counts().sort_index()

print(
    class_counts
)


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

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Number of Samples"
)

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


# ----------------------------------------------------------------------
# First split:
#
# 80% temporary data
# 20% final test data
# ----------------------------------------------------------------------

X_temp, X_test, y_temp, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


# ----------------------------------------------------------------------
# Second split:
#
# From remaining 80%:
#
# 80% training
# 20% validation
#
# Final:
#
# Training   = 64%
# Validation = 16%
# Testing    = 20%
# ----------------------------------------------------------------------

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

print(
    "Training   = 64%"
)

print(
    "Validation = 16%"
)

print(
    "Testing    = 20%"
)


# ======================================================================
# 9. STANDARDIZATION
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()


# ----------------------------------------------------------------------
# Fit scaler ONLY on training data
# ----------------------------------------------------------------------

X_train_scaled = scaler.fit_transform(
    X_train
)


# ----------------------------------------------------------------------
# Transform validation data
# ----------------------------------------------------------------------

X_val_scaled = scaler.transform(
    X_val
)


# ----------------------------------------------------------------------
# Transform test data
# ----------------------------------------------------------------------

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
# 10. CREATE ANN-2 MODEL
# ======================================================================

print("\n" + "=" * 70)
print("CREATING ANN-2 MODEL")
print("=" * 70)


model = MLPClassifier(

    # --------------------------------------------------------------
    # 16 neurons in one hidden layer
    # --------------------------------------------------------------

    hidden_layer_sizes=(
        HIDDEN_NEURONS,
    ),

    # --------------------------------------------------------------
    # TANH ACTIVATION
    # --------------------------------------------------------------

    activation="tanh",

    # --------------------------------------------------------------
    # ADAM OPTIMIZER
    # --------------------------------------------------------------

    solver="adam",

    # --------------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------------

    learning_rate_init=LEARNING_RATE,

    # --------------------------------------------------------------
    # Batch size
    # --------------------------------------------------------------

    batch_size=BATCH_SIZE,

    # --------------------------------------------------------------
    # One epoch for each partial_fit() call
    # --------------------------------------------------------------

    max_iter=1,

    random_state=RANDOM_STATE,

    # --------------------------------------------------------------
    # Validation is handled manually
    # --------------------------------------------------------------

    early_stopping=False,

    shuffle=True
)


# ======================================================================
# DISPLAY ANN ARCHITECTURE
# ======================================================================

print("\nANN Architecture:")

print(
    "Input Layer  : 3 features"
)

print(
    "   1. Temperature"
)

print(
    "   2. Humidity"
)

print(
    "   3. Motion"
)

print(
    "Hidden Layer : 16 neurons"
)

print(
    "Output Layer : 1 neuron (Normal/Abnormal)"
)

print(
    "\nArchitecture: 3 features -> 16 hidden neurons -> 1 output neuron"
)

print(
    "\nActivation Function: Tanh"
)

print(
    "Optimizer          : Adam"
)

print(
    "Batch Size         :",
    BATCH_SIZE
)

print(
    "Learning Rate      :",
    LEARNING_RATE
)

print(
    "Maximum Epochs     :",
    EPOCHS
)


# ======================================================================
# 11. TRAIN MODEL EPOCH BY EPOCH
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING ANN-2 - TANH")
print("=" * 70)


epoch_losses = []

epoch_accuracies = []

epoch_f1_scores = []


# ----------------------------------------------------------------------
# Classes for partial_fit
# ----------------------------------------------------------------------

classes = np.array(
    [0, 1]
)


# ----------------------------------------------------------------------
# Start training timer
# ----------------------------------------------------------------------

start_time = time.time()


# ======================================================================
# TRAINING LOOP
# ======================================================================

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


    # --------------------------------------------------------------
    # Validation accuracy
    # --------------------------------------------------------------

    validation_accuracy = accuracy_score(

        y_val,

        validation_pred
    )


    # --------------------------------------------------------------
    # Validation F1
    # --------------------------------------------------------------

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


# ======================================================================
# 12. TRAINING COMPLETED
# ======================================================================

training_time = (
    time.time() - start_time
)

training_epochs = len(
    epoch_losses
)


print("\n" + "=" * 70)
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
# 13. FINAL TEST PREDICTION
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
# 14. CALCULATE FINAL METRICS
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
# 15. PRINT FINAL RESULTS
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
# 16. CLASSIFICATION REPORT
# ======================================================================

print("\n" + "=" * 70)

print(
    "CLASSIFICATION REPORT"
)

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


print(
    report
)


# ======================================================================
# 17. CONFUSION MATRIX
# ======================================================================

cm = confusion_matrix(

    y_test,

    y_pred
)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm
)


plt.title(
    "ANN-2 Tanh - Confusion Matrix"
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

        "06_Tanh_Confusion_Matrix.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 18. TRAINING LOSS CURVE
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
    "ANN-2 Tanh - Training Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
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

        "07_Tanh_Training_Loss.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 19. VALIDATION ACCURACY CURVE
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
    "ANN-2 Tanh - Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Validation Accuracy"
)


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

        "08_Tanh_Validation_Accuracy.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 20. VALIDATION F1-SCORE CURVE
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
    "ANN-2 Tanh - Validation F1-Score"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "F1-Score"
)


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

        "09_Tanh_Validation_F1.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 21. TANH ACTIVATION FUNCTION CURVE
# ======================================================================

x_tanh = np.linspace(

    -5,

    5,

    500
)


y_tanh = np.tanh(
    x_tanh
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    x_tanh,

    y_tanh,

    linewidth=2
)


plt.title(
    "Tanh Activation Function"
)


plt.xlabel(
    "Input"
)


plt.ylabel(
    "Tanh Output"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "10_Tanh_Activation_Function.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 22. FINAL F1-SCORE
# ======================================================================

plt.figure(
    figsize=(7, 5)
)


plt.bar(

    ["Tanh"],

    [f1]
)


plt.title(
    "ANN-2 Tanh - Final Test F1-Score"
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

        "11_Tanh_Final_F1.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 23. ACTUAL VS PREDICTED
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
    "ANN-2 Tanh - Actual vs Predicted"
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

        "12_Tanh_Actual_vs_Predicted.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 24. PREDICTION PROBABILITY
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
    "ANN-2 Tanh - Abnormal Prediction Probability"
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

        "13_Tanh_Prediction_Probability.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 25. TEMPERATURE VS HUMIDITY
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


plt.xlabel(
    "Temperature"
)


plt.ylabel(
    "Humidity"
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

        "14_Temperature_vs_Humidity.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 26. METRICS BAR CHART
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
    "ANN-2 Tanh - Final Test Metrics"
)


plt.xlabel(
    "Metric"
)


plt.ylabel(
    "Score"
)


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

        "15_Tanh_Metrics.png"
    ),

    dpi=300
)


plt.close()


# ======================================================================
# 27. FIRST 30 TEST PREDICTIONS
# ======================================================================

print("\n" + "=" * 70)

print(
    "FIRST 30 TEST PREDICTIONS"
)

print("=" * 70)


num_predictions = min(
    30,
    len(y_test)
)


prediction_table = pd.DataFrame({

    "Sample": np.arange(
        1,
        num_predictions + 1
    ),

    "Actual": y_test[:num_predictions],

    "Predicted": y_pred[:num_predictions],

    "Abnormal_Probability":
        y_pred_probability[:num_predictions]
})


print(

    prediction_table.to_string(
        index=False
    )
)


# ======================================================================
# 28. SAVE PREDICTION RESULTS TO CSV
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

    "Tanh_Test_Predictions.csv"
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
# 29. SAVE FINAL METRICS TO CSV
# ======================================================================

metrics_results = pd.DataFrame({

    "Model": [
        "ANN-2 Tanh"
    ],

    "Activation": [
        "Tanh"
    ],

    "Input_Features": [
        3
    ],

    "Hidden_Neurons": [
        HIDDEN_NEURONS
    ],

    "Output_Neurons": [
        1
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

    "Tanh_Final_Metrics.csv"
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
# 30. SAVE MODEL USING JOBLIB
# ======================================================================

model_path = os.path.join(

    RESULT_FOLDER,

    "ANN2_Tanh_Model.joblib"
)


scaler_path = os.path.join(

    RESULT_FOLDER,

    "ANN2_Tanh_Scaler.joblib"
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

print(
    "MODEL SAVING"
)

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
# 31. FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)

print(
    "ANN-2 TANH - FINAL SUMMARY"
)

print("=" * 70)


print(
    "Input Features      : Temperature, Humidity, Motion"
)


print(
    f"Architecture        : "
    f"3 features -> {HIDDEN_NEURONS} hidden neurons -> 1 output neuron"
)


print(
    "Activation           : Tanh"
)


print(
    "Output               : 0 = Normal, 1 = Abnormal"
)


print(
    f"Batch Size           : "
    f"{BATCH_SIZE}"
)


print(
    f"Learning Rate        : "
    f"{LEARNING_RATE}"
)


print(
    f"Training Epochs      : "
    f"{training_epochs}"
)


print(
    f"Accuracy             : "
    f"{accuracy:.4f}"
)


print(
    f"Precision            : "
    f"{precision:.4f}"
)


print(
    f"Recall               : "
    f"{recall:.4f}"
)


print(
    f"F1-Score             : "
    f"{f1:.4f}"
)


print(
    f"Training Time        : "
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