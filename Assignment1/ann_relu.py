# ============================================================
# DEEP LEARNING & IoT ASSIGNMENT
# ANN-3: ReLU ACTIVATION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib


# ============================================================
# 2. SETTINGS
# ============================================================

DATA_PATH = (
    r"E:\HW\Deep-Learning\Assignments_Submission"
    r"\data\Iot_Sensor_data.csv"
)

RESULT_FOLDER = (
    r"E:\HW\Deep-Learning\Assignments_Submission"
    r"\Result\ANN3_ReLU_Results"
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. ANN SETTINGS
# ============================================================

HIDDEN_NEURONS = 16

LEARNING_RATE = 0.001

BATCH_SIZE = 32

EPOCHS = 100

RANDOM_STATE = 42


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING IoT SENSOR DATASET")
print("=" * 70)

data = pd.read_csv(
    DATA_PATH
)

print()
print("Dataset shape:")
print(data.shape)

print()
print("First 5 rows:")
print(data.head())


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "temperature",
    "humidity",
    "motion",
    "label"
]

for column in required_columns:

    if column not in data.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )

print()
print("All required columns are available.")


# ============================================================
# 6. PREPARE INPUT AND OUTPUT
# ============================================================

# INPUT FEATURES
#
# Temperature
# Humidity
# Motion

X = data[
    [
        "temperature",
        "humidity",
        "motion"
    ]
]


# OUTPUT

y = data[
    "label"
]


print()
print("=" * 70)
print("INPUT AND OUTPUT")
print("=" * 70)

print()
print("Input features:")
print("1. Temperature")
print("2. Humidity")
print("3. Motion")

print()
print("Number of input features:")
print(X.shape[1])

print()
print("Output:")
print("0 = Normal")
print("1 = Abnormal")


# ============================================================
# 7. ANN ARCHITECTURE
# ============================================================

print()
print("=" * 70)
print("ANN ARCHITECTURE")
print("=" * 70)

print()
print("Input Layer  : 3 features")

print("               Temperature")
print("               Humidity")
print("               Motion")

print()

print("Hidden Layer : 16 neurons")

print()

print("Activation   : ReLU")

print()

print("Output Layer : 1 neuron")

print("               0 = Normal")
print("               1 = Abnormal")


# ============================================================
# 8. CLASS DISTRIBUTION
# ============================================================

class_counts = (
    y.value_counts()
    .sort_index()
)

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(class_counts)


# ============================================================
# 9. CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    ["Normal", "Abnormal"],
    [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
)

plt.title(
    "Class Distribution"
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


# ============================================================
# 10. TEMPERATURE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    data["temperature"],
    bins=30
)

plt.title(
    "Temperature Distribution"
)

plt.xlabel(
    "Temperature"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "02_temperature_Distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 11. HUMIDITY DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    data["humidity"],
    bins=30
)

plt.title(
    "Humidity Distribution"
)

plt.xlabel(
    "Humidity"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "03_humidity_Distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 12. MOTION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    data["motion"],
    bins=30
)

plt.title(
    "Motion Distribution"
)

plt.xlabel(
    "Motion"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "04_motion_Distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


# ============================================================
# 14. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(

    X_temp,

    y_temp,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y_temp
)


# ============================================================
# 15. DISPLAY DATA SPLIT
# ============================================================

print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print()
print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 16. STANDARDIZATION
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_val_scaled = scaler.transform(
    X_val
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 17. CREATE ANN-3 RELU MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING ANN-3 RELU MODEL")
print("=" * 70)

model = MLPClassifier(

    hidden_layer_sizes=(
        HIDDEN_NEURONS,
    ),

    activation="relu",

    solver="adam",

    learning_rate_init=LEARNING_RATE,

    batch_size=BATCH_SIZE,

    max_iter=1,

    random_state=RANDOM_STATE,

    early_stopping=False,

    shuffle=True
)


# ============================================================
# 18. TRAINING HISTORY
# ============================================================

training_loss = []

validation_accuracy = []

validation_f1 = []


# ============================================================
# 19. CLASS VALUES
# ============================================================

classes = np.array(
    [0, 1]
)


# ============================================================
# 20. TRAIN MODEL EPOCH BY EPOCH
# ============================================================

print()
print("=" * 70)
print("TRAINING ANN-3 RELU")
print("=" * 70)

for epoch in range(
    EPOCHS
):

    # --------------------------------------------------------
    # TRAIN ONE EPOCH
    # --------------------------------------------------------

    model.partial_fit(

        X_train_scaled,

        y_train,

        classes=classes
    )


    # --------------------------------------------------------
    # TRAINING LOSS
    # --------------------------------------------------------

    current_loss = model.loss_


    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    validation_pred = model.predict(
        X_val_scaled
    )


    # --------------------------------------------------------
    # VALIDATION ACCURACY
    # --------------------------------------------------------

    current_accuracy = accuracy_score(

        y_val,

        validation_pred
    )


    # --------------------------------------------------------
    # VALIDATION F1
    # --------------------------------------------------------

    current_f1 = f1_score(

        y_val,

        validation_pred,

        zero_division=0
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    training_loss.append(
        current_loss
    )

    validation_accuracy.append(
        current_accuracy
    )

    validation_f1.append(
        current_f1
    )


    # --------------------------------------------------------
    # PRINT EPOCH RESULT
    # --------------------------------------------------------

    print(

        f"Epoch {epoch + 1:03d}/{EPOCHS} "

        f"| Loss: {current_loss:.6f} "

        f"| Val Accuracy: {current_accuracy:.4f} "

        f"| Val F1: {current_f1:.4f}"

    )


# ============================================================
# 21. FINAL TEST PREDICTION
# ============================================================

test_prediction = model.predict(
    X_test_scaled
)


# ============================================================
# 22. PREDICTION PROBABILITY
# ============================================================

test_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 23. CALCULATE FINAL ACCURACY
# ============================================================

final_accuracy = accuracy_score(

    y_test,

    test_prediction
)


# ============================================================
# 24. CALCULATE FINAL PRECISION
# ============================================================

final_precision = precision_score(

    y_test,

    test_prediction,

    zero_division=0
)


# ============================================================
# 25. CALCULATE FINAL RECALL
# ============================================================

final_recall = recall_score(

    y_test,

    test_prediction,

    zero_division=0
)


# ============================================================
# 26. CALCULATE FINAL F1-SCORE
# ============================================================

final_f1 = f1_score(

    y_test,

    test_prediction,

    zero_division=0
)


# ============================================================
# 27. PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("ANN-3 RELU FINAL RESULTS")
print("=" * 70)

print()

print(
    f"Accuracy  : {final_accuracy:.4f}"
)

print(
    f"Precision : {final_precision:.4f}"
)

print(
    f"Recall    : {final_recall:.4f}"
)

print(
    f"F1-Score  : {final_f1:.4f}"
)


# ============================================================
# 28. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    test_prediction,

    target_names=[
        "Normal",
        "Abnormal"
    ],

    zero_division=0
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()
print(report)


# ============================================================
# 29. SAVE CLASSIFICATION REPORT
# ============================================================

with open(

    os.path.join(
        RESULT_FOLDER,
        "ReLU_Classification_Report.txt"
    ),

    "w"
) as file:

    file.write(
        report
    )


# ============================================================
# 30. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    test_prediction
)


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print(cm)


# ============================================================
# 31. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Normal",
        "Abnormal"
    ]
)

display.plot()

plt.title(
    "ReLU Confusion Matrix"
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "06_ReLU_Confusion_Matrix.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 32. TRAINING LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    training_loss
)

plt.title(
    "ReLU Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "07_ReLU_Training_Loss.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 33. VALIDATION ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    validation_accuracy
)

plt.title(
    "ReLU Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "08_ReLU_Validation_Accuracy.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 34. VALIDATION F1 CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    validation_f1
)

plt.title(
    "ReLU Validation F1-Score"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "F1-Score"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "09_ReLU_Validation_F1.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 35. RELU ACTIVATION FUNCTION
# ============================================================

x_relu = np.linspace(
    -10,
    10,
    500
)


y_relu = np.maximum(
    0,
    x_relu
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    x_relu,
    y_relu
)

plt.title(
    "ReLU Activation Function"
)

plt.xlabel(
    "Input"
)

plt.ylabel(
    "ReLU Output"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "10_ReLU_Activation_Function.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 36. FINAL F1-SCORE
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.bar(
    ["ReLU"],
    [final_f1]
)

plt.ylim(
    0,
    1.05
)

plt.title(
    "ReLU Final F1-Score"
)

plt.ylabel(
    "F1-Score"
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "11_ReLU_Final_F1.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 37. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(

    range(
        len(y_test)
    ),

    y_test.to_numpy(),

    label="Actual"
)

plt.plot(

    range(
        len(test_prediction)
    ),

    test_prediction,

    label="Predicted"
)

plt.title(
    "ReLU Actual vs Predicted"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Class"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "12_ReLU_Actual_vs_Predicted.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 38. PREDICTION PROBABILITY
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    test_probability
)

plt.axhline(

    0.5,

    linestyle="--"
)

plt.title(
    "ReLU Prediction Probability"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Probability of Abnormal"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "13_ReLU_Prediction_Probability.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 39. TEMPERATURE VS HUMIDITY
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(

    data["temperature"],

    data["humidity"],

    c=data["label"]
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

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "14_Temperature_vs_Humidity.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 40. FINAL METRICS BAR CHART
# ============================================================

metric_names = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1-Score"
]


metric_values = [

    final_accuracy,

    final_precision,

    final_recall,

    final_f1
]


plt.figure(
    figsize=(8, 6)
)

plt.bar(

    metric_names,

    metric_values
)

plt.ylim(
    0,
    1.05
)

plt.title(
    "ReLU Final Metrics"
)

plt.ylabel(
    "Score"
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "15_ReLU_Metrics.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 41. FIRST 30 TEST PREDICTIONS
# ============================================================

first_30 = pd.DataFrame({

    "Actual":

        y_test.to_numpy()[:30],

    "Predicted":

        test_prediction[:30],

    "Abnormal_Probability":

        test_probability[:30]

})


print()
print("=" * 70)
print("FIRST 30 TEST PREDICTIONS")
print("=" * 70)

print()

print(
    first_30.to_string(
        index=False
    )
)


# ============================================================
# 42. SAVE PREDICTION RESULTS
# ============================================================

prediction_results = X_test.copy()


prediction_results[
    "Actual"
] = y_test.to_numpy()


prediction_results[
    "Predicted"
] = test_prediction


prediction_results[
    "Abnormal_Probability"
] = test_probability


prediction_results.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "ReLU_Test_Predictions.csv"
    ),

    index=False
)


# ============================================================
# 43. SAVE FINAL METRICS
# ============================================================

metrics_data = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-Score"

    ],

    "Score": [

        final_accuracy,

        final_precision,

        final_recall,

        final_f1

    ]

})


metrics_data.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "ReLU_Final_Metrics.csv"
    ),

    index=False
)


# ============================================================
# 44. SAVE TRAINING HISTORY
# ============================================================

history_data = pd.DataFrame({

    "Epoch":

        range(
            1,
            EPOCHS + 1
        ),

    "Training_Loss":

        training_loss,

    "Validation_Accuracy":

        validation_accuracy,

    "Validation_F1":

        validation_f1

})


history_data.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "ReLU_Training_History.csv"
    ),

    index=False
)


# ============================================================
# 45. SAVE MODEL
# ============================================================

joblib.dump(

    model,

    os.path.join(

        RESULT_FOLDER,

        "ANN3_ReLU_Model.joblib"
    )
)


# ============================================================
# 46. SAVE SCALER
# ============================================================

joblib.dump(

    scaler,

    os.path.join(

        RESULT_FOLDER,

        "ANN3_ReLU_Scaler.joblib"
    )
)


# ============================================================
# 47. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("ANN-3 RELU FINAL SUMMARY")
print("=" * 70)

print()

print(
    "Dataset              : IoT Sensor Data"
)

print(
    "Input Features       : 3"
)

print(
    "Hidden Layer         : 16 neurons"
)

print(
    "Activation Function  : ReLU"
)

print(
    "Output Layer         : 1 neuron"
)

print(
    "Optimizer            : Adam"
)

print(
    "Learning Rate        : 0.001"
)

print(
    "Batch Size           : 32"
)

print(
    "Epochs               : 100"
)

print()

print(
    f"Accuracy             : {final_accuracy:.4f}"
)

print(
    f"Precision            : {final_precision:.4f}"
)

print(
    f"Recall               : {final_recall:.4f}"
)

print(
    f"F1-Score             : {final_f1:.4f}"
)

print()

print(
    "Results saved in:"
)

print(
    RESULT_FOLDER
)

print()

print("=" * 70)
print("ANN-3 RELU COMPLETED")
print("=" * 70)