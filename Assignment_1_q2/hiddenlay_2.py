# ============================================================
# MLP - ONE HIDDEN LAYER - ReLU
# ============================================================

import os
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
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = r"E:\HW\Deep-Learning\Assignments_Submission\data\Iot_Sensor_data.csv"

RESULT_FOLDER = r"E:\HW\Deep-Learning\Assignments_Submission\Q2result\hiddenresult2"

os.makedirs(RESULT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)

X = data[["temperature", "humidity", "motion"]]
y = data["label"]

print("Dataset shape:", data.shape)
print(data.head())


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

print("\nClass Distribution:")
print(y.value_counts().sort_index())

plt.figure(figsize=(6, 4))
y.value_counts().sort_index().plot(kind="bar")
plt.xticks([0, 1], ["Normal", "Abnormal"], rotation=0)
plt.title("Class Distribution")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "01_Class_Distribution.png"))
plt.show()


# ============================================================
# 4. FEATURE DISTRIBUTIONS
# ============================================================

for feature in ["temperature", "humidity", "motion"]:

    plt.figure(figsize=(6, 4))
    plt.hist(data[feature], bins=30)
    plt.title(feature.capitalize() + " Distribution")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_FOLDER,
            feature + "_Distribution.png"
        )
    )

    plt.show()


# ============================================================
# 5. TRAIN / VALIDATION / TEST
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.20,
    random_state=42,
    stratify=y_temp
)

print("\nData Split:")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))


# ============================================================
# 6. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ============================================================
# 7. MLP MODEL
# ============================================================

# 3 inputs → 16 hidden neurons → 1 output

model = MLPClassifier(
    hidden_layer_sizes=(16,8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=1,
    random_state=42
)


# ============================================================
# 8. TRAINING
# ============================================================

loss_history = []
val_accuracy_history = []
val_f1_history = []

for epoch in range(100):

    model.partial_fit(
        X_train,
        y_train,
        classes=np.array([0, 1])
    )

    # Loss
    loss_history.append(model.loss_)

    # Validation prediction
    val_pred = model.predict(X_val)

    # Validation accuracy
    val_accuracy_history.append(
        accuracy_score(y_val, val_pred)
    )

    # Validation F1
    val_f1_history.append(
        f1_score(y_val, val_pred, zero_division=0)
    )

    print(
        f"Epoch {epoch + 1:03d}/100 "
        f"Loss={model.loss_:.4f} "
        f"Val Accuracy={val_accuracy_history[-1]:.4f}"
    )


# ============================================================
# 9. TEST PREDICTION
# ============================================================

y_pred = model.predict(X_test)

probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 10. FINAL METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n================================")
print("FINAL RESULTS")
print("================================")

print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Abnormal"],
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Abnormal"]
)

disp.plot()
plt.title("ReLU Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "05_Confusion_Matrix.png"
    )
)

plt.show()


# ============================================================
# 13. TRAINING LOSS
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(loss_history)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "06_Training_Loss.png"
    )
)

plt.show()


# ============================================================
# 14. VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(val_accuracy_history)

plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "07_Validation_Accuracy.png"
    )
)

plt.show()


# ============================================================
# 15. VALIDATION F1
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(val_f1_history)

plt.title("Validation F1-Score")
plt.xlabel("Epoch")
plt.ylabel("F1-Score")
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "08_Validation_F1.png"
    )
)

plt.show()


# ============================================================
# 16. FINAL METRICS
# ============================================================

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

plt.figure(figsize=(7, 5))

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0, 1.05)
plt.title("Final Model Performance")
plt.ylabel("Score")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "09_Final_Metrics.png"
    )
)

plt.show()


# ============================================================
# 17. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values[:50],
    label="Actual"
)

plt.plot(
    y_pred[:50],
    label="Predicted"
)

plt.title("Actual vs Predicted")
plt.xlabel("Test Sample")
plt.ylabel("Class")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "10_Actual_vs_Predicted.png"
    )
)

plt.show()


# ============================================================
# 18. PREDICTION PROBABILITY
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(probability)

plt.axhline(
    0.5,
    linestyle="--"
)

plt.title("Abnormal Prediction Probability")
plt.xlabel("Test Sample")
plt.ylabel("Probability")
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "11_Prediction_Probability.png"
    )
)

plt.show()


# ============================================================
# 19. ReLU FUNCTION
# ============================================================

x = np.linspace(-10, 10, 500)

relu = np.maximum(0, x)

plt.figure(figsize=(7, 5))

plt.plot(x, relu)

plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("ReLU Output")
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "12_ReLU_Function.png"
    )
)

plt.show()

# ============================================================
# TEMPERATURE + HUMIDITY: NORMAL vs ABNORMAL
# ============================================================

plt.figure(figsize=(8, 6))

normal = data[data["label"] == 0]
abnormal = data[data["label"] == 1]

plt.scatter(
    normal["temperature"],
    normal["humidity"],
    label="Normal (0)"
)

plt.scatter(
    abnormal["temperature"],
    abnormal["humidity"],
    label="Abnormal (1)"
)

plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Temperature vs Humidity: Normal vs Abnormal")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "13_Temperature_Humidity_Normal_Abnormal.png"
    )
)

plt.show()
# ============================================================
# 20. SAVE RESULTS
# ============================================================

result_table = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

result_table.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "Final_Metrics.csv"
    ),
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n================================")
print("MODEL STRUCTURE")
print("================================")

print("Input Layer  : 3")
print("Hidden Layer : 16 + 8+ReLU")
print("Output Layer : 1")
print("Optimizer    : Adam")

print("\nResults saved in:")
print(RESULT_FOLDER)