# ============================================================
# MLP - THREE HIDDEN LAYERS
# RMSprop Optimizer + ReLU Activation
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = r"E:\HW\Deep-Learning\Assignments_Submission\data\Iot_Sensor_data.csv"

RESULT_FOLDER = r"E:\HW\Deep-Learning\Assignments_Submission\opt_diff\rmsprop"

os.makedirs(RESULT_FOLDER, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_STATE = 42


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MLP - RMSprop Optimizer")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. SET RANDOM SEEDS
# ============================================================

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)


# ============================================================
# 4. LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)

print("\nDataset Shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 5. INPUT AND OUTPUT
# ============================================================

X = data[
    ["temperature", "humidity", "motion"]
].values

y = data["label"].values


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

print("\nClass Distribution:")
print(
    pd.Series(y)
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(6, 4))

pd.Series(y).value_counts().sort_index().plot(
    kind="bar"
)

plt.xticks(
    [0, 1],
    ["Normal", "Abnormal"],
    rotation=0
)

plt.title("Class Distribution")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "01_Class_Distribution.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 7. FEATURE DISTRIBUTIONS
# ============================================================

for feature in [
    "temperature",
    "humidity",
    "motion"
]:

    plt.figure(figsize=(6, 4))

    plt.hist(
        data[feature],
        bins=30
    )

    plt.title(
        feature.capitalize() + " Distribution"
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_FOLDER,
            feature + "_Distribution.png"
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# 8. TEMPERATURE vs HUMIDITY
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

plt.title(
    "Temperature vs Humidity: Normal vs Abnormal"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "05_Temperature_Humidity_Normal_Abnormal.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 9. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print("\nData Split:")

print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))


# ============================================================
# 10. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


# ============================================================
# 11. CONVERT DATA TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)


X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
).reshape(-1, 1)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# 12. DATA LOADER
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 13. MLP MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Input Layer
            # 3 input features
            nn.Linear(3, 16),
            nn.ReLU(),

            # Hidden Layer 2
            nn.Linear(16, 8),
            nn.ReLU(),

            # Hidden Layer 3
            nn.Linear(8, 4),
            nn.ReLU(),

            # Output Layer
            nn.Linear(4, 1)

        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 14. CREATE MODEL
# ============================================================

model = MLP().to(device)


# ============================================================
# 15. LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# 16. RMSprop OPTIMIZER
# ============================================================

optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 17. PRINT MODEL
# ============================================================

print("\n================================")
print("MODEL STRUCTURE")
print("================================")

print(model)

print("\nInput Layer  : 3")
print("Hidden Layer 1: 16")
print("Hidden Layer 2: 8")
print("Hidden Layer 3: 4")
print("Output Layer : 1")
print("Activation   : ReLU")
print("Optimizer    : RMSprop")
print("Learning Rate:", LEARNING_RATE)


# ============================================================
# 18. TRAINING
# ============================================================

loss_history = []

val_accuracy_history = []

val_f1_history = []


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    # --------------------------------------------------------
    # Mini-batch training
    # --------------------------------------------------------

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)

        # Calculate loss
        loss = criterion(
            outputs,
            y_batch
        )

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()


    # Average loss
    average_loss = (
        total_loss / len(train_loader)
    )

    loss_history.append(
        average_loss
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    with torch.no_grad():

        val_outputs = model(
            X_val_tensor.to(device)
        )

        val_probabilities = torch.sigmoid(
            val_outputs
        )

        val_predictions = (
            val_probabilities >= 0.5
        ).int()

        val_predictions = (
            val_predictions
            .cpu()
            .numpy()
            .flatten()
        )


    # Validation Accuracy

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_accuracy_history.append(
        val_accuracy
    )


    # Validation F1

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_f1_history.append(
        val_f1
    )


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} "
        f"Loss={average_loss:.4f} "
        f"Val Accuracy={val_accuracy:.4f} "
        f"Val F1={val_f1:.4f}"
    )


# ============================================================
# 19. TEST PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    test_outputs = model(
        X_test_tensor.to(device)
    )

    test_probabilities = torch.sigmoid(
        test_outputs
    )

    y_pred = (
        test_probabilities >= 0.5
    ).int().cpu().numpy().flatten()

    probability = (
        test_probabilities
        .cpu()
        .numpy()
        .flatten()
    )


# ============================================================
# 20. FINAL METRICS
# ============================================================

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


print("\n================================")
print("FINAL RESULTS - RMSprop")
print("================================")

print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Abnormal"
        ],
        zero_division=0
    )
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Abnormal"
    ]
)

disp.plot()

plt.title(
    "RMSprop Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "06_Confusion_Matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 23. TRAINING LOSS
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    range(1, EPOCHS + 1),
    loss_history
)

plt.title(
    "RMSprop Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "07_Training_Loss.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 24. VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    range(1, EPOCHS + 1),
    val_accuracy_history
)

plt.title(
    "RMSprop Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "08_Validation_Accuracy.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 25. VALIDATION F1
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    range(1, EPOCHS + 1),
    val_f1_history
)

plt.title(
    "RMSprop Validation F1-Score"
)

plt.xlabel("Epoch")
plt.ylabel("F1-Score")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "09_Validation_F1.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 26. FINAL METRICS BAR CHART
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

plt.ylim(
    0,
    1.05
)

plt.title(
    "RMSprop Final Model Performance"
)

plt.ylabel("Score")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "10_Final_Metrics.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 27. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    y_test[:50],
    label="Actual"
)

plt.plot(
    y_pred[:50],
    label="Predicted"
)

plt.title(
    "RMSprop - Actual vs Predicted"
)

plt.xlabel("Test Sample")
plt.ylabel("Class")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "11_Actual_vs_Predicted.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 28. PREDICTION PROBABILITY
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    probability
)

plt.axhline(
    0.5,
    linestyle="--"
)

plt.title(
    "RMSprop - Abnormal Prediction Probability"
)

plt.xlabel("Test Sample")
plt.ylabel("Probability")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "12_Prediction_Probability.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 29. ReLU FUNCTION
# ============================================================

x = np.linspace(
    -10,
    10,
    500
)

relu = np.maximum(
    0,
    x
)

plt.figure(figsize=(7, 5))

plt.plot(
    x,
    relu
)

plt.title(
    "ReLU Activation Function"
)

plt.xlabel("Input")
plt.ylabel("ReLU Output")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "13_ReLU_Function.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 30. SAVE FINAL METRICS
# ============================================================

result_table = pd.DataFrame({

    "Optimizer": [
        "RMSprop"
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

    "F1-Score": [
        f1
    ]

})


result_table.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "RMSprop_Final_Metrics.csv"
    ),
    index=False
)


# ============================================================
# 31. SAVE TRAINING HISTORY
# ============================================================

history_table = pd.DataFrame({

    "Epoch": range(
        1,
        EPOCHS + 1
    ),

    "Training_Loss": loss_history,

    "Validation_Accuracy": val_accuracy_history,

    "Validation_F1": val_f1_history

})


history_table.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "RMSprop_Training_History.csv"
    ),
    index=False
)


# ============================================================
# 32. FINAL
# ============================================================

print("\n================================")
print("FINAL MODEL INFORMATION")
print("================================")

print("Input Layer   : 3")
print("Hidden Layer 1: 16")
print("Hidden Layer 2: 8")
print("Hidden Layer 3: 4")
print("Output Layer  : 1")

print("Activation    : ReLU")
print("Optimizer     : RMSprop")
print("Learning Rate : 0.001")
print("Epochs        :", EPOCHS)
print("Batch Size    :", BATCH_SIZE)

print("\nResults saved in:")

print(RESULT_FOLDER)

print("\n================================")
print("RMSprop TRAINING COMPLETED")
print("================================")