import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

torch.manual_seed(42)
np.random.seed(42)

digits = load_digits()
X = digits.images.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

mean = X_train.mean()
std = X_train.std() + 1e-8
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

def make_loader(X, y, shuffle=False):
    X = torch.tensor(X[:, None, :, :], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=64, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

def run_epoch(loader, training=True):
    model.train(training)
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        if training:
            optimizer.zero_grad()

        output = model(xb)
        loss = criterion(output, yb)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (output.argmax(1) == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, correct / total

for epoch in range(15):
    train_loss, train_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(val_loader, False)

    print(
        f"Epoch {epoch+1:02d}/15 | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

model.eval()
predictions = []
actual = []

with torch.no_grad():
    for xb, yb in test_loader:
        output = model(xb.to(device))
        predictions.extend(output.argmax(1).cpu().numpy())
        actual.extend(yb.numpy())

print("\nTest Accuracy:", accuracy_score(actual, predictions))
print("\nClassification Report:")
print(classification_report(actual, predictions, digits=4))
print("\nConfusion Matrix:")
print(confusion_matrix(actual, predictions))

torch.save(
    {"model_state_dict": model.state_dict(), "mean": mean, "std": std},
    "results/digit_cnn.pt"
)
