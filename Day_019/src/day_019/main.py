import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.random.seed(42)

# 150 evenly spaced X values between 0 and 2*pi
X = np.linspace(0, 2*np.pi, 150)

# target values using y = sin(X), Gaussian noise of standard deviation 0.2
y = np.sin(X) + np.random.normal(0, 0.2, X.size)

# Randomly shuffles the indices before splitting.
indices = np.random.permutation(X.shape[0])

# Splits the records into 70% training, 15% validation, and 15% testing
train_indices = indices[:int(0.7 * X.shape[0])]
val_indices = indices[int(0.7 * X.shape[0]):int(0.85 * X.shape[0])]
test_indices = indices[int(0.85 * X.shape[0]):]

# Converts the datasets into PyTorch tensors with shape (N, 1) and dtype float32.
X_train = torch.tensor(X[train_indices], dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(y[train_indices], dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X[val_indices], dtype=torch.float32).reshape(-1, 1)
y_val = torch.tensor(y[val_indices], dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X[test_indices], dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y[test_indices], dtype=torch.float32).reshape(-1, 1)

class SineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

torch.manual_seed(42)
model = SineModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(model, epochs=300):
    train_losses = [] 
    val_losses = []
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: Training Loss {loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = loss_fn(val_predictions, y_val)
            val_losses.append(val_loss.item())
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}: Validation Loss {val_loss.item():.6f}")

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), "best_sine_model.pth")
    return train_losses, val_losses

train_loss_history, val_loss_history = train_model(model, 300)

best_epoch = np.argmin(val_loss_history) + 1
best_val_loss = np.min(val_loss_history)

print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Loss: {best_val_loss:.6f}")


best_model = SineModel()

best_model.load_state_dict(
    torch.load("best_sine_model.pth", weights_only=True)
)

best_model.eval()

with torch.no_grad():
    test_predictions = best_model(X_test)
    test_loss = loss_fn(test_predictions, y_test)

print(f"Test Loss: {test_loss.item():.6f}")
