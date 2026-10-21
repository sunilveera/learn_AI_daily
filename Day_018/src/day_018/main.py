import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Create a tensor X containing 100 evenly spaced numbers between -2 and 2
X = torch.linspace(-2, 2, 100)

# Reshape X into 100 rows and 1 column.
X = X.reshape(-1, 1)

# target values using y = 2 * X + 1
y = 2 * X + 1

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

default_model = SimpleNN()

xavier_model = SimpleNN()
with torch.no_grad():
    nn.init.xavier_uniform_(xavier_model.layer1.weight)

he_model = SimpleNN()
with torch.no_grad():
    nn.init.kaiming_uniform_(
        he_model.layer1.weight,
        nonlinearity="relu"
    )

def train_model(model, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

default_model_losses = train_model(default_model, 200)
xavier_model_losses = train_model(xavier_model, 200)
he_model_losses = train_model(he_model, 200)

print("Default:", default_model_losses[0], default_model_losses[-1])
print("Xavier:", xavier_model_losses[0], xavier_model_losses[-1])
print("He:", he_model_losses[0], he_model_losses[-1])

plt.plot(default_model_losses, label="Default Model")
plt.plot(xavier_model_losses, label="Xavier Model")
plt.plot(he_model_losses, label="HE Model")
plt.legend()
plt.show()