# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md

import warnings
import pandas as pd
import seaborn as sns
from torchviz import make_dot

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.datasets import load_boston

import components

warnings.filterwarnings("ignore", category=FutureWarning)


class MLP(nn.Module):
    """
    Multilayer Perceptron for regression.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)


torch.manual_seed(42)

X, y = load_boston(return_X_y=True)
trainloader = DataLoader(
    dataset=components.BostonDataset(X, y), batch_size=10, shuffle=True, num_workers=1
)
mlp = MLP()

# Plot model architecture
batch = next(iter(trainloader))
yhat = mlp(batch[0])
make_dot(yhat, params=dict(list(mlp.named_parameters()))).render(
    "torchviz", format="png"
)

loss_function = nn.LLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

# --- train
for epoch in range(0, 20):
    print(f"Starting epoch {epoch+1}")
    
    current_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))
    
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(inputs)
        
        loss = loss_function(outputs, targets)        
        loss.backward()

        # Perform optimization
        optimizer.step()        
        
        current_loss += loss.item()
        if i % 10 == 0:
            print("Loss after mini-batch %5d: %.3f" % (i + 1, current_loss / 500))
            current_loss = 0.0

print("Training process has finished.")

# --- predict
scaler = torch.load("scaler.pkl")
X_scale = scaler.transform(X)
y_pred = [float(mlp(torch.FloatTensor(X_scale[i]))) for i in range(0, X_scale.shape[0])]
y_pred = pd.DataFrame({"y": y, "y_pred": y_pred}, index=range(0, len(y)))

sns.lmplot(x="y", y="y_pred", data=y_pred)
# plt.show()
