# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# %%
# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root='.', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

# %%
# Modelo CNN


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# %%
# Configuración de entrenamiento
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: AlexNet = AlexNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs: int = 3
history: dict[str, list[float]] = {'train_loss': [], 'train_accuracy': []}

# %%
# Entrenamiento y seguimiento de métricas por minibatch
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        predictions = output.argmax(dim=1)
        batch_accuracy = (predictions == target).float().mean().item()
        history['train_loss'].append(loss.item())
        history['train_accuracy'].append(batch_accuracy)
        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)}"
                f" | Loss: {loss.item():.4f} | Accuracy: {batch_accuracy:.4f}"
            )

# %%
# Evaluación final en el conjunto de test
model.eval()
test_loss: float = 0.0
correct: int = 0
total: int = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        test_loss += loss.item() * data.size(0)
        predictions = output.argmax(dim=1)
        correct += (predictions == target).sum().item()
        total += data.size(0)

average_test_loss: float = test_loss / total
test_accuracy: float = correct / total
print(f"Test loss: {average_test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")

# %%
# Visualización del progreso durante el entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_loss'], label='Train loss por minibatch')
axes[0].set_title('Evolución de la pérdida')
axes[0].set_xlabel('Minibatch')
axes[0].set_ylabel('Pérdida')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(history['train_accuracy'], label='Train accuracy por minibatch')
axes[1].set_title('Evolución de la accuracy')
axes[1].set_xlabel('Minibatch')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True)
axes[1].legend()

fig.suptitle('Seguimiento del entrenamiento por minibatch')
plt.tight_layout()
plt.show()
