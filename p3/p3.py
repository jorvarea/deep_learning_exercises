# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% [markdown]
# # Cargar datos

#%%

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

#%% [markdown]
# # Definición de modelos

# %%

class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 16, kernel_size=4),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(16*9*9, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
# %% [markdown]
# # Definición de función para evaluar el modelo

# %%

def evaluate_on_test_set(model, test_loader, criterion, device):
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calcular accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Guardar predicciones
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    return {
        'loss': avg_test_loss,
        'accuracy': test_accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }


def evaluate_model(model, test_loader, criterion, device, epoch):
    print(f"\n{'='*60}")
    print(f"📊 Evaluating at Epoch {epoch}")
    print(f"{'='*60}")
    
    test_metrics = evaluate_on_test_set(model, test_loader, criterion, device)
    test_metrics['epoch'] = epoch
    
    print(f"\n📈 Test Set Evaluation:")
    print(f"  • Loss: {test_metrics['loss']:.4f}")
    print(f"  • Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"{'='*60}\n")
    
    return test_metrics

# %% [markdown]
# # Training

# %%

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)


# %%

model = CNN().to(device)
lr = 0.0002
num_epochs = 50

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
test_results = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = train_model(model, train_loader, criterion, optimizer, device)
    losses.append(epoch_loss)

    print(f"Epoch {epoch}: Loss {epoch_loss}")

    if (epoch + 1) % 5 == 0:
        test_metrics = evaluate_model(model, test_loader, criterion, device, epoch)
        test_results.append(test_metrics)

torch.save(model.state_dict(), 'model.pth')

# %% [markdown]
# # Save metrics to CSV

# %%

import os

test_metrics = {
    'epoch': [result['epoch'] for result in test_results],
    'loss': [result['loss'] for result in test_results],
    'accuracy': [result['accuracy'] for result in test_results]
}

training_metrics = {
    'epoch': list(range(num_epochs)),
    'loss': losses,
}

os.makedirs('results', exist_ok=True)

df_training_metrics = pd.DataFrame(training_metrics)
df_training_metrics.to_csv('results/training_metrics.csv', index=False)

df_test_metrics = pd.DataFrame(test_metrics)
df_test_metrics.to_csv('results/test_metrics.csv', index=False)

# %% [markdown]
# # Plot losses

# %%

plt.figure(figsize=(10, 5))
plt.plot(test_metrics['loss'], label='Training Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/training_losses.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(test_metrics['loss'], label='Test Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Losses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/test_losses.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(test_metrics['accuracy'], label='Test Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/test_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()