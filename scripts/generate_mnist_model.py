#!/usr/bin/env python3
"""
Generate a small MNIST classifier model in ONNX format.
This creates a trained 2-layer MLP that can classify handwritten digits.
"""

import subprocess
import sys
import os

def ensure_packages():
    """Install required packages if not present."""
    packages = ['torch', 'torchvision', 'onnx']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure_packages()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx


class MNISTNet(nn.Module):
    """
    CNN for MNIST: 
    Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> Dropout -> FC
    """
    def __init__(self):
        super().__init__()
        # 1. Convolutional Block 1
        # Input: (1, 28, 28) -> Output: (32, 28, 28) (due to padding=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Normalizes data for faster, stable training

        # 2. Convolutional Block 2
        # Input: (32, 14, 14) -> Output: (64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 3. Pooling Layer (Reduces dimension by half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4. Fully Connected Layers
        self.flatten = nn.Flatten()
        
        # Calculation for linear input: 
        # Image reduces 28 -> 14 -> 7. Depth is 64. Total: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5) # Prevents overfitting
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x) # Image size becomes 14x14

        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x) # Image size becomes 7x7

        # Classification Head
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x) # Randomly zeroes neurons during training
        x = self.fc2(x)
        
        return x


def train_model(model, device, train_loader, epochs=3):
    """Train the model on MNIST dataset."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    return model


def evaluate_model(model, device, test_loader):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def export_to_onnx(model, filepath):
    """Export PyTorch model to ONNX format using legacy exporter."""
    model.eval()

    # Create dummy input (batch_size=1, channels=1, height=28, width=28)
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export using legacy exporter to avoid onnxscript compatibility issues
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Use legacy exporter
    )

    # Verify the model
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)

    print(f"Model exported to: {filepath}")
    print(f"Input shape: [batch, 1, 28, 28] (grayscale image)")
    print(f"Output shape: [batch, 10] (logits for digits 0-9)")


def main():
    # Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Success: Using Apple Metal (MPS) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Success: Using NVIDIA CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    print("\nDownloading/loading MNIST dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create model
    model = MNISTNet().to(device)
    print(f"\nModel architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\nTraining model...")
    model = train_model(model, device, train_loader, epochs=3)

    # Evaluate
    print("\nEvaluating model...")
    evaluate_model(model, device, test_loader)

    # Export to ONNX
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    onnx_path = os.path.join(models_dir, 'mnist.onnx')
    print(f"\nExporting to ONNX...")
    model = model.to('cpu')
    export_to_onnx(model, onnx_path)

    # Also copy to wasm output directory if it exists
    wasm_dir = os.path.join(script_dir, '..', 'zig-out', 'wasm')
    if os.path.exists(wasm_dir):
        import shutil
        wasm_model_path = os.path.join(wasm_dir, 'mnist.onnx')
        shutil.copy(onnx_path, wasm_model_path)
        print(f"Copied to: {wasm_model_path}")

    print("\nDone! Model ready for WASM inference.")


if __name__ == '__main__':
    main()
