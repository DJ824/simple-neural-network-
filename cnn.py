import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("using mps device")
else:
    device = torch.device("cpu")
    print("mps device not found, using cpu")


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        # cifar 10 is 32x32x3 (3 color channels)
        # initial input shape = (batch_size, # channels, h, w)
        # formula for conv output: (n - f + 2p) / s + 1
        # formula for maxpool output: (input_size - kernel_size) / s + 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # 32 filters, output is now (batch_size, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output is now (batch_size, 32, 16, 16)
            nn.Dropout2d(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # output = (batch_size, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # output = (batch_size, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output = (batch_size, 64, 8, 8)
            nn.Dropout2d(0.3)
        )

        # input = outpur from conv2
        self.fc = nn.Sequential(
            # flatten all dims
            nn.Flatten(),
            # output = (batch_size, 64 * 8 * 8)
            nn.Linear(64 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            # output = (batch_size, 256)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
            # output = (batch_size, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.003)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'epoch [{epoch + 1}/{epochs}], step [{i + 1}/{len(train_loader)}], '
                      f'loss: {running_loss / 100:.3f}, '
                      f'acc: {100. * correct / total:.3f}%, '
                      f'lr: {current_lr:.8f}')
                running_loss = 0.0

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'epoch [{epoch + 1}/{epochs}], '
              f'test loss: {test_loss / len(test_loader):.3f}, '
              f'test acc: {100. * correct / total:.3f}%, '
              f'time: {time.time() - start_time:.2f}s')


def load_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_cifar10()
    model = CIFAR10CNN()
    train_model(model, train_loader, test_loader, epochs=50)
