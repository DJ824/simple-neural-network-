import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os

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
            nn.Dropout2d(0.3)
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

        # input = output from conv2
        self.fc = nn.Sequential(
            # flatten all dims
            nn.Flatten(),
            # output = (batch_size, 64 * 8 * 8)
            nn.Linear(64 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            # output = (batch_size, 256)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
            # output = (batch_size, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


class ContinuousLRScheduler:
    def __init__(self, initial_lr, decay_rate):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.current_step = 0

    def get_lr(self):
        return self.initial_lr * (1.0 / (1.0 + self.decay_rate * self.current_step))

    def step(self):
        self.current_step += 1


def train_model(model, train_loader, test_loader, epochs, initial_lr, decay_rate,
                save_dir='model_checkpoints', resume_from=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=decay_rate)
    lr_scheduler = ContinuousLRScheduler(initial_lr, decay_rate)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_accuracy = 0.0
    start_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    if resume_from:
        print(f"loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['accuracy']
        lr_scheduler.current_step = checkpoint['scheduler_step']
        history = checkpoint['history']
        print(f"resuming from epoch {start_epoch} with accuracy {best_accuracy:.3f}%")

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            current_lr = lr_scheduler.get_lr()
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'epoch [{epoch + 1}/{epochs}], step [{i + 1}/{len(train_loader)}], '
                      f'loss: {running_loss / 10:.3f}, '
                      f'acc: {100. * correct / total:.3f}%, '
                      f'lr: {current_lr:.8f}')
                running_loss = 0.0

        train_accuracy = 100. * correct / total
        train_loss = epoch_loss / len(train_loader)
        history['train_acc'].append(train_accuracy)
        history['train_loss'].append(train_loss)

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

        current_accuracy = 100. * correct / total
        val_loss = test_loss / len(test_loader)
        history['val_acc'].append(current_accuracy)
        history['val_loss'].append(val_loss)

        print(f'epoch [{epoch + 1}/{epochs}], '
              f'train loss: {train_loss:.3f}, '
              f'val loss: {val_loss:.3f}, '
              f'train acc: {train_accuracy:.3f}%, '
              f'val acc: {current_accuracy:.3f}%, '
              f'time: {time.time() - start_time:.2f}s')

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'scheduler_step': lr_scheduler.current_step,
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model_checkpoint.pth'))
            print(f"saved new best model with accuracy: {best_accuracy:.3f}%")

    return best_accuracy, history


def fine_tune_model(base_model_path, train_loader, test_loader,
                    new_lr=0.0001, new_weight_decay=1e-5,
                    freeze_layers=None, epochs=10):
    model = CIFAR10CNN()
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if freeze_layers:
        layer_dict = {
            'conv1': model.conv1,
            'conv2': model.conv2,
            'fc': model.fc
        }

        for layer_name in freeze_layers:
            if layer_name in layer_dict:
                print(f"freezing {layer_name} layer")
                for param in layer_dict[layer_name].parameters():
                    param.requires_grad = False
            else:
                print(f": layer {layer_name} not found in model")

    for name, param in model.named_parameters():
        print(f"Layer: {name}, Trainable: {param.requires_grad}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=new_lr,
        weight_decay=new_weight_decay
    )

    best_accuracy, history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=epochs,
        initial_lr=new_lr,
        decay_rate=1e-5,
        save_dir='fine_tuned_checkpoints'
    )

    return model, best_accuracy, history


def load_cifar10(batch_size=512):
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

    print("initial training phase:")
    model = CIFAR10CNN()
    initial_accuracy, initial_history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=50,
        initial_lr=0.0005,
        decay_rate=1e-4
    )

    # print("\nfine-tuning-phase:")
    #
    # fine_tuned_model1, accuracy1, history1 = fine_tune_model(
    #     base_model_path='model_checkpoints/best_model_checkpoint.pth',
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     new_lr=0.0005,
    #     new_weight_decay=2e-5,
    #     freeze_layers=['conv1', 'conv2'],
    #     epochs=10
    # )

    # fine_tuned_model2, accuracy2, history2 = fine_tune_model(
    #     base_model_path='model_checkpoints/best_model_checkpoint.pth',
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     new_lr=0.00005,
    #     new_weight_decay=1e-6,
    #     freeze_layers=None,
    #     epochs=15
    # )
    #
    # print(f"initial training accuracy: {initial_accuracy:.3f}%")
    # print(f"fine tuning results with frozen convolutions {accuracy1:.3f}%")
    # print(f"Fine-tuning results {accuracy2:.3f}%")
