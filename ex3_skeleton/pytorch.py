# """PyTorch MNIST Example."""
#
#
# from __future__ import print_function
# import argparse
# import sys
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
#
#
# class Net(nn.Module):
#     """A neural network implementation."""
#     def __init__(self):
#         super(Net, self).__init__()
#
#         # TODO: Define the member variables for your layers.
#         # Use the appropriate layers from torch.nn
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
#         self.pool = nn.MaxPool2d(kernel_size=2)
#         self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Feature map size after conv2 and pooling
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#
#         # TODO: Implement one forward pass of your neural network.
#         x = F.relu(self.conv1(x))  # [batch, 32, 26, 26]
#         x = F.relu(self.conv2(x))  # [batch, 64, 24, 24]
#         x = self.pool(x)  # [batch, 64, 12, 12]
#         x = x.view(x.size(0), -1)  # Flatten: [batch, 9216]
#         x = F.relu(self.fc1(x))  # [batch, 128]
#         x = self.fc2(x)  # [batch, 10]
#         return F.log_softmax(x, dim=1)  # Log probabilities
#
#
#
# def train(model, device, train_loader, optimizer, epoch, args):
#     """Train the model for one epoch."""
#     # This indicates to the model that it is used for training.
#     # Will, e.g., change how dropout layers operate.
#     model.train()
#
#     # remember the loss values
#     running_loss = []
#
#     # Inner training loop: Iterate over batches
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if batch_idx >= args['batches_per_episode']:
#             break
#
#         # Move data and target to the correct device (cpu/gpu).
#         data, target = data.to(device), target.to(device)
#         # zero out gradients
#         optimizer.zero_grad()
#
#         # TODO: implement one step of the optimization:
#         # loss = None
#         # # * Calculate predictions
#         # # * Calculate the loss (i.e. cross-entropy loss)
#         # running_loss.append(loss.data.numpy())
#         # # * Backpropagate the loss to find the gradients
#         # # * Take one gradient step with your optimizer
#
#         # Forward pass: compute predictions
#         output = model(data)
#
#         # Compute loss using negative log likelihood (since model uses log_softmax)
#         loss = F.nll_loss(output, target)
#
#         # Backpropagation
#         loss.backward()
#
#         # Optimization step
#         optimizer.step()
#
#         # Save loss value
#         running_loss.append(loss.item())
#
#         if batch_idx % args["log_interval"] == 0:
#             sys.stdout.write('\rTrain Epoch: {} [{}/{}]\tAverage Loss: {:.6f}'.format(
#                 epoch, (batch_idx+args["log_interval"]) * len(data), args['batches_per_episode'] * len(data),
#                 np.average(running_loss)
#             ))
#             sys.stdout.flush()
#             if args["dry_run"]:
#                 break
#
#
# def test(model, device, test_loader):
#     """Test the model on the specified test set, and print test loss and accuracy."""
#     # Similar to .train() above, this will tell the model it is used for inference.
#     model.eval()
#
#     # Accumulator for the loss over the test dataset
#     test_loss = 0
#     # Accumulator for the number of correctly classified items
#     correct = 0
#
#     # This block will not compute any gradients
#     with torch.no_grad():
#         # Similar to the inner training loop, only over the test_loader
#         for data, target in test_loader:
#
#             data, target = data.to(device), target.to(device)
#             # Use the same loss function as in training
#             criterion = torch.nn.CrossEntropyLoss()
#
#             # TODO: Implement the same loss calculations as in training
#             # No optimizer step here.
#             # Forward pass
#             output = model(data)
#
#             # Compute loss
#             loss = criterion(output, target)
#             test_loss += loss.item() * data.size(0)  # Multiply by batch size
#             # Calculate the predictions (choose class with maximum predicted value) of your model over the batch
#             # Get predicted class (the one with the max score)
#             pred = output.argmax(dim=1, keepdim=False)
#             # Calculate how many predictions were correct, and add them here
#
#             # Count correct predictions
#             correct += (pred == target).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#
#
# def get_mnist_loaders(batch_size):
#     """Creates train- and test-DataLoaders for MNIST with the specified batch size."""
#     train_kwargs = {'batch_size': batch_size, 'shuffle': True}
#     test_kwargs = {'batch_size': batch_size}
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
#     dataset2 = datasets.MNIST('data', train=False, download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
#
#     return train_loader, test_loader
#
#
# def main():
#
#     # Seed your model for reproducibility
#     torch.manual_seed(4711)
#
#     # If possible, use CUDA (i.e., your GPU) for computations.
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     # Training Parameters
#     learning_rate = 0.5
#     batch_size = 64
#     epochs = 10
#     training_args = dict(
#         log_interval=10,
#         dry_run=False,
#         batches_per_episode=50
#     )
#
#     # Retrieve DataLoaders for the train- and test-dataset.
#     train_loader, test_loader = get_mnist_loaders(batch_size)
#
#     # Create your network, and move it to the specified device
#     model = Net().to(device)
#
#     # TODO: Create your optimizer here
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#
#     # The outer training loop (over epochs)
#     test(model, device, test_loader)
#     for epoch in range(1, epochs + 1):
#         train(model, device, train_loader, optimizer, epoch, training_args)
#         test(model, device, test_loader)
#
#     # Save the trained model.
#     torch.save(model.state_dict(), "mnist_cnn.pt")
#
#
# if __name__ == '__main__':
#     main()
"""PyTorch MNIST Example."""

from __future__ import print_function
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    """A neural network implementation."""

    def __init__(self):
        super(Net, self).__init__()

        # TODO: Define the member variables for your layers.
        # Use the appropriate layers from torch.nn
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Feature map size after conv2 and pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # TODO: Implement one forward pass of your neural network.
        x = F.relu(self.conv1(x))  # [batch, 32, 26, 26]
        x = F.relu(self.conv2(x))  # [batch, 64, 24, 24]
        x = self.pool(x)  # [batch, 64, 12, 12]
        x = x.view(x.size(0), -1)  # Flatten: [batch, 9216]
        x = F.relu(self.fc1(x))  # [batch, 128]
        x = self.fc2(x)  # [batch, 10]
        return F.log_softmax(x, dim=1)  # Log probabilities


def train(model, device, train_loader, optimizer, epoch, args):
    """Train the model for one epoch."""
    # This indicates to the model that it is used for training.
    # Will, e.g., change how dropout layers operate.
    model.train()

    # remember the loss values
    running_loss = []

    # Inner training loop: Iterate over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= args['batches_per_episode']:
            break

        # Move data and target to the correct device (cpu/gpu).
        data, target = data.to(device), target.to(device)
        # zero out gradients
        optimizer.zero_grad()

        # TODO: implement one step of the optimization:
        # loss = None
        # # * Calculate predictions
        # # * Calculate the loss (i.e. cross-entropy loss)
        # running_loss.append(loss.data.numpy())
        # # * Backpropagate the loss to find the gradients
        # # * Take one gradient step with your optimizer

        # Forward pass: compute predictions
        output = model(data)

        # Compute loss using negative log likelihood (since model uses log_softmax)
        loss = F.nll_loss(output, target)

        # Backpropagation
        loss.backward()

        # Optimization step
        optimizer.step()

        # Save loss value
        running_loss.append(loss.item())

        if batch_idx % args["log_interval"] == 0:
            sys.stdout.write('\rTrain Epoch: {} [{}/{}]\tAverage Loss: {:.6f}'.format(
                epoch, (batch_idx + args["log_interval"]) * len(data), args['batches_per_episode'] * len(data),
                np.average(running_loss)
            ))
            sys.stdout.flush()
            if args["dry_run"]:
                break


def test(model, device, test_loader):
    """Test the model on the specified test set, and print test loss and accuracy."""
    # Similar to .train() above, this will tell the model it is used for inference.
    model.eval()

    # Accumulator for the loss over the test dataset
    test_loss = 0
    # Accumulator for the number of correctly classified items
    correct = 0

    # This block will not compute any gradients
    with torch.no_grad():
        # Similar to the inner training loop, only over the test_loader
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Use the same loss function as in training
            criterion = torch.nn.CrossEntropyLoss()

            # TODO: Implement the same loss calculations as in training
            # No optimizer step here.
            # Forward pass
            output = model(data)

            # Compute loss
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # Multiply by batch size
            # Calculate the predictions (choose class with maximum predicted value) of your model over the batch
            # Get predicted class (the one with the max score)
            pred = output.argmax(dim=1, keepdim=False)
            # Calculate how many predictions were correct, and add them here

            # Count correct predictions
            correct += (pred == target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_mnist_loaders(batch_size):
    """Creates train- and test-DataLoaders for MNIST with the specified batch size."""
    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': batch_size}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def main():
    # Seed your model for reproducibility
    torch.manual_seed(4711)

    # If possible, use CUDA (i.e., your GPU) for computations.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Training Parameters
    learning_rate = 0.001
    batch_size = 64
    epochs = 20
    training_args = dict(
        log_interval=10,
        dry_run=False,
        batches_per_episode=50
    )

    # Retrieve DataLoaders for the train- and test-dataset.
    train_loader, test_loader = get_mnist_loaders(batch_size)

    # Create your network, and move it to the specified device
    model = Net().to(device)

    # TODO: Create your optimizer here
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # The outer training loop (over epochs)
    test(model, device, test_loader)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, training_args)
        test(model, device, test_loader)

    # Save the trained model.
    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
