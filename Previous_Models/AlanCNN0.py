import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10


# Define data preprocessing
transform = transforms.Compose([
    transforms.RandomResizedCrop(224), # random cropping
    transforms.RandomHorizontalFlip(), # random horizontal flip
    transforms.ToTensor(), # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalized
])

# load dataset
train_dataset = datasets.ImageFolder(root='train', transform=transform) # Training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Training set data loader

test_dataset = datasets.ImageFolder(root='test', transform=transform) # test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Test set data loader

# Define the neural network cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define the convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # Define the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define the fully connected layer
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 102)

    def forward(self, x):
        # Convolution layer, using ReLU activation function for nonlinear transformation
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = self.pool(nn.functional.relu(self.conv5(x)))
        # Expand the feature map into a one-dimensional vector
        x = x.view(-1, 512 * 7 * 7)
        # Fully connected layer, using ReLU activation function for nonlinear transformation
        x = nn.functional.relu(self.fc1(x))
        # Use dropout to improve the generalization ability of the model
        x = nn.functional.dropout(x, training=self.training)
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.dropout(x, training=self.training)
        # Output layer, without activation function
        x = self.fc3(x)
        return x

# Optimizer loss function
model = CNN().cuda() # Put the model on the GPU for acceleration
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

