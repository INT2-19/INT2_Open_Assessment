import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10            # Maximum Number of epochs to train the network
batch_size = 64             
learning_rate = 0.001

# Define transforms with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomRotation(45),
    transforms.ColorJitter(contrast=0.25, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Flower102: Unknown number and various sized color images in 102 classes, with 40 to 258 images per class
train_dataset = torchvision.datasets.Flowers102(root='./data', split='train',
                                                download=True, transform=train_transform)
test_dataset = torchvision.datasets.Flowers102(root='./data', split='test',
                                               download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes=102):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avPool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(25088, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.drop = nn.Dropout(0.4)
        # Batch normalisation and Max Pooling used on convolution layers, dropout 40% used on fully connected linear layers
        # All layers have Relu Applied

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = self.avPool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.drop(self.fc1(out)))
        out = self.fc2(out)
        return out

# Create model and push to device
model = ConvNet().to(device)

# Get loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
Runs the current model on the training data to calculate the current loss and accuracy of the epoch.
For the training set, label smoothness is added to introduce randomness to prevent overfitting.

Args:
  model: Current version of the training model
  train_loader: The data loader responsible for loading the training set from the Flowers102 dataset 
  criterion: The loss function
  optimiser: The optimising function applied to the learning model
  device: The hardware accelerator that is being used to train the model
  smoothing: Value applied for label smoothing

Returns:
  Two values, a float value which is the loss from the current epoch and
  and accuracy a float value representing a percentage of the current accuracy for the training data
'''
def train(model, train_loader, criterion, optimiser, device, epoch, smoothing=0.1):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply label smoothing
        num_classes = model.fc2.out_features
        one_hot_labels = torch.zeros(labels.size(0), num_classes).to(device)
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
        one_hot_labels = one_hot_labels * (1 - smoothing) + smoothing / num_classes

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, one_hot_labels)

        # Backward and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    loss = train_loss / len(train_loader)

    return loss, acc

'''
Runs the current model on the validation data to calculate the test the current accuracy and loss of the epoch.

Args:
  model: Current version of the training model
  train_loader: The data loader responsible for loading the validation set from the Flowers102 dataset 
  criterion: The loss function
  device: The hardware accelerator that is being used to train the model

Returns:
  Two values, a float value which is the loss from the current epoch and
  and accuracy a float value representing a percentage of the current accuracy for the validation data
'''
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    loss = test_loss / len(test_loader)

    return loss, acc

# Training the model
train_losses, train_accs, valid_losses, valid_accs = [], [], [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimiser, device, epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Times validation accuracy is calculated is adjusted for debugging and data analysis purposes. 
    if ((epoch + 1) % 25 == 0):
      valid_loss, valid_acc = test(model, test_loader, criterion, device)
      valid_losses.append(valid_loss)   
      valid_accs.append(valid_acc)
      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%')
    else:
      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

# Save the model
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)