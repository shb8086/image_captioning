import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

# Define the caption dataset
class CaptionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_name = str(img_name) + ".jpg"  # Add .jpg extension
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        caption = self.data.iloc[idx, 2]

        return image, caption

# Define the caption generator model
class CaptionGenerator(nn.Module):
    def __init__(self, backbone, hidden_size, output_size):
        super(CaptionGenerator, self).__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, image, caption):
        features = self.backbone(image)
        caption = self.embedding(caption)
        output, _ = self.lstm(caption, features.unsqueeze(0))
        output = self.fc(output.squeeze(0))
        return output

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set constants
csv_file = "../dataset/captions.csv"
image_dir =  "../dataset/images/"
batch_size = 32
hidden_size = 512
output_size = 1000
learning_rate = 0.001
num_epochs = 10

# Load the ResNet model
backbone = models.resnet50(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone.to(device)

# Initialize the caption generator model
caption_generator = CaptionGenerator(backbone, hidden_size, output_size)
caption_generator.to(device)

# Define the transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Create the dataset and data loader
dataset = CaptionDataset(csv_file, image_dir, transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(caption_generator.parameters(), lr=learning_rate)

# Training loop
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(data_loader):
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        outputs = caption_generator(images, captions)

        # Compute loss
        loss = criterion(outputs.view(-1, output_size), captions.view(-1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}")

print("Training complete!")