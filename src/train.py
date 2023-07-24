import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from collections import Counter
import torch.nn.utils.rnn as rnn_utils
import warnings
from tqdm import tqdm 

# Add this before the code that produces warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Preprocess the image data and captions
train_dir = "../dataset/train_dir"
test_dir = "../dataset/test_dir"
caption_file_train = "../dataset/train_captions.csv"

# Load captions from CSV file
captions_df = pd.read_csv(caption_file_train)

# Step 2: Create a custom dataset and data loaders
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, target_size=None):
        self.data = pd.read_csv(csv_file)
        self.data.dropna(inplace=True)  # Drop rows with missing data (if any)
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.vocab = self.build_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data['image_name'].values[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}")
        image = Image.open(img_path)

        if self.target_size:
            # Resize the image to the target size
            image = transforms.Resize(self.target_size)(image)

        if self.transform:
            image = self.transform(image)

        caption = self.data['comment-fa'].values[idx]
        return image, caption

    def build_vocab(self):
        counter = Counter()
        for caption in self.data["comment-fa"].values:
            counter.update(caption.split())

        # Append new tokens to the existing vocabulary
        vocab = getattr(self, 'vocab', {})
        for token in counter:
            if token not in vocab:
                vocab[token] = len(vocab)
        return vocab

target_size = (256, 256)
train_dataset = ImageCaptionDataset(caption_file_train, train_dir, 
                                    transform=transforms.ToTensor(), 
                                    target_size=target_size)
# print(type(train_dataset[1][1]))
# test_dataset = ImageCaptionDataset(caption_file_test, test_dir, transform=transforms.ToTensor())
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Step 3: Define the neural network architecture
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(ImageCaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim + 2048, hidden_dim)  # Update LSTM input size
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, captions):
        embeddings = self.embedding(captions)
        # Replicate image features along the sequence dimension to match captions length
        image_features = image_features.unsqueeze(1).expand(-1, captions.size(1), -1)
        combined_features = torch.cat((image_features, embeddings), dim=2)
        output, _ = self.rnn(combined_features)
        output = self.fc(output)
        return output

# Step 4: Define the image encoder architecture
# Replace the image encoder with preferred pre-trained model
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Example: pre-trained ResNet model without classification head
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] # Remove the last classification layer
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        return self.resnet(images).squeeze()


# Step 5: Define the loss function and optimizer
vocab_size = len(train_dataset.vocab)
embed_dim = 256
hidden_dim = 512
learning_rate = 0.001

image_encoder = ImageEncoder()
model = ImageCaptioningModel(vocab_size, embed_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def caption_to_indices(caption, vocab):
    return [vocab[word] for word in caption.split()]

def train_model(model, image_encoder, train_loader, criterion, optimizer, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_encoder.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Wrap the data loader with tqdm to display the progress bar
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, captions in train_loader_iter:
            images = images.to(device)

            # Convert captions to numerical indices and pad them
            captions = [caption_to_indices(c, vocab) for c in captions]
            captions_padded = rnn_utils.pad_sequence([torch.tensor(c) for c in captions], batch_first=True)
            captions_padded = captions_padded.to(device)

            # Pass the images through the image encoder
            image_features = image_encoder(images)

            # Forward pass
            outputs = model(image_features, captions_padded[:, :-1])  # Ignore the last token in the captions

            # Calculate the loss (excluding the padding tokens)
            targets = captions_padded[:, 1:].contiguous()  # Ignore the first token in the captions
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update the tqdm progress bar with the current loss
            train_loader_iter.set_postfix({"Loss": loss.item()})

        # Print the average loss for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {running_loss / len(train_loader)}")


# Define hyperparameters
num_epochs = 25
learning_rate = 0.001

# Initialize the image encoder and model
image_encoder = ImageEncoder()
model = ImageCaptioningModel(vocab_size, embed_dim, hidden_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, image_encoder, train_loader, criterion, optimizer, train_dataset.vocab)

# Step 6: Save the trained model
save_path = "trained_model.pth"  # Define the file path and name for saving the model
model_info = {
    'state_dict': model.state_dict(),
    'vocab': vocab,
    'embed_dim': embed_dim,
    'hidden_dim': hidden_dim,
}
torch.save(model_info, save_path)
print("Trained model saved successfully!")

