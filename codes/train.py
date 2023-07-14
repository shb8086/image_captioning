import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
from PIL import Image
import os
import time
import max_min

# Set the path to the dataset and CSV file
dataset_path = 'path_to_dataset_folder'
csv_file = 'path_to_csv_file.csv'
column_index = 2

# Set the device to be used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the desired dimensions for input images
img_height = 224
img_width = 224

# Set the maximum number of words in a caption
word_lengths, min_length, num_min_length_rows, max_length, num_max_length_rows, total_word_length, row_count = max_min::count_words_in_column(csv_file, column_index)
max_caption_length = max_length

# Set the batch size and number of epochs for training
batch_size = 32
num_epochs = 10

# Define the image captioning model with attention
class AttentionCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(AttentionCaptionModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.cnn = models.resnet152(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        self.attention = nn.Linear(embed_size + hidden_size, 1)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images, captions, lengths):
        batch_size = images.size(0)
        embeddings = self.embed(captions)
        features = self.cnn(images)
        hidden = self.init_hidden(batch_size).to(device)

        caption_length = captions.size(1)
        alphas = torch.zeros(batch_size, caption_length, 1).to(device)
        for t in range(caption_length):
            feature_attention = torch.cat((features, hidden[0]), dim=1)
            attention = self.attention(feature_attention)
            attention_weights = torch.softmax(attention, dim=1)
            alphas[:, t, :] = attention_weights
            context = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1)
            lstm_input = torch.cat((embeddings[:, t, :], context), dim=1)
            hidden = self.lstm(lstm_input, hidden)
        outputs = self.fc(self.dropout(hidden[0]))
        return outputs, alphas

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.lstm.hidden_size).to(device)
        c = torch.zeros(batch_size, self.lstm.hidden_size).to(device)
        return h, c

# Define the custom dataset for image captioning
class CaptionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = self.build_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        caption = self.data.iloc[idx, 2]
        caption = caption.lower()
        caption = caption.replace('.', ' .')
        caption = caption.replace(',', ' ,')
        caption = caption.replace('"', ' "')
        caption = caption.replace("'", " '")
        caption = caption.split()

        return image, caption

    def build_vocab(self):
        word_index = {}
        index_word = {}
        word_index['<pad>'] = 0
        index_word[0] = '<pad>'
        word_index['<start>'] = 1
        index_word[1] = '<start>'
        word_index['<end>'] = 2
        index_word[2] = '<end>'
        vocab = set()
        for _, _, caption in self.data.iterrows():
            caption = caption[2]
            caption = caption.lower()
            caption = caption.replace('.', ' .')
            caption = caption.replace(',', ' ,')
            caption = caption.replace('"', ' "')
            caption = caption.replace("'", " '")
            vocab.update(caption.split())

        start_index = len(word_index)
        for i, word in enumerate(vocab, start=start_index):
            word_index[word] = i
            index_word[i] = word

        return word_index

# Define the image transformations
image_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the dataset
dataset = CaptionDataset(csv_file=csv_file, root_dir=dataset_path, transform=image_transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize the captioning model with different CNN backbones
vocab_size = len(dataset.vocab)
embed_size = 512
hidden_size = 512

# Captioning model with ResNet-152
model_resnet = AttentionCaptionModel(embed_size, hidden_size, vocab_size).to(device)

# Captioning model with InceptionV3
model_inception = AttentionCaptionModel(embed_size, hidden_size, vocab_size).to(device)
model_inception.cnn = models.inception_v3(pretrained=True, aux_logits=False)
model_inception.cnn.fc = nn.Linear(model_inception.cnn.fc.in_features, embed_size)

# Captioning model with MobileNetV2
model_mobilenet = AttentionCaptionModel(embed_size, hidden_size, vocab_size).to(device)
model_mobilenet.cnn = models.mobilenet_v2(pretrained=True).features
model_mobilenet.cnn.fc = nn.Linear(model_mobilenet.cnn.fc.in_features, embed_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<pad>'])

# Define the optimizer for each model
optimizer_resnet = torch.optim.Adam(model_resnet.parameters(), lr=0.001)
optimizer_inception = torch.optim.Adam(model_inception.parameters(), lr=0.001)
optimizer_mobilenet = torch.optim.Adam(model_mobilenet.parameters(), lr=0.001)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()
    for i, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = [[dataset.vocab['<start>']] + caption + [dataset.vocab['<end>']] for caption in captions]
        lengths = [len(caption) for caption in captions]
        targets = []
        for caption in captions:
            target = caption[1:]
            target.append(dataset.vocab['<pad>'])
            targets.append(target)

        captions = [caption[:max_caption_length] for caption in captions]
        targets = [target[:max_caption_length] for target in targets]
        captions = torch.LongTensor(captions).to(device)
        targets = torch.LongTensor(targets).to(device)

        # Update the ResNet-152 model
        optimizer_resnet.zero_grad()
        outputs_resnet, _ = model_resnet(images, captions, lengths)
        loss_resnet = criterion(outputs_resnet.reshape(-1, vocab_size), targets.reshape(-1))
        loss_resnet.backward()
        optimizer_resnet.step()

        # Update the InceptionV3 model
        optimizer_inception.zero_grad()
        outputs_inception, _ = model_inception(images, captions, lengths)
        loss_inception = criterion(outputs_inception.reshape(-1, vocab_size), targets.reshape(-1))
        loss_inception.backward()
        optimizer_inception.step()

        # Update the MobileNetV2 model
        optimizer_mobilenet.zero_grad()
        outputs_mobilenet, _ = model_mobilenet(images, captions, lengths)
        loss_mobilenet = criterion(outputs_mobilenet.reshape(-1, vocab_size), targets.reshape(-1))
        loss_mobilenet.backward()
        optimizer_mobilenet.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], '
                  f'Loss (ResNet-152): {loss_resnet.item():.4f}, '
                  f'Loss (InceptionV3): {loss_inception.item():.4f}, '
                  f'Loss (MobileNetV2): {loss_mobilenet.item():.4f}')

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s')

# Evaluation loop
with torch.no_grad():
    model_resnet.eval()
    model_inception.eval()
    model_mobilenet.eval()
    total_correct_resnet = 0
    total_correct_inception = 0
    total_correct_mobilenet = 0
    total_samples = 0
    for images, captions in test_loader:
        images = images.to(device)
        captions = [[dataset.vocab['<start>']] + caption + [dataset.vocab['<end>']] for caption in captions]
        lengths = [len(caption) for caption in captions]
        targets = []
        for caption in captions:
            target = caption[1:]
            target.append(dataset.vocab['<pad>'])
            targets.append(target)

        captions = [caption[:max_caption_length] for caption in captions]
        targets = [target[:max_caption_length] for target in targets]
        captions = torch.LongTensor(captions).to(device)
        targets = torch.LongTensor(targets).to(device)

        # ResNet-152 evaluation
        outputs_resnet, _ = model_resnet(images, captions, lengths)
        _, predicted_resnet = torch.max(outputs_resnet.data, 2)
        correct_resnet = (predicted_resnet == targets).sum().item()
        total_correct_resnet += correct_resnet

        # InceptionV3 evaluation
        outputs_inception, _ = model_inception(images, captions, lengths)
        _, predicted_inception = torch.max(outputs_inception.data, 2)
        correct_inception = (predicted_inception == targets).sum().item()
        total_correct_inception += correct_inception

        # MobileNetV2 evaluation
        outputs_mobilenet, _ = model_mobilenet(images, captions, lengths)
        _, predicted_mobilenet = torch.max(outputs_mobilenet.data, 2)
        correct_mobilenet = (predicted_mobilenet == targets).sum().item()
        total_correct_mobilenet += correct_mobilenet

        total_samples += targets.size(0) * targets.size(1)

    accuracy_resnet = total_correct_resnet / total_samples
    accuracy_inception = total_correct_inception / total_samples
    accuracy_mobilenet = total_correct_mobilenet / total_samples
    print(f'Accuracy on test set (ResNet-152): {accuracy_resnet * 100:.2f}%')
    print(f'Accuracy on test set (InceptionV3): {accuracy_inception * 100:.2f}%')
    print(f'Accuracy on test set (MobileNetV2): {accuracy_mobilenet * 100:.2f}%')