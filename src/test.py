import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout

# Set the paths to your image folder and CSV file
IMAGE_FOLDER = '../dataset/test_dir/'
CSV_FILE = '../dataset/test_captions.csv'

# Load image data from folder and create a dataframe with image filenames and captions
def load_image_data_from_csv(CSV_FILE, IMAGE_FOLDER):
    df = pd.read_csv(CSV_FILE)
    df['image_path'] = df['image_name'].apply(lambda x: f"{IMAGE_FOLDER}/{x}")
    return df

# Load and preprocess the images using VGG16 pre-processing
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        images.append(img)
    return np.vstack(images)

# Tokenize captions and create word-to-index and index-to-word dictionaries
def tokenize_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(captions)
    return tokenizer, vocab_size, sequences

# Generate input-output pairs for the RNN model
def create_input_output_sequences(sequences):
    input_sequences, output_sequences = [], []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[:i])
            output_sequences.append(sequence[i])
    return input_sequences, output_sequences

# Pad sequences to have the same length
def pad_sequence_sequences(sequences, max_length):
    return pad_sequences(sequences, maxlen=max_length, padding='post')

# Generate caption for given image using trained model
def generate_caption(image_path, model, tokenizer, max_length):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    feature_vector = model.predict(img)
    caption = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        next_word_index = np.argmax(model.predict([feature_vector, sequence]))
        next_word = index_to_word[next_word_index]
        caption += ' ' + next_word
        if next_word == 'endseq':
            break
    return caption

# Load and preprocess the image data
df = load_image_data_from_csv(CSV_FILE, IMAGE_FOLDER)
image_paths = df['image_path'].values
images = preprocess_images(image_paths)

# Tokenize captions and create word-to-index and index-to-word dictionaries
tokenizer, vocab_size, sequences = tokenize_captions(df['comment-fa'])
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# Create input-output sequences for the RNN model
input_sequences, output_sequences = create_input_output_sequences(sequences)
max_length = max([len(sequence) for sequence in input_sequences])

# Pad sequences to have the same length
input_sequences = pad_sequence_sequences(input_sequences, max_length)
output_sequences = np.array(output_sequences)
output_sequences = to_categorical(output_sequences, num_classes=vocab_size)

# Load VGG16 model and extract features
vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
vgg_model.trainable = False
image_input = Input(shape=(224, 224, 3))
image_features = vgg_model(image_input)
image_features = tf.keras.layers.Flatten()(image_features)
image_features = Dense(256, activation='relu')(image_features)

# LSTM model for caption generation
caption_input = Input(shape=(max_length,))
embedding_layer = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
lstm_layer = LSTM(256)(embedding_layer)
lstm_layer = Dropout(0.5)(lstm_layer)

decoder_output = Dense(vocab_size, activation='softmax')(lstm_layer)

# Combine the image and caption models
model = Model(inputs=[image_input, caption_input], outputs=decoder_output)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# # Train the model
model.fit([images, input_sequences], output_sequences, epochs=10, batch_size=32)
print("Done")

# # Save the model for later use
# model.save('image_caption_model.h5')

# # Example usage to generate captions for new images:
# new_image_path = 'path_to_new_image.jpg'
# generated_caption = generate_caption(new_image_path, model, tokenizer, max_length)
# print(generated_caption)
