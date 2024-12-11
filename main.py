import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
import logging
import faiss
import clip

# Initialize logging to track progress
logging.basicConfig(level=logging.INFO)

# Configure device for PyTorch operations (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom PyTorch Dataset class for image data
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess an image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Function to load image paths and corresponding labels
def load_images(image_dir, label):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg'))]
    labels = [label] * len(image_paths)
    return image_paths, labels

# Define a preprocessing pipeline for images
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for CLIP
    ])
    return transform(image)

# Function to set up FAISS index for fast similarity search
def setup_faiss(features):
    features = features.cpu().detach().numpy()
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / (norm + 1e-10)  # Avoid division by zero
    index = faiss.IndexFlatIP(features.shape[1])  # Inner product for cosine similarity
    index.add(normalized_features)
    return index

# Function to search FAISS index for nearest neighbors
def search_faiss(index, query_vector, k):
    query_vector = query_vector.cpu().detach().numpy()
    norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_vector = query_vector / (norm + 1e-10)
    D, I = index.search(query_vector, k)  # D: distances, I: indices
    return D, I

# Initialize the OpenAI CLIP model for feature extraction
logging.info("Initializing CLIP model...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Define paths to datasets
benign_dir = "Dataset_BUSI_with_GT/train/benign"
malignant_dir = "Dataset_BUSI_with_GT/train/malignant"
additional_data_path = "medmnist/breastmnist.npz"

# Load benign and malignant image datasets
logging.info("Loading datasets...")
benign_images, benign_labels = load_images(benign_dir, 0)  # Label 0 for benign
malignant_images, malignant_labels = load_images(malignant_dir, 1)  # Label 1 for malignant

# Load additional MedMNIST dataset
medmnist_data = np.load(additional_data_path)
additional_data = medmnist_data['train_images'].astype('float32') / 255.0  # Normalize images
additional_labels = medmnist_data['train_labels']
medmnist_data.close()

# Preprocess images and labels for additional data
logging.info("Preprocessing images...")
benign_dataset = CustomImageDataset(benign_images, benign_labels, transform=preprocess)
malignant_dataset = CustomImageDataset(malignant_images, malignant_labels, transform=preprocess)
add_images = [preprocess_image(Image.fromarray((img * 255).astype('uint8')).convert('RGB')) for img in additional_data]
add_labels = additional_labels.flatten()

# Encode images with CLIP to generate feature embeddings
logging.info("Encoding images using CLIP...")
with torch.no_grad():
    benign_embeddings = torch.stack([clip_model.encode_image(img.unsqueeze(0).to(device)) for img, _ in benign_dataset])
    malignant_embeddings = torch.stack([clip_model.encode_image(img.unsqueeze(0).to(device)) for img, _ in malignant_dataset])
    add_embeddings = torch.stack([clip_model.encode_image(img.unsqueeze(0).to(device)) for img in add_images]).to(device)

# Combine all embeddings and labels for training and testing
combined_images = torch.cat([benign_embeddings, malignant_embeddings, add_embeddings], dim=0)
combined_labels = torch.tensor(np.concatenate([benign_labels, malignant_labels, add_labels])).to(device)

# Set up FAISS index for image retrieval
logging.info("Setting up FAISS index...")
index = setup_faiss(combined_images)
query_vector = combined_images[0:1]  # Example query vector
k = 10  # Number of neighbors to retrieve
distances, indices = search_faiss(index, query_vector, k)
logging.info(f"Retrieved indices: {indices}")

# Train a Logistic Regression model on the combined dataset
logging.info("Training Logistic Regression model...")
X_train, X_test, y_train, y_test = train_test_split(combined_images.cpu(), combined_labels.cpu(), test_size=0.2, random_state=42)
X_train = X_train.numpy()
X_test = X_test.numpy()
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
logging.info(f"Logistic Regression Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, predictions))

# Function to display retrieved images
def show_images(images, title=None):
    plt.figure(figsize=(15, 15))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Display retrieved images from FAISS index
retrieved_images = [additional_data[i] for i in indices.flatten()]
show_images(retrieved_images, title="Retrieved Images")
