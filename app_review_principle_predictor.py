import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import os
import requests
import zipfile
import io

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(file_path):
    """
    Load the balanced dataset from the provided Excel file.
    
    Parameters:
    file_path (str): Path to the Excel file containing the balanced dataset
    
    Returns:
    pd.DataFrame: Loaded dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Print basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Total number of reviews: {len(df)}")
    
    return df

def preprocess_text(text):
    """
    Preprocess the text of app reviews.
    
    Parameters:
    text (str): Raw text from app review
    
    Returns:
    str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def multi_label_smote(df, labels, minority_classes, n_samples=1):
    """
    Apply a SMOTE-like approach for multi-label classification.
    Generate synthetic samples for minority classes.
    
    Parameters:
    df (DataFrame): Original dataset
    labels (list): List of label column names
    minority_classes (list): List of minority classes to oversample
    n_samples (int): Number of synthetic samples to generate for each minority instance
    
    Returns:
    DataFrame: Augmented dataset with synthetic samples
    """
    print(f"Applying ML-SMOTE for {len(minority_classes)} minority classes...")
    
    augmented_df = df.copy()
    
    # For each minority class
    for minority_class in minority_classes:
        if minority_class not in labels:
            print(f"Warning: {minority_class} not found in labels. Skipping.")
            continue
        
        # Get samples with this minority class
        minority_samples = df[df[minority_class] == 1]
        
        if len(minority_samples) == 0:
            print(f"Warning: No samples found for {minority_class}. Skipping.")
            continue
        
        print(f"  Class {minority_class}: Found {len(minority_samples)} samples, generating {len(minority_samples) * n_samples} synthetic samples")
        
        # Create synthetic samples
        synthetic_samples = []
        for _ in range(n_samples):
            # Sample with replacement from minority class instances
            synthetic = minority_samples.sample(len(minority_samples), replace=True).reset_index(drop=True)
            
            # Add small random noise to non-categorical features (assuming first column is text)
            text_column = synthetic.columns[0]
            
            # For text data, we can't add noise directly, but we can apply simple augmentations
            # like randomly removing some words or shuffling word order
            synthetic[text_column] = synthetic[text_column].apply(lambda x: simple_text_augmentation(x))
            
            synthetic_samples.append(synthetic)
        
        # Concatenate synthetic samples
        if synthetic_samples:
            synthetic_df = pd.concat(synthetic_samples, ignore_index=True)
            augmented_df = pd.concat([augmented_df, synthetic_df], ignore_index=True)
    
    print(f"Final dataset size after augmentation: {len(augmented_df)} samples")
    return augmented_df

def simple_text_augmentation(text):
    """
    Apply simple text augmentation techniques.
    
    Parameters:
    text (str): Original text
    
    Returns:
    str: Augmented text
    """
    if not isinstance(text, str):
        return text
    
    words = text.split()
    if len(words) <= 3:  # Don't augment very short texts
        return text
    
    # Randomly choose an augmentation method
    method = np.random.choice(['drop', 'shuffle', 'both'])
    
    if method == 'drop' or method == 'both':
        # Randomly drop 10-20% of words
        drop_ratio = np.random.uniform(0.1, 0.2)
        n_drop = max(1, int(len(words) * drop_ratio))
        drop_indices = np.random.choice(len(words), n_drop, replace=False)
        words = [w for i, w in enumerate(words) if i not in drop_indices]
    
    if method == 'shuffle' or method == 'both':
        # Shuffle word order (preserve some structure by shuffling chunks)
        chunk_size = max(2, len(words) // 3)
        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
        np.random.shuffle(chunks)
        words = [word for chunk in chunks for word in chunk]
    
    return ' '.join(words)

class ReviewVocabulary:
    """Enhanced vocabulary class with support for pre-trained embeddings."""
    def __init__(self, max_size=10000, embedding_dim=200, use_pretrained=True):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_count = Counter()
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained
        self.embeddings = None
        
        # Initialize pre-trained embeddings matrix
        if use_pretrained:
            # Check if embeddings file already exists
            self.pretrained_embeddings = check_embeddings_exist(embedding_dim)
            if not self.pretrained_embeddings:  # If not found locally, try downloading
                self.pretrained_embeddings = load_glove_embeddings(embedding_dim)
        else:
            self.pretrained_embeddings = {}
    
    def add_document(self, document):
        for word in document.split():
            self.word_count[word] += 1
    
    def build_vocabulary(self):
        # If using pre-trained embeddings, prioritize words that have embeddings
        if self.use_pretrained and self.pretrained_embeddings:
            # First add words that have pre-trained embeddings (by frequency)
            pretrained_words = set(self.pretrained_embeddings.keys())
            words_with_embeddings = [(word, count) for word, count in self.word_count.items() 
                                    if word in pretrained_words]
            words_with_embeddings.sort(key=lambda x: x[1], reverse=True)
            
            # Then add other frequent words
            other_words = [(word, count) for word, count in self.word_count.items() 
                          if word not in pretrained_words]
            other_words.sort(key=lambda x: x[1], reverse=True)
            
            # Combine and limit to max_size - 2 (PAD and UNK are already added)
            combined_words = words_with_embeddings + other_words
            combined_words = combined_words[:self.max_size - 2]
            
            for idx, (word, _) in enumerate(combined_words, 2):
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        else:
            # Sort by frequency (original implementation)
            most_common = self.word_count.most_common(self.max_size - 2)
            for idx, (word, _) in enumerate(most_common, 2):
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        # Create embeddings matrix
        self.create_embeddings_matrix()
    
    def create_embeddings_matrix(self):
        """Create embeddings matrix from pre-trained vectors or random initialization."""
        vocab_size = len(self.word2idx)
        
        # Initialize embedding matrix with random values
        self.embeddings = np.random.normal(scale=0.1, size=(vocab_size, self.embedding_dim))
        
        # Set padding token to zeros
        self.embeddings[0] = np.zeros(self.embedding_dim)
        
        # Fill in with pre-trained embeddings where available
        if self.use_pretrained and self.pretrained_embeddings:
            found = 0
            for word, idx in self.word2idx.items():
                if word in self.pretrained_embeddings:
                    self.embeddings[idx] = self.pretrained_embeddings[word]
                    found += 1
            
            print(f"Initialized {found}/{vocab_size} word vectors from pre-trained embeddings.")
        else:
            print(f"Using randomly initialized word vectors.")
    
    def get_embeddings(self):
        """Get the embeddings matrix as a tensor."""
        if self.embeddings is None:
            self.create_embeddings_matrix()
        return torch.FloatTensor(self.embeddings)
    
    def text_to_indices(self, text, max_length=200):
        words = text.split()
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Padding/truncating
        if len(indices) < max_length:
            indices = indices + [0] * (max_length - len(indices))  # Pad with 0
        else:
            indices = indices[:max_length]  # Truncate
        
        return indices
    
    def __len__(self):
        return len(self.word2idx)

def check_embeddings_exist(embedding_dim=200):
    """
    Check if GloVe embeddings already exist locally and load them if they do.
    
    Parameters:
    embedding_dim (int): Dimensionality of word embeddings
    
    Returns:
    dict: Dictionary of word embeddings or empty dict if not found
    """
    cache_dir = 'embeddings'
    embedding_file = os.path.join(cache_dir, f'glove.6B.{embedding_dim}d.txt')
    
    if os.path.exists(embedding_file):
        print(f"Found existing GloVe embeddings at {embedding_file}")
        embeddings = {}
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[word] = vector
            print(f"Loaded {len(embeddings)} word vectors from existing file.")
            return embeddings
        except Exception as e:
            print(f"Error loading existing embeddings: {e}")
            return {}
    else:
        print(f"No existing embeddings found at {embedding_file}")
        return {}

class ReviewDataset(Dataset):
    """Dataset class for app reviews"""
    def __init__(self, texts, labels_level1, labels_level2, labels_level3, vocab, max_length=200):
        self.texts = texts
        self.labels_level1 = labels_level1
        self.labels_level2 = labels_level2
        self.labels_level3 = labels_level3
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = self.vocab.text_to_indices(text, self.max_length)
        
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'labels_level1': torch.tensor(self.labels_level1[idx], dtype=torch.float),
            'labels_level2': torch.tensor(self.labels_level2[idx], dtype=torch.float),
            'labels_level3': torch.tensor(self.labels_level3[idx], dtype=torch.float)
        }

def prepare_hierarchical_data(df, hierarchy, review_text_column, oversample_minority=True, use_pretrained=True):
    """
    Prepare the data for hierarchical classification with improved handling of imbalanced data.
    
    Parameters:
    df (pd.DataFrame): Dataset containing app reviews and principle labels
    hierarchy (dict): Dictionary containing the hierarchical structure of principles
    review_text_column (str): Name of the column containing review text
    oversample_minority (bool): Whether to oversample minority classes
    use_pretrained (bool): Whether to use pre-trained word embeddings
    
    Returns:
    tuple: Training, validation, and test data, plus vocabulary and label encoders
    """
    # Preprocess text
    print("Preprocessing review text...")
    df['preprocessed_text'] = df[review_text_column].apply(preprocess_text)
    
    # Extract labels for Level 1 (Main Categories)
    level1_labels = []
    for _, row in df.iterrows():
        categories = []
        for category, principles in hierarchy["mainCategories"].items():
            if row[principles].any():
                categories.append(category)
        level1_labels.append(categories)
    
    # Extract labels for Level 2 (Sub-Categories)
    level2_labels = []
    for _, row in df.iterrows():
        subcategories = []
        for subcategory, principles in hierarchy["subCategories"].items():
            if row[principles].any():
                subcategories.append(subcategory)
        level2_labels.append(subcategories)
    
    # Extract labels for Level 3 (Individual Principles)
    level3_labels = []
    for _, row in df.iterrows():
        principles = []
        for category, category_principles in hierarchy["mainCategories"].items():
            for principle in category_principles:
                if row[principle]:
                    principles.append(principle.strip())
        level3_labels.append(principles)
    
    # Encode multi-label data
    mlb_level1 = MultiLabelBinarizer()
    mlb_level2 = MultiLabelBinarizer()
    mlb_level3 = MultiLabelBinarizer()
    
    y_level1 = mlb_level1.fit_transform(level1_labels)
    y_level2 = mlb_level2.fit_transform(level2_labels)
    y_level3 = mlb_level3.fit_transform(level3_labels)
    
    # Identify minority classes for oversampling
    if oversample_minority:
        # For each level, identify classes with fewer than threshold samples
        level2_counts = y_level2.sum(axis=0)
        level3_counts = y_level3.sum(axis=0)
        
        # Use 20th percentile as threshold for minority classes
        level2_threshold = np.percentile(level2_counts, 20)
        level3_threshold = np.percentile(level3_counts, 20)
        
        # Get minority class names
        level2_minority = [mlb_level2.classes_[i] for i, count in enumerate(level2_counts) if count < level2_threshold]
        level3_minority = [mlb_level3.classes_[i] for i, count in enumerate(level3_counts) if count < level3_threshold]
        
        # Map back to original column names
        level2_minority_cols = []
        for subcategory in level2_minority:
            for principles_list in hierarchy["subCategories"].values():
                if subcategory in principles_list:
                    level2_minority_cols.extend(principles_list)
        
        level3_minority_cols = []
        for principle in level3_minority:
            for original_col in df.columns:
                if original_col.strip() == principle:
                    level3_minority_cols.append(original_col)
        
        # Combine minority classes (remove duplicates)
        minority_classes = list(set(level2_minority_cols + level3_minority_cols))
        
        print(f"Identified {len(minority_classes)} minority classes for oversampling")
        print(f"Minority Level 2 categories: {level2_minority}")
        print(f"Minority Level 3 principles: {level3_minority}")
        
        # Apply oversampling
        original_size = len(df)
        df = multi_label_smote(df, df.columns, minority_classes, n_samples=2)
        print(f"Dataset size increased from {original_size} to {len(df)} samples after oversampling")
        
        # Recompute preprocessed text and labels
        df['preprocessed_text'] = df[review_text_column].apply(preprocess_text)
        
        # Recalculate labels after augmentation
        level1_labels = []
        level2_labels = []
        level3_labels = []
        
        for _, row in df.iterrows():
            # Level 1
            categories = []
            for category, principles in hierarchy["mainCategories"].items():
                if row[principles].any():
                    categories.append(category)
            level1_labels.append(categories)
            
            # Level 2
            subcategories = []
            for subcategory, principles in hierarchy["subCategories"].items():
                if row[principles].any():
                    subcategories.append(subcategory)
            level2_labels.append(subcategories)
            
            # Level 3
            principles = []
            for category, category_principles in hierarchy["mainCategories"].items():
                for principle in category_principles:
                    if row[principle]:
                        principles.append(principle.strip())
            level3_labels.append(principles)
        
        # Re-encode labels
        y_level1 = mlb_level1.transform(level1_labels)
        y_level2 = mlb_level2.transform(level2_labels)
        y_level3 = mlb_level3.transform(level3_labels)
    
    # Split data into train, validation, and test sets
    X = df['preprocessed_text'].values
    X_train, X_temp, y_train_level1, y_temp_level1 = train_test_split(X, y_level1, test_size=0.3, random_state=42)
    X_val, X_test, y_val_level1, y_test_level1 = train_test_split(X_temp, y_temp_level1, test_size=0.5, random_state=42)
    
    _, _, y_train_level2, y_temp_level2 = train_test_split(X, y_level2, test_size=0.3, random_state=42)
    _, _, y_val_level2, y_test_level2 = train_test_split(X_temp, y_temp_level2, test_size=0.5, random_state=42)
    
    _, _, y_train_level3, y_temp_level3 = train_test_split(X, y_level3, test_size=0.3, random_state=42)
    _, _, y_val_level3, y_test_level3 = train_test_split(X_temp, y_temp_level3, test_size=0.5, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create vocabulary with pre-trained embeddings
    print(f"Building vocabulary {'with pre-trained embeddings' if use_pretrained else 'without pre-trained embeddings'}...")
    vocab = ReviewVocabulary(max_size=15000, embedding_dim=200, use_pretrained=use_pretrained)
    for text in X_train:
        vocab.add_document(text)
    vocab.build_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = ReviewDataset(X_train, y_train_level1, y_train_level2, y_train_level3, vocab)
    val_dataset = ReviewDataset(X_val, y_val_level1, y_val_level2, y_val_level3, vocab)
    test_dataset = ReviewDataset(X_test, y_test_level1, y_test_level2, y_test_level3, vocab)
    
    return (train_dataset, val_dataset, test_dataset, 
            vocab, mlb_level1, mlb_level2, mlb_level3,
            y_test_level1, y_test_level2, y_test_level3)

class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on important parts of sequences."""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * x, dim=1)  # (batch_size, hidden_dim)
        return context_vector, attention_weights

class HierarchicalClassifier(nn.Module):
    """Enhanced hierarchical classification model with pre-trained embeddings and attention mechanism."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes_level1, num_classes_level2, num_classes_level3, 
                 dropout_rate=0.4, pretrained_embeddings=None, freeze_embeddings=False):
        super(HierarchicalClassifier, self).__init__()
        
        # Embedding layer (with pre-trained embeddings if provided)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        # Increased model capacity
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Use attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # Shared dense layers with increased capacity
        self.shared_dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Level-specific output layers
        self.level1_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes_level1),
            nn.Sigmoid()
        )
        
        self.level2_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes_level2),
            nn.Sigmoid()
        )
        
        self.level3_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes_level3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Apply attention
        x, _ = self.attention(x)
        
        # Shared features
        shared_features = self.shared_dense(x)
        
        # Level-specific outputs
        level1 = self.level1_output(shared_features)
        level2 = self.level2_output(shared_features)
        level3 = self.level3_output(shared_features)
        
        return level1, level2, level3

def train_model(model, train_loader, val_loader, device, epochs=30, patience=5):
    """
    Train the hierarchical model with improved techniques for handling imbalanced data.
    
    Parameters:
    model: The model to train
    train_loader, val_loader: DataLoader objects for training and validation
    device: Device to run the model on (cpu or cuda)
    epochs: Number of training epochs
    patience: Number of epochs to wait for improvement before early stopping
    
    Returns:
    dict: Training history
    """
    # Move model to device
    model = model.to(device)
    
    # Calculate class weights for each level from the training data
    y_train_level1 = torch.cat([batch['labels_level1'] for batch in train_loader], dim=0)
    y_train_level2 = torch.cat([batch['labels_level2'] for batch in train_loader], dim=0)
    y_train_level3 = torch.cat([batch['labels_level3'] for batch in train_loader], dim=0)
    
    weight_level1 = calculate_class_weights(y_train_level1.cpu().numpy())
    weight_level2 = calculate_class_weights(y_train_level2.cpu().numpy())
    weight_level3 = calculate_class_weights(y_train_level3.cpu().numpy())
    
    print(f"Class weights Level 1: {weight_level1}")
    print(f"Class weights Level 2: {weight_level2}")
    print(f"Class weights Level 3: {weight_level3}")
    
    # Weighted loss functions for each level
    criterion_level1 = nn.BCEWithLogitsLoss(pos_weight=weight_level1.to(device))
    criterion_level2 = nn.BCEWithLogitsLoss(pos_weight=weight_level2.to(device))
    criterion_level3 = nn.BCEWithLogitsLoss(pos_weight=weight_level3.to(device))
    
    # Alternative: Use focal loss for severe imbalance
    # focal_loss = FocalLoss(gamma=2.0)
    
    # Optimizer with reduced learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Learning rate scheduler - removed 'verbose' parameter to avoid warning
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'level1_train_loss': [],
        'level2_train_loss': [],
        'level3_train_loss': [],
        'level1_val_loss': [],
        'level2_val_loss': [],
        'level3_val_loss': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        level1_loss, level2_loss, level3_loss = 0.0, 0.0, 0.0
        
        for batch in train_loader:
            # Move batch to device
            texts = batch['text'].to(device)
            labels_level1 = batch['labels_level1'].to(device)
            labels_level2 = batch['labels_level2'].to(device)
            labels_level3 = batch['labels_level3'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs_level1, outputs_level2, outputs_level3 = model(texts)
            
            # Calculate weighted loss for each level
            loss_level1 = criterion_level1(outputs_level1, labels_level1)
            loss_level2 = criterion_level2(outputs_level2, labels_level2)
            loss_level3 = criterion_level3(outputs_level3, labels_level3)
            
            # Weight the losses based on importance - emphasize levels 2 and 3
            combined_loss = loss_level1 + 1.5 * loss_level2 + 2.0 * loss_level3
            
            # Backward pass and optimization
            combined_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            train_loss += combined_loss.item()
            level1_loss += loss_level1.item()
            level2_loss += loss_level2.item()
            level3_loss += loss_level3.item()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        level1_loss /= len(train_loader)
        level2_loss /= len(train_loader)
        level3_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_level1_loss, val_level2_loss, val_level3_loss = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                texts = batch['text'].to(device)
                labels_level1 = batch['labels_level1'].to(device)
                labels_level2 = batch['labels_level2'].to(device)
                labels_level3 = batch['labels_level3'].to(device)
                
                # Forward pass
                outputs_level1, outputs_level2, outputs_level3 = model(texts)
                
                # Calculate weighted loss
                loss_level1 = criterion_level1(outputs_level1, labels_level1)
                loss_level2 = criterion_level2(outputs_level2, labels_level2)
                loss_level3 = criterion_level3(outputs_level3, labels_level3)
                
                # Combined loss
                combined_loss = loss_level1 + 1.5 * loss_level2 + 2.0 * loss_level3
                
                # Update statistics
                val_loss += combined_loss.item()
                val_level1_loss += loss_level1.item()
                val_level2_loss += loss_level2.item()
                val_level3_loss += loss_level3.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_level1_loss /= len(val_loader)
        val_level2_loss /= len(val_loader)
        val_level3_loss /= len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"L1: {level1_loss:.4f}/{val_level1_loss:.4f} - "
              f"L2: {level2_loss:.4f}/{val_level2_loss:.4f} - "
              f"L3: {level3_loss:.4f}/{val_level3_loss:.4f} - "
              f"LR: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['level1_train_loss'].append(level1_loss)
        history['level2_train_loss'].append(level2_loss)
        history['level3_train_loss'].append(level3_loss)
        history['level1_val_loss'].append(val_level1_loss)
        history['level2_val_loss'].append(val_level2_loss)
        history['level3_val_loss'].append(val_level3_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model
                model.load_state_dict(best_model_state)
                break
    
    # If training completed without early stopping, ensure we use the best model
    if best_model_state is not None and no_improvement_count < patience:
        model.load_state_dict(best_model_state)
    
    return history

def evaluate_model(model, test_loader, mlb_level1, mlb_level2, mlb_level3, device, y_test_level1, y_test_level2, y_test_level3):
    """
    Evaluate the model on test data.
    
    Parameters:
    model: Trained model
    test_loader: DataLoader for test data
    mlb_level*: MultiLabelBinarizer objects for each level
    device: Device to run the model on
    y_test_level*: Test labels for each level
    
    Returns:
    dict: Evaluation metrics
    """
    model.eval()
    
    # Predictions
    all_preds_level1 = []
    all_preds_level2 = []
    all_preds_level3 = []
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            
            # Forward pass
            outputs_level1, outputs_level2, outputs_level3 = model(texts)
            
            # Convert to binary predictions
            preds_level1 = (outputs_level1 > 0.5).float().cpu().numpy()
            preds_level2 = (outputs_level2 > 0.5).float().cpu().numpy()
            preds_level3 = (outputs_level3 > 0.5).float().cpu().numpy()
            
            all_preds_level1.extend(preds_level1)
            all_preds_level2.extend(preds_level2)
            all_preds_level3.extend(preds_level3)
    
    # Convert to numpy arrays
    all_preds_level1 = np.array(all_preds_level1)
    all_preds_level2 = np.array(all_preds_level2)
    all_preds_level3 = np.array(all_preds_level3)
    
    # Calculate metrics
    metrics = {}
    
    # Level 1 metrics
    metrics['level1'] = {
        'f1_micro': f1_score(y_test_level1, all_preds_level1, average='micro', zero_division=0),
        'f1_macro': f1_score(y_test_level1, all_preds_level1, average='macro', zero_division=0),
        'precision_micro': precision_score(y_test_level1, all_preds_level1, average='micro', zero_division=0),
        'precision_macro': precision_score(y_test_level1, all_preds_level1, average='macro', zero_division=0),
        'recall_micro': recall_score(y_test_level1, all_preds_level1, average='micro', zero_division=0),
        'recall_macro': recall_score(y_test_level1, all_preds_level1, average='macro', zero_division=0)
    }
    
    # Level 2 metrics
    metrics['level2'] = {
        'f1_micro': f1_score(y_test_level2, all_preds_level2, average='micro', zero_division=0),
        'f1_macro': f1_score(y_test_level2, all_preds_level2, average='macro', zero_division=0),
        'precision_micro': precision_score(y_test_level2, all_preds_level2, average='micro', zero_division=0),
        'precision_macro': precision_score(y_test_level2, all_preds_level2, average='macro', zero_division=0),
        'recall_micro': recall_score(y_test_level2, all_preds_level2, average='micro', zero_division=0),
        'recall_macro': recall_score(y_test_level2, all_preds_level2, average='macro', zero_division=0)
    }
    
    # Level 3 metrics
    metrics['level3'] = {
        'f1_micro': f1_score(y_test_level3, all_preds_level3, average='micro', zero_division=0),
        'f1_macro': f1_score(y_test_level3, all_preds_level3, average='macro', zero_division=0),
        'precision_micro': precision_score(y_test_level3, all_preds_level3, average='micro', zero_division=0),
        'precision_macro': precision_score(y_test_level3, all_preds_level3, average='macro', zero_division=0),
        'recall_micro': recall_score(y_test_level3, all_preds_level3, average='micro', zero_division=0),
        'recall_macro': recall_score(y_test_level3, all_preds_level3, average='macro', zero_division=0)
    }
    
    # Print classification reports
    print("\nLevel 1 Classification Report:")
    print(classification_report(y_test_level1, all_preds_level1, target_names=mlb_level1.classes_, zero_division=0))
    
    print("\nLevel 2 Classification Report:")
    print(classification_report(y_test_level2, all_preds_level2, target_names=mlb_level2.classes_, zero_division=0))
    
    print("\nLevel 3 Classification Report:")
    print(classification_report(y_test_level3, all_preds_level3, target_names=mlb_level3.classes_, zero_division=0))
    
    return metrics

def plot_training_history(history):
    """
    Plot training and validation metrics over epochs.
    
    Parameters:
    history: Training history
    """
    plt.figure(figsize=(15, 10))
    
    # Overall loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Overall Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Level 1 loss
    plt.subplot(2, 2, 2)
    plt.plot(history['level1_train_loss'], label='Train Loss')
    plt.plot(history['level1_val_loss'], label='Val Loss')
    plt.title('Level 1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Level 2 loss
    plt.subplot(2, 2, 3)
    plt.plot(history['level2_train_loss'], label='Train Loss')
    plt.plot(history['level2_val_loss'], label='Val Loss')
    plt.title('Level 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Level 3 loss
    plt.subplot(2, 2, 4)
    plt.plot(history['level3_train_loss'], label='Train Loss')
    plt.plot(history['level3_val_loss'], label='Val Loss')
    plt.title('Level 3 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

def save_model_and_components(model, vocab, mlb_level1, mlb_level2, mlb_level3):
    """
    Save the trained model and components.
    
    Parameters:
    model: Trained model
    vocab: Vocabulary object
    mlb_level*: MultiLabelBinarizer objects for each level
    """
    import pickle
    
    # Save model
    torch.save(model.state_dict(), 'app_review_principle_model.pt')
    print("Model saved as 'app_review_principle_model.pt'")
    
    # Save vocabulary and label encoders
    with open('model_components.pkl', 'wb') as f:
        pickle.dump({
            'vocab': vocab,
            'mlb_level1': mlb_level1,
            'mlb_level2': mlb_level2,
            'mlb_level3': mlb_level3
        }, f)
    
    print("Model components saved as 'model_components.pkl'")

def predict_principles_for_new_review(model, vocab, mlb_level1, mlb_level2, mlb_level3, review_text, device):
    """
    Predict principles for a new app review.
    
    Parameters:
    model: Trained model
    vocab: Vocabulary object
    mlb_level*: MultiLabelBinarizer objects for each level
    review_text: Text of the new app review
    device: Device to run the model on
    
    Returns:
    dict: Predicted principles at each level
    """
    # Preprocess the text
    preprocessed_text = preprocess_text(review_text)
    
    # Convert to indices
    indices = vocab.text_to_indices(preprocessed_text)
    
    # Convert to tensor and add batch dimension
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        preds_level1, preds_level2, preds_level3 = model(tensor)
    
    # Convert to binary predictions
    binary_level1 = (preds_level1 > 0.5).float().cpu().numpy()
    binary_level2 = (preds_level2 > 0.5).float().cpu().numpy()
    binary_level3 = (preds_level3 > 0.5).float().cpu().numpy()
    
    # Convert binary predictions to labels
    labels_level1 = mlb_level1.inverse_transform(binary_level1)
    labels_level2 = mlb_level2.inverse_transform(binary_level2)
    labels_level3 = mlb_level3.inverse_transform(binary_level3)
    
    # Create result dictionary
    result = {
        'level1': labels_level1[0],
        'level2': labels_level2[0],
        'level3': labels_level3[0]
    }
    
    return result

class FocalLoss(nn.Module):
    """Focal Loss for handling imbalanced classes."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = 1e-7
        
    def forward(self, inputs, targets):
        # Binary cross entropy loss
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        
        # Focal loss weighting
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
            
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
def calculate_class_weights(labels):
    """Calculate class weights based on inverse frequency for handling imbalanced data."""
    pos_counts = np.sum(labels, axis=0)
    neg_counts = labels.shape[0] - pos_counts
    
    # Ensure no division by zero (add small constant)
    pos_weights = np.where(pos_counts > 0, 
                          neg_counts / (pos_counts + 1e-5), 
                          1.0)
    
    # Clip weights to prevent extreme values
    pos_weights = np.clip(pos_weights, 0.1, 10.0)
    
    return torch.tensor(pos_weights, dtype=torch.float32)

def main():
    """Main function to run the app review principle prediction with all enhancements."""
    # Define the hierarchical structure
    hierarchy = {
        # Level 1: Main categories
        "mainCategories": {
            "Usability": ["Learnability ", "Memorability ", "User Error Protection", 
                         "Operability", "Accessability", "Satisfaction", "Efficiency", "Effectiveness"],
            "Security": ["Confidentiality ", "Integrity ", "Availability ", "Authenticity ", 
                        "Accountability", "Non repudation", "Traceability", "Authorization", "Resiliance "]
        },
        
        # Level 2: Sub-categories
        "subCategories": {
            "Interaction": ["Learnability ", "Memorability ", "Operability"],
            "User Experience": ["Satisfaction", "Accessability"],
            "Performance": ["Efficiency", "Effectiveness"],
            "Error Handling": ["User Error Protection"],
            "Data Protection": ["Confidentiality ", "Integrity ", "Authenticity "],
            "Access Control": ["Authorization"],
            "System Health": ["Availability ", "Resiliance "],
            "Accountability": ["Accountability", "Non repudation", "Traceability"]
        }
    }
    
    # Define improved parameters
    embedding_dim = 200  # GloVe dimension
    hidden_dim = 256
    batch_size = 16
    epochs = 30
    dropout_rate = 0.4
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create embeddings directory if it doesn't exist
    cache_dir = 'embeddings'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Instructions for manual download
    manual_download_instructions = """
    If the automatic download fails, you can manually download the GloVe embeddings:
    
    1. Download from: http://nlp.stanford.edu/data/glove.6B.zip
    2. Extract the zip file
    3. Copy the file 'glove.6B.200d.txt' to the 'embeddings' folder in your project directory
    4. Restart the script
    """
    
    # Load balanced dataset (use level 2 balanced dataset for best results)
    try:
        file_path = "balanced_level2.xlsx"
        df = load_data(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find the dataset file '{file_path}'.")
        print("Please make sure the file exists in the current directory.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Print available columns to help identify the text column
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Try to identify a text column
    text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                   ['text', 'review', 'comment', 'feedback', 'description'])]
    
    if text_columns:
        print(f"\nPotential text columns found: {text_columns}")
        review_text_column = text_columns[0]  # Use the first one found
        print(f"Using '{review_text_column}' as the review text column")
    else:
        # Use a non-principle column with likely string values as a fallback
        principle_columns = list(hierarchy["mainCategories"]["Usability"] + hierarchy["mainCategories"]["Security"])
        non_principle_columns = [col for col in df.columns if col not in principle_columns]
        
        if non_principle_columns:
            review_text_column = non_principle_columns[0]
            print(f"\nNo obvious text column found. Using '{review_text_column}' as a fallback.")
        else:
            raise ValueError("Could not identify a suitable text column in the dataset.")
    
    # Check if the selected column contains text data by examining a sample
    sample = df[review_text_column].iloc[0] if not df[review_text_column].empty else ""
    print(f"\nSample from selected column: {sample[:100]}...")
    
    # Analyze class distribution before balancing
    print("\nAnalyzing class distribution in the dataset...")
    analyze_class_distribution(df, hierarchy)
    
    # Determine whether to use pre-trained embeddings
    embedding_file = os.path.join(cache_dir, f'glove.6B.{embedding_dim}d.txt')
    embeddings_exist = os.path.exists(embedding_file)
    
    use_pretrained = True
    if not embeddings_exist:
        try:
            # Check if we can access the embeddings URL
            test_connection = requests.head("http://nlp.stanford.edu/data/", timeout=5)
            if test_connection.status_code != 200:
                print("\nWarning: Could not connect to Stanford NLP server. Will run without pre-trained embeddings.")
                print(manual_download_instructions)
                use_pretrained = False
        except:
            print("\nWarning: Could not connect to Stanford NLP server. Will run without pre-trained embeddings.")
            print(manual_download_instructions)
            use_pretrained = False
    
    # Ask user if they want to use pre-trained embeddings
    if embeddings_exist:
        prompt = f"\nUse existing pre-trained word embeddings found in '{embedding_file}'? (yes/no, default: yes): "
    else:
        prompt = f"\nDownload and use pre-trained word embeddings? This requires ~800MB of data. (yes/no, default: {use_pretrained}): "
    
    user_choice = input(prompt)
    if user_choice.lower() in ['n', 'no']:
        use_pretrained = False
    elif user_choice.lower() in ['y', 'yes'] or user_choice.strip() == '':
        use_pretrained = True
    
    print(f"Running with {'pre-trained' if use_pretrained else 'random'} word embeddings.\n")
    
    # Prepare data with oversampling for minority classes
    try:
        (train_dataset, val_dataset, test_dataset, 
        vocab, mlb_level1, mlb_level2, mlb_level3,
        y_test_level1, y_test_level2, y_test_level3) = prepare_hierarchical_data(
            df, hierarchy, review_text_column, oversample_minority=True, use_pretrained=use_pretrained)
    except Exception as e:
        print(f"Error preparing data: {e}")
        if use_pretrained:
            print("\nError occurred with pre-trained embeddings. Would you like to try again without them? (yes/no): ")
            retry_choice = input()
            if retry_choice.lower() in ['y', 'yes']:
                print("Retrying without pre-trained embeddings...")
                (train_dataset, val_dataset, test_dataset, 
                vocab, mlb_level1, mlb_level2, mlb_level3,
                y_test_level1, y_test_level2, y_test_level3) = prepare_hierarchical_data(
                    df, hierarchy, review_text_column, oversample_minority=True, use_pretrained=False)
            else:
                print("Exiting. Please try again later or follow the manual download instructions:")
                print(manual_download_instructions)
                return
        else:
            print("Exiting due to error.")
            return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get pre-trained embeddings from vocabulary
    pretrained_embeddings = vocab.get_embeddings()
    
    # Create enhanced model with pre-trained embeddings
    model = HierarchicalClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes_level1=len(mlb_level1.classes_),
        num_classes_level2=len(mlb_level2.classes_),
        num_classes_level3=len(mlb_level3.classes_),
        dropout_rate=dropout_rate,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=False  # Set to True to freeze embeddings
    )
    
    # Print model summary
    print(model)
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model with improved training function
    history = train_model(model, train_loader, val_loader, device, epochs=epochs)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, mlb_level1, mlb_level2, mlb_level3, device, 
                            y_test_level1, y_test_level2, y_test_level3)
    
    # Save model and components
    save_model_and_components(model, vocab, mlb_level1, mlb_level2, mlb_level3)
    
    # Example of predicting principles for a new review
    new_review = "The app is very intuitive and easy to learn, but it crashed several times when I tried to log in."
    predictions = predict_principles_for_new_review(model, vocab, mlb_level1, mlb_level2, mlb_level3, new_review, device)
    
    print("\nPredictions for new review:")
    print(f"Review: {new_review}")
    print(f"Level 1 (Main Categories): {predictions['level1']}")
    print(f"Level 2 (Sub-Categories): {predictions['level2']}")
    print(f"Level 3 (Individual Principles): {predictions['level3']}")

def analyze_class_distribution(df, hierarchy):
    """Analyze and print class distribution information."""
    # Level 1 distribution
    print("\nLevel 1 (Main Categories) distribution:")
    for category, principles in hierarchy["mainCategories"].items():
        count = df[df[principles].any(axis=1)].shape[0]
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} samples ({percentage:.2f}%)")
    
    # Level 2 distribution
    print("\nLevel 2 (Sub-Categories) distribution:")
    for subcategory, principles in hierarchy["subCategories"].items():
        count = df[df[principles].any(axis=1)].shape[0]
        percentage = (count / len(df)) * 100
        print(f"  {subcategory}: {count} samples ({percentage:.2f}%)")
    
    # Level 3 distribution (individual principles)
    print("\nLevel 3 (Individual Principles) distribution:")
    for category, principles in hierarchy["mainCategories"].items():
        print(f"\n  {category} principles:")
        for principle in principles:
            if principle in df.columns:
                count = df[principle].sum()
                percentage = (count / len(df)) * 100
                print(f"    {principle.strip()}: {count} samples ({percentage:.2f}%)")
            else:
                print(f"    {principle.strip()}: Column not found")

# Function to download and load GloVe embeddings
def load_glove_embeddings(embedding_dim=200):
    """
    Download and load GloVe word embeddings.
    
    Parameters:
    embedding_dim (int): Dimensionality of word embeddings (50, 100, 200, or 300)
    
    Returns:
    dict: Dictionary mapping words to embedding vectors
    """
    valid_dims = [50, 100, 200, 300]
    if embedding_dim not in valid_dims:
        print(f"Warning: {embedding_dim} is not a standard GloVe dimension. Using 200 instead.")
        embedding_dim = 200
    
    cache_dir = 'embeddings'
    os.makedirs(cache_dir, exist_ok=True)
    
    embedding_file = os.path.join(cache_dir, f'glove.6B.{embedding_dim}d.txt')
    
    # Check if file already exists
    if not os.path.exists(embedding_file):
        print(f"Downloading GloVe embeddings ({embedding_dim}d)...")
        
        # URL for GloVe embeddings
        url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        
        try:
            # Download with progress feedback and timeout
            print("This may take a few minutes. Please be patient...")
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()  # Raise exception for HTTP errors
            
            # Get file size if available
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            # Create a temporary file to store the zip
            zip_path = os.path.join(cache_dir, 'glove.6B.zip')
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Show progress
                        if total_size > 0:
                            percent = downloaded / total_size * 100
                            print(f"\rDownloading: {percent:.1f}% ({downloaded} / {total_size} bytes)", end='')
            
            print("\nExtracting embeddings...")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(cache_dir)
            
            # Remove the zip file after extraction
            os.remove(zip_path)
            print("GloVe embeddings downloaded and extracted successfully.")
            
        except requests.exceptions.Timeout:
            print("Error: Download timed out. Please try again later or manually download the embeddings.")
            print("You can download from: http://nlp.stanford.edu/data/glove.6B.zip")
            print("Extract the file and place glove.6B.{}d.txt in the 'embeddings' folder.".format(embedding_dim))
            return {}
        except requests.exceptions.RequestException as e:
            print(f"Error downloading GloVe embeddings: {e}")
            print("Proceeding without pre-trained embeddings.")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Proceeding without pre-trained embeddings.")
            return {}
    
    # Load embeddings
    print(f"Loading GloVe embeddings from {embedding_file}...")
    embeddings = {}
    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
        print(f"Loaded {len(embeddings)} word vectors.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}
    
    return embeddings

if __name__ == "__main__":
    main() 