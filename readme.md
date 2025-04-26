# App Review Principle Predictor: A Hierarchical Multi-Label Classification Framework

## 1. Introduction and Overview

This repository implements a sophisticated machine learning framework for the hierarchical multi-label classification of mobile app reviews based on usability and security principles. The system employs advanced deep learning techniques centered on a bidirectional LSTM architecture enhanced with attention mechanisms to effectively classify app reviews across three hierarchical levels of principle categorization.

The framework addresses several key challenges in automated app review analysis:
- **Multi-label classification**: App reviews often address multiple principles simultaneously, requiring a model that can predict multiple labels per review.
- **Hierarchical structure**: Usability and security principles naturally organize into hierarchical structures that should be preserved in the classification process.
- **Class imbalance**: Real-world app review datasets exhibit significant imbalance, with some principles appearing much more frequently than others.
- **Text representation**: Converting unstructured text reviews into meaningful feature representations that capture semantic content.

Our technical approach employs:
- Advanced text preprocessing with lemmatization and stopword removal
- Custom-designed hierarchical balancing strategies
- Pre-trained word embeddings (GloVe) for enhanced text representation
- Bidirectional LSTM layers with attention mechanisms for sequence modeling
- Specialized loss functions and weighting schemes for imbalanced classification
- Comprehensive evaluation metrics suitable for multi-label hierarchical classification

This README provides detailed technical documentation of the implementation, algorithmic choices, and theoretical foundations of the approach.

## 2. Background & Research Motivation

### 2.1 Challenges in App Review Analysis

Mobile app reviews represent a valuable source of user feedback and insights into user experiences, expectations, and concerns. These reviews often contain critical information about usability issues, security vulnerabilities, feature requests, and satisfaction levels. However, manually analyzing these reviews to extract usability and security concerns is:

- **Time-consuming**: Human analysts cannot feasibly process thousands of reviews.
- **Subjective**: Manual categorization varies between analysts.
- **Difficult to scale**: The volume of app reviews grows continuously.
- **Inconsistent**: Manual categorization may lack standardized criteria.

### 2.2 Significance of the Hierarchical Multi-Label Approach

Our research addresses these challenges by developing an automated system that categorizes app reviews according to established usability and security principles using a hierarchical multi-label classification approach. This approach is particularly significant for several reasons:

1. **Multi-label representation**: App reviews frequently address multiple principles simultaneously. For example, a single review might mention both the "learnability" of an interface and a "confidentiality" concern. Traditional single-label classification would force an artificial choice between these aspects.

2. **Hierarchical structure**: Usability and security principles naturally organize into hierarchical taxonomies, with general categories (e.g., "Security") containing more specific sub-categories (e.g., "Data Protection"), which further contain individual principles (e.g., "Confidentiality"). This structure allows for analysis at different levels of granularity.

3. **Class imbalance mitigation**: Real-world app review datasets exhibit significant imbalance, with some principles (e.g., "Satisfaction") appearing much more frequently than others (e.g., "Non-repudiation"). A hierarchical approach allows for more effective balancing strategies across different levels.

4. **Improved model performance**: By modeling the hierarchical relationships between principles, the model can leverage information from higher levels to improve prediction performance at more granular levels.

### 2.3 Research Contributions

This work makes several contributions to the fields of:

- **Natural Language Processing for Software Engineering**: Demonstrating the effectiveness of deep learning approaches for automated analysis of user feedback in software engineering contexts.

- **Automated Requirements Engineering**: Providing a tool for extracting structured requirements and concerns from unstructured user feedback.

- **User Feedback Analysis**: Developing methods to systematically categorize and quantify user concerns expressed in natural language.

- **Hierarchical Multi-Label Classification with Imbalanced Data**: Advancing methodological approaches for machine learning with hierarchically structured, imbalanced, multi-label data.

The significance extends beyond the research contributions, offering practical solutions for app developers to efficiently process large volumes of user feedback and prioritize improvements based on quantifiable data about usability and security concerns.

## 3. Dataset Description and Characteristics

### 3.1 Dataset Composition and Structure

The model is trained on a specialized dataset of mobile app reviews that have been meticulously labeled according to a structured hierarchy of usability and security principles. Each review in the dataset is annotated with binary indicators for the presence or absence of each principle at three hierarchical levels.

The raw dataset characteristics include:

- **Source**: Reviews collected from popular mobile application platforms (iOS App Store and Google Play Store)
- **Format**: Tabular data (Excel format) with reviews as text and binary labels for principles
- **Size**: Several thousand labeled reviews with varying length and complexity
- **Language**: English-language reviews with diverse linguistic patterns and domain-specific terminology

### 3.2 Dataset Challenges and Properties

The dataset exhibits several challenging properties that the modeling approach must address:

1. **Multi-label nature**: Each review can be associated with multiple principles simultaneously, ranging from none to several principles across different categories. This requires specialized classification techniques beyond traditional single-label approaches.

2. **Hierarchical label structure**: The principles are organized in a three-level taxonomy (main categories → sub-categories → individual principles), with dependencies and relationships between levels that should be preserved in the modeling.

3. **Significant class imbalance**: The dataset exhibits extreme imbalance, with some principles appearing hundreds of times more frequently than others. For example:
   - Common principles (e.g., "Satisfaction", "Learnability") may appear in 20-30% of reviews
   - Rare principles (e.g., "Traceability", "Non-repudiation") may appear in less than 1% of reviews

4. **Textual characteristics**:
   - Variable length (from single sentences to paragraphs)
   - Informal language with abbreviations, slang, and typographical errors
   - App-specific terminology and references
   - Diverse expression of similar concerns (vocabulary inconsistency)

5. **Annotation complexity**: The manual annotation process requires domain expertise in both usability and security principles, leading to potential subjective variations in the labeling.

### 3.3 Data Exploration and Analysis

Prior to model development, we conduct extensive exploratory data analysis to understand the dataset characteristics:

- Distribution analysis of principles at each hierarchical level
- Co-occurrence patterns between principles
- Text length and complexity metrics
- Vocabulary analysis and domain-specific terminology
- Class imbalance quantification (imbalance ratios)

This analysis, implemented in the `hierarchical_balance_analysis.py` module, provides visualizations of class distributions and quantitative metrics of imbalance that inform our subsequent balancing strategies.

## 4. Hierarchical Structure of Principles

### 4.1 Three-Level Taxonomy

The classification framework is structured around a comprehensive three-level taxonomy of usability and security principles derived from established standards in human-computer interaction and security engineering. This hierarchical organization allows for analyzing reviews at different levels of granularity.

The complete taxonomy is implemented in the code as hierarchical dictionaries:

```python
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
```

### 4.2 Level 1: Main Categories

At the highest level, principles are divided into two fundamental categories:

1. **Usability**: Encompassing principles focused on the quality of user interaction and experience with the application. These principles are derived from established usability frameworks such as ISO 9241-11 and Nielsen's heuristics, adapted for mobile application contexts.

2. **Security**: Encompassing principles related to the protection of data, system integrity, and user privacy. These principles are based on established security frameworks such as ISO 27001 and OWASP mobile security principles.

This top-level division provides a broad categorization that separates user-centered design concerns from security and protection concerns.

### 4.3 Level 2: Sub-Categories

Each main category is further divided into more specific sub-categories:

**Usability Sub-Categories:**
- **Interaction**: Principles concerning how users interact with the application interface
- **User Experience**: Principles related to subjective user satisfaction and accessibility
- **Performance**: Principles focused on application efficiency and effectiveness
- **Error Handling**: Principles addressing error prevention and recovery

**Security Sub-Categories:**
- **Data Protection**: Principles concerning the safeguarding of user and system data
- **Access Control**: Principles addressing authentication and authorization mechanisms
- **System Health**: Principles related to availability and resilience of the application
- **Accountability**: Principles focused on tracking and attributing system and user actions

These sub-categories provide an intermediate level of granularity that groups related principles together for more coherent analysis.

### 4.4 Level 3: Individual Principles

At the most granular level, specific principles define particular aspects of usability or security:

**Usability Principles:**
- **Learnability**: How easily users can learn to use the application
- **Memorability**: How easily users can remember how to use the application after a period of non-use
- **Operability**: How easily users can operate and control the application
- **Satisfaction**: The degree of user satisfaction with the application
- **Accessibility**: How accessible the application is to users with different abilities
- **Efficiency**: How efficiently users can perform tasks
- **Effectiveness**: How effectively users can achieve their goals
- **User Error Protection**: How well the application prevents and helps users recover from errors

**Security Principles:**
- **Confidentiality**: Protection of data from unauthorized access
- **Integrity**: Ensuring data remains accurate and unmodified
- **Availability**: Ensuring system resources are available when needed
- **Authenticity**: Verification of identity and origin of data or users
- **Accountability**: Ability to trace actions to responsible entities
- **Non-repudiation**: Prevention of denial of having performed an action
- **Traceability**: Ability to track actions and events within the system
- **Authorization**: Granting appropriate access rights to authenticated users
- **Resilience**: Ability of the system to withstand adverse conditions and attacks

### 4.5 Hierarchical Relationships and Dependencies

The hierarchical structure encodes important relationships between principles:

1. **Part-whole relationships**: Each individual principle is part of a broader sub-category, which in turn is part of a main category. This encodes domain knowledge about how principles relate to each other.

2. **Inheritance relationships**: Lower-level principles inherit properties and characteristics from their parent categories, which can be leveraged for improved classification.

3. **Co-occurrence patterns**: Principles within the same sub-category tend to co-occur more frequently, creating natural correlations that can be exploited by the model.

This hierarchical structure not only organizes the classification space in a meaningful way but also provides valuable information that can be used to improve model performance through hierarchical learning approaches.

## 5. Methodological Approach

### 5.1 Data Preprocessing and Text Representation

Text preprocessing is a critical component of the pipeline that significantly impacts the quality of feature representation and subsequent model performance. Our implementation incorporates a comprehensive text preprocessing pipeline designed specifically for app review text.

#### 5.1.1 Text Cleaning and Normalization

The text preprocessing pipeline is implemented in the `preprocess_text()` function in `app_review_principle_predictor.py`:

```python
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
```

The pipeline includes:

1. **Case normalization**: Converting all text to lowercase to reduce vocabulary size and improve generalization.
   - Rationale: Capital letters rarely carry meaning relevant to principles; normalizing case reduces feature dimensionality.

2. **Special character and digit removal**: Eliminating punctuation, special characters, and numerical digits.
   - Rationale: These elements typically don't contribute to the semantic meaning related to usability or security principles and can introduce noise.

3. **Whitespace normalization**: Removing excessive spaces and standardizing whitespace.
   - Rationale: Consistent spacing helps with tokenization and eliminates meaningless variations.

4. **Stopword removal**: Filtering out common words (e.g., "the", "and", "is") that carry little semantic value.
   - Rationale: Stopwords add computational complexity without contributing much to distinguishing between principles.

5. **Lemmatization**: Reducing words to their base or dictionary form using NLTK's WordNetLemmatizer.
   - Rationale: Lemmatization consolidates different inflected forms of a word, reducing vocabulary size while preserving meaning better than stemming.

This preprocessing produces cleaner text representations that focus on meaningful content words while reducing noise and dimensionality, which is particularly important for the subsequent word embedding and neural network stages.

#### 5.1.2 Vocabulary Construction and Word Embeddings

The `ReviewVocabulary` class implements a sophisticated approach to vocabulary construction and word embedding:

1. **Vocabulary building with frequency-based filtering**:
   - Words are counted across all documents in the training set
   - Top N most frequent words are retained (default max_size=10000)
   - Special tokens for padding (`<PAD>`) and unknown words (`<UNK>`) are added

2. **Integration with pre-trained embeddings**:
   - GloVe embeddings (200-dimensional) are loaded from Stanford NLP
   - Words with pre-trained embeddings are prioritized in the vocabulary
   - Embeddings matrix is initialized with pre-trained vectors where available
   - Random initialization is used for words without pre-trained embeddings

3. **Text-to-indices conversion**:
   - Each text is converted to a sequence of numerical indices
   - Sequences are padded or truncated to a fixed length for batch processing
   - Out-of-vocabulary words are mapped to the `<UNK>` token

The use of pre-trained word embeddings provides several advantages:
- Captures semantic relationships between words learned from large corpus
- Transfers knowledge from general language understanding to domain-specific task
- Improves model performance, especially with limited labeled data
- Accelerates model convergence during training

The embeddings integration is implemented with careful handling of the vocabulary-embedding alignment:

```python
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
```

This approach creates a robust foundation for text representation that balances vocabulary coverage, semantic richness, and computational efficiency.

### 5.2 Data Balancing Strategy for Hierarchical Multi-Label Classification

Class imbalance presents a significant challenge in app review classification, where some principles (e.g., "Satisfaction") appear much more frequently than others (e.g., "Non-repudiation"). Our implementation incorporates a sophisticated multi-faceted approach to addressing class imbalance that considers the hierarchical structure of the labels.

#### 5.2.1 Hierarchical Balance Analysis

Before applying balancing techniques, we perform a comprehensive analysis of the class distribution at each hierarchical level, implemented in the `hierarchical_balance_analysis.py` module:

1. **Quantitative analysis** at each hierarchical level:
   - Counts and percentages of reviews for each main category, sub-category, and principle
   - Calculations of imbalance ratios (ratio of most common to least common class)
   - Multi-label co-occurrence patterns

2. **Visualization of class distributions**:
   - Pie charts for main categories (Level 1)
   - Bar charts for sub-categories (Level 2)
   - Bar charts for individual principles (Level 3)

3. **Imbalance metrics**:
   ```python
   def calculate_imbalance_ratio(counts_dict):
       """Calculate the ratio between the most and least common categories."""
       values = [count for count in counts_dict.values() if count > 0]
       if not values:
           return 0
       return max(values) / min(values)
   ```

This analysis provides critical insights that guide the selection and parametrization of balancing strategies. For example, if Level 3 shows much higher imbalance than Level 1, more aggressive balancing techniques may be applied at the more granular level.

#### 5.2.2 Multi-Label SMOTE Implementation

To address class imbalance, we implement a custom version of the Synthetic Minority Over-sampling Technique (SMOTE) adapted for multi-label classification. The `multi_label_smote()` function creates synthetic samples for minority classes:

```python
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
```

The approach incorporates several key components:

1. **Minority class identification** based on percentile thresholds:
   ```python
   # Use 20th percentile as threshold for minority classes
   level2_threshold = np.percentile(level2_counts, 20)
   level3_threshold = np.percentile(level3_counts, 20)
   
   # Get minority class names
   level2_minority = [mlb_level2.classes_[i] for i, count in enumerate(level2_counts) if count < level2_threshold]
   level3_minority = [mlb_level3.classes_[i] for i, count in enumerate(level3_counts) if count < level3_threshold]
   ```

2. **Text augmentation techniques** for generating synthetic samples:
   - **Random word dropping**: Randomly removing 10-20% of words from the original text
   - **Chunk-based word order shuffling**: Dividing text into chunks and shuffling their order
   - **Combination strategies**: Applying multiple augmentation techniques with probability-based selection

3. **Careful handling of multi-label dependencies**:
   - Preserving label co-occurrence patterns in synthetic samples
   - Maintaining hierarchical relationships between labels
   - Sampling with replacement from existing minority class instances

The text augmentation is implemented in the `simple_text_augmentation()` function:

```python
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
```

The synthetic sampling approach is significantly more sophisticated than standard SMOTE:
- Works with text data rather than numerical features
- Preserves semantic meaning through careful augmentation
- Handles multi-label dependencies appropriately
- Considers the hierarchical structure of labels

#### 5.2.3 Class Weighting for Loss Functions

In addition to oversampling, we apply class-specific weights in the loss function to further address imbalance. The weights are calculated based on the inverse frequency of each class:

```python
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
```

These weights ensure that errors on minority classes are penalized more heavily during training, forcing the model to pay more attention to these underrepresented classes.

#### 5.2.4 Hierarchical Balancing Strategies

The system implements two different approaches to hierarchical balancing:

1. **Level 1 balancing**: Balances the dataset at the main category level, ensuring equal representation of Usability and Security reviews.

2. **Level 2 balancing**: Balances the dataset at the sub-category level, ensuring equal representation of each sub-category (Interaction, User Experience, Data Protection, etc.).

The `create_balanced_dataset()` function in `hierarchical_balance_analysis.py` implements these approaches:

```python
def create_balanced_dataset(file_path, output_path=None, method="hierarchical", level=2):
    """
    Create a balanced dataset using the hierarchical approach.
    
    Parameters:
    file_path (str): Path to the Excel file containing the dataset
    output_path (str): Path to save the balanced dataset (if None, will not save)
    method (str): Method to use for balancing ('hierarchical' or 'random')
    level (int): Hierarchical level to balance at (1 or 2)
    
    Returns:
    pd.DataFrame: Balanced dataset
    """
```

The implementation includes careful handling of reviews that belong to multiple categories, ensuring that even after balancing, the multi-label nature of the data is preserved.

These comprehensive balancing strategies work together to address the inherent class imbalance, improving model performance on rare principles without sacrificing performance on common ones.

### 5.3 Neural Network Architecture for Hierarchical Classification

The model architecture is specifically designed to address the challenges of hierarchical multi-label classification of text data. The architecture combines state-of-the-art techniques in natural language processing with specialized components for hierarchical prediction.

#### 5.3.1 Embedding Layer with Pre-trained Vectors

The foundation of the model is a word embedding layer that maps tokenized text to dense vector representations:

```python
# Embedding layer (with pre-trained embeddings if provided)
self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
if pretrained_embeddings is not None:
    self.embedding.weight.data.copy_(pretrained_embeddings)
    if freeze_embeddings:
        self.embedding.weight.requires_grad = False
```

Key design choices:
- **Dimension**: 200-dimensional GloVe embeddings, providing a good balance between expressiveness and computational efficiency
- **Initialization**: Pre-trained GloVe vectors for words in vocabulary, random initialization for other words
- **Padding handling**: Special padding index (0) with zero embedding
- **Fine-tuning option**: Embeddings can be frozen or fine-tuned during training

#### 5.3.2 Bidirectional LSTM Encoder

The core of the model is a bidirectional LSTM (BiLSTM) encoder that processes the embedded text sequence:

```python
# Increased model capacity
self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
self.dropout1 = nn.Dropout(dropout_rate)
```

The BiLSTM architecture offers several advantages:
- **Bidirectional processing**: Captures context from both directions, essential for understanding text meaning
- **Multi-layer design**: Two stacked LSTM layers for increased model capacity and abstraction level
- **Hidden dimension**: 256 hidden units per direction (512 total) to model complex patterns
- **Regularization**: Dropout applied to outputs to prevent overfitting

The bidirectional LSTM effectively transforms the word embeddings into contextual representations that capture the sequential dependencies and semantic patterns in the review text.

#### 5.3.3 Self-Attention Mechanism

A key innovation in the architecture is the incorporation of a self-attention mechanism, implemented in the `AttentionLayer` class:

```python
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
```

The attention mechanism provides several benefits:
- **Adaptive focus**: Learns to focus on the most relevant parts of the text for each principle
- **Word importance weighting**: Assigns different weights to different words based on their relevance
- **Improved interpretability**: Attention weights can be visualized to understand which parts of a review influenced the classification
- **Effective handling of variable-length text**: Creates a fixed-length context vector regardless of input length

The attention layer transforms the LSTM outputs into a single context vector that captures the most salient features of the review for classification.

#### 5.3.4 Shared Feature Extraction

The architecture includes shared dense layers that process the context vector before the hierarchical outputs:

```python
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
```

This component includes:
- **Two fully-connected layers** with decreasing dimensions for feature transformation
- **Batch normalization** for faster convergence and improved generalization
- **ReLU activation** for non-linearity
- **Dropout regularization** to prevent overfitting

The shared feature extraction creates a common representation that captures the general aspects of the review, which is then used by level-specific output layers.

#### 5.3.5 Hierarchical Output Layers

The model implements separate output branches for each hierarchical level, allowing specialized prediction for each level:

```python
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
```

Each level-specific branch includes:
- **Fully-connected layers** with task-specific weights
- **ReLU activation** for non-linearity
- **Dropout regularization** to prevent overfitting
- **Sigmoid activation** for multi-label classification (producing independent probabilities for each class)

The hierarchical output structure allows the model to make predictions at different levels of granularity while sharing common learned representations.

#### 5.3.6 Forward Pass and Integration

The forward pass integrates all components into a cohesive processing pipeline:

```python
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
```

This architecture design offers several advantages for hierarchical multi-label classification:
- **Parameter efficiency**: Shared layers reduce total parameter count and improve generalization
- **Level-specific specialization**: Dedicated output branches for each hierarchical level
- **Integrated training**: The entire model is trained end-to-end, allowing interactions between levels
- **Flexible prediction**: Can predict at all levels simultaneously or at selected levels

The neural network architecture represents a carefully crafted solution that combines proven NLP techniques (embeddings, BiLSTM) with specialized components for hierarchical multi-label classification (attention mechanism, level-specific outputs).

### 5.4 Advanced Training Methodology

The training strategy incorporates specialized techniques for dealing with the challenges of hierarchical multi-label classification with imbalanced data. The training process is implemented in the `train_model()` function with several key components designed to optimize performance.

#### 5.4.1 Weighted Loss Functions for Imbalanced Data

One of the central challenges in the training process is addressing class imbalance. We implement class-specific weighted loss functions using PyTorch's BCEWithLogitsLoss with custom positive weights:

```python
# Calculate class weights for each level from the training data
y_train_level1 = torch.cat([batch['labels_level1'] for batch in train_loader], dim=0)
y_train_level2 = torch.cat([batch['labels_level2'] for batch in train_loader], dim=0)
y_train_level3 = torch.cat([batch['labels_level3'] for batch in train_loader], dim=0)

weight_level1 = calculate_class_weights(y_train_level1.cpu().numpy())
weight_level2 = calculate_class_weights(y_train_level2.cpu().numpy())
weight_level3 = calculate_class_weights(y_train_level3.cpu().numpy())

# Weighted loss functions for each level
criterion_level1 = nn.BCEWithLogitsLoss(pos_weight=weight_level1.to(device))
criterion_level2 = nn.BCEWithLogitsLoss(pos_weight=weight_level2.to(device))
criterion_level3 = nn.BCEWithLogitsLoss(pos_weight=weight_level3.to(device))
```

The weights are calculated based on the inverse frequency of positive examples for each class:
- Higher weights for minority classes increase their impact on gradient updates
- Weights are clipped to prevent extreme values from destabilizing training
- Each hierarchical level gets its own set of weights based on its specific class distribution

Additionally, the code includes an alternative Focal Loss implementation for cases of severe imbalance:

```python
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
```

Focal Loss dynamically adjusts the weight based on the prediction confidence, focusing more on hard examples (those with low prediction confidence), which is particularly useful for severe imbalance cases.

#### 5.4.2 Hierarchical Loss Weighting

To balance the learning process across the hierarchical levels, we implement level-specific loss weighting:

```python
# Weight the losses based on importance - emphasize levels 2 and 3
combined_loss = loss_level1 + 1.5 * loss_level2 + 2.0 * loss_level3
```

This weighting scheme:
- Assigns higher importance to more granular levels (2 and 3)
- Helps the model focus more on specific principles rather than general categories
- Counteracts the tendency of the model to perform better on higher levels due to their simpler classification boundaries

The specific weights (1.0, 1.5, 2.0) were determined through experimental validation, finding the optimal balance for overall hierarchical performance.

#### 5.4.3 Optimization Strategy

The training process uses Adam optimization with carefully tuned hyperparameters:

```python
# Optimizer with reduced learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
```

Key optimization choices:
- **Adam optimizer**: Combines the benefits of AdaGrad and RMSProp with adaptive learning rates
- **Learning rate**: 0.0005, lower than default to ensure stable training with complex loss landscape
- **Weight decay**: 1e-5 for L2 regularization to prevent overfitting
- **Gradient clipping**: Prevents exploding gradients and stabilizes training
  ```python
  # Gradient clipping to prevent exploding gradients
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

#### 5.4.4 Learning Rate Scheduling

To further optimize training, we implement learning rate scheduling using PyTorch's ReduceLROnPlateau:

```python
# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# In the training loop:
scheduler.step(val_loss)
```

This scheduler:
- Monitors validation loss for improvement
- Reduces learning rate by half (factor=0.5) when validation loss plateaus
- Waits for 2 epochs (patience=2) before reducing learning rate
- Helps the model converge to a better minimum by fine-tuning the search as training progresses

#### 5.4.5 Early Stopping Implementation

To prevent overfitting and save computation time, we implement early stopping with best model restoration:

```python
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
```

This mechanism:
- Tracks the best model state based on validation loss
- Counts epochs without improvement
- Stops training when patience threshold is exceeded
- Restores the best model weights instead of using the final weights

The early stopping patience is set to 5 epochs, allowing the model sufficient opportunity to recover from temporary plateaus while preventing excessive overfitting.

#### 5.4.6 Training Progress Monitoring

The training process includes comprehensive progress monitoring and history tracking:

```python
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
```

This detailed tracking:
- Records both overall and level-specific losses
- Monitors both training and validation performance
- Tracks learning rate changes
- Provides data for post-training visualization and analysis

The comprehensive training methodology combines multiple advanced techniques to address the specific challenges of hierarchical multi-label classification with imbalanced data, resulting in a robust and effective training process.

## 6. Implementation Details and Components

The implementation consists of multiple interrelated components that work together to create a cohesive hierarchical classification system. This section details the key classes and functions that form the core of the implementation.

### 6.1 Key Components and Their Functionality

#### 6.1.1 `ReviewVocabulary` Class

The `ReviewVocabulary` class serves as the interface between raw text and numerical representations suitable for neural networks:

```python
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
            self.pretrained_embeddings = load_glove_embeddings(embedding_dim)
        else:
            self.pretrained_embeddings = {}
```

Key functionality:
- **Vocabulary construction** with frequency-based filtering
- **Special token handling** for padding and unknown words
- **Pre-trained embedding integration** with GloVe vectors
- **Text-to-indices conversion** for model input preparation
- **Embedding matrix creation** for model initialization

The class implements sophisticated methods for handling vocabulary construction, including prioritization of words with pre-trained embeddings:

```python
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
```

This approach significantly improves the quality of the embedding layer by maximizing the use of pre-trained vectors while maintaining vocabulary coverage.

#### 6.1.2 `ReviewDataset` Class

The `ReviewDataset` class implements PyTorch's Dataset interface, enabling efficient data loading and batching:

```python
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
```

Key functionality:
- **Multi-label data handling** at three hierarchical levels
- **Text indexing** using the vocabulary
- **Tensor conversion** for model consumption
- **Batching support** through the Dataset interface

The class is designed to efficiently handle the complex data structure of hierarchical multi-label classification, providing a clean interface for the PyTorch DataLoader.

#### 6.1.3 `AttentionLayer` Class

The `AttentionLayer` class implements a self-attention mechanism that allows the model to focus on relevant parts of the input sequence:

```python
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
```

Key functionality:
- **Attention score calculation** for each token in the sequence
- **Softmax normalization** to create a probability distribution over tokens
- **Weighted aggregation** of token representations
- **Context vector creation** for downstream classification

The attention mechanism is a crucial component that enhances the model's ability to identify and focus on the most relevant parts of a review for classifying different principles.

#### 6.1.4 `HierarchicalClassifier` Class

The `HierarchicalClassifier` class implements the complete neural network architecture:

```python
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
        
        # Bidirectional LSTM encoder
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # Shared dense layers
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
```

Key functionality:
- **End-to-end architecture** integration
- **Modular component design** for flexibility
- **Hierarchical prediction** with shared representations
- **Forward pass implementation** through all components
- **Level-specific output generation** for each hierarchical level

The class represents the core of the implementation, tying together all components into a cohesive neural network architecture specifically designed for hierarchical multi-label classification.

#### 6.1.5 `FocalLoss` Class

The `FocalLoss` class implements an alternative loss function specifically designed for severe class imbalance:

```python
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
```

Key functionality:
- **Dynamic weighting** based on prediction confidence
- **Hard example focusing** with configurable gamma parameter
- **Optional alpha weighting** for class-specific weights
- **Reduction modes** for aggregating loss values

The Focal Loss implementation provides an alternative to the standard weighted BCE loss, particularly useful for extreme imbalance cases by focusing more on hard examples where the model makes confident but incorrect predictions.

### 6.2 Comprehensive Evaluation Metrics

The evaluation of hierarchical multi-label classification requires specialized metrics that can properly assess performance at different levels and handle the multi-label nature of the data. Our implementation incorporates a comprehensive set of evaluation metrics.

#### 6.2.1 Hierarchical Evaluation Strategy

The evaluation is performed separately for each hierarchical level, allowing for detailed analysis of model performance at different granularities:

```python
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
```

The hierarchical evaluation allows stakeholders to understand:
- Performance on broad categories vs. specific principles
- Where the model excels or struggles in the hierarchy
- How performance cascades through the hierarchical levels

#### 6.2.2 Multi-Label Classification Metrics

For each hierarchical level, the implementation calculates multiple metrics suitable for multi-label classification:

```python
# Level 1 metrics
metrics['level1'] = {
    'f1_micro': f1_score(y_test_level1, all_preds_level1, average='micro', zero_division=0),
    'f1_macro': f1_score(y_test_level1, all_preds_level1, average='macro', zero_division=0),
    'precision_micro': precision_score(y_test_level1, all_preds_level1, average='micro', zero_division=0),
    'precision_macro': precision_score(y_test_level1, all_preds_level1, average='macro', zero_division=0),
    'recall_micro': recall_score(y_test_level1, all_preds_level1, average='micro', zero_division=0),
    'recall_macro': recall_score(y_test_level1, all_preds_level1, average='macro', zero_division=0)
}
```

The metrics include:

1. **F1 Score**: A balanced measure of precision and recall:
   - **Micro-averaged F1**: Calculates metrics globally, giving equal weight to each sample and better reflecting performance on frequent classes
   - **Macro-averaged F1**: Calculates metrics for each label and finds their unweighted mean, giving equal weight to each class regardless of frequency

2. **Precision**: Measures exactness or quality of positive predictions:
   - **Micro-averaged precision**: Ratio of correctly predicted positive observations to all predicted positive observations
   - **Macro-averaged precision**: Average of precision scores for each class

3. **Recall**: Measures completeness or quantity of positive predictions:
   - **Micro-averaged recall**: Ratio of correctly predicted positive observations to all actual positive observations
   - **Macro-averaged recall**: Average of recall scores for each class

The use of both micro and macro averaging is crucial for imbalanced datasets:
- Micro-averaging provides an overall performance view, typically dominated by frequent classes
- Macro-averaging provides equal weight to all classes, better reflecting performance on rare classes

#### 6.2.3 Detailed Classification Reports

In addition to aggregate metrics, the implementation generates detailed classification reports for each hierarchical level:

```python
# Print classification reports
print("\nLevel 1 Classification Report:")
print(classification_report(y_test_level1, all_preds_level1, target_names=mlb_level1.classes_, zero_division=0))
```

These reports provide class-specific metrics including:
- Precision, recall, and F1-score for each individual class
- Support (number of actual occurrences) for each class
- Weighted and unweighted averages across all classes

The detailed reports are essential for identifying specific strengths and weaknesses in the model's performance across different principles.

### 6.3 Visualization and Analysis Tools

The implementation includes comprehensive visualization tools for understanding both data characteristics and model performance. These tools are critical for interpreting results and communicating findings.

#### 6.3.1 Training History Visualization

The `plot_training_history()` function creates visualizations of the training process:

```python
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
    # ...
    
    # Level 3 loss
    # ...
```

The visualizations include:
- Overall loss curves for training and validation
- Level-specific loss curves for each hierarchical level
- Learning rate changes during training
- Training vs. validation performance over epochs

These visualizations help in:
- Identifying overfitting or underfitting
- Understanding convergence behavior
- Comparing performance across hierarchical levels
- Assessing the effectiveness of learning rate scheduling

#### 6.3.2 Hierarchical Balance Analysis

The `hierarchical_balance_analysis.py` module provides comprehensive visualizations of class distributions:

```python
def plot_hierarchical_results(results, hierarchy):
    """Create visualizations of the hierarchical analysis results."""
    # Set up the figure
    plt.figure(figsize=(15, 12))
    
    # 1. Level 1 Categories (Pie chart)
    plt.subplot(2, 2, 1)
    level1_data = {k: v["count"] for k, v in results["level1"].items()}
    plt.pie(level1_data.values(), labels=level1_data.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Level 1: Main Categories')
    
    # 2. Level 2 Sub-categories (Bar chart)
    # ...
    
    # 3. Usability principles (Bar chart)
    # ...
    
    # 4. Security principles (Bar chart)
    # ...
```

The visualizations include:
- Pie charts of main categories (Level 1)
- Bar charts of sub-categories (Level 2)
- Bar charts of individual principles (Level 3)
- Imbalance ratio calculations

These visualizations provide critical insights into:
- The degree of class imbalance at each hierarchical level
- The distribution of reviews across different principles
- The relative frequency of usability vs. security concerns
- The need for and effectiveness of balancing strategies

Together, these implementation components create a comprehensive framework for hierarchical multi-label classification of app reviews, addressing the specific challenges of the task and providing robust tools for training, evaluation, and analysis.

## 7. Usage Instructions and Implementation Workflow

This section provides detailed instructions for using the framework, from data preparation to model training and prediction.

### 7.1 Data Preparation and Analysis

The first step in the workflow is preparing and analyzing the dataset to understand its characteristics and prepare it for model training.

#### 7.1.1 Dataset Format Requirements

The input dataset should be an Excel file with the following structure:
- One column containing the review text
- Binary (0/1) columns for each principle in the hierarchy
- Column names matching the principle names used in the hierarchy dictionary

Example format:
```
| Review_Text                           | Learnability  | Memorability  | ... | Confidentiality  | Integrity  | ... |
|---------------------------------------|---------------|---------------|-----|------------------|------------|-----|
| The app is very easy to use and...    | 1             | 1             | ... | 0                | 0          | ... |
| I'm concerned about my data being...  | 0             | 0             | ... | 1                | 1          | ... |
```

#### 7.1.2 Running Hierarchical Balance Analysis

To understand the class distribution and imbalance in your dataset, run the hierarchical balance analysis script:

```
python hierarchical_balance_analysis.py
```

This script:
1. Loads the dataset (default path: "30K Final Reviews for Thesis - Relabelled.xlsx")
2. Analyzes class distribution at each hierarchical level
3. Calculates imbalance ratios and statistics
4. Generates visualizations of class distributions
5. Creates balanced datasets at different hierarchical levels

The script outputs:
- Terminal statistics on class distributions
- Visual analysis as "hierarchical_balance_analysis.png"
- Two balanced datasets: "balanced_level1.xlsx" and "balanced_level2.xlsx"

#### 7.1.3 Understanding Analysis Results

The analysis provides critical insights into dataset characteristics:

- **Class distribution**: Percentage of reviews in each category and principle
- **Imbalance ratios**: Ratio between most common and least common classes
- **Hierarchical relationships**: How principles relate across levels
- **Co-occurrence patterns**: Which principles frequently appear together

These insights inform the choice of balancing strategy and model parameters.

### 7.2 Model Training and Evaluation

Once the data is prepared and analyzed, the next step is training and evaluating the model.

#### 7.2.1 Running the Training Script

To train the hierarchical classifier on your balanced dataset:

```
python app_review_principle_predictor.py
```

By default, the script:
1. Loads the balanced dataset (preferring Level 2 balanced dataset)
2. Preprocesses the review text
3. Downloads or loads GloVe embeddings (200-dimensional)
4. Builds vocabulary with pre-trained embeddings
5. Prepares hierarchical multi-label data with oversampling
6. Initializes and trains the hierarchical classifier
7. Evaluates performance on test data
8. Saves the trained model and components

#### 7.2.2 Configuration Options

The main script includes several configurable parameters that can be adjusted in the `main()` function:

```python
# Define improved parameters
embedding_dim = 200  # GloVe dimension
hidden_dim = 256     # LSTM hidden dimension
batch_size = 16      # Training batch size
epochs = 30          # Maximum training epochs
dropout_rate = 0.4   # Dropout for regularization
```

Additional configuration options include:
- `file_path`: Path to the dataset file
- `oversample_minority`: Whether to apply minority class oversampling
- `freeze_embeddings`: Whether to freeze or fine-tune embeddings
- `patience`: Number of epochs for early stopping

#### 7.2.3 Training Output and Monitoring

During training, the script outputs comprehensive information:

```
Epoch 1/30 - Loss: 0.6823 - Val Loss: 0.6215 - L1: 0.2341/0.2156 - L2: 0.3012/0.2876 - L3: 0.4521/0.4012 - LR: 0.000500
```

This includes:
- Overall loss for training and validation
- Level-specific losses for each hierarchical level
- Current learning rate

After training, the script generates:
- A training history plot ("training_history.png")
- Classification reports for each hierarchical level
- Performance metrics for micro and macro-averaged metrics

#### 7.2.4 Model and Component Saving

The trained model and necessary components are saved for later use:

```python
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
```

This creates:
- Model weights file: "app_review_principle_model.pt"
- Component pickle file: "model_components.pkl"

### 7.3 Making Predictions on New Reviews

After training, the model can be used to predict principles for new, unseen app reviews.

#### 7.3.1 Loading the Trained Model

To use the trained model, first load the saved model and components:

```python
import torch
import pickle

# Load model components
with open('model_components.pkl', 'rb') as f:
    components = pickle.load(f)
    vocab = components['vocab']
    mlb_level1 = components['mlb_level1']
    mlb_level2 = components['mlb_level2']
    mlb_level3 = components['mlb_level3']

# Initialize model with same architecture
model = HierarchicalClassifier(
    vocab_size=len(vocab),
    embedding_dim=200,
    hidden_dim=256,
    num_classes_level1=len(mlb_level1.classes_),
    num_classes_level2=len(mlb_level2.classes_),
    num_classes_level3=len(mlb_level3.classes_),
    dropout_rate=0.4
)

# Load saved weights
model.load_state_dict(torch.load('app_review_principle_model.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
```

#### 7.3.2 Predicting Principles for New Reviews

The `predict_principles_for_new_review()` function handles prediction for new reviews:

```python
# Example usage
new_review = "The app is very intuitive and easy to learn, but it crashed several times when I tried to log in."
predictions = predict_principles_for_new_review(model, vocab, mlb_level1, mlb_level2, mlb_level3, new_review, device)

print("\nPredictions for new review:")
print(f"Review: {new_review}")
print(f"Level 1 (Main Categories): {predictions['level1']}")
print(f"Level 2 (Sub-Categories): {predictions['level2']}")
print(f"Level 3 (Individual Principles): {predictions['level3']}")
```

The function:
1. Preprocesses the new review text
2. Converts text to numerical indices
3. Passes the input through the model
4. Applies thresholding to get binary predictions
5. Converts binary predictions back to principle labels
6. Returns predictions at all three hierarchical levels

#### 7.3.3 Interpreting Prediction Results

The prediction results include identified principles at each level of the hierarchy:

```
Predictions for new review:
Review: The app is very intuitive and easy to learn, but it crashed several times when I tried to log in.
Level 1 (Main Categories): ('Usability',)
Level 2 (Sub-Categories): ('Interaction', 'Error Handling')
Level 3 (Individual Principles): ('Learnability', 'User Error Protection')
```

These predictions can be used for:
- Prioritizing development efforts based on identified issues
- Categorizing and routing user feedback to appropriate teams
- Aggregating statistics on common usability or security concerns
- Monitoring trends in user feedback over time

## 8. Technical Implementation Notes and Considerations

### 8.1 Dependencies and Environment

The implementation relies on several key libraries and frameworks:

#### 8.1.1 Core Dependencies

- **PyTorch**: Core deep learning framework for neural network implementation
  - Version 1.8.0 or higher recommended
  - GPU acceleration strongly recommended for training

- **pandas and NumPy**: Data manipulation and numerical operations
  - Used for dataset loading, transformation, and numerical operations
  - Critical for efficiency in data preprocessing stages

- **scikit-learn**: Machine learning utilities
  - MultiLabelBinarizer for encoding multi-label data
  - Metrics functions for evaluation
  - train_test_split for dataset partitioning

- **NLTK**: Natural Language Processing toolkit
  - WordNetLemmatizer for text preprocessing
  - Stopwords corpus for filtering common words
  - Required NLTK downloads: 'stopwords', 'wordnet'

- **matplotlib and seaborn**: Visualization libraries
  - Training history plots
  - Class distribution visualization
  - Performance metrics visualization

#### 8.1.2 Environment Setup

The recommended environment setup includes:

```
Python 3.7+
torch==1.10.0
pandas==1.3.4
numpy==1.21.4
scikit-learn==1.0.1
nltk==3.6.5
matplotlib==3.5.0
seaborn==0.11.2
requests==2.26.0
```

For GPU acceleration (optional but recommended for large datasets):
```
torch==1.10.0+cu113  # CUDA 11.3 version
```

### 8.2 Performance Optimization Techniques

The implementation incorporates several optimizations to improve model performance:

#### 8.2.1 Training Speed Optimization

- **Batch Normalization**: Normalizes activations between layers for faster convergence
  - Applied after dense layers in the shared feature extraction component
  - Helps mitigate internal covariate shift during training

- **Gradient Clipping**: Prevents exploding gradients and stabilizes training
  - Maximum norm threshold of 1.0 prevents extreme gradient values
  - Particularly important with complex loss functions and imbalanced data

- **Learning Rate Scheduling**: Adapts learning rate during training
  - ReduceLROnPlateau monitors validation loss and reduces learning rate when progress plateaus
  - Helps navigate complex loss landscapes efficiently

- **Early Stopping**: Prevents unnecessary computation and overfitting
  - Monitors validation loss and stops training when no improvement occurs
  - Restores best model weights from earlier epochs

#### 8.2.2 Memory Efficiency Considerations

- **Batch Processing**: Data is processed in batches (default size 16)
  - Balances between memory efficiency and computation efficiency
  - Allows training on GPUs with limited memory

- **Fixed Sequence Length**: Review texts are truncated or padded to a fixed length (200 tokens)
  - Enables efficient batching and parallel processing
  - Length chosen to balance information retention with memory usage

- **Vocabulary Size Limitation**: Vocabulary is limited to most frequent words (default 15,000)
  - Reduces embedding layer size
  - Focuses on most informative tokens

### 8.3 Extensibility and Customization

The framework is designed to be extensible and adaptable for different scenarios and requirements:

#### 8.3.1 Adding New Principles

New principles can be added to the hierarchy by modifying the hierarchy dictionary:

```python
# Example: Adding new principles to the hierarchy
hierarchy["mainCategories"]["Usability"].append("New_Principle")
hierarchy["subCategories"]["User Experience"].append("New_Principle")
```

The model architecture automatically adapts to the new dimensions as long as the dataset includes binary labels for the new principles.

#### 8.3.2 Alternative Architectures

The modular design allows for replacing components with alternatives:

- **Embedding Module Alternatives**:
  - Word2Vec embeddings instead of GloVe
  - FastText for better handling of out-of-vocabulary words
  - Domain-specific embeddings trained on app review corpora

- **Encoder Alternatives**:
  - Transformer-based encoders (e.g., BERT, RoBERTa)
  - GRU instead of LSTM for faster training
  - Convolutional networks for text representation

- **Attention Mechanism Variants**:
  - Multi-head attention for capturing different aspects
  - Hierarchical attention for document-level understanding
  - Cross-attention between hierarchical levels

These alternatives can be implemented by modifying the respective components in the `HierarchicalClassifier` class.

#### 8.3.3 Balancing Strategy Customization

The balancing strategies can be customized for different datasets:

- **Adjustable thresholds** for minority class identification
  - Default 20th percentile can be adjusted based on dataset characteristics

- **Different text augmentation techniques**
  - Synonym replacement for semantic preservation
  - Back-translation for paraphrase generation
  - More sophisticated word dropping strategies

- **Alternative balancing methods**
  - Class-weighted sampling instead of oversampling
  - Custom loss functions with dynamic weighting
  - Hierarchical resampling strategies

## 9. Conclusion and Future Research Directions

### 9.1 Summary of Contributions

This implementation provides a comprehensive framework for the hierarchical multi-label classification of app reviews based on usability and security principles. The key contributions include:

1. **Hierarchical Multi-Label Architecture**: A specialized neural network design that effectively handles the hierarchical structure of usability and security principles.

2. **Advanced Data Balancing**: A sophisticated approach to addressing class imbalance that considers both the hierarchical structure and multi-label nature of the data.

3. **Attention-Enhanced BiLSTM**: An effective text encoding mechanism that combines bidirectional LSTM with attention for capturing relevant aspects of app reviews.

4. **Pre-trained Embedding Integration**: A carefully designed approach for integrating GloVe embeddings into the model, prioritizing words with pre-trained vectors.

5. **Comprehensive Evaluation Framework**: A multi-faceted evaluation strategy that assesses performance at each hierarchical level using metrics suitable for multi-label classification.

The implementation successfully addresses the key challenges in automated app review analysis:
- Converting unstructured text to meaningful representations
- Handling significant class imbalance
- Preserving hierarchical relationships between principles
- Supporting multi-label classification at multiple granularities

### 9.2 Future Research Directions

While the current implementation provides a robust solution, several promising directions for future research and development include:

#### 9.2.1 Advanced Model Architectures

- **Transformer-Based Models**: Exploring BERT, RoBERTa, or other transformer architectures to potentially improve text representation.
  - Benefits: Stronger contextual understanding, better handling of long-range dependencies
  - Challenges: Higher computational requirements, potential overfitting with limited data

- **Graph Neural Networks**: Modeling explicit relationships between principles using graph structures.
  - Benefits: Direct encoding of hierarchical relationships, improved information flow
  - Challenges: More complex implementation, additional hyperparameters to tune

#### 9.2.2 Enhanced Hierarchical Learning

- **Hierarchical Loss Functions**: Developing specialized loss functions that explicitly model dependencies between hierarchical levels.
  - Examples: Tree-structured loss, hierarchical cross-entropy
  - Benefits: More direct optimization for hierarchical performance

- **Level-Specific Feature Extraction**: Designing different feature extractors optimized for each hierarchical level.
  - Benefits: Specialized representations for different granularities
  - Challenges: Increased model complexity, potential for overfitting

#### 9.2.3 Active Learning for Targeted Data Collection

- **Uncertainty-Based Sampling**: Identifying reviews where the model is uncertain to prioritize for manual labeling.
  - Benefits: More efficient use of labeling resources
  - Implementation: Monte Carlo dropout for uncertainty estimation

- **Diversity-Based Sampling**: Selecting diverse reviews to maximize coverage of the principle space.
  - Benefits: Better representation of rare principles
  - Implementation: Clustering-based selection strategies

#### 9.2.4 Explainable AI Techniques

- **Attention Visualization**: Enhancing the visualization of attention weights to understand model focus.
  - Benefits: Increased transparency and trust in model predictions
  - Implementation: Word-level highlighting based on attention scores

- **LIME or SHAP Integration**: Implementing model-agnostic explainability techniques.
  - Benefits: Local explanations for individual predictions
  - Challenges: Computational overhead, interpretation of multi-label results

#### 9.2.5 Cross-Domain Adaptation

- **Transfer Learning Across App Categories**: Developing techniques to adapt the model to different app domains.
  - Examples: Gaming apps to productivity apps, social media to finance
  - Implementation: Domain adaptation techniques, fine-tuning strategies

- **Multi-Lingual Extensions**: Extending the framework to handle reviews in multiple languages.
  - Benefits: Broader applicability across global app markets
  - Implementation: Multilingual embeddings, language-agnostic architecture

### 9.3 Practical Applications

The framework has numerous practical applications for app developers, UX designers, and security teams:

1. **Automated Feedback Analysis**: Efficiently processing thousands of user reviews to extract actionable insights.

2. **Issue Prioritization**: Quantitatively identifying which usability or security principles are most frequently mentioned.

3. **Competitive Analysis**: Comparing principle distributions across competitor apps to identify relative strengths and weaknesses.

4. **Trend Monitoring**: Tracking changes in user concerns over time, especially after updates.

5. **Targeted Testing**: Focusing QA efforts on areas identified as problematic through review analysis.

By providing a means to automatically classify app reviews according to established principles, this framework bridges the gap between unstructured user feedback and structured, actionable insights for development teams.

## 10. References and Acknowledgments

### 10.1 Research Foundations

This work builds upon research in several interconnected fields:

1. **Hierarchical Multi-Label Classification**:
   - Silla, C.N., & Freitas, A.A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22(1-2), 31-72.
   - Cerri, R., Barros, R.C., & de Carvalho, A.C. (2014). Hierarchical multi-label classification using local neural networks. Journal of Computer and System Sciences, 80(1), 39-56.

2. **Text Classification with Deep Learning**:
   - Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1746-1751.
   - Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical Attention Networks for Document Classification. Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 1480-1489.

3. **Imbalanced Learning Techniques**:
   - Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.
   - Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. Proceedings of the IEEE International Conference on Computer Vision, 2980-2988.

4. **App Review Analysis**:
   - Maalej, W., & Nabil, H. (2015). Bug report, feature request, or simply praise? On automatically classifying app reviews. 2015 IEEE 23rd International Requirements Engineering Conference (RE), 116-125.
   - Guzman, E., & Maalej, W. (2014). How do users like this feature? A fine grained sentiment analysis of app reviews. 2014 IEEE 22nd International Requirements Engineering Conference (RE), 153-162.

5. **Usability and Security Principles**:
   - Nielsen, J. (1994). Enhancing the explanatory power of usability heuristics. Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 152-158.
   - Saltzer, J.H., & Schroeder, M.D. (1975). The protection of information in computer systems. Proceedings of the IEEE, 63(9), 1278-1308.

### 10.2 Acknowledgments

The implementation uses several open-source resources:

- **GloVe embeddings** from Stanford NLP Group:
  - Pennington, J., Socher, R., & Manning, C.D. (2014). GloVe: Global Vectors for Word Representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532-1543.
  - Available at: https://nlp.stanford.edu/projects/glove/

- **PyTorch** deep learning framework:
  - Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32, 8026-8037.

- **NLTK** (Natural Language Toolkit):
  - Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media Inc.

- **scikit-learn** machine learning library:
  - Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
