# Deep-Learning---Exp6

 **DL- Developing a Deep Learning Model for NER using LSTM**

# AIM

To develop an LSTM-based model for recognizing the named entities in the text.

# THEORY


**Neural Network Model**


<img width="609" height="317" alt="Screenshot 2025-10-26 210344" src="https://github.com/user-attachments/assets/a0a3921a-9211-4361-8bf2-8bb22f0b3691" />


# DESIGN STEPS

**STEP 1: Data Preprocessing**
   
   - Load the dataset (ner_dataset.csv) using pandas.

   - Fill missing values using forward fill (.ffill() method).

   - Extract unique words and tags from the dataset and create mappings (word2idx, tag2idx).

**STEP 2: Sentence Grouping**

   - Combine words, their POS tags, and entity tags into complete sentences using groupby("Sentence #").

   - Each sentence becomes a list of (word, POS, tag) tuples to preserve word-level tagging structure.

**STEP 3: Token Indexing and Padding**

   - Convert each word and tag into their corresponding integer indices using the mappings.

   - Apply padding (using Keras pad_sequences) to make all sequences equal in length (e.g., max_len = 50).

   - Split data into training and testing sets using train_test_split.

**STEP 4: Model Construction**

   - Define an Embedding layer to convert word indices into dense vectors.

   - Apply SpatialDropout1D for regularization.

   - Use a Bidirectional LSTM layer to capture contextual information from both directions.

   - Add a TimeDistributed Dense layer with a softmax activation to predict entity tags at each word position.

**STEP 5: Model Compilation and Training**

   - Compile the model with Adam optimizer and sparse_categorical_crossentropy loss.

   - Train the model for multiple epochs (e.g., 3) with the training data and validate using the test set.

**STEP 6: Evaluation and Prediction**

   - Plot training vs. validation accuracy and loss to monitor learning.

   - Predict tags for a sample sentence from the test set.

   - Compare the true tags and predicted tags word by word to evaluate model performance.


# Name: Moulishwar G

# Register Number: 2305001020



# OUTPUT


**Training loss, validation Loss Vs iteration Plot**


<img width="701" height="516" alt="Screenshot 2025-10-26 211138" src="https://github.com/user-attachments/assets/ca4130a5-0fa0-47b1-b969-8d2f118adc5c" />


<img width="690" height="513" alt="Screenshot 2025-10-26 211306" src="https://github.com/user-attachments/assets/ceee8d21-51bb-42f8-83ea-a1022d5e3f92" />



# Sample Text Prediction



<img width="376" height="858" alt="Screenshot 2025-10-26 211522" src="https://github.com/user-attachments/assets/d62c081d-c8fd-4b4b-afe9-280918a97d2b" />


**RESULT**


Thus, an LSTM-based model for recognizing the named entities in the text is successfully developed.
