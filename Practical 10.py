 import numpy as np
 import tensorflow as tf
 from tensorflow import keras
 from tensorflow.keras.preprocessing.text import Tokenizer
 from tensorflow.keras.utils import to_categorical
 from tensorflow.keras.layers import Input, Dense, Embedding, Lambda
 from tensorflow.keras.models import Model
 import matplotlib.pyplot as plt
 from sklearn.decomposition import PCA
 print("TensorFlow Version:", tf.__version__)

 corpus = [
 'the cat sat on the mat',
 'the dog sat on the log',
 'the cat chased the dog'
 ]

 # 1. Tokenization: Convert words to integers
 tokenizer = Tokenizer()
 tokenizer.fit_on_texts(corpus)

 # The dictionary mapping words to integers
 word_index = tokenizer.word_index
 print("Word Index:", word_index)

 # Vocabulary size is the number of unique words + 1 (for padding)
 vocab_size = len(word_index) + 1

 # Convert sentences to sequences of integers
 sequences = tokenizer.texts_to_sequences(corpus)
 print("Integer Sequences:", sequences)

 # Create context/target pairs
 window_size = 1
 X, y = [], []
 for sequence in sequences:
 for i in range(window_size, len(sequence) - window_size):
 # Target word is the word at the center of the window
        target_word = sequence[i]
 # Context words are the words around the target
        context_words = sequence[i - window_size : i] + sequence[i + 1 : i + 1 + window_size]
        X.append(context_words)
        y.append(target_word)
   
 # Convert to NumPy arrays for the model
 X = np.array(X)
 y = np.array(y)
   
 # One-hot encode the target variable, as this is a classification task
 y = to_categorical(y, num_classes=vocab_size)
 print("Context (X) shape:", X.shape)
 print("Target (y) shape:", y.shape)
 print("\nSample Context:", X[0])
 print("Sample Target (one-hot):", y[0])

 # Define model parameters
 embedding_dim = 10
 context_size = 2 * window_size
 # Define the model using the Keras Functional API
 # Input layer for context words
 input_layer = Input(shape=(context_size,))

 # Embedding layer
 embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding_layer')(input_layer)
   
 # Average the embeddings of the context words
 avg_layer = Lambda(lambda x: tf.reduce_mean(x, axis=1))(embedding_layer)
   
 # Output layer to predict the target word
 output_layer = Dense(vocab_size, activation='softmax')(avg_layer)
   
 # Create the CBOW model
 cbow_model = Model(inputs=input_layer, outputs=output_layer)
   
 # Compile the model
 cbow_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   
 # Print the model summary
 cbow_model.summary()
   
 # Train the model
 history = cbow_model.fit(X, y, epochs=200, verbose=1)

 # 1. Extract the embedding for a sample word
 word_embeddings = cbow_model.get_layer('embedding_layer').get_weights()[0]

 # 2. Create a dictionary to map words to their learned vectors
 embedding_dict = {}
 for word, index in word_index.items():
     embedding_dict[word]=word_embeddings[index]
   
 #Print the embedding for a sample word
 print(f"Embedding for the word 'cat':\n{embedding_dict['cat']}")
   
 # 3. Visualize the embeddings in 2D space
 # We use PCA to reduce the 10-dimensional embeddings to 2 dimensions
 pca = PCA(n_components=2)
 reduced_embeddings = pca.fit_transform(list(embedding_dict.values()))
 plt.figure(figsize=(10, 8))
 for i, word in enumerate(embedding_dict.keys()):
    x, y_coord = reduced_embeddings[i]
    plt.scatter(x, y_coord)
    plt.annotate(word, (x, y_coord), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
 plt.title("CBOW Word Embeddings Visualized with PCA")
 plt.xlabel("PCA Component 1")
 plt.ylabel("PCA Component 2")
 plt.grid(True)
 plt.show()
  
