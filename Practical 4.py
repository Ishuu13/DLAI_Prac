import tensorflow as tf
 from tensorflow import keras
 import numpy as np
 import matplotlib.pyplot as plt
 print("TensorFlow version:", tf.__version__)

# Load the dataset
 (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
 # --- Data Preprocessing --
# Normalize the images to be between 0 and 1
 x_train = x_train.astype("float32") / 255.0
 x_test = x_test.astype("float32") / 255.0
 # Flatten the images from 28x28 to a 784-element vector
 x_train = x_train.reshape(60000, 784)
 x_test = x_test.reshape(10000, 784)
 # One-hot encode the labels
 # There are 10 classes (digits 0-9)
 y_train = keras.utils.to_categorical(y_train, 10)
 y_test = keras.utils.to_categorical(y_test, 10)
 print("Training data shape:", x_train.shape)
 print("Test data shape:", x_test.shape)
 print("Sample one-hot encoded label:", y_train[0])

# Define the model using the Sequential API
 model = keras.Sequential([
 # Input layer: specify the input shape for the first layer
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
 # First hidden layer
    keras.layers.Dense(128, activation='relu'),
 # Output layer: 10 neurons for 10 classes, softmax for probabilities
    keras.layers.Dense(10, activation='softmax')
 ])
 # Print a summary of the model's architecture
 model.summary()

# Compile the model
 model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
 # Train the model
 # We'll save the training history to plot it later
 history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=40,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Evaluate the model on the test set
 score = model.evaluate(x_test, y_test, verbose=0)
 print(f"Test Loss: {score[0]:.4f}")
 print(f"Test Accuracy: {score[1]:.4f}")

# Get training and validation accuracy and loss from history object
 acc = history.history['accuracy']
 val_acc = history.history['val_accuracy']
 loss = history.history['loss']
 val_loss = history.history['val_loss']
 epochs = range(1, len(acc) + 1)
 # Plot training and validation accuracy
 plt.figure(figsize=(12, 5))
 plt.subplot(1, 2, 1)
 plt.plot(epochs, acc, 'bo-', label='Training accuracy')
 plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
 plt.title('Training and Validation Accuracy')
 plt.xlabel('Epochs')
 plt.ylabel('Accuracy')
 plt.legend()

# Plot training and validation loss
 plt.subplot(1, 2, 2)
 plt.plot(epochs, loss, 'bo-', label='Training loss')
 plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
 plt.title('Training and Validation Loss')
 plt.xlabel('Epochs')
 plt.ylabel('Loss')
 plt.legend()
 plt.tight_layout()
 plt.show()

# Load the dataset again to access the original unflattened data
 (x_train_original, y_train_original), (x_test_original, y_test_original) = keras.datasets.mnist.load_data()
 # Select the first image from the original training data
 image_to_plot = x_train_original[0]
 # Plot the image
 plt.matshow(image_to_plot, cmap=plt.get_cmap('gray'))
 plt.title("Sample Image from Training Data")
 plt.show()

