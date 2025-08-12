import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define constants
IMG_SIZE = 224  # Standard input size for many CNNs
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5  # Abrasion, Bruise, Burn, Cut, Ulcer
MODEL_PATH = 'models/injury_model.h5'

def create_cnn_model():
    
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_generators(data_dir):
    """
    Prepares training and validation data generators.
    
    Args:
        data_dir: Directory containing the dataset organized in folders by class
        
    Returns:
        train_generator: Generator for training data
        validation_generator: Generator for validation data
        class_names: List of class names
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, validation_generator, class_names

def train_model(model, train_generator, validation_generator):
    """
    Trains the CNN model.
    
    Args:
        model: The compiled Keras model
        train_generator: Generator for training data
        validation_generator: Generator for validation data
        
    Returns:
        history: Training history
    """
    # Create callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

def evaluate_model(model, validation_generator):
    """
    Evaluates the trained model.
    
    Args:
        model: The trained Keras model
        validation_generator: Generator for validation data
        
    Returns:
        evaluation: Evaluation metrics
    """
    evaluation = model.evaluate(validation_generator)
    print(f"Validation Loss: {evaluation[0]:.4f}")
    print(f"Validation Accuracy: {evaluation[1]:.4f}")
    
    return evaluation

def plot_training_history(history):
    """
    Plots the training and validation accuracy/loss.
    
    Args:
        history: Training history
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.close()

def save_model_summary(model):
    """
    Saves the model summary to a text file.
    
    Args:
        model: The Keras model
    """
    with open('models/model_summary.txt', 'w') as f:
        # Redirect stdout to the file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        model.summary()
        sys.stdout = original_stdout

def preprocess_image(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Preprocesses an image for prediction.
    
    Args:
        img_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        preprocessed_img: Preprocessed image ready for prediction
    """
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    preprocessed_img = np.expand_dims(img_array, axis=0) / 255.0
    
    return preprocessed_img

def predict_injury(model, img_path, class_names):
    """
    Predicts the injury type from an image.
    
    Args:
        model: The trained Keras model
        img_path: Path to the image file
        class_names: List of class names
        
    Returns:
        predicted_class: Predicted class name
        confidence: Prediction confidence
    """
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    return predicted_class, confidence

def main():
    """
    Main function to run the entire training pipeline.
    """
    # Create the model
    model = create_cnn_model()
    
    # Save model summary
    save_model_summary(model)
    
    # Define the dataset directory path (make sure it points to the correct location)
    data_dir = 'dataset'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please provide the dataset.")
        return
    
    # Prepare data generators
    train_generator, validation_generator, class_names = prepare_data_generators(data_dir)
    
    # Train the model
    history = train_model(model, train_generator, validation_generator)
    
    # Evaluate the model
    evaluate_model(model, validation_generator)
    
    # Plot training history
    plot_training_history(history)
    
    # Save class names
    np.save('models/class_names.npy', class_names)
    
    print(f"Model training completed. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()