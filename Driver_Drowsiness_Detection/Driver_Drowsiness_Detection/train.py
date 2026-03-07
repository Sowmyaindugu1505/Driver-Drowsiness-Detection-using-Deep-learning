from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image settings
img_size = (24, 24)
batch_size = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
valid_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'data/train',
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)

valid_data = valid_gen.flow_from_directory(
    'data/valid',
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10
)

# Save model
model.save('models/cnnCat2.h5')

print("Model training completed and saved successfully.")
