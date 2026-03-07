from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = load_model("models/cnnCat2.h5")

# Load validation data
datagen = ImageDataGenerator(rescale=1./255)

valid_data = datagen.flow_from_directory(
    'data/valid',
    target_size=(24,24),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32
)

# Evaluate model
loss, accuracy = model.evaluate(valid_data)

print("Validation Accuracy:", accuracy * 100, "%")
