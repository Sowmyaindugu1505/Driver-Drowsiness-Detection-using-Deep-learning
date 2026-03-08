from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_yawn"
MODEL_PATH = BASE_DIR / "models" / "yawn_cnn.h5"

model = load_model(MODEL_PATH)

test_gen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_gen.flow_from_directory(
    DATA_DIR / "test",
    target_size=(64, 64),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

loss, accuracy = model.evaluate(test_data)
print(f"Yawn Test Loss: {loss:.4f}")
print(f"Yawn Test Accuracy: {accuracy * 100:.2f}%")
