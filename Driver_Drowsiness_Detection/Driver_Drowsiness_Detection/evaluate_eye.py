from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cnnCat2.h5"
TEST_DIR = BASE_DIR / "data" / "test"

model = load_model(str(MODEL_PATH))

test_gen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(24, 24),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

loss, accuracy = model.evaluate(test_data)
print(f"Eye Test Loss: {loss:.4f}")
print(f"Eye Test Accuracy: {accuracy * 100:.2f}%")
