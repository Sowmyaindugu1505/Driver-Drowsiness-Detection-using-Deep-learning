from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app_config import EYE_DATA_DIR, EYE_MODEL_PATH


def main() -> None:
    model = load_model(str(EYE_MODEL_PATH))

    test_data = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        EYE_DATA_DIR / "test",
        target_size=(24, 24),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
    )

    loss, accuracy = model.evaluate(test_data)
    print(f"Eye Test Loss: {loss:.4f}")
    print(f"Eye Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
