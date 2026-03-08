from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app_config import YAWN_DATA_DIR, YAWN_MODEL_PATH


def main() -> None:
    model = load_model(str(YAWN_MODEL_PATH))

    test_data = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        YAWN_DATA_DIR / "test",
        target_size=(64, 64),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
    )

    loss, accuracy = model.evaluate(test_data)
    print(f"Yawn Test Loss: {loss:.4f}")
    print(f"Yawn Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
