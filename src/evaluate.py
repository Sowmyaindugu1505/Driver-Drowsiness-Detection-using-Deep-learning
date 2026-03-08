from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    BATCH_SIZE,
    EYE_DATA_DIR,
    EYE_MODEL_PATH,
    IMG_SIZE_EYES,
    IMG_SIZE_YAWNS,
    YAWN_DATA_DIR,
    YAWN_MODEL_PATH,
)


def main() -> None:
    args = get_args()

    if args.target == "eyes":
        data_dir = EYE_DATA_DIR
        img_size = IMG_SIZE_EYES
        model_path = EYE_MODEL_PATH
    else:
        # yawns
        data_dir = YAWN_DATA_DIR
        img_size = IMG_SIZE_YAWNS
        model_path = YAWN_MODEL_PATH

    if not model_path.exists():
        print(f"[Error] Model not found at {model_path}. Please run train.py first.")
        return

    if not (data_dir / "test").exists():
        print(f"[Error] Test data not found at {data_dir}. Did you run prepare_dataset.py?")
        return

    print(f"\n--- Evaluating Model for: {args.target.upper()} ---")

    # Load Model
    model = load_model(str(model_path))

    # Load Test Data
    test_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_data = test_gen.flow_from_directory(
        data_dir / "test",
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Evaluate
    loss, accuracy = model.evaluate(test_data)
    print(f"\n[Results] {args.target.capitalize()}")
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
