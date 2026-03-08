#!/usr/bin/env python3
import argparse
import sys

def _inject_target_arg(target: str):
    """Dynamically injects --target args so the src modules don't break."""
    if "--target" in sys.argv:
        return
    sys.argv.extend(["--target", target])


def main():
    parser = argparse.ArgumentParser(
        description="🚗 Ultimate Driver Drowsiness Detection System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prep", "train", "eval", "detect"],
        help=(
            "prep   : Extract archive.zip and prepare the dataset\n"
            "train  : Train the CNN model (requires --target)\n"
            "eval   : Evaluate the trained model (requires --target)\n"
            "detect : Start the live webcam driver monitoring"
        ),
    )

    parser.add_argument(
        "--target",
        type=str,
        choices=["eyes", "yawns"],
        help="Required only for 'train' and 'eval' modes.",
    )

    # Parse only known args here so we can inject them back for the modules that expect `get_args()`
    args, _ = parser.parse_known_args()

    try:
        if args.mode == "prep":
            print("\n🚀 Starting Dataset Preparation...")
            from src.data_prep import main as prep_main
            prep_main()

        elif args.mode == "train":
            if not args.target:
                parser.error("--target (eyes|yawns) is required when using --mode train")
            print(f"\n🧠 Starting Training for {args.target.upper()}...")
            _inject_target_arg(args.target)
            from src.model import main as train_main
            train_main()

        elif args.mode == "eval":
            if not args.target:
                parser.error("--target (eyes|yawns) is required when using --mode eval")
            print(f"\n📊 Starting Evaluation for {args.target.upper()}...")
            _inject_target_arg(args.target)
            from src.evaluate import main as eval_main
            eval_main()

        elif args.mode == "detect":
            print("\n🎥 Starting Live Webcam Detector...")
            from src.detector import main as detect_main
            detect_main()

    except KeyboardInterrupt:
        print("\n[Stopped] Operation interrupted by user.")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")


if __name__ == "__main__":
    main()
