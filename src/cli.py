import sys
import log

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py <command>")
        return

    command = sys.argv[1]

    if command == "fetch":
        from data.database import main as fetch_videos
        fetch_videos()
    elif command == "preprocess":
        from data.preprocess import preprocess_dataset
        preprocess_dataset()
    elif command == "train":
        from model.train import create_model
        create_model()
    else:
        print(f"Unknown command: {command}")
        return

if __name__ == "__main__":
    main()