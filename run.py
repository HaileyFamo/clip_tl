# run.py
from pathlib import Path
from src.main import main
# from src.analyze_lens import main

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    main(project_root)
