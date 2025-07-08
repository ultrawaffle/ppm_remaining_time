from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).parents[0].resolve()

DATA_ROOT = PROJECT_ROOT / "data"
LOG_ROOT = DATA_ROOT / "logs"