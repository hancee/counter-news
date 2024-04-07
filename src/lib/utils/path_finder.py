from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).parent.parent.parent.parent.resolve()
CONFIG_PATH = PROJECT_DIRECTORY.joinpath(".config/config.json")