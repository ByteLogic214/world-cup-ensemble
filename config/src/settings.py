import os
import yaml
from dataclasses import dataclass

@dataclass
class Secrets:
    api_football_key: str
    odds_api_key: str

def load_config(path="config/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_secrets():
    return Secrets(
        api_football_key=os.getenv("API_FOOTBALL_KEY", ""),
        odds_api_key=os.getenv("ODDS_API_KEY", "")
    )
