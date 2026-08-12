"""Rebuild projections.json from projections.csv backup without dropping columns."""
import json
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "server" / "models"

df = pd.read_csv(MODELS_DIR / "projections.csv", low_memory=False)
df = df.where(pd.notna(df), None)
records = df.to_dict(orient="records")

with open(MODELS_DIR / "projections.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"Recovered {len(records)} players to projections.json")
