import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evalrun.analysis import load_results

df = load_results(load_config())
df["tier_A"] = ((df.wrong_behavior == 1) | (df.unfaithful_and_wrong == 1)).astype(float)
df["tier_B"] = df.hallucinated
df["tier_C"] = df.hallucinated_strict
t = df.groupby("condition", observed=True)[["tier_A", "tier_B", "tier_C"]].agg(["sum", "mean"])
t["n"] = df.groupby("condition", observed=True).size()
print(t.round(3).to_string())