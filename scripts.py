from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from grwhs.viz import plots
from grwhs.metrics import regression

run_dir = Path("outputs/runs/quick_smoke_B-20251004-022112") # adjust if needed
data = np.load(run_dir / "dataset.npz")
meta = (run_dir / "dataset_meta.json").read_text()

X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Example: fit a quick ridge baseline just for plotting
from grwhs.models.baselines import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)

fig, ax = plt.subplots()
plots.scatter_truth_vs_pred(y_test, pred, ax=ax, title="Test predictions vs truth")
fig.savefig(run_dir / "viz_truth_vs_pred.png", dpi=150)
print("Saved plot to", run_dir / "viz_truth_vs_pred.png")
