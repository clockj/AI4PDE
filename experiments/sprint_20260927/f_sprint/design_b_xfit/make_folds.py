"""Design B cross-fit folds: the 74 non-validation real trajectories, 5 folds stratified by (AoA, Re) round-robin
(kfold.folds' 'stratified' rule applied after removing model.VAL_TRAJ).  Writes folds.json: {"k": [names...]}.
Fold model k trains with --exclude (fold k + VAL_TRAJ), so val_unseen stays a clean evaluation set for the twins."""
import json, os, sys
DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
VAL = ["3750_0", "6300_10", "8850_20", "11400_20", "13950_10", "16500_5", "21600_15", "26700_0"]
names = sorted(f[:-3] for f in os.listdir(DATA) if f.endswith(".h5"))
names = [n for n in names if n not in VAL]
names.sort(key=lambda n: (int(n.split("_")[1]), int(n.split("_")[0])))
folds = {str(k): sorted(names[k::5]) for k in range(5)}
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "folds.json")
json.dump({"val": VAL, "folds": folds}, open(out, "w"), indent=1)
for k, v in folds.items():
    print(k, len(v), " ".join(v))
