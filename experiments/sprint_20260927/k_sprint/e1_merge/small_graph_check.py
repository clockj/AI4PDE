import sys, os, numpy as np
pkg = sys.argv[1]
X = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["x"].astype(np.float32)
os.chdir(pkg); sys.path.insert(0, pkg)
import submission
ref = {k: v.copy() for k, v in submission.predict(X[:56]).items()}
for i in range(4):
    j = 5 * i
    o = submission.predict(X[j:j + 1])
    print(pkg.split("/")[-1], "call", i, "win", j, " ".join("%s maxabs %.2e" % (k, np.abs(o[k][0] - ref[k][j]).max()) for k in ("prediction", "lower", "upper")),
          "graph", getattr(submission, "SMALL_GRAPH_ACTIVE", None), flush=True)
