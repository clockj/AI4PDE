# EA: time one extracted package in a fresh process. usage: rp_f27a_time.py t1|t2 <extract_dir> <val_unseen.npz>
import sys, os, time, importlib.util, numpy as np, torch
trk, D, V = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("submission", os.path.join(D, "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
d = np.load(V); x = d["x"].astype(np.float32); y = d["y"].astype(np.float32)
if trk == "t1":
    sub.predict(x[:64])  # warm-up
    ts = []
    for r in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter(); out = sub.predict(x); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / len(x) * 1e3)
    print("T1 %s ms/window median %.4f min %.4f all %s" % (D, np.median(ts), min(ts), ["%.4f" % t for t in ts]))
else:
    mi = np.array([0.154960856, -0.000513992854, 0.0], np.float32); si = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
    mt = np.array([0.154962569, -0.000517793698, 0.0], np.float32); st = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
    xn = torch.from_numpy((x - mi) / si); yn = torch.from_numpy((y - mt) / st)
    dev = "cuda"
    m = sub.get_ttt_model(D, dev)
    ts = []
    for r in range(3):
        m.reset_ttt_state(); prev = None
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(len(xn)):
            p, info = m.ttt_step(xn[i:i + 1], prev)
            prev = yn[i:i + 1]
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / len(xn) * 1e3)
    print("T2 %s ms/step median %.4f min %.4f all %s" % (D, np.median(ts), min(ts), ["%.4f" % t for t in ts]))
