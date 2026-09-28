import sys, os, time, importlib.util, numpy as np, torch
pkg = os.path.abspath(sys.argv[1]); os.chdir("/tmp")
z = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz"); X, Y = z["x"][:42], z["y"][:42]
MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
I = [torch.from_numpy(((x - MI) / SI)[None].astype(np.float32)) for x in X]; T = [torch.from_numpy(((y - MT) / ST)[None].astype(np.float32)) for y in Y]
spec = importlib.util.spec_from_file_location("ps", pkg + "/submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m = mod.get_ttt_model(pkg, "cuda"); ts = []
for rep in range(8):
    m.reset_ttt_state(); prev = None
    for i in range(42):
        t0 = time.perf_counter(); p, info = m.ttt_step(I[i], prev); ts.append(time.perf_counter() - t0); prev = T[i]
ts = np.array(ts[42:]) * 1e3
print(f"TIME {os.path.basename(pkg)} median {np.median(ts):.3f} mean {ts.mean():.3f} ms/step", flush=True)
