"""9900 steps in ONE process, 100 trajectories of 14..99 windows; keep every returned object, crc at return, re-check at end."""
import sys, os, time, zlib, importlib.util, numpy as np, torch
pkg = os.path.abspath(sys.argv[1]); os.chdir("/tmp")
def mem():
    return {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmLck", "VmPin", "VmHWM"))}
z = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz"); XR, YR = z["x"], z["y"]
MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
XN = ((XR - MI) / SI).astype(np.float32); YN = ((YR - MT) / ST).astype(np.float32)
spec = importlib.util.spec_from_file_location("ps", pkg + "/submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m = mod.get_ttt_model(pkg, "cuda"); print("MEM_LOAD", mem(), "arena", getattr(m, "arena", None) is not None, flush=True)
rng = np.random.default_rng(1)
lens = [99] * 99 + [99]  # 100 x 99 = 9900
keep, crcs, ts = [], [], []; step = 0; bad = 0
crc = lambda t: zlib.crc32(np.ascontiguousarray(np.asarray(t)).tobytes())
for k, L in enumerate(lens):
    a = int(rng.integers(0, len(XR) - 42)); idx = [(a + (i % 42)) for i in range(L)]
    m.reset_ttt_state(); prev = None
    for i in idx:
        t0 = time.perf_counter(); p, info = m.ttt_step(torch.from_numpy(XN[i][None]), prev); ts.append(time.perf_counter() - t0)
        outs = (p, info["lower"], info["upper"])
        for o in outs:
            o_ = torch.as_tensor(o)
            if not bool(torch.isfinite(o_).all()): bad += 1
        if not bool((torch.as_tensor(outs[0 + 1]) <= torch.as_tensor(outs[2])).all()): bad += 1
        keep.append(outs); crcs.append(tuple(crc(o) for o in outs))
        prev = torch.from_numpy(YN[i][None]); step += 1
    if k in (10, 11, 20, 21, 50, 99): print(f"traj {k} steps {step} mem {mem()} arena_i {getattr(m,'arena_i',None)}", flush=True)
changed = [j for j, (o, c) in enumerate(zip(keep, crcs)) if tuple(crc(x) for x in o) != c]
ts = np.array(ts) * 1e3
print(f"SPLIT ms med steps<1024 {np.median(ts[:1024]):.3f} (mean {ts[:1024].mean():.3f}) | 1024..2047 {np.median(ts[1024:2048]):.3f} | >=2048 {np.median(ts[2048:]):.3f} (mean {ts[2048:].mean():.3f}) | steps 1020..1028 " + " ".join(f"{t:.2f}" for t in ts[1020:1029]), flush=True)
print("TRANSITION types", [type(keep[j][0]).__name__ for j in (1022,1023,1024,1025)], "shares_memory 1023~1024", np.shares_memory(np.asarray(keep[1023][0]), np.asarray(keep[1024][0])), "1024~1025", np.shares_memory(np.asarray(keep[1024][0]), np.asarray(keep[1025][0])), flush=True)
print(f"LONG steps {step} nonfinite/badband {bad} changed_after_return {len(changed)} first {changed[:5]} "
      f"ms med {np.median(ts):.3f} mean {ts.mean():.3f} total_s {ts.sum()/1e3:.1f}", flush=True)
print("MEM_END", mem(), flush=True)
