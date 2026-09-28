"""Does it matter that the official runner hands prev_target normalized with TARGET stats (our stream_eval uses INPUT stats)?
R1's unscored test compares yp to r1_z = -MEAN_in/STD_in with tol 1e-6; tgt-normalized masked zeros differ by 6e-5 / 2.3e-4."""
import sys, os, importlib.util, itertools, numpy as np, torch
pkg = os.path.abspath(sys.argv[1]); os.chdir("/tmp")
sys.path.insert(0, "/users/PAS1187/qwu/RealPDE/official/t2_v6/realpde_t2_starting_kit_v6"); import scoring as off
z = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz"); XR, YR, IDS = z["x"], z["y"], z["ids"]
MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
first = [i == 0 or IDS[i].split(":")[0] != IDS[i - 1].split(":")[0] for i in range(len(IDS))]
spec = importlib.util.spec_from_file_location("ps", pkg + "/submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m = mod.get_ttt_model(pkg, "cuda")
print("zero-target frac", float((YR[..., :2] == 0).mean()))
for tag, (mm, ss) in (("tgt-stats (official runner)", (MT, ST)), ("in-stats (our stream_eval)", (MI, SI))):
    P, L, U = [], [], []; prev = None
    for i in range(len(XR)):
        if first[i]: m.reset_ttt_state(); prev = None
        p, info = m.ttt_step(torch.from_numpy(((XR[i] - MI) / SI)[None].astype(np.float32)), prev)
        P.append(np.asarray(p)[0] * ST + MT); L.append(np.asarray(info["lower"])[0] * ST + MT); U.append(np.asarray(info["upper"])[0] * ST + MT)
        prev = torch.from_numpy(((YR[i] - mm) / ss)[None].astype(np.float32))
    P, L, U = map(np.stack, (P, L, U)); c = 2
    rl = float(np.mean(off.rel_l2_per_sample(P, YR, c))); tk = float(np.mean(off.tke_rel_l2_per_sample(P, YR, c))); mv = off.mvpe_rel_l2(P, YR)
    sps, cov = off.aggregate_sps(P, YR, c, lower=L, upper=U)
    print(tag, "rel", off.score_error(rl), "tke", off.score_error(tk), "mvpe", off.score_error(mv), "sps", sps, flush=True)
