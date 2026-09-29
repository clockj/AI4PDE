"""X3 adversarial QA harness for Track-2 packages. Usage: qa_t2_harness.py <pkgdir> <mode> <outprefix>
mode: graph (default path) | eager (CUDAGraph capture forced to raise -> package fallback) | eagerfp32 (also AMP off)
      | fp32 (AMP off, graph kept) | pinfail (page-locked allocations over 10 M elements refused -> no arena)
Copied unchanged from experiments/sprint_20260927/f_sprint/qa/qa_t2_harness.py (gate R2, 2026-09-28), with the one added
mode `fp32` and VAL taken from RPDE_QA_VAL when set; nothing else differs.
Runs through the package's real get_ttt_model / reset_ttt_state / ttt_step, mimicking t2_v6 local_eval
(reset + prev_target=None on each trajectory's first step; CPU tensors in; device 'cuda')."""
import sys, os, time, json, importlib.util, warnings, itertools, traceback
import numpy as np
import torch

pkg, mode, outp = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3])
os.chdir("/tmp")
VAL = os.environ.get("RPDE_QA_VAL", "/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")
MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
z = np.load(VAL); XR, YR, IDS = z["x"], z["y"], z["ids"]
groups = []
for k, g in itertools.groupby(range(len(IDS)), key=lambda i: IDS[i].split(":")[0]):
    g = list(g); groups.append((k, g[0], g[-1] + 1))
def nin(a): return ((a - MI) / SI).astype(np.float32)
def ntg(a): return ((a - MT) / ST).astype(np.float32)
def traj(i):
    k, a, b = groups[i]; return XR[a:b].copy(), YR[a:b].copy()

warnings.simplefilter("always")
wlog = []
_old_show = warnings.showwarning
def _show(msg, cat, fn, ln, file=None, line=None):
    wlog.append(f"{cat.__name__}: {msg} ({os.path.basename(fn)}:{ln})")
warnings.showwarning = _show

spec = importlib.util.spec_from_file_location("participant_submission", os.path.join(pkg, "submission.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if mode.startswith("eager"):
    class _NoGraph:
        def __init__(self, *a, **k): raise RuntimeError("QA: capture disabled")
    torch.cuda.CUDAGraph = _NoGraph
def memstat():
    d = {}
    for l in open("/proc/self/status"):
        if l.startswith(("VmRSS", "VmLck", "VmPin", "VmHWM")): d[l.split(":")[0]] = l.split(":")[1].strip()
    return d
if mode == "pinfail":
    _opin = torch.Tensor.pin_memory
    def _pin(self, *a, **k):
        if self.numel() > 10_000_000: raise RuntimeError("QA: pin_memory refused")
        return _opin(self, *a, **k)
    torch.Tensor.pin_memory = _pin
if mode == "eagerfp32":
    mod.AMP = None; mod._AMP = None
if mode == "fp32":
    mod.AMP = None; mod._AMP = None
t0 = time.perf_counter()
model = mod.get_ttt_model(pkg, "cuda")
load_s = time.perf_counter() - t0
info_load = dict(mode=mode, load_s=load_s, graph=model.graph is not None, AMP_ACTIVE=getattr(mod, "AMP_ACTIVE", None),
                 GRAPH_ERR=getattr(mod, "GRAPH_ERR", None), GRAPH_WORST=getattr(mod, "GRAPH_WORST", None),
                 arena=(getattr(model, "arena", None) is not None), mem=memstat())
print("LOAD", info_load, flush=True)

R = {"load": info_load, "tests": {}}
def rec(name, **kw):
    R["tests"][name] = kw; print("TEST", name, json.dumps(kw, default=str)[:1500], flush=True)

class Fail(Exception): pass

def check(pred, info, step, tag):
    if not isinstance(info, dict) or "adapt_loss" not in info: raise Fail(f"{tag} step {step}: info lacks adapt_loss")
    if ("lower" in info) != ("upper" in info) or "lower" not in info: raise Fail(f"{tag} step {step}: lower/upper missing")
    if info["lower"] is None or info["upper"] is None: raise Fail(f"{tag} step {step}: lower/upper None")
    P, L, U = (torch.as_tensor(t) for t in (pred, info["lower"], info["upper"]))
    for nm, t in (("pred", P), ("lower", L), ("upper", U)):
        if tuple(t.shape) != (1, 20, 32, 64, 3): raise Fail(f"{tag} step {step}: {nm} shape {tuple(t.shape)}")
        if t.dtype != torch.float32: raise Fail(f"{tag} step {step}: {nm} dtype {t.dtype}")
        if not bool(torch.isfinite(t).all()):
            bad = (~torch.isfinite(t)).sum().item()
            raise Fail(f"{tag} step {step}: {nm} has {bad} non-finite; finite max {t[torch.isfinite(t)].abs().max().item() if bad < t.numel() else 'n/a'}")
        if bool((t[..., 2] != 0).any()): raise Fail(f"{tag} step {step}: {nm} p channel nonzero max {t[...,2].abs().max().item()}")
    if bool((L > U).any()):
        raise Fail(f"{tag} step {step}: lower>upper on {(L > U).sum().item()} elements, max gap {(L - U).max().item()}")
    return P, L, U

def run(inps, tgts, firsts, tag, do_reset=True, prev_mode="target", keep=True, timing=False, tgt_override=None):
    """inps/tgts: normalized arrays (N,20,32,64,3). firsts: bool per step. prev_mode 'target' (runner) | 'pred' (zero error)."""
    outs, copies, times = [], [], []
    prev = None; lastpred = None
    for i in range(len(inps)):
        x = torch.from_numpy(inps[i][None].copy())
        rt = 0.0
        if firsts[i]:
            if do_reset:
                t1 = time.perf_counter(); model.reset_ttt_state(); rt = time.perf_counter() - t1
                prev = None
        if prev_mode == "pred" and prev is not None and lastpred is not None:
            prev = lastpred
        t1 = time.perf_counter()
        pred, info = model.ttt_step(x, prev)
        times.append(time.perf_counter() - t1 + rt)
        P, L, U = check(pred, info, i, tag)
        if keep:
            outs.append((pred, info["lower"], info["upper"]))
            copies.append((P.numpy().copy(), L.numpy().copy(), U.numpy().copy()))
        lastpred = torch.as_tensor(pred).clone()
        prev = torch.from_numpy(tgts[i][None].copy())
    fresh = True
    if keep:
        for (a, b, c), (ca, cb, cc) in zip(outs, copies):
            for t, cp in ((a, ca), (b, cb), (c, cc)):
                if not np.array_equal(torch.as_tensor(t).numpy(), cp): fresh = False
    stack = None
    if keep:
        stack = [np.concatenate([c[j] for c in copies], 0) for j in range(3)]
    return stack, times, fresh

def val_stream(n=None, xfun=None, yfun=None):
    I, T, F = [], [], []
    for g in range(len(groups)):
        x, y = traj(g)
        if n is not None: x, y = x[:n], y[:n]
        if xfun: x = xfun(x)
        if yfun: y = yfun(y)
        I.append(nin(x)); T.append(ntg(y)); F += [True] + [False] * (len(x) - 1)
    return np.concatenate(I), np.concatenate(T), F

def safe(name, fn):
    try:
        fn()
    except Fail as e:
        rec(name, status="RED", err=str(e))
    except Exception as e:
        rec(name, status="RED-EXC", err=repr(e)[:500], tb=traceback.format_exc()[-1500:])

# ---- A. ordinary validation stream, freshness, determinism ----
I0, T0, F0 = val_stream()
base = {}
def tA():
    st, tm, fr = run(I0, T0, F0, "A")
    base["A"] = st
    np.savez(outp + "_A.npz", pred=st[0], lower=st[1], upper=st[2])
    P, L, U = st
    x = torch.from_numpy(I0[0][None].copy()); model.reset_ttt_state(); pr, inf = model.ttt_step(x, None)
    R["out_fmt"] = [(type(t).__name__, str(getattr(t, "dtype", None)), tuple(t.shape), bool(t.flags["C_CONTIGUOUS"]) if isinstance(t, np.ndarray) else bool(t.is_contiguous()),
                     bool(t.flags["WRITEABLE"]) if isinstance(t, np.ndarray) else None) for t in (pr, inf["lower"], inf["upper"])]
    print("OUTFMT", R["out_fmt"], flush=True)
    rec("A_val", status="OK" if fr else "RED", fresh=fr, steps=len(I0), ms_med=1e3 * float(np.median(tm)),
        ms_mean=1e3 * float(np.mean(tm)), pred_in_band=bool(((L <= P) & (P <= U)).all()),
        half_mean=float(((U - L) / 2)[..., :2].mean()))
safe("A_val", tA)
def tB():
    st, tm, fr = run(I0, T0, F0, "B")
    same = all(np.array_equal(a, b) for a, b in zip(st, base["A"]))
    rec("B_repeat_bitident", status="OK" if same else "RED", same=same,
        maxdiff=max(float(np.abs(a - b).max()) for a, b in zip(st, base["A"])))
safe("B_repeat_bitident", tB)

# ---- C. stream shapes ----
def tC():
    res = {}
    for n in (1, 2, 3, 14):
        I, T, F = val_stream(n)
        st, tm, fr = run(I, T, F, f"C_len{n}")
        # prefix consistency: first n windows of each traj must equal A's (state reset, causal)
        ok = True
        for g in range(len(groups)):
            a0 = groups[g][1] - groups[0][1]
            for j in range(3):
                if not np.array_equal(st[j][g * n:(g + 1) * n], base["A"][j][a0:a0 + n]): ok = False
        res[n] = dict(fresh=fr, prefix_equal_A=ok)
    bad = [n for n in res if not (res[n]["fresh"] and res[n]["prefix_equal_A"])]
    rec("C_short_streams", status="OK" if not bad else "RED", res=res)
safe("C_short_streams", tC)
def tC99():
    # 99-window trajectories: a val trajectory run forward then backward (continuous-ish), as ONE trajectory; x3 trajectories
    I, T, F = [], [], []
    for g in (0, 3, 6):
        x, y = traj(g)
        xs = np.concatenate([x, x[::-1], x])[:99]; ys = np.concatenate([y, y[::-1], y])[:99]
        I.append(nin(xs)); T.append(ntg(ys)); F += [True] + [False] * 98
    I, T = np.concatenate(I), np.concatenate(T)
    st, tm, fr = run(I, T, F, "C99")
    st2, tm2, _ = run(I, T, F, "C99b", keep=False)
    tm3 = tm + tm2
    rec("C_99_windows", status="OK" if fr else "RED", fresh=fr, ms_med=1e3 * float(np.median(tm3)), ms_mean=1e3 * float(np.mean(tm3)),
        ms_p99=1e3 * float(np.percentile(tm3, 99)), ms_max=1e3 * float(np.max(tm3)),
        worst_case_100x99_s=float(np.mean(tm3)) * 9900, half_last_mean=float(((st[2] - st[1]) / 2)[-20:, ..., :2].mean()))
safe("C_99_windows", tC99)
def tCnr():
    st, tm, fr = run(I0, T0, F0, "C_noreset", do_reset=False)
    rec("C_no_reset_between_traj", status="OK", note="not reachable per runner (it always resets at is_first); finiteness only", fresh=fr)
safe("C_no_reset_between_traj", tCnr)
def tCprev():
    # prev_target handed on a trajectory's first step (after reset) instead of None -- must be ignored
    outs = []
    I, T, F = val_stream(5)
    model.reset_ttt_state()
    res = []
    prev = torch.from_numpy(T0[40][None].copy())
    for i in range(len(I)):
        if F[i]: model.reset_ttt_state()
        pred, info = model.ttt_step(torch.from_numpy(I[i][None].copy()), prev)
        P, L, U = check(pred, info, i, "Cprev")
        res.append((P.numpy().copy(), L.numpy().copy(), U.numpy().copy()))
        prev = torch.from_numpy(T[i][None].copy())
    ok = True
    for g in range(len(groups)):
        a0 = groups[g][1]
        for j in range(3):
            if not np.array_equal(np.concatenate([r[j] for r in res[g * 5:(g + 1) * 5]]), base["A"][j][a0:a0 + 5]): ok = False
    rec("C_prev_on_first_step", status="OK" if ok else "INFO-diff", equal_A=ok, note="runner passes None on first step; this is robustness only")
safe("C_prev_on_first_step", tCprev)
def tCcuda():
    x0, y0 = traj(0)
    I, T = nin(x0[:6]), ntg(y0[:6])
    model.reset_ttt_state(); prev = None; ok = True
    for i in range(6):
        pred, info = model.ttt_step(torch.from_numpy(I[i][None].copy()).cuda(), prev)
        P, L, U = check(torch.as_tensor(pred).cpu(), {k: (torch.as_tensor(v).cpu() if v is not None else None) for k, v in info.items()}, i, "Ccuda")
        if not np.array_equal(P.numpy(), base["A"][0][i:i + 1]): ok = False
        prev = torch.from_numpy(T[i][None].copy()).cuda()
    rec("C_cuda_inputs", status="OK" if ok else "INFO-diff", equal_A=ok)
safe("C_cuda_inputs", tCcuda)

# ---- D. value stress ----
def one(name, xfun=None, yfun=None, n=None, prev_mode="target", gi=(0, 5, 7), inorm=None, tnorm=None):
    def f():
        I, T, F = [], [], []
        for g in gi:
            x, y = traj(g)
            if n: x, y = x[:n], y[:n]
            if xfun: x = xfun(x.copy())
            if yfun: y = yfun(y.copy())
            xi, yt = nin(x), ntg(y)
            if inorm: xi = inorm(xi)
            if tnorm: yt = tnorm(yt)
            I.append(xi); T.append(yt); F += [True] + [False] * (len(x) - 1)
        I, T = np.concatenate(I), np.concatenate(T)
        st, tm, fr = run(I, T, F, name, prev_mode=prev_mode)
        P, L, U = st
        h = (U - L)[..., :2] / 2
        rec(name, status="OK" if fr else "RED", fresh=fr, pred_absmax=float(np.abs(P).max()), half_max=float(h.max()),
            half_min=float(h.min()), half_mean=float(h.mean()), pred_in_band=bool(((L <= P) & (P <= U)).all()))
    safe(name, f)
def mask_block(a):
    a[:, :, 0:6, 0:12, :2] = 0.0; return a
def mask_half(a):
    a[:, :, :, 32:, :2] = 0.0; return a
def mask_all(a):
    a[:, :, :, :, :2] = 0.0; return a
one("D_a_mask_block", mask_block, mask_block)
one("D_a_mask_half", mask_half, mask_half)
one("D_a_mask_all_phys0", mask_all, mask_all, n=20)
one("D_b_target_eq_prevpred", prev_mode="pred")
def const_t(a):
    a[:] = a[:, :1]; return a
one("D_c_const_target", None, const_t)
one("D_c_const_input_and_target", const_t, const_t)
one("D_d_speed_x0.2", lambda a: a * 0.2, lambda a: a * 0.2)
one("D_d_speed_x3", lambda a: a * 3.0, lambda a: a * 3.0)
one("D_d_speed_x7", lambda a: a * 7.0, lambda a: a * 7.0)
def out_in(a):
    a[5, 10, :, :, 0] = 50.0; a[9, 3, 10, 20, :2] = -50.0; return a
def out_tg(a):
    a[5, 10, :, :, 0] = 50.0; a[9, 3, 10, 20, :2] = -50.0; return a
one("D_e_outlier50", inorm=out_in, tnorm=out_tg)
def out_all(a):
    a[5:8] = 50.0; a[..., 2] = 0.0; return a
one("D_e_outlier50_whole_windows", inorm=out_all, tnorm=out_all)
rng = np.random.default_rng(0)
def pgarb(a):
    a[..., 2] = rng.standard_normal(a[..., 2].shape).astype(np.float32) * 1e3; return a
one("D_f_p_garbage", inorm=pgarb, tnorm=pgarb)
def zero_err_masked(a):  # masked block AND zero realised error
    a[:, :, 0:6, 0:12, :2] = 0.0; return a
one("D_ab_mask_plus_target_eq_pred", mask_block, mask_block, prev_mode="pred")
def spikes(a):
    a[::7, :, 16, 32, 1] = 1e3; return a
one("D_e_v_spike_1e3_input", inorm=spikes)

# ---- E. state fully reset after stress: rerun A ----
def tE():
    st, tm, fr = run(I0, T0, F0, "E")
    same = all(np.array_equal(a, b) for a, b in zip(st, base["A"]))
    rec("E_after_stress_bitident_A", status="OK" if same else "RED", same=same,
        maxdiff=max(float(np.abs(a - b).max()) for a, b in zip(st, base["A"])))
safe("E_after_stress_bitident_A", tE)

# ---- F. causality: perturb targets from window j and inputs from j+1 of trajectory 2 ----
def tF():
    x, y = traj(2); I, T = nin(x), ntg(y); F = [True] + [False] * (len(x) - 1)
    st, _, _ = run(I, T, F, "F0")
    j = 17
    I2, T2 = I.copy(), T.copy()
    T2[j:] = rng.standard_normal(T2[j:].shape).astype(np.float32); I2[j + 1:] = rng.standard_normal(I2[j + 1:].shape).astype(np.float32)
    I2[..., 2] = 0; T2[..., 2] = 0
    st2, _, _ = run(I2, T2, F, "F1")
    ok = all(np.array_equal(a[:j + 1], b[:j + 1]) for a, b in zip(st, st2))
    rec("F_causality", status="OK" if ok else "RED", ok=ok, maxdiff=max(float(np.abs(a[:j + 1] - b[:j + 1]).max()) for a, b in zip(st, st2)))
safe("F_causality", tF)

def nanin(a):
    a[3, 5, 4, 4, 0] = np.nan; return a
one("D_g_nan_input_INFO", inorm=nanin, n=8, gi=(0,))
def tG():
    # after a NaN window in trajectory k: is trajectory k+1 (after reset) clean?
    x, y = traj(1); I, T = nin(x[:5]), ntg(y[:5])
    try:
        run(I, T, [True] + [False] * 4, "G_after_nan")
        rec("G_after_nan_next_traj_INFO", status="OK")
    except Fail as e:
        rec("G_after_nan_next_traj_INFO", status="INFO-poisoned", err=str(e))
tG()
R["warnings"] = sorted(set(wlog))[:50]
R["mem_end"] = memstat(); print("MEM_END", R["mem_end"], "arena_i", getattr(model, "arena_i", None), flush=True)
R["red"] = [k for k, v in R["tests"].items() if str(v.get("status", "")).startswith("RED") and "INFO" not in k]
json.dump(R, open(outp + ".json", "w"), indent=1, default=str)
print("WARNINGS", R["warnings"])
print("SUMMARY", mode, os.path.basename(pkg), "RED:" if R["red"] else "ALL OK", R["red"], flush=True)
