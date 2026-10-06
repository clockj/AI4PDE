"""RealPDE Track 2 (LTTTA), team xtgg233: the two entry points of the training package.

    train(data_dir, output_dir, device)    the released training data -> everything inference needs
    get_ttt_model(output_dir, device)      the trained submission, under the Codabench contract

`train` is our Codabench submission 948792 rebuilt from scratch: it trains the eleven networks (six
members, five twins), fits the interval band on the twins' held-out errors, bakes the members and the
band into the submission archive, runs the archive through the platform's contract and the causality
check, and unpacks it into `output_dir`. `get_ttt_model` hands that directory to the archive's own
`submission.py`, whose `get_ttt_model(submission_dir, device)` is what Codabench called.

The eleven trainings are independent: about 37 GPU-hours in all on an A100 (3.7 h for an F-FNO or a ViT with
the CFD term, 1.8 h for a ViT without). `train` runs them one per GPU over every GPU the container sees
(`JOBS` per GPU; a second training on the same GPU gains nothing, the step is GPU-bound), longest first.
Everything else runs in sequence. Checkpoints, caches and logs go under `<output_dir>/work`; a stage whose
result exists is skipped, so an interrupted run resumes.

Optional environment variables (none is set by the organizers; the defaults are the submission):
    RPDE_JOBS     trainings run at once per GPU (default JOBS)
    RPDE_EPOCHS   epochs per network (default 400; anything else is a smoke test of the pipeline)
    RPDE_WORK     the working directory (default <output_dir>/work)
    RPDE_CPUS     CPU cores to share among the trainings (default: the affinity mask, at most 16)
    RPDE_SIMCACHE where the CFD-map tables are cached (default <work>/simcache, removed at the end)
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
UNET, BAND, BUILD = (os.path.join(HERE, d) for d in ("unet", "band", "build"))

JOBS = 1            # trainings at once per GPU
EPOCHS = 400

# ---- the recipes ---------------------------------------------------------------------------------
# Every network: 400 epochs over the real windows at stride 5, batch 32, AdamW, one-cycle to 2e-4, weight
# decay 1e-5, dropout 0.3. The loss weights are train.py's: field MSE (lam-f), Welsch field loss (lam-cov,
# knee cov-c sigma), velocity-gradient (lam-grad) and time-derivative (lam-tgrad) matching, log-sigma
# (lam-v), turbulent kinetic energy (lam-t); the CFD term (lam-sim) on the mapped train_sim windows.
COMMON = "--batch 32 --stride 5 --lr 2e-4 --wd 1e-5"
LOSS_C = "--lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15"
LOSS_P = "--lam-f 1.0 --lam-cov 0.0 --lam-grad 1.0 --lam-tgrad 0.0 --lam-v 0.3 --lam-t 0.15"
FFNO = f"--arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 {LOSS_C} {COMMON}"
VITC = f"--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 {LOSS_C} {COMMON}"
VITP = f"--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 {LOSS_P} {COMMON}"
SIMF = "--lam-sim 0.3 --sim-stride 5"           # + --sim <train_sim>, added by train()

# name -> flags. A MEMBER trains on every real trajectory (--all-data) and ships; a TWIN holds out the
# eight trajectories of unet/data.py:VAL_TRAJ and is used only to fit the band. Listed longest first,
# which is the order the scheduler starts them in.
RECIPES = {
    "member_ffno_s1": f"{FFNO} {SIMF} --seed 1 --all-data",
    "member_ffno_s2": f"{FFNO} {SIMF} --seed 2 --all-data",
    "member_ffno_s3": f"{FFNO} {SIMF} --seed 3 --all-data",
    "member_ffno_s4": f"{FFNO} {SIMF} --seed 4 --all-data",
    "member_vit_c":   f"{VITC} {SIMF} --seed 3 --all-data",
    "twin_ffno_s1":   f"{FFNO} {SIMF} --sim-exclude-held --seed 1",
    "twin_ffno_s2":   f"{FFNO} {SIMF} --sim-exclude-held --seed 2",
    "twin_ffno_s3":   f"{FFNO} {SIMF} --sim-exclude-held --seed 3",
    "twin_vit_c":     f"{VITC} {SIMF} --sim-exclude-held --seed 1",
    "member_vit_p":   f"{VITP} --seed 1 --all-data",
    "twin_vit_p":     f"{VITP} --seed 1",
}
MEMBERS = ["member_ffno_s1", "member_ffno_s2", "member_ffno_s3", "member_ffno_s4",
           "member_vit_p", "member_vit_c"]                      # build.py's order: fno.pth, fno_2 ... fno_6
TWINS = ["twin_ffno_s1", "twin_ffno_s2", "twin_ffno_s3", "twin_vit_p", "twin_vit_c"]


# ---- running the stages --------------------------------------------------------------------------
T0 = time.time()


def say(msg: str) -> None:
    print(f"[train {time.time() - T0:7.0f}s] {msg}", flush=True)


def run(cmd: list[str], cwd: str, log: str, env: dict, must: str | None = None) -> None:
    """Run one stage to completion, its output in `log`; fail with the log's name, and with `must`
    missing from it when that line is required."""
    what = os.path.basename(log)[:-4]
    with open(log, "ab") as f:
        rc = subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    text = open(log, "rb").read().decode("utf-8", "replace")
    if rc != 0:
        raise RuntimeError(f"{what} failed (rc {rc}); see {log}\n{text[-2000:]}")
    if must and must not in text:
        raise RuntimeError(f"{what} did not print '{must.strip()}'; see {log}\n{text[-2000:]}")


def train_all(py: str, real: str, sim: str, ck: str, env: dict, epochs: int, jobs: int,
              cpus: int, gpus: list[str]) -> None:
    """The eleven trainings, `jobs` at a time on each of `gpus` (CUDA_VISIBLE_DEVICES values), longest
    first; a checkpoint that exists is skipped. A run restarts from its last `--ckpt-every` boundary."""
    todo = [n for n in RECIPES if not os.path.exists(os.path.join(ck, n + ".pth"))]
    for n in RECIPES:
        if n not in todo:
            say(f"  skip  {n} (checkpoint exists)")
    slots = jobs * len(gpus)
    threads = max(1, cpus // max(1, min(slots, len(todo) or 1)))
    env = {**env, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    live: dict[str, tuple[subprocess.Popen, str]] = {}
    started: dict[str, float] = {}

    def launch(n: str) -> None:
        gpu = min(gpus, key=lambda g: sum(1 for _, h in live.values() if h == g))
        flags = RECIPES[n].split()
        if "--lam-sim" in flags:
            flags += ["--sim", sim]
        cmd = [py, "train.py", "--data", real, "--epochs", str(epochs), *flags,
               "--out", os.path.join(ck, n + ".pth"), "--ckpt-every", "20", "--resume"]
        with open(os.path.join(ck, n + ".log"), "ab") as log:
            live[n] = (subprocess.Popen(cmd, cwd=UNET, stdout=log, stderr=subprocess.STDOUT,
                                        env={**env, "CUDA_VISIBLE_DEVICES": gpu}), gpu)
        started[n] = time.time()
        say(f"  start {n}  (GPU {gpu}, {threads} threads)")

    while todo or live:
        while todo and len(live) < slots:
            launch(todo.pop(0))
        time.sleep(10)
        for n, (p, _) in list(live.items()):
            if p.poll() is None:
                continue
            del live[n]
            log = os.path.join(ck, n + ".log")
            tail = open(log, "rb").read().decode("utf-8", "replace").strip().splitlines()[-1:]
            if p.returncode != 0 or not os.path.exists(os.path.join(ck, n + ".pth")):
                for q, _ in live.values():
                    q.terminate()
                raise RuntimeError(f"training {n} failed (rc {p.returncode}); see {log}\n"
                                   + "\n".join(tail))
            say(f"  done  {n}  {(time.time() - started[n]) / 3600:.2f} h   {tail[0] if tail else ''}")


def train(data_dir: str, output_dir: str, device: str = "cuda") -> None:
    """From the released training data under `data_dir` to the submission's files in `output_dir`."""
    if device != "cuda":
        raise ValueError("train() needs device='cuda'")
    sys.path.insert(0, UNET)
    from data import VAL_TRAJ, release_dirs  # noqa: E402
    real, sim = release_dirs(data_dir)
    output_dir = os.path.abspath(output_dir)
    work = os.path.abspath(os.environ.get("RPDE_WORK") or os.path.join(output_dir, "work"))
    epochs = int(os.environ.get("RPDE_EPOCHS", EPOCHS))
    jobs = int(os.environ.get("RPDE_JOBS", JOBS))
    cpus = int(os.environ.get("RPDE_CPUS") or min(16, len(os.sched_getaffinity(0))))
    ck, log, val, band, stage = (os.path.join(work, d) for d in ("ckpt", "log", "val", "band", "stage"))
    for d in (output_dir, ck, log, val, band):
        os.makedirs(d, exist_ok=True)
    py = sys.executable
    simcache = os.environ.get("RPDE_SIMCACHE") or os.path.join(work, "simcache")
    env = {**os.environ, "RPDE_REAL": real, "RPDE_SIMCACHE": simcache,
           "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "HDF5_USE_FILE_LOCKING": "FALSE",
           "PYTHONDONTWRITEBYTECODE": "1"}
    n_real = len([f for f in os.listdir(real) if f.endswith(".h5")])
    n_sim = len([f for f in os.listdir(sim) if f.endswith(".h5")])
    say(f"data: {n_real} real and {n_sim} CFD trajectories under {data_dir}; work under {work}")
    say(f"{epochs} epochs per network, {cpus} CPU cores"
        + ("" if epochs == EPOCHS else "   ** NOT the submission: a smoke test **"))
    run([py, "-c", "import torch, numpy, h5py; print('torch', torch.__version__, 'numpy', numpy.__version__, "
         "'h5py', h5py.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))"],
        HERE, os.path.join(log, "env.log"), env)
    line = open(os.path.join(log, "env.log")).read().strip().splitlines()[-1]
    say(line)
    n_gpu = int(line.split("gpus")[1].split()[0])
    # The GPUs as the trainings will name them: CUDA_VISIBLE_DEVICES of this process, or 0 .. n-1.
    visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    gpus = visible[:n_gpu] if len(visible) >= n_gpu else [str(i) for i in range(n_gpu)]
    say(f"{len(gpus)} GPU(s): {jobs} training(s) at a time on each")

    # The CFD -> measurement map, once per table (the all-data one and the twins' one, which withholds
    # the held-out trajectories' statistics), before the trainings read them concurrently.
    for tag, rex in (("members", []), ("twins", VAL_TRAJ)):
        run([py, "-c", "from simreal import load_sim_mapped; "
             f"load_sim_mapped({sim!r}, {real!r}, exclude=None, real_exclude={rex!r})"],
            UNET, os.path.join(log, f"simmap_{tag}.log"), env)
        say(f"CFD map: the {tag}' table is ready")

    say("training")
    train_all(py, real, sim, ck, env, epochs, jobs, cpus, gpus)
    members = [os.path.join(ck, n + ".pth") for n in MEMBERS]
    twins = [os.path.join(ck, n + ".pth") for n in TWINS]

    say("validation windows")
    if not (os.path.exists(os.path.join(val, "val_seen.npz")) and os.path.exists(os.path.join(val, "val_unseen.npz"))):
        run([py, "make_val.py", "--data", real, "--out", val], BAND, os.path.join(log, "make_val.log"), env)
    say("band: the static fit on the twins' errors, then the sigma power law")
    run([py, "fit_band.py", "--members", *members, "--twins", *twins, "--val", val,
         "--out", os.path.join(band, "band_static.npz")], BAND, os.path.join(log, "fit_band.log"), env)
    run([py, "dump_twins.py", "--twins", *twins, "--val", val, "--out", os.path.join(band, "dump_unseen.npz")],
        BAND, os.path.join(log, "dump_twins.log"), env)
    run([py, "sigma_band.py", "--dump", os.path.join(band, "dump_unseen.npz"),
         "--band", os.path.join(band, "band_static.npz"), "--out", band],
        BAND, os.path.join(log, "sigma_band.log"), env, must="SIGPOW")
    for line in open(os.path.join(log, "sigma_band.log")):
        if "q (u, v)" in line or "SIGPOW" in line:
            say("  " + line.strip())

    say("the archive")
    archive = os.path.join(work, "submission_track2.zip")
    run([py, "build.py", "--members", *members, "--band", os.path.join(band, "band_sigblend_lam0.5.npz"),
         "--sigblend", os.path.join(band, "band_sigblend_lam0.5.json"), "--stage", stage, "--out", archive],
        BUILD, os.path.join(log, "build.log"), env, must="wrote")
    say("the gates: the platform's contract and the scores on the held-out trajectories, then causality")
    run([py, "validate.py", "--zip", archive, "--extract", os.path.join(work, "validate")],
        BUILD, os.path.join(log, "validate.log"), env, must="\nSUBMITTABLE")
    for line in open(os.path.join(log, "validate.log")):
        if line.startswith("  scores") or line.startswith("  errors") or "ms/step" in line:
            say("  " + line.strip())
    run([py, "causal_check.py", "--submission", os.path.join(work, "validate")],
        BUILD, os.path.join(log, "causal_check.log"), env, must="CAUSALITY OK")

    # What inference needs: the archive's nine files, flat in output_dir (and the archive itself).
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        z.extractall(output_dir)
    shutil.copy2(archive, os.path.join(output_dir, "submission_track2.zip"))
    if simcache.startswith(work):                   # 3 GB of CFD-map tables: not needed any more
        shutil.rmtree(simcache, ignore_errors=True)
    sha = hashlib.sha256(open(archive, "rb").read()).hexdigest()
    say(f"wrote {output_dir}: {' '.join(sorted(names))}")
    say(f"submission_track2.zip sha256 {sha}")
    say(f"TRAIN OK  ({(time.time() - T0) / 3600:.2f} h)")


def get_ttt_model(output_dir: str, device: str = "cuda"):
    """The trained submission: `output_dir/submission.py`'s `get_ttt_model(output_dir, device)`, an
    object with `reset_ttt_state()` and `ttt_step(input_norm, prev_target_norm=None)`."""
    output_dir = os.path.abspath(output_dir)
    path = os.path.join(output_dir, "submission.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run train() first")
    spec = importlib.util.spec_from_file_location("xtgg233_track2_submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.get_ttt_model(output_dir, device)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python solution.py <data_dir> <output_dir>     (the same as train(data_dir, output_dir, 'cuda'))")
    train(sys.argv[1], sys.argv[2], "cuda")
