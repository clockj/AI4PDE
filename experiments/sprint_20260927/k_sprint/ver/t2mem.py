import runpy, sys, os
# wrap t2run: print /proc/self/status after the run (load + 336 steps)
sys.argv = ["t2run.py"] + sys.argv[1:]
runpy.run_path("/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/ver/t2run.py", run_name="__main__")
d = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/self/status") if l.split(":")[0] in ("VmRSS", "VmLck", "VmPin", "VmHWM")}
print("MEM t2", os.path.basename(sys.argv[1]), d)
