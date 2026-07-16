"""Wrapper to run batch_fitting and capture output to file"""
import sys, os, subprocess
sys.stdout.reconfigure(encoding='utf-8')

script = os.path.join(os.path.dirname(__file__), "batch_fitting.py")
log = os.path.join(os.path.dirname(__file__), "fitting_run_log.txt")

with open(log, "w", encoding="utf-8") as f:
    proc = subprocess.run(
        [sys.executable, script, "--target", "42000"],
        capture_output=True, text=True, timeout=1800,  # 30 min timeout
        cwd=os.path.dirname(__file__)
    )
    f.write("STDOUT:\n" + proc.stdout + "\n")
    f.write("STDERR:\n" + proc.stderr + "\n")
    f.write(f"EXIT CODE: {proc.returncode}\n")

print(f"Exit code: {proc.returncode}")
print(f"Log saved: {log}")
# Print last 20 lines of stdout
lines = proc.stdout.splitlines()
for l in lines[-30:]:
    print(l)
