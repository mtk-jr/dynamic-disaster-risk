import subprocess
import sys

commands = [
    [sys.executable, "scripts/generate_demo_data.py"],
    [sys.executable, "scripts/build_dataset.py"],
    [sys.executable, "scripts/train.py"],
    [sys.executable, "scripts/evaluate.py"],
]

for command in commands:
    print("RUNNING:", " ".join(command))
    subprocess.run(command, check=True)

print("Pipeline completed.")
