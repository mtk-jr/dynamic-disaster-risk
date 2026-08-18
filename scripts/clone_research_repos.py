from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "research_sources"

REPOS = {
    "ml4floods": "https://github.com/spaceml-org/ml4floods.git",
    "udl4fl": "https://github.com/kipoju/udl4fl.git",
    "ImpactMesh": "https://github.com/IBM/ImpactMesh.git",
    "crisis_datasets_benchmarks": "https://github.com/firojalam/crisis_datasets_benchmarks.git",
    "h3h": "https://github.com/MWieland/h3h.git",
    "geospatial-risk-mapper": "https://github.com/L-Gardiner/geospatial-risk-mapper.git",
}

DEST.mkdir(parents=True, exist_ok=True)

for name, url in REPOS.items():
    target = DEST / name
    if target.exists():
        print(f"[SKIP] {name}")
        continue
    print(f"[CLONE] {name}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)

print("Research repositories cloned.")
