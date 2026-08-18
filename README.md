# Dynamic Disaster Risk Assessment

Four-modality early-fusion project for Remote Sensing + IoT + GIS + Social Media.

The first implementation target is **flood risk**. The architecture is disaster-agnostic so other hazards can be added later.

## Quick start

Python 3.11 is recommended.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the complete MVP pipeline:
```bash
python scripts/clone_research_repos.py
python scripts/run_pipeline.py
```

Start API:
```bash
uvicorn src.api.app:app --reload
```

Start dashboard:
```bash
streamlit run dashboard/app.py
```

Docker:
```bash
docker compose up --build
```

## Architecture

```text
Satellite ─┐
IoT       ─┤
GIS       ─┼─> event/spatial alignment -> modality encoders
Social    ─┘                              -> early fusion -> risk
```

Every real sample must eventually have:
- event_id
- timestamp
- latitude
- longitude
- h3_cell
- risk_label

The current MVP uses synthetic data so the entire software pipeline can be tested before real datasets are aligned.

## Git

```bash
git init
git add .
git commit -m "Initial multimodal disaster risk architecture"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

Feature branches:
```bash
git checkout -b feature/satellite
git add .
git commit -m "Add satellite preprocessing"
git push -u origin feature/satellite
```
