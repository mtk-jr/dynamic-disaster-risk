# Real-data migration

## Remote sensing
Start with a manageable flood dataset. Candidate repositories include ML4Floods, UDL4FL and ImpactMesh.

## GIS
Use OSM, DEM, river networks, roads, buildings and permitted exposure/population layers.

## IoT
Begin with CSV/simulation. Then replace ingestion with MQTT and ESP32/hardware if available.

## Social
Start with CrisisBench/CrisisNLP for reproducible experiments. Add live APIs only after the offline pipeline works.

## Alignment
For each event/spatial unit create:
- event_id
- timestamp
- latitude
- longitude
- h3_cell
- hazard/risk label

Do not combine unrelated datasets without a defensible alignment strategy.
