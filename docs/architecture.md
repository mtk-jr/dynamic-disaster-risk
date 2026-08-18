# Architecture

## Four pipelines

Remote sensing -> preprocessing -> environmental/flood features -> vector

IoT -> validation -> temporal aggregation -> sensor features -> vector

GIS -> spatial joins/raster statistics -> terrain/infrastructure features -> vector

Social -> text cleaning -> classifier/embedding -> social features -> vector

## Alignment

Common keys:
```text
event_id
timestamp
latitude
longitude
h3_cell
```

The MVP uses event_id for deterministic joining. The production system should also perform spatial and temporal window matching.

## Early fusion

```text
satellite -> encoder -> z_sat
iot       -> encoder -> z_iot
gis       -> encoder -> z_gis
social    -> encoder -> z_social

[z_sat | z_iot | z_gis | z_social]
                 |
                 v
             fusion MLP
                 |
                 v
       LOW/MODERATE/HIGH/CRITICAL
```
