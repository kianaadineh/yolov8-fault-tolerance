# Fault-Tolerant YOLOv8 for Resource-Constrained Computing Systems

Research on fault injection, error propagation, and selective
fault tolerance in YOLOv8 object detection systems.

## Research model

- Architecture: YOLOv8s
- Parameters: 11,166,560
- Fault-eligible Conv2d layers: 63
- Fault-eligible Conv2d weights: 11,146,080

## Current research status

| Activity | Status |
|---|---|
| YOLOv8s baseline | Completed |
| COCO 2017 fault-free validation | Completed |
| Official COCOeval baseline | Completed |
| INT8 bit-flip utilities | Completed |
| INT4 weight fault-injection study | Preliminary results published |
| NMS-aware single-image analysis | Preliminary results published |
| Activation fault injection | Planned |
| Hybrid fault-tolerance mechanisms | Planned |
| Faulty-model COCO evaluation | Planned |
| Runtime and energy evaluation | Planned |

## Published fault-injection experiments

- 36 initial targeted INT4 scenarios.
- 96 extended balanced INT4 scenarios across 12 selected layers.
- 12 additional preliminary TMR scenarios.
- The 132 INT4 scenarios exclude the additional TMR scenarios.

These experiments do not establish dataset-wide fault tolerance.
Single-image NMS matching is not equivalent to COCO mAP.

## Research artifacts

- [Notebook 04 closure report](docs/research_reports/Notebook_04_Closure_Report_v1.md)
- [Research file index](docs/research_reports/Notebook_04_Closure_File_Index_v1.csv)
- [Fault-injection results](results/fault_injection/)

## Repository organization

- `notebooks/`: research notebooks 00–04.
- `src/`: INT4 and INT8 bit-flip utilities.
- `experiments/baseline/`: baseline metadata.
- `results/fault_injection/`: versioned research outputs.
- `docs/research_reports/`: research reports and file index.
- `paper/`: experimental evaluation documentation.

Some other repository directories are currently empty.
Git does not preserve empty directories in a fresh clone.

## Environment

Research experiments use Python, PyTorch, and Ultralytics.
The current requirements.txt is incomplete and not fully pinned.
Model weights and large datasets are excluded from Git.

## Limitations

The research is ongoing. Published fault-injection findings
are preliminary and do not establish deployment-level reliability.
Theoretical INT4 storage estimates are not measured runtime
or energy overhead.

## Research roadmap

1. Expand fault sensitivity and error propagation analysis.
2. Investigate activation faults.
3. Develop hybrid selective fault-tolerance mechanisms.
4. Evaluate accuracy, reliability, and resource overhead.
5. Study multi-objective optimization.

