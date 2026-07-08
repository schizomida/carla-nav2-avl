# Execution plan (status-tracked)

1. [x] Folder + git + PROMPT.md + design spec
2. [ ] Core package: config, fusion, model wrappers (lazy), pipeline, demo
       — offline tests green with no weights installed
3. [ ] Weights: yolo11n-seg (or research agent's pick) + YOLOPv2; demo runs
       on testdata/ street scenes (road green, cars blue, people red)
4. [ ] Course model: dataset (research agent's ranked pick) -> fine-tune on
       5090 -> cones orange + white lines white on test photos
5. [ ] Bench table (5090) + tools/export_trt.py for Jetson FP16 engines
6. [ ] Results report: sample overlays + timings sent to user; commit log
       clean; deploy-to-dinosaur checklist (blocked: car offline)

Agents: research (models/datasets) + test-media acquisition run in
parallel with step 2.
