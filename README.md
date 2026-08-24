# hypospadias-object-detection

Code for Stage 1 (object detection) of the surgical phase recognition pipeline:
a YOLOv8 instrument detector trained on the AVOS bounding-box dataset
(bovie, needle driver, forceps), evaluated zero-shot on hypospadias video
frames, with the full statistics battery from the Methods and a
dataset-agnostic comparison report so additional datasets can be scored
through the same pipeline for context (e.g. a held-out AVOS procedure type,
or an external instrument-detection benchmark).

## Layout

```
configs/stage1_datasets.yaml   dataset registry: model path, classes, per-dataset image/label paths
src/stage1_detection/
  extract_frames.py            randomly samples frames from source videos + writes an UNLABELED
                                labeling template for an expert to fill in presence/absence
  metrics.py                   confusion counts, accuracy/precision/recall/F1, Wilson 95% CI,
                                binomial test vs. chance, two-proportion z-test between datasets
  predict.py                   runs YOLOv8 over a frame set -> per-frame per-class presence/absence
  report.py                    builds the per-class / pooled / pairwise-comparison CSV report
scripts/run_stage1_eval.py     CLI: runs every dataset in the config through the pipeline
tests/                         unit tests for the statistics + frame-extraction modules
```

## Building a labeled eval set from raw videos

`hypospadias_eval` (and any other video-derived dataset) starts as **unlabeled**
video. Extract random frames and generate a blank labeling template:

```bash
python -m stage1_detection.extract_frames \
  --videos_dir data/hypospadias_videos \
  --output_dir data/hypospadias_eval/images \
  --n_per_video 8 \
  --seed 42 \
  --model_path models/yolov8_avos_best.pt
```

This writes the JPEG frames, a `manifest.csv` (frame_id/video_id/frame_index/
timestamp), and a `labels_template.csv` with every `label` cell **blank** --
sampling is random and reproducible via `--seed`, but presence/absence is not
inferred. The template's class list is read from `--model_path`'s own class
names (the AVOS bounding-box classes it was trained on) rather than typed out
by hand; pass `--classes a,b,c` instead only if you want to template a subset.
An expert fills in 0/1 per row, and the completed file is saved as
that dataset's `labels_csv`. `load_expert_labels` (in `predict.py`) refuses to
load a file with any blank or non-0/1 label cells, so an unfinished template
can't accidentally be scored as ground truth.

## Expected inputs

- `model_path` in the config: a YOLOv8 checkpoint trained on AVOS (COCO-pretrained init).
- Per dataset: an `images_dir` of frames and a `labels_csv` of expert
  presence/absence labels with columns `frame_id,class,label` (label is 0/1).
  This matches the Stage 1 protocol: presence/absence per class per frame,
  not box-level localization.

## Running an evaluation

```bash
pip install -r requirements.txt
python scripts/run_stage1_eval.py --config configs/stage1_datasets.yaml --out results/stage1
```

This scores every dataset listed in the config (e.g. `avos_test`,
`hypospadias_eval`, and any additional comparator you add) and writes to
`results/stage1/`:

- `per_class_metrics.csv` — accuracy, precision, recall, F1, Wilson 95% CI,
  binomial p-value vs. chance, per class per dataset
- `pooled_metrics.csv` — same, pooled across classes, per dataset
- `pairwise_comparisons.csv` — two-proportion z-test (pooled and per-class)
  between every pair of datasets in the config, e.g. AVOS test accuracy vs.
  hypospadias zero-shot accuracy
- `summary.json`

## Adding another comparison dataset

Add an entry under `datasets:` in `configs/stage1_datasets.yaml` pointing at
its `images_dir` and `labels_csv`; it's automatically included in the
per-class/pooled tables and in every pairwise comparison against the other
configured datasets. See the commented-out `avos_heldout_procedure` example
in that file for the recommended comparator: an AVOS procedure type held out
of training, which isolates "generalizes to a new open-surgery procedure"
from "generalizes to the hypospadias imaging setup specifically."

## Tests

```bash
pytest tests/ -q
```

The statistics module (`metrics.py`) is fully unit-tested. `predict.py`
imports `ultralytics` lazily so the metrics/report code can be tested
without a model checkpoint or image data present.
