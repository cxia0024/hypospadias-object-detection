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
  train.py                     fine-tunes a COCO-pretrained YOLOv8 checkpoint on your AVOS train/val split
  avos_labels.py               converts AVOS's own YOLO box .txt labels into frame_id,class,label
                                ground truth -- no new expert labeling needed for avos_test
  extract_frames.py            randomly samples frames from source videos + writes an UNLABELED
                                labeling template for an expert to fill in presence/absence
  metrics.py                   confusion counts, accuracy/precision/recall/F1, Wilson 95% CI,
                                binomial test vs. chance, two-proportion z-test between datasets
  predict.py                   runs YOLOv8 over a frame set -> per-frame per-class presence/absence
  report.py                    builds the per-class / pooled / pairwise-comparison CSV report
scripts/run_stage1_eval.py     CLI: runs every dataset in the config through the pipeline
notebooks/
  train_yolo_colab.ipynb       Colab notebook: train on your prepared AVOS split
  run_stage1_eval_colab.ipynb  Colab notebook: run the Stage 1 zero-shot eval
tests/                         unit tests for the statistics + frame-extraction + training modules
```

## Training on your AVOS train/val split

Once your train/val split is ready as a standard Ultralytics YOLO dataset
(a `data.yaml` with `train:`, `val:`, and `names:` keys, pointing at your
image/label folders -- this script doesn't create or modify the split itself):

```bash
pip install -r requirements.txt
python -m stage1_detection.train \
  --data path/to/avos_data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --out models/yolov8_avos_best.pt
```

- `--model yolov8s.pt` is the COCO-pretrained starting point (per the
  Methods); ultralytics downloads it automatically if not already local.
  Swap in `yolov8n.pt` for a faster/smaller run or `yolov8m.pt`/`yolov8l.pt`
  for more capacity if you have the GPU memory.
- Class names come entirely from your `data.yaml`'s `names:` list -- nothing
  in this script hardcodes AVOS's instrument classes.
- `validate_data_yaml` checks the config and that the `train`/`val` paths
  actually exist *before* training starts, so a typo fails immediately
  instead of after an hour of training.
- After training, the best checkpoint (`runs/train/avos_yolov8/weights/best.pt`)
  is copied to `--out` (default `models/yolov8_avos_best.pt`), which is
  exactly the `model_path` the Stage 1 eval config expects -- so you can go
  straight from training into `run_stage1_eval.py` / `extract_frames.py`.

No GPU locally? Use `notebooks/train_yolo_colab.ipynb` -- same steps, run on
a Colab GPU runtime, with the resulting checkpoint saved back to Drive.

## Validating with AVOS

`avos_test` needs the same `frame_id,class,label` ground truth format as
every other dataset in the eval config -- but unlike `hypospadias_eval`,
that ground truth already exists: it's the YOLO bounding-box `.txt` files
from your train/val split. Convert them instead of relabeling anything:

```bash
python -m stage1_detection.avos_labels \
  --images_dir path/to/avos/images/val \
  --labels_dir path/to/avos/labels/val \
  --model_path models/yolov8_avos_best.pt \
  --out_csv data/avos_test/labels.csv
```

This reduces "which boxes are in this image" to "which classes are present"
(the unit Stage 1 is scored on) -- a missing/empty `.txt` file means no
objects were annotated in that image (a legitimate background frame in YOLO
convention), not a labeling gap. Class order comes from `--model_path` (the
checkpoint's own class names), which matches the `.txt` files' class indices
since they were trained on the same AVOS data. Point `avos_test.images_dir`
at the same `images/val` folder and `avos_test.labels_csv` at the CSV this
writes, then run the evaluation scoped to just that dataset (see below) --
`hypospadias_eval` isn't expert-labeled yet, so it can't be scored alongside it.

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

Only have `avos_test` ready so far? Scope the run with `--datasets` so it
doesn't try (and fail) to load an unlabeled `hypospadias_eval`:

```bash
python scripts/run_stage1_eval.py --config configs/stage1_datasets.yaml --out results/stage1 --datasets avos_test
```

Drop `--datasets` once every dataset you want in the comparison has a real
`labels_csv`. This scores every dataset listed in the config (e.g.
`avos_test`, `hypospadias_eval`, and any additional comparator you add) and
writes to `results/stage1/`:

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
