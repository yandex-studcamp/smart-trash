# smart-trash
Smart Trash: Classifier and Spotter for waste

## Spotter Through anomalib

### What is implemented now

The `spotter` is implemented as an image anomaly detector based on `anomalib` and the `PatchCore` model.

Current source files:

- config loading and path resolution: [src/spotter/config.py](/C:/Users/memel/Studcamp/smart-trash/src/spotter/config.py:160)
- dataset preparation: [src/spotter/dataset.py](/C:/Users/memel/Studcamp/smart-trash/src/spotter/dataset.py:79)
- training and evaluation: [src/spotter/training.py](/C:/Users/memel/Studcamp/smart-trash/src/spotter/training.py:139)
- runtime inference wrapper: [src/spotter/predictor.py](/C:/Users/memel/Studcamp/smart-trash/src/spotter/predictor.py:120)

The logic is the following:

1. `PatchCore` sees only normal images during training.
2. It builds a memory bank of normal feature patches.
3. During inference it compares the new frame against the learned representation of normal frames.
4. If the frame differs too much, the anomaly score grows and the frame is marked as anomalous.

### Is the implementation good enough right now

Overall, for a first practical baseline, yes.

What is good:

- The architecture is simple and maintainable.
- `PatchCore` is a strong baseline for few-shot anomaly detection.
- The code already has a clean split into config, dataset prep, training, and runtime prediction.
- The runtime wrapper can be used directly from the pipeline code with a saved checkpoint.

What is important to keep in mind:

- The model is frame-based, not sequence-based. It works on one image at a time.
- The current split is random by frame. If many frames come from the same short video fragment, train and test can become too similar, and metrics may look better than they really are.
- All anomaly images are currently used only for evaluation, not for training. This is correct for classical anomaly detection, but it means test quality depends a lot on how representative those anomalies are.
- The final decision in production should usually not rely on one isolated frame. It is better to aggregate anomaly scores over a short sliding window on the server side.

### What data format the model expects

There are three different stages: raw data, prepared dataset, and runtime inference.

#### 1. Raw data before preparation

The preparation script expects two folders:

```text
data/spotter/captures_esp/
├── normal/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
└── anomaly/
    ├── frame_101.jpg
    ├── frame_102.jpg
    └── ...
```

Requirements:

- files must be images with one of the configured extensions, by default: `.jpg`, `.jpeg`, `.png`, `.bmp`
- `normal/` must contain frames without trash or without the target anomaly
- `anomaly/` must contain frames where the anomaly is present

The model does not expect masks in the current implementation.

#### 2. Prepared dataset for anomalib

The preparation script transforms the raw folders into the format expected by `anomalib.data.Folder`:

```text
data/spotter/prepared/<exp_name>/
├── train/
│   └── good/
│       ├── ...
└── test/
    ├── good/
    │   ├── ...
    └── anomaly/
        ├── ...
```

Meaning:

- `train/good` is used for training the normal-state representation
- `test/good` is used to verify that normal frames stay normal
- `test/anomaly` is used to evaluate anomaly detection quality

With the current code:

- `normal` frames are split into `train/good` and `test/good`
- all `anomaly` frames go into `test/anomaly`

#### 3. Runtime inference input

At inference time the wrapper `TorchSpotterPredictor.predict(...)` expects one image at a time.

Supported input types:

- image path as `str` or `Path`
- `numpy.ndarray`
- `torch.Tensor`
- `PIL.Image.Image`

So the current spotter is:

- good for single-frame scoring
- usable inside a loop over frames
- not a temporal model by itself

If you want sliding-window logic, it should be implemented outside the model in the pipeline layer.

## How to create the dataset and run training

### 1. Prepare raw data

Put your frames into:

```text
data/spotter/captures_esp/normal
data/spotter/captures_esp/anomaly
```

If needed, adjust the config file:

- [src/config/spotter_patchcore.yaml](/C:/Users/memel/Studcamp/smart-trash/src/config/spotter_patchcore.yaml:1)

Main parameters there:

- `raw_data.normal_dir`, `raw_data.anomaly_dir`
- `dataset.train_ratio`
- `model.image_size`, `model.center_crop_size`
- `model.backbone`, `model.layers`
- `engine.accelerator`, `engine.export_type`

### 2. Prepare anomalib dataset

Run:

```powershell
.venv\Scripts\python scripts\prepare_spotter_dataset.py `
  --exp_name esp_patchcore `
  --config src\config\spotter_patchcore.yaml `
  --force
```

What this does:

- reads raw frames from `normal` and `anomaly`
- randomly splits normal frames into train and test parts
- creates the prepared dataset under `data/spotter/prepared/esp_patchcore`
- writes `manifest.json` with the exact split

### 3. Train the spotter

CPU:

```powershell
.venv\Scripts\python scripts\train_spotter_patchcore.py `
  --exp_name esp_patchcore `
  --config src\config\spotter_patchcore.yaml `
  --accelerator cpu
```

GPU:

```powershell
.venv\Scripts\python scripts\train_spotter_patchcore.py `
  --exp_name esp_patchcore `
  --config src\config\spotter_patchcore.yaml `
  --accelerator gpu
```

Optional export after training:

```powershell
.venv\Scripts\python scripts\train_spotter_patchcore.py `
  --exp_name esp_patchcore `
  --config src\config\spotter_patchcore.yaml `
  --accelerator gpu `
  --export_type onnx
```

Training outputs are written to:

```text
experiments/spotter/<exp_name>/
├── weights/patchcore.ckpt
├── train_summary.json
└── resolved_config.yaml
```

### 4. What happens during training and testing

The training script:

1. loads config
2. prepares dataset if it is missing
3. creates `anomalib.data.Folder`
4. creates `PatchCore`
5. runs `engine.fit(...)`
6. saves checkpoint to `weights/patchcore.ckpt`
7. runs `engine.test(...)`
8. writes summary JSON with metrics and paths

## How to run inference

### From Python code

Example:

```python
from pathlib import Path
from src.spotter import TorchSpotterPredictor, load_spotter_config

workspace_root = Path.cwd()
config = load_spotter_config("src/config/spotter_patchcore.yaml", workspace_root=workspace_root)

predictor = TorchSpotterPredictor(
    checkpoint_path="experiments/spotter/esp_patchcore/weights/patchcore.ckpt",
    config=config,
    device="cpu",
)

prediction = predictor.predict("data/spotter/captures_esp/anomaly/frame_0001.jpg")

print("score:", prediction.score)
print("label:", prediction.label)
print("is_anomaly:", prediction.is_anomaly)
print("anomaly_map shape:", None if prediction.anomaly_map is None else prediction.anomaly_map.shape)
```

Returned fields:

- `score`: anomaly score
- `label`: binary decision from the model post-processing
- `is_anomaly`: boolean wrapper around `label`
- `anomaly_map`: per-pixel anomaly intensity map
- `pred_mask`: predicted binary anomaly mask

### How to use it in the pipeline

The current recommended way is:

1. receive frames from ESP
2. score each frame independently with `predictor.predict(frame)`
3. keep the last `N` scores in a short sliding window
4. trigger the downstream classifier only if the anomaly signal is stable enough

Pseudo-code:

```python
from collections import deque

window = deque(maxlen=8)

def handle_frame(frame):
    prediction = predictor.predict(frame)
    window.append(prediction.score or 0.0)

    high_scores = [score for score in window if score > 0.55]
    if max(window, default=0.0) > 0.85 or len(high_scores) >= 4:
        run_classifier(frame)
```

This is usually better than triggering the classifier on every single anomalous frame.

## Practical recommendations

- If your frames come from the same short video sessions, split by session, not only by random frame.
- Keep the camera position and lighting as stable as possible.
- Do not over-interpret very high metrics on tiny datasets.
- Start with server-side inference and sliding-window aggregation before trying to compress the logic onto ESP.
