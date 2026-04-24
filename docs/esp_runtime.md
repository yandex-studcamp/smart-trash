# ESP32-CAM Runtime Pipeline

## Network

Connect the ESP32-CAM and the laptop to the same Wi-Fi network. A phone hotspot is fine.

## Start the laptop server

From the project root:

```powershell
.\scripts\start-server.ps1 -ListenHost 0.0.0.0 -Port 8000
```

The script prints one or more LAN URLs, for example:

```text
LAN URL: http://192.168.43.120:8000
```

## ESP request

Send each camera frame as raw JPEG bytes:

```text
POST http://192.168.43.120:8000/api/pipeline/frame
Content-Type: image/jpeg
Body: JPEG bytes
```

The response is JSON. Read the `command` field:

```text
-1 none, no stable object decision yet
 0 other
 1 plastic
 2 paper
```

Example local test from PowerShell:

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/pipeline/frame" -Method POST -ContentType "image/jpeg" -InFile "captures/example.jpg"
```

## Runtime behavior

The server keeps a spotter sliding window. Until the window is full, it returns `command = -1`. Once the anomaly ratio is above the configured threshold, it crops anomalous frames using the PatchCore heatmap or mask, classifies clean crops with the YOLO classifier, and returns the majority class.

Duplicate command protection is controlled by `PIPELINE_COMMAND_COOLDOWN_FRAMES`. The default is `0`, so duplicate suppression is disabled. If it is greater than zero, repeated identical commands inside the cooldown are suppressed as `command = -1` with `duplicate = true` and `suppressed_command` set to the original command.

## Main environment knobs

```text
PIPELINE_SPOTTER_WINDOW_SIZE=8
PIPELINE_SPOTTER_TRUE_RATIO_THRESHOLD=0.6
PIPELINE_CLASSIFIER_CONF_THRESHOLD=0.3
PIPELINE_CROP_PADDING=0.10
PIPELINE_CROP_MIN_PADDING=6
PIPELINE_SAVE_DEBUG_ARTIFACTS=true
PIPELINE_DEBUG_DIR=runs/runtime_pipeline
PIPELINE_BACKGROUND_REF=
PIPELINE_COMMAND_COOLDOWN_FRAMES=0
```
