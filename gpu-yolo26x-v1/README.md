# gpu-yolo26x-v1

**Backbone**: Ultralytics YOLO26-x (extra-large) · ~57M params after trim

**Target runtime**: NVIDIA GPU via DeepStream nvinfer.

**Trim**: stock 80-class COCO head sliced to 15 classes. No fine-tuning.
6 classification convs sliced.

**Files**:

- `best.onnx` (213 MB) — biggest of the family
- `labels.txt`
- `trim.json`

**Deploy** — identical to [gpu-yolo26m-v1/README.md](../gpu-yolo26m-v1/README.md)
except `yolo26_variant=fnvr-yolo26x-v1`.

**Performance** (untested on fnvr's Orin AGX as of 2026-04-25 — extrapolation):

- Expected GPU (GR3D): pinned at 95-98 % with 6 cameras
- Expected engine size: ~110 MB (vs ~289 MB stock yolo26x)
- Expected accuracy: best on small/occluded objects at distance

**When to pick this**:

- Single-camera or low-camera-count deployments
- High-end NVIDIA hardware (e.g. dedicated dGPU, Orin used only for
  detection)
- Site footage has lots of distant / partially-occluded targets

**When NOT to pick it**:

- Multi-camera (3+) Orin AGX — runs out of GPU headroom
- Live recording running in parallel

**Reproducing**:

```sh
docker run --rm -v $PWD:/work -w /work fnvr-train-detector \
    python trim-classes.py \
        --target gpu \
        --source yolo26x.pt \
        --keep 0 1 2 3 4 5 6 7 8 15 16 17 18 19 21 \
        --names person bicycle car motorcycle airplane bus train truck boat cat dog horse sheep cow bear \
        --out fnvr-yolo26x-v1
```
