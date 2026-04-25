# gpu-yolo26l-v1

**Backbone**: Ultralytics YOLO26-l (large) · 26.4M params after trim

**Target runtime**: NVIDIA GPU via DeepStream nvinfer (yolo26 NMS-free
parser, FP16 TensorRT engine).

**Trim**: stock 80-class COCO head sliced to 15 classes (see
`trim.json`). No fine-tuning. 6 classification convs sliced.

**Files**:

- `best.onnx` (95 MB) — ONNX from `export_yolo26.py`
- `labels.txt`
- `trim.json`

**Deploy** — identical to [gpu-yolo26m-v1/README.md](../gpu-yolo26m-v1/README.md)
except `yolo26_variant=fnvr-yolo26l-v1`.

**Performance** (fnvr, 6 cameras × 1080p):

- GPU (GR3D): avg ~82 % (vs ~95 % stock yolo26l)
- GPU power: 30-33 W
- Engine size: 53 MB
- Detection accuracy: marginally better than yolo26m on small / distant
  objects (small people, far cars). Gap shrinks once you've fine-tuned
  on actual site footage.

**When to pick this over gpu-yolo26m-v1**:

- The Orin has spare GPU after ANPR/face-ID is loaded
- You want best accuracy on the kept classes, accepting ~17 pp more
  GPU load
- Single Orin doing nothing else compute-heavy

**Reproducing**:

```sh
docker run --rm -v $PWD:/work -w /work fnvr-train-detector \
    python trim-classes.py \
        --target gpu \
        --source yolo26l.pt \
        --keep 0 1 2 3 4 5 6 7 8 15 16 17 18 19 21 \
        --names person bicycle car motorcycle airplane bus train truck boat cat dog horse sheep cow bear \
        --out fnvr-yolo26l-v1
```
