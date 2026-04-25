# hailo-yolo11m-v1

**Backbone**: Ultralytics YOLOv11-m (medium) · ~20M params after trim

**Target runtime**: Hailo-8 / Hailo-8L PCIe accelerator via libhailort
4.23 (NMS-on-chip).

Smaller sibling to [`hailo-yolo11l-v1`](../hailo-yolo11l-v1/) — same
trim, lighter backbone. Should fit at higher batch sizes on the
hailo8l (yolov11l forces batch=2 due to 8-context layout; yolov11m
*may* configure cleanly at batch=4, freeing more aggregate
throughput when multiple cameras share the device).

**Trim**: stock 80-class COCO head sliced to 15 classes (see
`trim.json`). No fine-tuning. 3 classification convs sliced.

**Files**:

- `best.onnx` (77 MB) — vanilla Ultralytics export, opset 11, NMS
  stripped, FP32. Input to the Hailo HEF compile.
- `best.hef` (43 MB) — INT8-quantised, hailo8l target, 15-class NMS
  on chip. Compiled with DFC 3.32.0 + hailo_model_zoo v2.16, 281
  calibration frames sampled from real fnvr camera footage.
- `labels.txt` · `trim.json`

## Compile reproducer

Same machinery as [`hailo-yolo11l-v1`](../hailo-yolo11l-v1/README.md);
just pass `yolov11m` as the model name when invoking the compile
script:

```sh
docker run --rm \
    -e USER=fnvr \
    -v "$PWD:/work" \
    -v "$PWD/onnx-src:/work/onnx-src:ro" \
    -v "$PWD/dataset:/work/dataset:ro" \
    fnvr-compile-hef \
    /work/compile.sh fnvr-yolo11m-v1 yolov11m
```

The bundled per-variant `.alls` recipes (yolov11m.alls vs yolov11l.alls)
hard-code different conv numbers — passing the wrong model name to
the recipe surfaces as `Given layers yolov11l/conv120 not exist in
the HN`.

## Deploy

Identical to [`hailo-yolo11l-v1/README.md`](../hailo-yolo11l-v1/README.md)
except `hailo_model_version=fnvr-yolo11m-v1`.

## Performance

Expected (vs the L variant on this hardware):

- Per-inference time: ~30-40 % faster (smaller backbone)
- Engine size: 43 MB (vs 47 MB)
- Likely max batch on hailo8l: 4 (vs 2 for L) — TBD on first deploy
- Detection accuracy on the kept classes: marginally lower than L
  (yolov11m mAP ~51.5 vs L ~53.4 on COCO at imgsz=640), but
  likely indistinguishable on residential-CCTV-grade scenes
