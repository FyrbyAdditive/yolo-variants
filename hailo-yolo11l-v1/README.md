# hailo-yolo11l-v1

**Backbone**: Ultralytics YOLOv11-l (large) · 25.3M params after trim

**Target runtime**: Hailo-8 / Hailo-8L PCIe accelerator via libhailort
4.23 (NMS-on-chip). Designed for the fnvr-hailo-broker.

NOT yolo26 — Hailo's compiler doesn't support yolo26's NMS-free
attention head. yolov11 is the closest Hailo-Zoo-compatible family.

**Trim**: stock 80-class COCO head sliced to 15 classes (see
`trim.json`). No fine-tuning. 3 classification convs sliced (yolov11l
has the standard `cv3` head only — half as many trim points as
yolo26's dual `cv3` + `one2one_cv3`).

**Files**:

- `best.onnx` (97 MB) — vanilla Ultralytics export, opset 11, NMS
  stripped, FP32. Shape `[1, 19, 8400]` = `[boxes(4) + classes(15),
  anchors]` — what Hailo's parser maps onto its NMS post-process layer.
  This is the **input** to the HEF compile; not loadable directly by
  the broker.
- `best.hef` — produced by `tools/compile-hef/` from `best.onnx`
  (this file lands here once compilation finishes; see "Building the
  HEF" below).
- `labels.txt` · `trim.json`

## Building the HEF (one-time, x86 only)

Hailo's Dataflow Compiler (DFC) is closed-source, x86-only, and
version-locked to specific `hailo_model_zoo` revisions. The fnvr
recipe pairs:

| DFC | hailort (broker) | model_zoo |
|-----|------------------|-----------|
| **3.32.0** | **4.23** | **v2.16** |

Newer DFC versions (3.33.x, 3.34.x) have a pyparsing parseAll bug
that breaks the bundled `.alls` script loading. 3.32.0 doesn't have
this bug. HEFs from DFC 3.32 run fine on hailort 4.23 (HEF format
stable across DFC 3.x patch revs).

```sh
# From the fnvr repo, with DFC 3.32 wheel in tools/compile-hef/:
cd tools/compile-hef/
docker build -t fnvr-compile-hef .

# Stage inputs: ONNX from this dir + 281 calibration JPEGs from the
# Orin's recorded footage (real on-camera frames produce the best
# INT8 quantisation — the calibrator observes activation
# distributions from your actual site).
mkdir -p /tmp/hef-build/onnx-src /tmp/hef-build/dataset/images/train /tmp/hef-build/out
cp ../../yolo-variants/hailo-yolo11l-v1/best.onnx /tmp/hef-build/onnx-src/
rsync -az --rsync-path="sudo rsync" \
    tim@172.16.4.23:/var/lib/docker/volumes/fnvr_fnvr-data/_data/models/yolo26/calib_images/ \
    /tmp/hef-build/dataset/images/train/

# Compile (~10-20 min, INT8 calibration is the long bit).
docker run --rm \
    -v "/tmp/hef-build:/work" \
    -v "/tmp/hef-build/onnx-src:/work/onnx-src:ro" \
    -v "/tmp/hef-build/dataset:/work/dataset:ro" \
    fnvr-compile-hef \
    /work/compile.sh fnvr-yolo11l-v1
```

Output: `/tmp/hef-build/out/fnvr-yolo11l-v1.hef` — copy back into this
directory as `best.hef` for archival, plus deploy to the Orin.

## Deploy on the fnvr Orin

```sh
# 1. Copy the HEF into the broker's model dir.
scp best.hef tim@172.16.4.23:/tmp/
ssh tim@172.16.4.23 'sudo mv /tmp/best.hef \
    /var/lib/docker/volumes/fnvr_fnvr-data/_data/models/hailo/fnvr-yolo11l-v1.hef'

# 2. Flip detector.hailo_model_version (Settings → Object detector,
#    "Hailo model" field: type "fnvr-yolo11l-v1") or via API:
CJ=$(mktemp)
curl -sSc $CJ -d '{"username":"admin","password":"admin"}' \
    -H 'Content-Type: application/json' \
    http://172.16.4.23:8081/api/v1/auth/login >/dev/null
curl -b $CJ -X PUT -H 'Content-Type: application/json' \
    -d '{"yolo26_variant":"fnvr-yolo26m-v1","yolo26_precision":"fp16",
         "anpr_enabled":true,"face_id_enabled":true,
         "hailo_model_version":"fnvr-yolo11l-v1"}' \
    http://172.16.4.23:8081/api/v1/settings/detector

# 3. Restart the hailo-broker (the entrypoint resolves the HEF path
#    from detector.hailo_model_version on startup).
ssh tim@172.16.4.23 'cd /home/tim/fnvr && \
    sudo docker compose -f deploy/docker/docker-compose.yml \
                        -f deploy/docker/docker-compose.hailo.yml \
                        restart hailo-broker'

# 4. Verify in the broker log:
ssh tim@172.16.4.23 'sudo docker logs fnvr-hailo-broker-1 2>&1 | head -5'
# Should read:
#   hailo-broker: using fine-tuned HEF: /var/lib/fnvr/models/hailo/fnvr-yolo11l-v1.hef
#   hailo: configured 15 classes, up to 100 bboxes/class, batch_size=4, ...
```

If the file is missing or the broker can't load it, the entrypoint
gracefully falls back to `stock` (`yolov11l.hef`, 80 COCO classes)
with a warning log.

## Performance

To be measured once the first compile completes. Expectations vs the
stock 80-class HEF:

- Per-inference time: marginally faster (smaller NMS post-process)
- Output size: ~30 KB / inference (15 × 100 × 16-byte bboxes max)
  vs ~160 KB stock — 5× smaller
- INT8 accuracy on kept classes: should match stock; the calibration
  set draws from real site footage so the quantisation grid is
  better-tuned than the stock recipe's COCO calibration.

## Reproducing

```sh
# 1. Trim the ONNX (CPU only, fast).
docker run --rm -v $PWD:/work -w /work fnvr-train-detector \
    python trim-classes.py \
        --target hailo \
        --keep 0 1 2 3 4 5 6 7 8 15 16 17 18 19 21 \
        --names person bicycle car motorcycle airplane bus train truck boat cat dog horse sheep cow bear \
        --out fnvr-yolo11l-v1

# 2. Compile to HEF (see "Building the HEF" above).
```
