# gpu-yolo26m-v1

**Backbone**: Ultralytics YOLO26-m (medium) · 20.1M params before trim

**Target runtime**: NVIDIA GPU via DeepStream nvinfer (yolo26 NMS-free
parser, FP16 TensorRT engine). Designed for the fnvr pipeline.

**Trim**: stock 80-class COCO head sliced to 15 classes (see
`trim.json`). No fine-tuning; all kept-class weights identical to
upstream Ultralytics yolo26m.pt. Three classification convs trimmed
per scale × two head branches (`cv3` + `one2one_cv3`) = 6 layers
sliced.

**Files**:

- `best.onnx` (78 MB) — ONNX export via DeepStream-Yolo's
  `export_yolo26.py`. Output shape `[1, 8400, 6]` = `[boxes(4) +
  score(1) + label(1)]` per anchor — what `NvDsInferParseYolo`
  expects.
- `labels.txt` — 15 class slugs in `yolo_id` order (0=person…14=bear)
- `trim.json` — kept-id list for reproducibility

**Deploy on the fnvr Orin**:

```sh
# 1. Push the ONNX into the pipeline's model cache.
scp best.onnx tim@172.16.4.23:/tmp/
ssh tim@172.16.4.23 'sudo mv /tmp/best.onnx \
    /var/lib/docker/volumes/fnvr_fnvr-data/_data/models/yolo26/fnvr-yolo26m-v1.onnx'

# 2. Flip the detector setting via the api (or Settings → Object
#    detector in the UI: pick "Custom fine-tuned" + name "fnvr-yolo26m-v1").
CJ=$(mktemp)
curl -sSc $CJ -d '{"username":"admin","password":"admin"}' \
    -H 'Content-Type: application/json' \
    http://172.16.4.23:8081/api/v1/auth/login >/dev/null
curl -b $CJ -X PUT -H 'Content-Type: application/json' \
    -d '{"yolo26_variant":"fnvr-yolo26m-v1","yolo26_precision":"fp16",
         "anpr_enabled":true,"face_id_enabled":true,
         "hailo_model_version":"stock"}' \
    http://172.16.4.23:8081/api/v1/settings/detector

# 3. Restart pipeline. nvinfer rebuilds the TRT engine on first
#    inference (~5-10 min for yolo26m), the entrypoint watcher
#    relocates the freshly-built engine into the volume so
#    subsequent restarts deserialise instantly.
curl -b $CJ -X POST http://172.16.4.23:8081/api/v1/system/pipeline/restart
```

**Performance** (fnvr deployment, 6 cameras × 1080p, all live):

- GPU (GR3D): avg ~65 % (vs ~95 % on stock 80-class yolo26l)
- GPU power: 27-29 W (vs 33 W stock)
- Engine size: 44 MB (vs ~95 MB stock)

**Tradeoffs**:

- Lower-capacity backbone than `gpu-yolo26l-v1` — slightly worse
  small-object recall, similar performance on people/vehicles at
  reasonable distance.
- Best balance for 6-camera Orin AGX deployments where GPU headroom
  matters (e.g. running ANPR + face-ID alongside).

**Reproducing this artefact** (off-Orin, on a CUDA host):

```sh
docker run --rm -v $PWD:/work -w /work fnvr-train-detector \
    python trim-classes.py \
        --target gpu \
        --source yolo26m.pt \
        --keep 0 1 2 3 4 5 6 7 8 15 16 17 18 19 21 \
        --names person bicycle car motorcycle airplane bus train truck boat cat dog horse sheep cow bear \
        --out fnvr-yolo26m-v1
```
