# yolo-variants

Class-trimmed YOLO models for the [fnvr](https://github.com/fnvr-project)
NVR. Each variant slices the stock 80-class COCO detection head down
to the 15 classes that matter for residential security (people,
vehicles, large mammals — the full list is in [`trim.json`](trim.json)).
**No fine-tuning** — all kept-class weights are byte-identical to
upstream Ultralytics.

## Why trim instead of fine-tune

A trimmed head is smaller, faster, and uses less GPU/NPU memory than
the stock 80-class head, with **zero accuracy regression on kept
classes** (their weights are unchanged). Fine-tuning is the next
step, but it requires a labelled dataset; trimming is free and
deployable today.

## License

**AGPL-3.0** (inherited from Ultralytics' weight licensing). See
[LICENSE](LICENSE) and [NOTICE](NOTICE).

## Storage

Binary weight files (`*.onnx`, `*.hef`) are tracked via
[Git LFS](https://git-lfs.com/). Clone with:

```sh
brew install git-lfs && git lfs install
git clone https://github.com/FyrbyAdditive/yolo-variants
```

If you deploy these weights as part of a network service, AGPL-3.0
requires you to make the corresponding source available to users
who interact with that service over a network.

## Variants

| Directory | Backbone | Target | Format | Status |
|---|---|---|---|---|
| [`gpu-yolo26m-v1/`](gpu-yolo26m-v1/) | YOLO26-m | NVIDIA GPU (DeepStream) | ONNX | Live on fnvr |
| [`gpu-yolo26l-v1/`](gpu-yolo26l-v1/) | YOLO26-l | NVIDIA GPU (DeepStream) | ONNX | Tested on fnvr |
| [`gpu-yolo26x-v1/`](gpu-yolo26x-v1/) | YOLO26-x | NVIDIA GPU (DeepStream) | ONNX | Built, untested |
| [`hailo-yolo11l-v1/`](hailo-yolo11l-v1/) | YOLOv11-l | Hailo-8 / Hailo-8L | ONNX + HEF | Tested on fnvr (batch=2) |
| [`hailo-yolo11m-v1/`](hailo-yolo11m-v1/) | YOLOv11-m | Hailo-8 / Hailo-8L | ONNX + HEF | Live on fnvr (batch=4, ~15 fps/cam) |

The YOLO26 family is the preferred GPU detector (better accuracy
than YOLOv11 at the same parameter count) but Hailo's compiler
doesn't support its NMS-free attention head — so the Hailo path
falls back to YOLOv11-l.

## Class set

The same 15-class trim applies to every variant. Class IDs in the
trimmed model are 0..14 in this order:

| ID | Slug | Original COCO ID |
|----|------|------------------|
| 0 | person | 0 |
| 1 | bicycle | 1 |
| 2 | car | 2 |
| 3 | motorcycle | 3 |
| 4 | airplane | 4 |
| 5 | bus | 5 |
| 6 | train | 6 |
| 7 | truck | 7 |
| 8 | boat | 8 |
| 9 | cat | 15 |
| 10 | dog | 16 |
| 11 | horse | 17 |
| 12 | sheep | 18 |
| 13 | cow | 19 |
| 14 | bear | 21 |

Pick the set you actually want by editing `--keep` and `--names` in
the trim invocation; everything downstream (label files, dataset
yaml, runtime decoders) reads from the same trim.

## Choosing a variant

```
                       │ Hailo-8 hardware? │
                       │       no          │       yes
                       ▼                   ▼
              ┌─────────────────┐     hailo-yolo11l-v1
              │   on a GPU      │
              ▼                 ▼
       ≥3 cameras live?    1-2 cameras
              │                 │
              ▼                 ▼
       gpu-yolo26m-v1    gpu-yolo26l-v1
                             (or 26x for
                              max accuracy)
```

## Reproducing

The trim script lives in the fnvr repo at
[`tools/train-detector/trim-classes.py`](https://github.com/fnvr-project/fnvr/blob/main/tools/train-detector/trim-classes.py).
Each variant's README has a copy-pasteable Docker invocation. CPU-only
work — no GPU needed for the trim itself; ~30 sec per variant.

For the Hailo HEF compile, see
[`hailo-yolo11l-v1/README.md`](hailo-yolo11l-v1/README.md). That step
needs Hailo's Dataflow Compiler (closed-source, x86-only, requires
a free Hailo developer-zone account to download).
