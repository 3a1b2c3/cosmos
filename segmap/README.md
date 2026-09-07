# segmap

Build a Cosmos 3 transfer **segmentation control video** from ordinary footage.

The transfer cookbook consumes precomputed control videos and contains no code
to produce them — the shipped `assets/seg/control_seg.mp4` is a sample, not a
tool. This fills that gap for the `seg` control.

## Why SAM 2

For engine-rendered sources, the renderer's own semantic-segmentation pass beats
anything estimated from pixels: exact boundaries, stable IDs, no inference cost.
Use that when it is available.

This exists for the case where it is not — recorded gameplay, captured video,
anything where the frames are all you have. SAM 2's **video predictor**
propagates masks from prompts on the first frame, so an object keeps one colour
for the whole clip. Per-frame segmentation is the wrong tool: it flickers at
boundaries, and a flickering control signal becomes shimmer in the generated
video.

## What the target format actually requires

Read off the shipped asset rather than assumed, by extracting frame 0 and
counting colours:

- **No fixed semantic palette.** The colours are flat and arbitrary — not
  ADE20K or Cityscapes conventions. Any palette works as long as it is
  *consistent across frames*, which is what makes propagation the requirement
  rather than the labels themselves.
- **Partial coverage is fine.** 41% of the reference frame is unlabeled black.
  Segmenting the few salient objects is enough; wall-to-wall labelling is not
  expected.
- **Resolution is flexible.** The asset is 1920x1080 while `seg.json` asks for
  720p, so the model resizes the control. Match the aspect ratio and frame
  count; the exact pixel dimensions matter less.
- **Compression tolerance is high.** The reference is visibly H.264-degraded —
  ragged boundaries, speckle — and works anyway. About 36,000 distinct RGB
  values appear in a frame holding roughly seven real classes. No need to
  over-engineer encoding quality.

`seg.json` wants **121 frames at 30 fps, 16:9**. `wsm` is the outlier at 101
frames and 10 fps.

## Setup

```bat
setup.bat
```

```bash
bash setup.sh
```

Both build a Python 3.11 venv beside this file. Torch is installed **first and
alone**, from the cu130 index on Windows and via `uv --torch-backend` on Linux,
so a later resolve cannot quietly replace it with a CPU wheel. The optional SAM 2
CUDA extension is skipped with `SAM2_BUILD_CUDA=0`: it needs a compiler and is
only used for mask post-processing, not for video propagation.

## Use

Automatic — segments frame 0, keeps the largest regions, propagates them:

```bat
.venv\Scripts\python.exe make_seg_control.py <video> -o control_seg.mp4
```

Manual, when automatic picks the wrong things. One point per object, in source
pixel coordinates on the first extracted frame:

```bat
.venv\Scripts\python.exe make_seg_control.py <video> -o control_seg.mp4 --points "640,400;1200,700"
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--frames` | `121` | Frame count, matching `seg.json`. |
| `--fps` | `30` | Output frame rate, matching `seg.json`. |
| `--start` | `0` | Seconds into the source to begin. |
| `--stride` | source fps / target fps | Frame step. |
| `--points` | automatic | Explicit prompts instead of automatic segmentation. |
| `--max-objects` | `8` | Cap on automatic prompting. |
| `--model` | `facebook/sam2.1-hiera-large` | SAM 2 checkpoint. |
| `--keep-frames` | off | Keep the extracted JPEGs, to find coordinates. |

### Two things that decide whether the output is usable

**Stride.** 60 fps footage written as 121 frames at 30 fps would otherwise be
two seconds of action in slow motion, and the generated video inherits that.
The default stride keeps wall-clock motion intact; `--stride 1` disables it if
slow motion is what you want.

**Cuts.** Propagation tracks objects forward from frame 0 and cannot survive a
scene change — after a cut, the masks follow whatever now occupies those
positions. Trailers and edited footage need `--start` chosen to land inside one
continuous shot. Roughly four seconds of source at the default stride, so any
shot shorter than that will contain a cut.

The script prints the share of labelled pixels against the reference's ~59%.
Much lower means automatic prompting missed the salient objects, and `--points`
is the fix.

## Feeding it back into transfer

Copy the spec and repoint it:

```bat
copy ..\cookbooks\cosmos3\generator\transfer\specs\seg.json ..\cookbooks\cosmos3\generator\transfer\specs\seg_mine.json
```

Edit `seg.control_path` to the generated file and `prompt_path` to a caption for
the scene you want *generated*. The shipped `prompt.json` files are structured
scene descriptions — subjects, lighting, cinematography, per-segment actions —
not one-liners, and the caption carries as much weight as the control does.

Note that the framework path runs guardrails for every control, including edge
and depth, and that needs `nvidia/Cosmos-Guardrail1` — gated separately from
`nvidia/Cosmos-1.0-Guardrail`. See `RUNNING.md` in the repository root.
