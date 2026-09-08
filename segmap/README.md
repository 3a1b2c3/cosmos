# segmap

Build Cosmos 3 transfer **control videos** from ordinary footage, and the
prompts to go with them.

The transfer cookbook consumes precomputed control videos and contains no code
to produce them — the shipped `assets/seg/control_seg.mp4` is a sample, not a
tool. This fills that gap.

## Three pipelines

| | Script | Runner | Control means |
| --- | --- | --- | --- |
| **Instance** | `make_seg_control.py` | `convert.bat` / `.sh` | SAM 2 regions, tracked per shot |
| **Semantic** | `make_seg_control_semantic.py` | `convert_semantic.bat` / `.sh` | A fixed class per colour, globally |
| **Geometric** | `make_geo_control.py` | `convert_geo.bat` / `.sh` | Canny edges, or relative depth |

Which to reach for depends entirely on the source material, and the answer is
not obvious — see [Choosing a pipeline](#choosing-a-pipeline).

## Setup

```bat
setup.bat
```

```bash
bash setup.sh
```

Both build a Python 3.11 venv beside this file. Torch is installed **first and
alone**, from the cu130 index on Windows and via `uv --torch-backend` on Linux,
so a later resolve cannot quietly replace it with a CPU wheel.

Three install details that cost time to discover:

- The SAM 2 dependency is given as a **bare git URL**, not `sam2 @ git+…`.
  Upstream declares the distribution as `SAM-2` while the import package is
  `sam2`, and uv rejects a named direct reference for the mismatch.
- `huggingface_hub` and `timm` are listed **explicitly**. SAM 2 imports the
  first inside `from_pretrained`, and DETR's ResNet backbone needs the second;
  neither declares the dependency, so both fail at first use rather than at
  install time.
- `SAM2_BUILD_CUDA=0` skips the optional mask post-processing extension, which
  needs a compiler and is not used by video propagation. Its absence prints a
  `cannot import name '_C'` warning on every run, which is harmless.

## What the target format actually requires

Measured from the shipped assets rather than assumed:

- **There is no standard palette.** `assets/seg/control_seg.mp4` and
  `assets/wsm/control_wsm.mp4` use *different* arbitrary colours, and neither
  matches Cityscapes — nearest-colour distances of 18–103, with a 36%-of-frame
  region best-matching "motorcycle". Nothing in the cookbook documents a
  class-to-colour mapping. So the model appears to treat a seg control as
  *distinct region identity*, not as a semantic lookup.
- **Sparse is normal.** The shipped seg asset is 41% unlabeled black and the
  wsm asset 77%. High coverage is not a requirement.
- **Resolution is flexible.** The seg asset is 1920x1080 while `seg.json` asks
  for 720p, so the model resizes. Match aspect ratio and frame count.
- **Compression tolerance is high.** The reference is visibly H.264-degraded
  and works anyway: about 36,000 distinct RGB values appear in a frame holding
  roughly seven real regions.

`seg.json` wants **121 frames at 30 fps, 16:9**. `wsm` is the outlier at 101
frames and 10 fps.

## Instance control (SAM 2)

```bat
.venv\Scripts\python.exe make_seg_control.py <video> -o control_seg.mp4
```

SAM 2's video predictor propagates masks forward from prompts on the first frame
of a shot, so an object keeps one colour for as long as it is tracked. Per-frame
segmentation is the wrong tool here: it flickers at boundaries, and a flickering
control becomes shimmer in the generated video.

| Flag | Default | Effect |
| --- | --- | --- |
| `--frames` | `121` | Frame count; `0` for the whole source. |
| `--fps` | `30` | Output frame rate. |
| `--start` / `--stride` | `0` / auto | Where to begin, and the frame step. |
| `--detect` | `car,truck,bus` | COCO classes always labelled; empty disables. |
| `--max-objects` | `14` | Cap on automatic prompting, up to the 24-colour palette. |
| `--cut-threshold` | `25` | Frame difference counted as a shot change. |
| `--reprompt` | `0` | Re-prompt every N frames within a shot. |
| `--points` | automatic | `"x,y;x,y"` explicit prompts instead. |
| `--points-per-side` | `32` | Sampling grid; higher finds smaller regions. |
| `--min-area-frac` | `0.0` | Drop regions below this fraction of the frame. |

### Three things that decide whether the output is usable

**Detection, or cars go missing.** SAM 2 is class-agnostic: it ranks regions by
area and knows nothing about what they are, so a car that is not among the
largest simply never gets a colour. `--detect` runs DETR on each shot's first
frame and hands those boxes to SAM 2 *before* the automatic pass fills the
remaining slots. Detections take the leading object ids, so they are painted
last and win any overlap.

**`--cut-threshold` must match the footage.** Fast camera motion produces frame
differences as large as real cuts. On a lateral tracking shot the default of 25
fragmented the clip into sub-6-frame "shots" which were then dropped, blanking
35 of 48 frames; at 70 it found the two genuine shots and coverage went from 8%
to 72%. Conversely a static portrait needs a low threshold to catch real cuts.
There is no safe universal value — check the reported shot count.

**`--reprompt` for fast motion.** Propagation decays when subjects move far from
where they were prompted; on fast lateral footage the objects picked on frame 0
have left the frame before the shot ends and coverage falls to zero. Subdividing
gives each window fresh prompts. The cost is that object identity, and so
colour, resets at each boundary.

Measured on the same 2-second clip: no re-prompting gave 23.3% mean coverage and
a blank final frame; `--reprompt 15` gave 27.1% with the tail recovered;
`--reprompt 8` gave 33.5% with the fewest blank frames.

**Flashes are not cuts, but they end tracking.** A white flash or dissolve
spreads its change over several frames, so no single delta trips the threshold
while SAM 2 still loses every object at once. One clip held ~25% coverage for 72
frames and went to exactly zero at a flash the detector called clean. Brightness
jumps find these where frame differencing does not.

The script reports coverage for the **first, mean and last** frame and names the
frame where tracking collapsed. A mean alone hides this: half a good clip and
half a blank one averages to a plausible-looking number.

## Semantic control (SegFormer)

```bat
.venv\Scripts\python.exe make_seg_control_semantic.py <video> -o control_seg.mp4
```

Labels every pixel with a class, so a colour means the same thing in every frame
of every video. `--palette alpasim` (default) uses the ten-class AV scheme from
alpasim's transfuser plugin, where everything not relevant to driving collapses
to black — matching the shipped controls' sparsity. `--palette cityscapes` gives
the full nineteen.

No shot detection or propagation: class labels are stable frame to frame by
construction, so cuts stop mattering entirely.

**It failed on stylised game footage**, and the failure is instructive.
SegFormer/Cityscapes labelled a bridge deck as `fence`, a UI element as
`traffic sign`, and found only **2.2% vehicles in a driving trailer**. Cityscapes
is trained on real German dashcam footage; the domain gap was too large. It also
churned 4.6% of pixels per frame against SAM 2's 1.3%, because per-frame
inference has no temporal mechanism and an uncertain model flips borderline
pixels. Expect this path to work on real driving video and struggle on anything
stylised.

## Geometric control (edge, depth)

```bat
convert_geo.bat
```

```bat
set CONTROL=depth ^& convert_geo.bat
```

Edge is pure Canny — no model, no download, near-instant. It pre-blurs by
default so compression noise does not become fake edges, and warns when the
result is too sparse or too dense. Depth uses Depth Anything V2, normalised per
frame because the model predicts relative depth with no fixed scale.

Neither has classes to get wrong, so neither has a domain gap. For stylised or
synthetic sources this is the path with the fewest ways to fail.

## Choosing a pipeline

- **Engine-rendered source, and you can re-render** — use the renderer's own
  semantic pass. Exact boundaries, stable ids, no inference cost. None of these
  scripts beat it.
- **Real driving footage** — semantic, `--palette alpasim`.
- **Stylised, synthetic or unusual footage** — geometric. The measured failure
  above is what semantic segmentation does outside its training domain.
- **You need specific objects tracked as individuals** — instance, with
  `--detect` naming their classes.

## Prompts

`make_style_prompts.py --tag <clip>` and `make_portrait_prompts.py` generate
prompt and spec pairs. Both write one file per style so a single control can be
run against several appearances.

Conventions taken from the shipped `assets/seg/prompt.json`:

- **Prompts are structured JSON, not sentences** — 18 fields covering content,
  look and time. The whole object is serialised and handed to the model.
- **A prompt never references the control.** The shipped one contains zero
  mentions of `segmentation`, `mask`, `control`, `label` or `map`; its colour
  words all describe objects in the target scene. Phrases like "as given by the
  control signal" are not a convention the model is trained on — fill those
  fields in concretely, describing what the control actually shows, so the two
  reinforce rather than conflict.
- **The framework overrides `duration` with the chunk length.** A 60-second
  script was rewritten to `4s` and fed identically into every chunk, so
  time-ranged `actions` spanning a long clip never reach the model. Write them
  for the window the model actually sees.
- **Vary appearance, hold camera.** Across style variants the camera and
  composition fields stay identical, because they describe motion the control
  already encodes. Only materials, light, palette and background change.
- The negative prompt uses the same schema plus `physical_realism`. Naming
  specific observed artifacts there works better than generic badness.

## Frame counts

The framework warns that anything outside **[24, 200]** frames may degrade.
`num_video_frames_per_chunk` is the size of one pass; `num_frames` beyond that
is produced by chaining, each chunk conditioned on a single frame from the
previous one.

An 80-second run at 2405 frames took 21 chunks and ~39 minutes, against the
warning. Setting `num_frames` and `num_video_frames_per_chunk` equal and within
range gives a single pass with no hand-off, which is the cleanest configuration.

Note the chunk arithmetic: with `num_conditional_frames: 1`, each chunk after
the first contributes only 120 new frames, so 2405 frames needs 21 chunks rather
than 20.

## Running a transfer

The notebooks hardcode the shipped specs, so custom controls are not reachable
from them. Use the runners in the repository root:

```bash
SPEC=.../specs/seg_drive_a.json bash run_custom_transfer.sh
```

`run_custom_transfer.sh` resolves the spec's control paths, checks they exist,
and verifies `num_frames` matches the control's real frame count with ffprobe —
all before any GPU time. That mismatch is the most likely way a run goes wrong,
since the spec is edited by hand and the video is produced here.

`FRAMES` and `STEPS` override the spec without editing it, `EXTRA_ARGS` passes
flags through verbatim, and `GUARDRAILS=1` restores the default guardrail.

`run_segmented_transfer.sh` cuts a long control into independent in-range
segments and skips any that already have output, so an interrupted job resumes
rather than restarting.

## Measured results

| Control | Coverage (mean) | Notes |
| --- | --- | --- |
| Portrait, 180 frames | **61.6%** | Person and car detected in every window |
| Driving A, 48 frames | 73.0% | `--cut-threshold 70`, vehicles detected |
| Driving C, 60 frames | 33.5% | `--reprompt 8`; heavy motion blur and VFX |
| Cityscapes, 300 frames | ~100% | Largely misclassified on this source |

Sampling on the GB300 with Cosmos3-Nano runs about 1 min 53 s per 121-frame
chunk at 50 steps with the diffusion cache active, which serves roughly half the
transformer evaluations from cache (`full=69 skipped=81`). That is a fidelity
trade; disabling it is the second lever after `num_steps` if output looks noisy.
