# Build a Cosmos 3 transfer segmentation control video by semantic class.
#
# The SAM 2 version in make_seg_control.py segments *regions* with no idea what
# they are, so a colour means "the third-largest blob in this shot" and changes
# meaning at every cut. Cosmos then renders those arbitrary shapes literally, as
# flat slabs of colour with hard edges.
#
# This labels every pixel with a Cityscapes class instead. Road is one colour
# everywhere in the video, car another, and the boundaries follow real objects,
# so the control carries meaning the transfer model can act on rather than
# shapes it can only copy.
#
# Consequences of the different approach:
#   - No shot detection or mask propagation. Class labels are stable frame to
#     frame by construction, so cuts and flash transitions stop mattering.
#   - Coverage is ~100% rather than the ~27% SAM 2 was producing.
#   - No instance separation: two adjacent cars become one region of "car".
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# The Cityscapes 19-class evaluation palette, in trainId order. Fixed by the
# dataset rather than chosen here, which is the whole point: the mapping from
# colour to meaning is the same in every frame of every video.
CITYSCAPES = [
    ("road", (128, 64, 128)),
    ("sidewalk", (244, 35, 232)),
    ("building", (70, 70, 70)),
    ("wall", (102, 102, 156)),
    ("fence", (190, 153, 153)),
    ("pole", (153, 153, 153)),
    ("traffic light", (250, 170, 30)),
    ("traffic sign", (220, 220, 0)),
    ("vegetation", (107, 142, 35)),
    ("terrain", (152, 251, 152)),
    ("sky", (70, 130, 180)),
    ("person", (220, 20, 60)),
    ("rider", (255, 0, 0)),
    ("car", (0, 0, 142)),
    ("truck", (0, 0, 70)),
    ("bus", (0, 60, 100)),
    ("train", (0, 80, 100)),
    ("motorcycle", (0, 0, 230)),
    ("bicycle", (119, 11, 32)),
]
PALETTE = np.array([colour for _, colour in CITYSCAPES], dtype=np.uint8)
CLASS_NAMES = [name for name, _ in CITYSCAPES]

# The AV palette from alpasim's transfuser plugin (TRANSFUSER_SEMANTIC_COLORS),
# which has a SIM2REAL converter -- sim2real being this exact use case. Ten
# classes rather than nineteen, everything not relevant to driving collapsed to
# black. That matches the shipped Cosmos controls, which are 41% (seg) and 77%
# (wsm) unlabeled: the sparsity is the convention, not a shortcoming.
ALPASIM = [
    ("unlabeled", (0, 0, 0)),
    ("vehicle", (31, 119, 180)),
    ("road", (128, 64, 128)),
    ("traffic light", (250, 170, 30)),
    ("pedestrian", (0, 255, 60)),
    ("road line", (157, 234, 50)),
    ("obstacle", (255, 0, 0)),
    ("special vehicle", (255, 255, 0)),
    ("stop sign", (125, 0, 0)),
    ("biker", (220, 20, 60)),
]

# Cityscapes trainId -> index into ALPASIM. Buildings, sky, vegetation and the
# rest of the static scenery become unlabeled: the transfer model infers those
# from the prompt, and marking them only adds shapes it might copy literally.
CITYSCAPES_TO_ALPASIM = {
    0: 2,    # road
    1: 0,    # sidewalk    -> unlabeled
    2: 0,    # building    -> unlabeled
    3: 0,    # wall        -> unlabeled
    4: 0,    # fence       -> unlabeled
    5: 0,    # pole        -> unlabeled
    6: 3,    # traffic light
    7: 8,    # traffic sign -> stop sign, the nearest available class
    8: 0,    # vegetation  -> unlabeled
    9: 0,    # terrain     -> unlabeled
    10: 0,   # sky         -> unlabeled
    11: 4,   # person      -> pedestrian
    12: 9,   # rider       -> biker
    13: 1,   # car         -> vehicle
    14: 1,   # truck       -> vehicle
    15: 7,   # bus         -> special vehicle
    16: 7,   # train       -> special vehicle
    17: 9,   # motorcycle  -> biker
    18: 9,   # bicycle     -> biker
}


def build_palette(name):
    """Return (lookup table, palette array, class names) for a palette choice.

    The lookup maps SegFormer's 19 Cityscapes trainIds onto whichever scheme is
    wanted, so the model stays the same and only the output encoding changes.
    """
    if name == "cityscapes":
        table = np.arange(len(CITYSCAPES), dtype=np.uint8)
        scheme = CITYSCAPES
    elif name == "alpasim":
        table = np.array([CITYSCAPES_TO_ALPASIM[i] for i in range(len(CITYSCAPES))], dtype=np.uint8)
        scheme = ALPASIM
    else:
        raise SystemExit(f"unknown palette '{name}'")
    return table, np.array([colour for _, colour in scheme], dtype=np.uint8), [n for n, _ in scheme]


def extract_frames(video_path, frames_dir, want_frames, target_fps, start_seconds, stride):
    """Frames are strided so the control covers the same wall-clock motion.

    Taking consecutive frames from 60fps footage and writing them at 30fps would
    halve the speed of everything in the clip, and the generated video would
    inherit that.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
    if stride is None:
        stride = max(1, round(source_fps / target_fps))
    if start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_seconds * source_fps)))

    limit = want_frames if want_frames > 0 else float("inf")
    written = 0
    read_index = 0
    size = None
    while written < limit:
        ok, frame = capture.read()
        if not ok:
            break
        if read_index % stride == 0:
            if size is None:
                size = (frame.shape[1], frame.shape[0])
            cv2.imwrite(str(frames_dir / f"{written:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
        read_index += 1
    capture.release()
    if written == 0:
        raise SystemExit(f"no frames decoded from {video_path}")
    print(f"source {source_fps:g} fps, stride {stride}, from {start_seconds:g}s")
    return written, size


def segment_batch(paths, processor, model, size, device):
    """Class-id maps for a batch of frames, at the frames' own resolution."""
    images = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    # SegFormer emits logits at a quarter of the input resolution, so they are
    # resized back before argmax rather than after: taking argmax first and then
    # scaling would interpolate between class ids, which are not a continuum.
    upsampled = torch.nn.functional.interpolate(
        logits, size=(size[1], size[0]), mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1).cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="source footage")
    parser.add_argument("-o", "--out", type=Path, required=True, help="control video to write")
    parser.add_argument("--frames", type=int, default=121, help="frame count; 0 for the whole source")
    parser.add_argument("--fps", type=float, default=30.0, help="output fps; seg.json wants 30")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the source to begin")
    parser.add_argument("--stride", type=int, help="frame step; defaults to source fps over target fps")
    parser.add_argument("--batch", type=int, default=4, help="frames per forward pass")
    parser.add_argument("--palette", choices=("alpasim", "cityscapes"), default="alpasim",
                        help="alpasim: 10 AV classes, scenery black, matching the shipped controls")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    parser.add_argument("--keep-frames", action="store_true", help="keep the extracted JPEGs")
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"missing: {args.video}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; per-frame segmentation on CPU is impractically slow")
    device = "cuda"

    work_dir = Path(tempfile.mkdtemp(prefix="segmap-sem-"))
    frames_dir = work_dir / "frames"
    frames_dir.mkdir()
    try:
        count, size = extract_frames(args.video, frames_dir, args.frames, args.fps, args.start, args.stride)
        print(f"extracted {count} frames at {size[0]}x{size[1]}")

        lookup, palette, class_names = build_palette(args.palette)
        print(f"palette: {args.palette}, {len(class_names)} classes")

        processor = SegformerImageProcessor.from_pretrained(args.model)
        model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
        if model.config.num_labels != len(CITYSCAPES):
            raise SystemExit(
                f"{args.model} has {model.config.num_labels} classes, not the {len(CITYSCAPES)} "
                "this palette is built for. Use a Cityscapes-finetuned checkpoint."
            )

        # Written as each batch completes rather than collected first: holding a
        # minute of 1080p canvases costs about 15 GB.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=1)
        histogram = np.zeros(len(class_names), dtype=np.int64)
        try:
            for first in range(0, count, args.batch):
                paths = [frames_dir / f"{index:05d}.jpg"
                         for index in range(first, min(first + args.batch, count))]
                for class_map in segment_batch(paths, processor, model, size, device):
                    mapped = lookup[class_map]
                    writer.append_data(palette[mapped])
                    histogram += np.bincount(mapped.ravel(), minlength=len(class_names))
                done = min(first + args.batch, count)
                print(f"\r  {done}/{count} frames", end="", flush=True)
        finally:
            writer.close()
        print()

        print(f"\nwrote {args.out}")
        print(f"  {count} frames at {args.fps} fps, {size[0]}x{size[1]}, {count / args.fps:.1f}s")
        print("  classes present, by share of pixels:")
        total = histogram.sum()
        for index in np.argsort(-histogram):
            share = 100 * histogram[index] / total
            if share < 0.05:
                continue
            print(f"    {class_names[index]:<16} {share:5.1f}%  rgb{tuple(int(v) for v in palette[index])}")
    finally:
        if args.keep_frames:
            print(f"  frames kept in {frames_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
