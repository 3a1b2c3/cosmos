# Build a Cosmos 3 transfer segmentation control video from game footage.
#
# The transfer cookbook consumes precomputed control videos; nothing in that
# repository produces them. This uses SAM 2's video predictor, which propagates
# masks from prompts on the first frame, so each object keeps one colour for the
# whole clip. Per-frame segmentation flickers at boundaries and that flicker
# becomes shimmer in the generated video.
#
# Matching the shipped assets/seg/control_seg.mp4: flat colour per object,
# black background, no fixed semantic palette. 41% of that frame is unlabeled,
# so covering only the salient objects is enough.
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Distinct flat colours, background stays black. Chosen far apart in RGB so
# H.264 ringing at boundaries cannot blend one class into another.
PALETTE = [
    (0, 0, 248),
    (177, 127, 67),
    (137, 0, 0),
    (169, 221, 249),
    (0, 67, 251),
    (128, 128, 0),
    (0, 160, 80),
    (240, 120, 200),
    (255, 200, 0),
    (100, 0, 160),
    (0, 200, 200),
    (200, 60, 0),
]


def parse_points(text):
    """"x,y;x,y" into [(x, y), ...]; one object per point."""
    points = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        points.append((float(x), float(y)))
    return points


def extract_frames(video_path, frames_dir, want_frames, target_fps, start_seconds, stride):
    """SAM 2 reads a directory of JPEGs named 00000.jpg upward.

    Frames are strided so the control covers the same wall-clock motion as the
    source. Taking consecutive frames from 60fps footage and writing them at
    30fps would halve the speed of everything in the clip, and the generated
    video would inherit that.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
    if stride is None:
        stride = max(1, round(source_fps / target_fps))
    if start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_seconds * source_fps)))

    written = 0
    read_index = 0
    size = None
    while written < want_frames:
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


def auto_prompts(first_frame_path, max_objects, model_id, device):
    """Segment the first frame, keep the largest regions, return their centroids.

    The video predictor takes point prompts, so each automatic mask is reduced
    to one interior point. Largest-first because small regions survive neither
    propagation nor the model's downscaling of the control video.
    """
    generator = SAM2AutomaticMaskGenerator.from_pretrained(model_id, device=device)
    image = cv2.cvtColor(cv2.imread(str(first_frame_path)), cv2.COLOR_BGR2RGB)
    masks = generator.generate(image)
    masks.sort(key=lambda entry: entry["area"], reverse=True)

    points = []
    for entry in masks[:max_objects]:
        segmentation = entry["segmentation"]
        ys, xs = np.nonzero(segmentation)
        if len(xs) == 0:
            continue
        # Centroid can land outside a concave mask; snap to the nearest pixel
        # that is actually inside it.
        cx, cy = xs.mean(), ys.mean()
        nearest = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        points.append((float(xs[nearest]), float(ys[nearest])))
    if not points:
        raise SystemExit("automatic segmentation found no regions; pass --points instead")
    del generator
    torch.cuda.empty_cache()
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="source footage")
    parser.add_argument("-o", "--out", type=Path, required=True, help="control video to write")
    parser.add_argument("--frames", type=int, default=121, help="frame count; seg.json wants 121")
    parser.add_argument("--fps", type=float, default=30.0, help="output fps; seg.json wants 30")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the source to begin")
    parser.add_argument("--stride", type=int, help="frame step; defaults to source fps over target fps")
    parser.add_argument("--points", type=str, help='"x,y;x,y" in source pixels, one per object')
    parser.add_argument("--max-objects", type=int, default=8, help="cap for automatic prompting")
    parser.add_argument("--model", type=str, default="facebook/sam2.1-hiera-large")
    parser.add_argument("--keep-frames", action="store_true", help="keep the extracted JPEGs")
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"missing: {args.video}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; SAM 2 video propagation on CPU is impractically slow")
    device = "cuda"

    work_dir = Path(tempfile.mkdtemp(prefix="segmap-"))
    frames_dir = work_dir / "frames"
    frames_dir.mkdir()
    try:
        count, size = extract_frames(args.video, frames_dir, args.frames, args.fps, args.start, args.stride)
        print(f"extracted {count} frames at {size[0]}x{size[1]}")
        if count < args.frames:
            print(f"WARNING: source has {count} frames, {args.frames} requested. The spec expects")
            print(f"         exactly {args.frames}; a shorter control may be rejected or padded.")

        if args.points:
            points = parse_points(args.points)
            print(f"prompting {len(points)} object(s) from --points")
        else:
            print(f"segmenting frame 0 automatically, keeping the {args.max_objects} largest...")
            points = auto_prompts(frames_dir / "00000.jpg", args.max_objects, args.model, device)
            print(f"found {len(points)} object(s)")
        if len(points) > len(PALETTE):
            raise SystemExit(f"{len(points)} objects exceeds the {len(PALETTE)}-colour palette")

        predictor = SAM2VideoPredictor.from_pretrained(args.model, device=device)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(frames_dir))
            for object_id, (x, y) in enumerate(points):
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=object_id,
                    points=np.array([[x, y]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )

            # Painted lowest-id-last so the first prompted object wins overlaps;
            # automatic prompts are largest-first, which puts big background
            # surfaces underneath the smaller objects sitting on them.
            canvases = {}
            for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
                canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                order = sorted(range(len(object_ids)), key=lambda i: object_ids[i], reverse=True)
                for i in order:
                    mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    canvas[mask] = PALETTE[object_ids[i]]
                canvases[frame_idx] = canvas

        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264", quality=8)
        for frame_idx in sorted(canvases):
            writer.append_data(canvases[frame_idx])
        writer.close()

        coverage = [float((canvases[idx].any(axis=2)).mean()) for idx in sorted(canvases)]
        print(f"\nwrote {args.out}")
        print(f"  {len(canvases)} frames at {args.fps} fps, {size[0]}x{size[1]}")
        print(f"  labelled pixels: first {100 * coverage[0]:.1f}%, "
              f"mean {100 * np.mean(coverage):.1f}%, last {100 * coverage[-1]:.1f}% "
              f"(the shipped asset is ~59%)")

        # A mean alone hides the characteristic failure: propagation holds for a
        # while, then every object is lost at once and the rest of the clip is
        # blank. Averaged with the good frames that still reads as a plausible
        # number, so the collapse is reported by where it happens.
        lost = next((i for i, value in enumerate(coverage) if value < 0.01 * coverage[0]), None)
        if lost is not None:
            seconds = args.start + lost / args.fps
            print(f"\n  WARNING: tracking collapsed at frame {lost} (source {seconds:.2f}s);")
            print(f"           {len(coverage) - lost} of {len(coverage)} frames are effectively blank.")
            print( "           Usually a flash or dissolve, which scene-cut detection misses")
            print( "           because the change is spread over several frames. Move --start")
            print( "           past it, or shorten the clip to end before it.")
        elif np.mean(coverage) < 0.05:
            print("\n  WARNING: almost nothing was labelled. Try --points, or raise --max-objects.")
    finally:
        if args.keep_frames:
            print(f"  frames kept in {frames_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
