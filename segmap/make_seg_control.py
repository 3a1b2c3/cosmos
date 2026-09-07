# Build a Cosmos 3 transfer segmentation control video from ordinary footage.
#
# The transfer cookbook consumes precomputed control videos; nothing in that
# repository produces them. This uses SAM 2's video predictor, which propagates
# masks from prompts on the first frame of a shot, so an object keeps one colour
# for as long as it is tracked. Per-frame segmentation flickers at boundaries and
# that flicker becomes shimmer in the generated video.
#
# Propagation cannot cross a scene change, so anything longer than a single shot
# is split at its cuts and re-prompted for each one. Cuts are found by frame
# difference rather than ffmpeg's scene filter, which misses flashes and
# dissolves: their change is spread over several frames, so each individual delta
# stays under the threshold while SAM 2 still loses every object at once.
#
# Matching the shipped assets/seg/control_seg.mp4: flat colour per object, black
# background, no fixed semantic palette. 41% of that frame is unlabeled, so
# covering only the salient objects is enough.
import argparse
import os
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
    (60, 220, 60),
    (255, 255, 255),
    (120, 90, 255),
    (200, 200, 120),
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


def find_shot_boundaries(frames_dir, count, threshold):
    """Indices where a new shot begins, by mean absolute frame difference.

    Deliberately not ffmpeg's scene filter: a flash or dissolve spreads its
    change across several frames, so no single delta trips that threshold even
    though tracking dies. Comparing downscaled greyscale catches both, and the
    cost is trivial next to propagation.
    """
    boundaries = [0]
    previous = None
    for index in range(count):
        frame = cv2.imread(str(frames_dir / f"{index:05d}.jpg"))
        small = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None and float(np.abs(small - previous).mean()) > threshold:
            boundaries.append(index)
        previous = small
    return boundaries


def auto_prompts(frame_path, max_objects, generator):
    """Segment one frame, keep the largest regions, return interior points.

    The video predictor takes point prompts, so each automatic mask is reduced
    to one interior point. Largest-first because small regions survive neither
    propagation nor the model's downscaling of the control video.
    """
    image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    masks = generator.generate(image)
    masks.sort(key=lambda entry: entry["area"], reverse=True)

    points = []
    for entry in masks[:max_objects]:
        ys, xs = np.nonzero(entry["segmentation"])
        if len(xs) == 0:
            continue
        # Centroid can land outside a concave mask; snap to the nearest pixel
        # that is actually inside it.
        cx, cy = xs.mean(), ys.mean()
        nearest = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        points.append((float(xs[nearest]), float(ys[nearest])))
    return points


def link_segment(frames_dir, segment_dir, first, last):
    """Renumber a shot's frames from zero, since init_state expects that.

    Hard links rather than copies: same volume, and a minute of 1080p JPEGs is
    large enough that duplicating it is worth avoiding.
    """
    segment_dir.mkdir()
    for offset, index in enumerate(range(first, last)):
        source = frames_dir / f"{index:05d}.jpg"
        target = segment_dir / f"{offset:05d}.jpg"
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)


def propagate_segment(predictor, segment_dir, points, size, device):
    """Masks for one shot, as {local frame index: HxWx3 canvas}."""
    canvases = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(segment_dir))
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
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
            canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            order = sorted(range(len(object_ids)), key=lambda i: object_ids[i], reverse=True)
            for i in order:
                mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                canvas[mask] = PALETTE[object_ids[i] % len(PALETTE)]
            canvases[frame_idx] = canvas
        predictor.reset_state(state)
    return canvases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="source footage")
    parser.add_argument("-o", "--out", type=Path, required=True, help="control video to write")
    parser.add_argument("--frames", type=int, default=121, help="frame count; 0 for the whole source")
    parser.add_argument("--fps", type=float, default=30.0, help="output fps; seg.json wants 30")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the source to begin")
    parser.add_argument("--stride", type=int, help="frame step; defaults to source fps over target fps")
    parser.add_argument("--points", type=str, help='"x,y;x,y" in source pixels, one per object')
    parser.add_argument("--max-objects", type=int, default=8, help="cap for automatic prompting")
    parser.add_argument("--cut-threshold", type=float, default=25.0, help="frame difference counted as a cut")
    parser.add_argument("--min-shot", type=int, default=6, help="shots shorter than this are left unlabeled")
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

        boundaries = find_shot_boundaries(frames_dir, count, args.cut_threshold)
        shots = [(boundaries[i], boundaries[i + 1] if i + 1 < len(boundaries) else count)
                 for i in range(len(boundaries))]
        shots = [(first, last) for first, last in shots if last - first >= args.min_shot]
        print(f"{len(shots)} shot(s) over {count} frames "
              f"(cut threshold {args.cut_threshold:g}, shots under {args.min_shot} frames dropped)")

        fixed_points = parse_points(args.points) if args.points else None
        if fixed_points and len(shots) > 1:
            print("NOTE: --points is applied to every shot. Coordinates chosen for one shot")
            print("      rarely land on anything meaningful in another.")

        generator = None
        if not fixed_points:
            generator = SAM2AutomaticMaskGenerator.from_pretrained(args.model, device=device)
        predictor = SAM2VideoPredictor.from_pretrained(args.model, device=device)

        # Frames are written as each shot finishes rather than collected first:
        # holding a minute of 1080p canvases costs about 15 GB, which the machine
        # will not survive. Shots are in order and cover contiguous ranges, so
        # the writer only needs blanks padded across the gaps between them.
        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264", quality=8)
        coverage = []
        written = 0
        failed = 0

        def emit(canvas):
            nonlocal written
            writer.append_data(canvas)
            coverage.append(float(canvas.any(axis=2).mean()))
            written += 1

        try:
            for number, (first, last) in enumerate(shots, start=1):
                while written < first:
                    emit(blank)
                label = f"  shot {number}/{len(shots)} frames {first}-{last - 1}"
                segment_dir = work_dir / f"shot{number:03d}"
                canvases = {}
                try:
                    link_segment(frames_dir, segment_dir, first, last)
                    points = fixed_points or auto_prompts(segment_dir / "00000.jpg", args.max_objects, generator)
                    if points:
                        canvases = propagate_segment(predictor, segment_dir, points, size, device)
                    else:
                        print(f"{label}: nothing found, left black")
                except Exception as error:
                    # One shot failing must not discard the other fifty-six. A
                    # long job that dies at the end has produced nothing at all.
                    failed += 1
                    print(f"{label}: FAILED ({type(error).__name__}: {error}), left black")
                    canvases = {}
                finally:
                    shutil.rmtree(segment_dir, ignore_errors=True)
                    torch.cuda.empty_cache()

                mark = len(coverage)
                for local_index in range(last - first):
                    emit(canvases.get(local_index, blank))
                if canvases:
                    shot_coverage = np.mean(coverage[mark:])
                    print(f"{label}: {len(points)} object(s), {100 * shot_coverage:.1f}% labelled")

            while written < count:
                emit(blank)
        finally:
            writer.close()

        print(f"\nwrote {args.out}")
        print(f"  {count} frames at {args.fps} fps, {size[0]}x{size[1]}, {count / args.fps:.1f}s")
        print(f"  labelled pixels: first {100 * coverage[0]:.1f}%, "
              f"mean {100 * np.mean(coverage):.1f}%, last {100 * coverage[-1]:.1f}% "
              f"(the shipped asset is ~59%)")
        dead = sum(1 for value in coverage if value < 0.01)
        if dead:
            print(f"  {dead} frame(s) effectively blank, from shots too short to prompt "
                  f"or where tracking found nothing")
        if failed:
            print(f"  {failed} shot(s) failed outright and were left black")
    finally:
        if args.keep_frames:
            print(f"  frames kept in {frames_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
