# Build a Cosmos 3 transfer edge or depth control video from ordinary footage.
#
# Both are geometric rather than semantic: they describe *shape*, not *meaning*.
# That matters for stylised or synthetic footage, where segmentation models fail
# badly -- SegFormer/Cityscapes on game footage labelled a bridge as "fence" and
# found only 2% vehicles in a driving trailer, because the source sits far
# outside its training domain. Edge and depth have no classes to get wrong.
#
#   edge   Canny edges, no model at all.
#   depth  Depth Anything V2, which generalises across domains far better than
#          semantic segmentation because relative depth is domain-independent.
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def extract_frames(video_path, frames_dir, want_frames, target_fps, start_seconds, stride):
    """Frames strided so the control covers the same wall-clock motion.

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


def edge_frame(path, low, high, blur):
    """White Canny edges on black, matching the shipped control_edge.mp4.

    Blurred first because Canny on raw frames turns compression noise and
    texture detail into edges, which gives the transfer model a control full of
    structure that is not really there.
    """
    grey = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    if blur > 0:
        grey = cv2.GaussianBlur(grey, (blur * 2 + 1, blur * 2 + 1), 0)
    edges = cv2.Canny(grey, low, high)
    return np.repeat(edges[:, :, None], 3, axis=2)


def depth_batch(paths, processor, model, size, device):
    """Greyscale relative depth, near bright, for a batch of frames."""
    images = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        predicted = model(**inputs).predicted_depth
    upsampled = torch.nn.functional.interpolate(
        predicted.unsqueeze(1), size=(size[1], size[0]), mode="bicubic", align_corners=False
    ).squeeze(1)

    frames = []
    for depth in upsampled:
        # Normalised per frame: the model predicts relative depth with no fixed
        # scale, so an absolute mapping would flicker as the scene's range
        # changes. Per-frame normalisation is what the shipped assets look like.
        low, high = depth.min(), depth.max()
        normalised = (depth - low) / (high - low + 1e-8)
        grey = (normalised * 255).clamp(0, 255).byte().cpu().numpy()
        frames.append(np.repeat(grey[:, :, None], 3, axis=2))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="source footage")
    parser.add_argument("-o", "--out", type=Path, required=True, help="control video to write")
    parser.add_argument("--control", choices=("edge", "depth"), default="edge")
    parser.add_argument("--frames", type=int, default=121, help="frame count; 0 for the whole source")
    parser.add_argument("--fps", type=float, default=30.0, help="output fps; the specs want 30")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the source to begin")
    parser.add_argument("--stride", type=int, help="frame step; defaults to source fps over target fps")
    parser.add_argument("--batch", type=int, default=4, help="frames per forward pass, depth only")
    parser.add_argument("--canny-low", type=int, default=100, help="Canny lower threshold")
    parser.add_argument("--canny-high", type=int, default=200, help="Canny upper threshold")
    parser.add_argument("--blur", type=int, default=2, help="pre-blur radius; 0 disables")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Large-hf")
    parser.add_argument("--keep-frames", action="store_true", help="keep the extracted JPEGs")
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"missing: {args.video}")
    if args.control == "depth" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; per-frame depth on CPU is impractically slow")

    work_dir = Path(tempfile.mkdtemp(prefix="segmap-geo-"))
    frames_dir = work_dir / "frames"
    frames_dir.mkdir()
    try:
        count, size = extract_frames(args.video, frames_dir, args.frames, args.fps, args.start, args.stride)
        print(f"extracted {count} frames at {size[0]}x{size[1]}")
        print(f"control: {args.control}")

        processor = model = None
        if args.control == "depth":
            processor = AutoImageProcessor.from_pretrained(args.model)
            model = AutoModelForDepthEstimation.from_pretrained(args.model).to("cuda").eval()

        # Written as each frame completes rather than collected first: holding a
        # minute of 1080p frames costs about 15 GB.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=1)
        coverage = []
        try:
            if args.control == "edge":
                for index in range(count):
                    frame = edge_frame(frames_dir / f"{index:05d}.jpg",
                                       args.canny_low, args.canny_high, args.blur)
                    writer.append_data(frame)
                    coverage.append(float(frame[:, :, 0].mean()) / 255)
                    print(f"\r  {index + 1}/{count} frames", end="", flush=True)
            else:
                for first in range(0, count, args.batch):
                    paths = [frames_dir / f"{index:05d}.jpg"
                             for index in range(first, min(first + args.batch, count))]
                    for frame in depth_batch(paths, processor, model, size, "cuda"):
                        writer.append_data(frame)
                        coverage.append(float(frame[:, :, 0].mean()) / 255)
                    print(f"\r  {min(first + args.batch, count)}/{count} frames", end="", flush=True)
        finally:
            writer.close()
        print()

        print(f"\nwrote {args.out}")
        print(f"  {count} frames at {args.fps} fps, {size[0]}x{size[1]}, {count / args.fps:.1f}s")
        if args.control == "edge":
            print(f"  edge pixels: {100 * np.mean(coverage):.1f}% of frame on average")
            if np.mean(coverage) < 0.01:
                print("  WARNING: almost no edges. Lower --canny-low, or reduce --blur.")
            elif np.mean(coverage) > 0.15:
                print("  WARNING: very dense edges, likely texture and compression noise rather")
                print("           than structure. Raise --canny-low, or increase --blur.")
        else:
            print(f"  mean brightness: {100 * np.mean(coverage):.1f}% (near is bright)")
    finally:
        if args.keep_frames:
            print(f"  frames kept in {frames_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
