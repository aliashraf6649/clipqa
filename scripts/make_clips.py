"""Turn the MOT17 and UA-DETRAC image sequences into MP4 clips. Run from the repo root."""
import configparser
import subprocess
from pathlib import Path

MOT_DIR = Path("data/MOT17/train")
MOT_SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"]
DETRAC_DIR = Path("data/DETRAC/DETRAC-Images")
DETRAC_SEQUENCES = ["MVI_39051", "MVI_40901"]
DETRAC_FPS = "25"  # every UA-DETRAC video runs at 25 frames per second
CLIPS_DIR = Path("data/clips")


def encode(frames_pattern: Path, fps: str, out_path: Path) -> None:
    """Encode numbered images as an H.264 MP4 at the given frame rate."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-framerate", fps, "-i", str(frames_pattern),
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         str(out_path)],
        check=True,
    )


def count_frames(path: Path) -> int:
    """Decode the whole clip and return how many frames it really has."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def make_clip(frames_pattern: Path, fps: str, expected: int, out_path: Path) -> None:
    """Encode one clip and check that no frame was lost or added."""
    encode(frames_pattern, fps, out_path)
    frames = count_frames(out_path)
    if frames != expected:
        raise RuntimeError(f"{out_path} has {frames} frames, expected {expected}")
    print(f"{out_path}: {frames} frames at {fps} fps")


def main() -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    for name in MOT_SEQUENCES:
        seq_dir = MOT_DIR / name
        config = configparser.ConfigParser()
        config.read(seq_dir / "seqinfo.ini")
        info = config["Sequence"]
        frames_pattern = seq_dir / info["imDir"] / f"%06d{info['imExt']}"
        out_path = CLIPS_DIR / f"{name.removesuffix('-FRCNN')}.mp4"
        make_clip(frames_pattern, info["frameRate"], int(info["seqLength"]), out_path)

    for name in DETRAC_SEQUENCES:
        seq_dir = DETRAC_DIR / name
        expected = len(list(seq_dir.glob("img*.jpg")))  # no seqinfo.ini here: count the images
        make_clip(seq_dir / "img%05d.jpg", DETRAC_FPS, expected, CLIPS_DIR / f"{name}.mp4")


if __name__ == "__main__":
    main()