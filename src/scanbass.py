import argparse
from pathlib import Path
import json
import sys


def main():
    p = argparse.ArgumentParser(description="ScanBass MVP — one CLI, two modes")
    p.add_argument("--mode", choices=["bass", "poly"], required=True,
                   help="bass: isolate instrument bass from mix; poly: lowest notes from polyphonic audio")
    p.add_argument("input", help="Path to audio file (wav/mp3/aiff/…)")
    p.add_argument("--out", default="outputs", help="Output root directory (default: outputs)")
    # poly controls
    p.add_argument("--frame-hz", type=int, default=40, help="Poly: frames per second for lowest-voice (default 40)")
    p.add_argument("--min-note-len", type=int, default=90, help="Poly: min note length in ms (default 90)")
    p.add_argument("--gap-merge", type=int, default=60, help="Poly: merge micro-gaps of same pitch in ms (default 60)")
    # bass controls
    p.add_argument("--voicing-threshold", type=float, default=0.5, help="Bass: torchcrepe voicing threshold (default 0.5)")
    p.add_argument("--segment-seconds", type=float, default=15.0, help="Bass: Demucs segment length in seconds (default 15.0)")
    p.add_argument("--segment-overlap", type=float, default=0.1, help="Bass: Demucs segment overlap ratio (default 0.1)")
    args = p.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        sys.exit(1)

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "poly":
        from modes.poly_mode import run_poly_mode
        out_dir = out_root / f"{in_path.stem}_poly"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_poly_mode(
            audio_path=str(in_path),
            out_dir=str(out_dir),
            frame_hz=args.frame_hz,
            min_note_len_ms=args.min_note_len,
            gap_merge_ms=args.gap_merge
        )
    else:
        from modes.bass_mode import run_bass_mode
        out_dir = out_root / f"{in_path.stem}_bass"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_bass_mode(
            audio_path=str(in_path),
            out_dir=str(out_dir),
            voicing_threshold=args.voicing_threshold,
            segment_seconds=args.segment_seconds,
            overlap=args.segment_overlap
        )

    with open(out_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n✅ Done! See outputs in:", out_dir)


if __name__ == "__main__":
    main()
