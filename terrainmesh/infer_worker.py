import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", required=True)
    p.add_argument("--sparse-depth", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    print("e")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import heavy ML deps only inside the 3.11 process
    from inference import run_inference

    result = run_inference(
        rgb_path=args.rgb,
        sparse_depth_path=args.sparse_depth,
        out_dir=str(out_dir),
    )

    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()