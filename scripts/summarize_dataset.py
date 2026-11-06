"""Merge a re-rendered dataset's per-file manifests into manifest.json and summary.md.

    python scripts/summarize_dataset.py --name eevee

Reads <rendered_datasets>/<name>/<suite>/*.manifest.json. Writes manifest.json (one entry per file,
without the alignment samples) and summary.md (per suite: files done, checks, frames, render time).
Exits non-zero if a file is missing or a check failed.
"""
import argparse
import json
import sys

from libero_blender.paths import LIBERO_DATASETS, RENDERED_DATASETS

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="render config name, e.g. eevee")
    parser.add_argument("--suites", nargs="+", default=SUITES)
    args = parser.parse_args()
    dataset = RENDERED_DATASETS / args.name

    entries, lines, complete = [], [f"# Re-rendered dataset: {args.name}", ""], True
    for suite in args.suites:
        expected = sorted(path.name for path in (LIBERO_DATASETS / suite).glob("*.hdf5"))
        manifests = {path.name.replace(".manifest.json", ".hdf5"): json.loads(path.read_text())
                     for path in (dataset / suite).glob("*.manifest.json")}
        done = [manifests[name] for name in expected if name in manifests]
        samples = [s for m in done for s in m["alignment_samples"]]
        worst = min(samples, key=lambda s: s["mean_iou"]) if samples else None
        # Every file must also agree on the stored image orientation.
        flips = sorted({m["stored_orientation_flip"] for m in done})
        aligned = sum(m["alignment_all_passed"] for m in done)
        complete = complete and len(done) == len(expected) and len(flips) == 1 and aligned == len(done)
        lines += [
            f"## {suite}", "",
            f"- Files: {len(done)} / {len(expected)}",
            f"- Stored orientation: {flips}; worst mean abs diff to MuJoCo re-renders "
            f"{max((m['orientation_mean_abs_diff'][m['stored_orientation_flip']] for m in done), default='-')} / 255",
            f"- Geom-id alignment passed: {aligned} / {len(done)} files ({len(samples)} camera-frames), "
            f"worst mean IoU {worst['mean_iou'] if worst else '-'}",
            f"- Demos: {sum(m['num_demos'] for m in done)}; frames: {sum(m['num_frames'] for m in done)}; "
            f"render time {sum(m['wall_seconds'] for m in done) / 3600:.1f} h", "",
        ]
        entries += [{key: value for key, value in m.items() if key != "alignment_samples"} for m in done]

    lines.append("Complete and all checks passed." if complete else "**Not complete, or a check failed.**")
    (dataset / "manifest.json").write_text(json.dumps(entries, indent=2))
    (dataset / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if complete else 1)


if __name__ == "__main__":
    main()
