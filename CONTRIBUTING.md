# Contributing

This is a personal tooling project, but contributions are welcome.

## Getting started

```bash
git clone <repo>
cd image-cropper
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running a step manually

Each script is self-contained and can be run independently. See
[docs/USAGE.md](docs/USAGE.md) for all options.

```bash
python align.py input/ output/aligned/ --debug
```

## Changing detection behaviour

The detector cascades in `align.py`, `crop_source.py`, and `crop_cutout.py`
all follow the same pattern: try the best detector first, fall back to
simpler ones. If you want to add or swap a detector, locate the
`detect_*` or `_landmarks_*` functions and slot in your implementation
at the appropriate level.

Padding constants at the top of each file control how much space is left
around the face — adjust them there rather than in the crop math.

## Changing pipeline flow

`pipeline.sh` is a linear Bash script with clearly delimited steps. Add or
remove steps by inserting/deleting the relevant `=== Step N/M ===` block
and updating the step counts in the headings.

## Code style

- Python: follow the style already in the file (no framework required)
- No tests currently exist; if you add a step with non-trivial geometry
  maths, a small unit test is appreciated
- Keep scripts self-contained — avoid cross-imports between step scripts

## Reporting issues

Open a GitHub issue with:
- The command you ran
- The error output (truncate model-loading noise)
- A sample image if the bug is image-specific (anonymise if needed)
