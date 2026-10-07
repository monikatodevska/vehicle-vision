# Original Research

`research/` includes all 128 original Python research files, along with the
supplied text and source-support files. Copied baseline
author/contact headers were removed at the project owner's request.
The maintained application lives in `vehicle_vision/` at the repository root.

The scripts span multiple detector architectures, training approaches, dataset
utilities, and traffic-analysis experiments. They retain hard-coded paths and
historical dependencies; some are incomplete. They are not part of the package
or automated checks. See the [training guide](../docs/training.md) for the main
training scripts and generated ground-truth formats, the
[inventory](../docs/research-inventory.json) for original file hashes and known
parsing failures, and the [migration notes](../docs/migration.md) for maintained
replacements.

Research source is tracked by Git. The checkpoint, Windows runtime, IDE state,
and caches remain local and ignored. No private datasets or validated pretrained
model are distributed with the package.
