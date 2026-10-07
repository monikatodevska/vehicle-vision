# Original Research

`research/` preserves the supplied experiment files locally. Copied baseline
author/contact headers were removed at the project owner's request.
The maintained application lives in `vehicle_vision/` at the repository root.

The scripts span multiple detector architectures, training approaches, dataset
utilities, and traffic-analysis experiments. They retain hard-coded paths and
historical dependencies; some are incomplete. They are not part of the package
or automated checks. See `docs/research-inventory.json` for original file hashes
and known parsing failures, and `docs/migration.md` for maintained replacements.

The entire research archive, including the checkpoint and Windows runtime, is
retained on disk but ignored by Git. No datasets or validated pretrained model
are distributed with the package.
