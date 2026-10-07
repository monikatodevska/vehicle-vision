# Research Migration

The supplied folder contained 128 Python files, numerous model variants,
temporary experiments, a Windows Python runtime, IDE state, and one checkpoint.
All original research files were moved together to `archive/research/` so their
sibling imports remain available. Copied baseline author/contact headers were
removed at the project owner's request. This local archive is excluded from Git;
the public repository contains the maintained package and documentation.

The [inventory](research-inventory.json) records original relative paths,
SHA-256 hashes, syntax failures, and repeated top-level definitions. It provides
an audit trail of the supplied source before header cleanup. Runtime binaries,
caches, IDE state, the checkpoint, and archived scripts are excluded from Git.

## Maintained Replacements

| Original area | Maintained module |
| --- | --- |
| IoU functions in postprocessing and YOLO evaluators | `geometry.py` |
| Clustering and output decoding in plotting scripts | `detection.py` |
| Repeated model loading and inception construction | `models.py` |
| YOLO-style detection metrics | `evaluation.py` |
| Video frame extraction scripts | `video.py` |
| Unfinished frame matching script | `tracking.py` (IoU baseline) |
| Hard-coded inference scripts | `cli.py`, `config.py`, `inference.py` |

The preserved experiments are historical references, not independent supported
applications. Training, custom losses, calibration, Siamese matching, wavelet
models, and specialized multi-output variants still need their original data,
environment, and experiment settings. The maintained package does not silently
select one of those variants.

Five archived Python files did not parse: `FCN_SSD_generator_main.py`,
`rename.py`, `track_vehicles.py`, `crop.py`, `GT_zapisuvanje_anchorless.py`,
as recorded in the inventory. `helper_model1.py` defines
`load_model` twice. These are recorded instead of guessing the intention of
incomplete historical experiments.

To examine an original script, use the archive as its working directory. Many
scripts execute immediately, reference private Windows paths, or modify data.
Read and configure them before running them. The archive is not imported by
the maintained package.
