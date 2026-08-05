"""simlingo_training package init.

NumPy 2.x compatibility: the original SimLingo environment pins numpy<1.24,
but Lilypad workers preinstall numpy 2.x (pinning numpy in the pip overlay is
not allowed). Restore the removed aliases that imgaug 0.4.0, dataset_base
(np.string_) and transfuser_utils (np.object) still use. This runs on any
`simlingo_training.*` import, before those modules load.
"""
import numpy as _np

for _alias, _repl in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode_", getattr(_np, "str_", str)),
    ("string_", getattr(_np, "bytes_", bytes)),
    ("float_", getattr(_np, "float64", float)),
):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _repl)
