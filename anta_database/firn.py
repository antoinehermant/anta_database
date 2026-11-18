"""The firn correction in the depths of the database, as text: from the compile settings (added by
the compilation, when the published depths have none) or from raw_md.json ("firn correction applied"
and "firn correction" of each IRH). Used by the compilation and the indexing (per IRH)."""

import numpy as np
import pandas as pd


def _missing(value) -> bool:
    return value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value))


def firn_text(applied, value, settings: dict | None = None) -> str:
    """"14.6 m, by the authors", "none", "not known"... An IRH dated at two cores can have one value
    per core (a list, in the order of "ice core")."""
    if settings and settings.get("firn_correction"):
        return f"{settings['firn_correction']:g} m, added by the compilation"
    if _missing(applied):
        return "not known"
    if not applied:
        return "none"
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    known = [v for v in values if not _missing(v)]
    if not known:
        return "by the authors, value not known"
    if len(values) > 1:
        return f"{' and '.join(f'{v:g} m' for v in known)}, by the authors (one per ice core)"
    return f"{known[0]:g} m, by the authors"


def summary(texts: list[str]) -> str:
    """The firn correction of a line, from those of its IRHs: one text, or each with its IRHs."""
    unique = list(dict.fromkeys(t for t in texts if t))
    return unique[0] if len(unique) == 1 else "; ".join(unique) if unique else "not known"
