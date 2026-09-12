from __future__ import annotations

import numpy as np


def oracle_caption(series, decimals: int = 2) -> str:
    y = np.asarray(series, dtype=float)
    n = y.size
    q = np.array_split(y, 4)
    f = lambda v: f"{v:.{decimals}f}"
    quarters = ", ".join(
        f"quarter {i+1} averages {f(float(b.mean()))}" for i, b in enumerate(q)
    )
    return (
        f"The series has {n} points. It starts at {f(y[0])} and ends at {f(y[-1])}. "
        f"Its maximum value {f(y.max())} occurs at time index {int(y.argmax())}, and its "
        f"minimum value {f(y.min())} occurs at time index {int(y.argmin())}. "
        f"Reading left to right, {quarters}. "
        f"The first three values are {f(y[0])}, {f(y[1])}, {f(y[2])} and the last three "
        f"are {f(y[-3])}, {f(y[-2])}, {f(y[-1])}."
    )
