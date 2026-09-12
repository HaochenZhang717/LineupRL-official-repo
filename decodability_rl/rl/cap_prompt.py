from __future__ import annotations

UPSTREAM_CAP_PROMPT = "Please describe this image in detail."

_BODY = (
    "Describe the series in detail: what values it takes, how it moves over time, and "
    "where along the time axis its notable features occur. Write it so that someone who "
    "cannot see the {what} could tell this series apart from other series that look "
    "broadly similar."
)

TS_CAP_PROMPT = (
    "This image is a line chart of a single time series. Describe the series in "
    "detail: what values it takes, how it moves over time, and where along the time "
    "axis its notable features occur. Write it so that someone who cannot see the "
    "chart could tell this series apart from other series that look broadly similar."
)


TS_CAP_PROMPT_TEXT = (
    "Below are the values of a single time series, in order. " + _BODY.format(what="values")
)

TS_CAP_PROMPT_CHATTS = (
    "I have a time series of length {n}: <ts><ts/>. " + _BODY.format(what="series")
)

assert TS_CAP_PROMPT == (
    "This image is a line chart of a single time series. " + _BODY.format(what="chart")
), "TS_CAP_PROMPT drifted from _BODY -- the trained arms' prompt must not change"

if __name__ == "__main__":
    print(TS_CAP_PROMPT)
