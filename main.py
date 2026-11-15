#!/usr/bin/env python3
"""btc-cent-forecast-01: forecast BTC/USD 240 minutes ahead.

Reads every <id>.in case in the input directory (JSON schema
makefun/btc-cent-case/v1: 4,320 one-minute closes over the 72 hours up to
the checkpoint, last price = close at the checkpoint) and writes
<id>.out with one forecast: {"forecasts": [{"case": id, "price": x}]}.

Method (calibrated by walk-forward backtest on the six practice cases,
every hour from t=1440 to t=4320-240, 264 forecasts, plus the five
checkpoints whose actuals are known from the overlapping cases):
  * persistence alone scores ~45,117 cents MAE walk-forward and 47,055
    on the five known checkpoints;
  * short-horizon trend extrapolation is much worse (60-min trend at
    damping 0.5: ~72,900 cents walk-forward) -- minute noise dominates;
  * mean reversion toward the 72-hour mean helps the walk-forward
    average but hurts the checkpoints badly (50,078 on the five known
    ones, all adjustments positive into a falling market), so it is
    NOT used: the final's checkpoints sit at window ends, like the
    practice checkpoints, not in the middle of the window;
  * the surviving blend: last price + a heavily damped 48-hour drift
    (damping 0.3 over 240 min, i.e. 2.5% of the 48-hour move). It beats
    persistence on the late-window walk-forward (43,380 vs 43,500) and
    on the five known checkpoints (45,934 vs 47,055), and is within
    0.5% of it on the full walk-forward. The edge is small and the
    market is nearly unforecastable at 4 hours -- the point is a small,
    consistent, low-variance tilt, never a big bet.
  The total adjustment is capped at +/-2% of the last price so a jumpy
  or gappy series can never produce a wild guess. On a flat series the
  drift is zero, so it reduces to persistence.

Deterministic, stdlib only, no network, no randomness. Fast: one pass
over the prices per case, far inside the 8 s cap.
"""

import json
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# Calibrated constants (see module docstring).
DRIFT_WINDOW = 2880      # minutes of history the drift is measured over
DRIFT_DAMP = 0.3         # damping applied to the 240-min drift extrapolation
MAX_ADJ_FRAC = 0.02      # cap on |adjustment| as a fraction of last price
HORIZON = 240            # minutes ahead (from the case's horizon_minutes)


def price5(x):
    """Keep x on the grid's 5 decimals and inside (0, 10_000_000)."""
    x = min(MAX_PRICE, max(MIN_PRICE, x))
    return round(x * 100000) / 100000


def forecast(prices, horizon):
    """One forecast from a list of closes, last = close at checkpoint."""
    last = prices[-1]
    n = len(prices)
    if n < 2 or not all(isinstance(v, (int, float)) for v in prices):
        return price5(last)
    # 48-hour drift, per minute, extrapolated over the horizon, damped.
    # Long window on purpose: the drift estimate over 48 h is far less
    # noisy than over a few hours, and the damping keeps the bet small.
    w = min(DRIFT_WINDOW, n - 1)
    drift_per_min = (last - prices[-1 - w]) / w
    adj = DRIFT_DAMP * horizon * drift_per_min
    cap = MAX_ADJ_FRAC * abs(last)
    if adj > cap:
        adj = cap
    elif adj < -cap:
        adj = -cap
    guess = last + adj
    if guess != guess or guess in (float("inf"), float("-inf")):
        return price5(last)
    return price5(guess)


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in in_dir.iterdir() if f.name.endswith(".in"))
    for path in files:
        case_id = path.name[: -len(".in")]
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            markets = case["markets"]
            prices = markets[0]["prices"]
            horizon = int(case.get("horizon_minutes") or HORIZON)
            guess = forecast(prices, horizon)
        except Exception:
            # Never leave a case without a valid output: fall back to the
            # checkpoint close when it can be read, else re-raise.
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                guess = price5(case["markets"][0]["prices"][-1])
            except Exception:
                raise
        payload = {"forecasts": [{"case": case_id, "price": guess}]}
        (out_dir / f"{case_id}.out").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
