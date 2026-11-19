"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy, five quantiles of its Exophase earn rate at the
case target, from the house's recorded reads of the trophy page before the
checkpoint. Pure Python 3.13 standard library, no network, deterministic.

Model (per trophy, all work done on the logit scale):
  * Holders and players are tracked separately.  Holders H = rate * players
    are fitted in log space against log(1 + hours since the first read), and
    the players-tracked count P is fitted the same way globally.  The
    holders-over-players forecast H_f / P_f therefore prices in the dilution
    every Exophase refresh causes: a jump in P drags every rate down even
    while holders keep growing.
  * A direct trend fit of logit(rate) against log(1 + hours) is blended 50/50
    (in logit space) with the holders-over-players forecast, so neither
    mechanism can run away on its own.
  * Slopes are shrunk toward the median slope of the trophy's tier
    (platinum / gold / silver / bronze).  A trophy whose history never moves
    carries no trend signal of its own at all, so it borrows its tier's slope
    outright; a trophy with a clear trend keeps most of its own.
  * The median is anchored on the last read and its move from the last rate
    is capped, so a two-day history cannot extrapolate a trophy to absurdity
    over a four-day horizon.
  * The spread is honest about refreshes: a random-walk term, a jump term
    that grows with the number of refreshes expected over the horizon, and
    the trend fit's own residual.  Exophase refreshes hours apart, not
    continuously, so a narrow band is wrong even over ninety minutes.

Usage (outside the grader): MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out
python3 main.py — reads every <id>.in, writes <id>.out beside it.
"""

import json
import math
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Standard-normal quantiles at the five scored levels, low to high.
NORM_Q = (-1.6448536269514722, -0.6744897501960817, 0.0,
          0.6744897501960817, 1.6448536269514722)

# Random-walk volatility, logit units per sqrt(day) — the house's own scale.
WALK_SIGMA = 0.5
# Typical logit move caused by one Exophase refresh (dilution as the tracked
# player base jumps), and how often refreshes arrive, in hours.
JUMP_SIGMA = 0.22
REFRESH_EVERY_HOURS = 5.0
# Extra spread for a trophy nobody holds yet: it can jump from zero.
ZERO_EXTRA_SIGMA = 0.6
# Largest logit move the median may make away from the last read.
MAX_MEDIAN_MOVE = 2.0
# Largest total spread allowed, logit units.
MAX_SIGMA = 2.5
# A series that never moves by more than this (percentage points) carries
# no trend signal of its own; it borrows its tier's slope outright.
FLAT_TOLERANCE = 0.005


def _parse_when(text):
    """ISO-8601 timestamp (with a trailing Z) as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hours_between(earlier, later):
    return (later - earlier).total_seconds() / 3600.0


def _clamp_pct(value):
    if value is None or not math.isfinite(value):
        return 0.01
    return min(99.99, max(0.01, value))


def _logit_of(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit_of(logit_value):
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    odds = math.exp(logit_value)
    return 100.0 * odds / (1.0 + odds)


def _round4(value):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    rounded = float(Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP))
    return int(rounded) if rounded.is_integer() else rounded


def _fit_line(xs, ys):
    """Least-squares line through (xs, ys); returns (slope, resid_std).

    With fewer than two distinct x values there is no slope to fit.
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return slope, resid


def _median(values):
    ordered = sorted(values)
    m = len(ordered)
    if m == 0:
        return 0.0
    if m % 2:
        return ordered[m // 2]
    return 0.5 * (ordered[m // 2 - 1] + ordered[m // 2])


def _fit_trophy(series):
    """Fit one trophy's history. series: [(x, rate_pct, players), ...].

    Returns the last read, both trend slopes, the logit-fit residual, and
    whether the series ever moved (a flat series has no trend of its own).
    """
    last_x, last_rate, last_players = series[-1]
    xs = [row[0] for row in series]
    logit_ys = [_logit_of(row[1]) for row in series]
    slope_logit, resid_logit = _fit_line(xs, logit_ys)

    holder_xs, holder_ys = [], []
    for x, rate, players in series:
        if rate is None or players is None or players <= 0:
            continue
        # Half a holder stands in for zero, so a 0.00% trophy still fits.
        holders = max(rate / 100.0 * players, 0.5)
        holder_xs.append(x)
        holder_ys.append(math.log(holders))
    slope_hold, _ = _fit_line(holder_xs, holder_ys)

    rates = [row[1] for row in series]
    moved = (max(rates) - min(rates)) > FLAT_TOLERANCE
    return {
        "last_logit": _logit_of(last_rate),
        "slope_logit": slope_logit,
        "resid_logit": resid_logit,
        "slope_hold": slope_hold,
        "last_rate": last_rate,
        "last_players": last_players,
        "last_x": last_x,
        "n_reads": len(series),
        "moved": moved,
    }


def _forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None:
        horizon_minutes = _hours_between(
            _parse_when(case["at"]), _parse_when(case["target"])) * 60.0
    horizon_hours = horizon_minutes / 60.0
    horizon_days = horizon_minutes / 1440.0

    if not history:
        # No reads at all: stay valid with a wide, low forecast.
        return {"trophies": [
            {"id": t["id"], "q": [0.01, 0.05, 0.5, 2.0, 10.0]}
            for t in trophies]}

    first_at = _parse_when(history[0]["at"])
    target_at = _parse_when(case["target"])

    def x_of(when_text):
        hours = _hours_between(first_at, _parse_when(when_text))
        return math.log1p(max(hours, 0.0))

    target_x = math.log1p(max(_hours_between(first_at, target_at), 0.0))

    # --- global players-tracked trend (shared by every trophy) ---
    p_xs, p_ys = [], []
    for read in history:
        players = read.get("players_tracked")
        if players and players > 0:
            p_xs.append(x_of(read["at"]))
            p_ys.append(math.log(players))
    p_slope, _ = _fit_line(p_xs, p_ys)
    p_slope = min(max(p_slope, 0.0), 2.0)
    last_players = history[-1].get("players_tracked") or 1
    last_p_x = x_of(history[-1]["at"])
    players_forecast = last_players * math.exp(
        p_slope * (target_x - last_p_x))

    # --- per-trophy series and fits ---
    per_trophy = []
    for idx, trophy in enumerate(trophies):
        series = []
        for read in history:
            rates = read.get("rates") or []
            rate = rates[idx] if idx < len(rates) else None
            if rate is None:
                continue
            series.append((x_of(read["at"]), float(rate),
                           read.get("players_tracked")))
        if not series:
            # Never observed: fall back to a low, wide forecast.
            per_trophy.append(None)
            continue
        per_trophy.append(_fit_trophy(series))

    # --- tier median slopes, for shrinkage ---
    tiers = {}
    for trophy, fit in zip(trophies, per_trophy):
        if fit is None:
            continue
        tiers.setdefault(trophy.get("tier", "bronze"), []).append(fit)
    tier_logit_slope, tier_hold_slope = {}, {}
    for tier, fits in tiers.items():
        tier_logit_slope[tier] = _median([f["slope_logit"] for f in fits])
        tier_hold_slope[tier] = _median([f["slope_hold"] for f in fits])

    rows = []
    for trophy, fit in zip(trophies, per_trophy):
        tid = trophy["id"]
        if fit is None:
            rows.append({"id": tid, "q": [0.01, 0.05, 0.5, 2.0, 10.0]})
            continue

        # A trophy everybody already holds stays at ~100 with a tight band.
        if fit["last_rate"] >= 99.99:
            rows.append({"id": tid, "q": [99.5, 99.8, 99.95, 99.99, 100]})
            continue

        tier = trophy.get("tier", "bronze")
        if fit["moved"]:
            # Weight on the trophy's own slope grows with the read count.
            own_weight = fit["n_reads"] / (fit["n_reads"] + 3.0)
        else:
            # Flat history: no signal of its own, borrow the tier's slope.
            own_weight = 0.0
        slope_logit = (own_weight * fit["slope_logit"]
                       + (1.0 - own_weight)
                       * tier_logit_slope.get(tier, 0.0))
        slope_hold = (own_weight * fit["slope_hold"]
                      + (1.0 - own_weight)
                      * tier_hold_slope.get(tier, 0.0))
        slope_hold = min(max(slope_hold, -0.5), 2.5)
        slope_logit = min(max(slope_logit, -2.0), 2.0)

        dx = target_x - fit["last_x"]
        # Direct trend median, anchored on the last read.
        median_logit_direct = fit["last_logit"] + slope_logit * dx
        # Holders-over-players median: holders extrapolated by their own
        # trend, divided by the extrapolated players-tracked count.
        last_holders = max(fit["last_rate"] / 100.0
                           * (fit["last_players"] or 1), 0.5)
        holders_forecast = last_holders * math.exp(slope_hold * dx)
        rate_hp = 100.0 * holders_forecast / max(players_forecast, 1.0)
        median_logit_hp = _logit_of(rate_hp)
        # Blend the two medians in logit space.
        median_logit = 0.5 * (median_logit_direct + median_logit_hp)
        # Cap how far the median may move from the last read.
        median_logit = min(max(median_logit,
                               fit["last_logit"] - MAX_MEDIAN_MOVE),
                           fit["last_logit"] + MAX_MEDIAN_MOVE)

        # --- spread: random walk + refresh jumps + fit residual ---
        walk = WALK_SIGMA * math.sqrt(max(horizon_days, 0.0))
        expected_refreshes = max(horizon_hours, 0.0) / REFRESH_EVERY_HOURS
        jumps = JUMP_SIGMA * math.sqrt(expected_refreshes)
        resid = min(max(fit["resid_logit"], 0.10), 1.0)
        sigma = math.sqrt(walk * walk + jumps * jumps + resid * resid)
        if fit["last_rate"] <= 0.01:
            sigma = math.sqrt(sigma * sigma
                              + ZERO_EXTRA_SIGMA * ZERO_EXTRA_SIGMA)
        sigma = min(sigma, MAX_SIGMA)

        quantiles = []
        previous = None
        for z in NORM_Q:
            value = _round4(_inv_logit_of(median_logit + z * sigma))
            value = min(max(value, 0.01), 100.0)
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": tid, "q": quantiles})

    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = _forecast_case(case)
        out_path = OUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
