# GTA VI Trophy Rarity Forecast — Week-One Earn Rates

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rizal556z/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rizal556z/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for all 55 PlayStation 5 trophies of Grand Theft Auto VI: given the
house's recorded reads of the Exophase trophy page up to a checkpoint, it
predicts five quantiles (5%, 25%, 50%, 75%, 95%) of each trophy's earn rate at
the case target — a week after launch for the final case — scored by pinball
loss on the logit scale.

## How it works

- **Holders and players, separately.** Holders H = rate × players-tracked are
  fitted in log space against log(1 + hours since the first read), and the
  players-tracked count is fitted the same way globally. The
  holders-over-players forecast therefore prices in the dilution every
  Exophase refresh causes: a jump in tracked players drags every rate down
  even while holders keep growing.
- **Blended median.** A direct trend fit of logit(rate) against log(1 + hours)
  is blended 50/50 (in logit space) with the holders-over-players forecast, so
  neither mechanism can run away on its own.
- **Hierarchical shrinkage.** Each trophy's trend slope is shrunk toward the
  median slope of its tier (platinum / gold / silver / bronze). A trophy whose
  history never moves carries no trend signal of its own, so it borrows its
  tier's slope outright.
- **Anchored, capped median.** The median is anchored on the last read and its
  move from the last rate is capped at 2 logit units.
- **Honest spread.** The spread combines a random-walk term, a refresh-jump
  term that grows with the number of refreshes expected over the horizon, and
  the trend fit's own residual — plus extra width for trophies nobody holds
  yet. The trophy everybody already holds (100%) is forecast at ~100 with a
  tight band.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` beside it:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}` — one row
per trophy, five non-decreasing quantiles in percent. Pure Python 3.13
standard library, no network, deterministic byte-for-byte across runs, well
under the 8-second limit per case.

## Local checks

- Practice cases `p00` / `p01`: valid shape (55 rows, quantiles
  non-decreasing, 0–100, file ≤ 65,536 bytes), byte-identical across two runs,
  ~0.1 s per run. Against proxy practice answers (uniform holder growth from
  the launch-night refresh: players 638 → 3,868, trophies earned 943 → 4,276),
  pinball loss ≈ 0.17 vs ≈ 0.20 for the house persistence baseline, with 90%
  coverage 1.00 vs 0.00 — the wider band covers the refresh the narrow
  baseline misses.
- A synthetic 2-day, 289-read final-style history (10-minute reads, refresh
  jumps, rising early trophies): sensible medians, wide honest bands, 0.25 s.
- Edge cases: a single-read history and a trophy at 0.00% both produce valid
  output with wide bands.
