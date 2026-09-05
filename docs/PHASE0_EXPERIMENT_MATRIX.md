# Phase-0 Experiment Matrix

**Protocol:** `docs/PHASE0_PREREGISTRATION.md`  
**Output root:** `results/adaptation_locus_phase0/`

## 1. Factors

| Factor | Levels |
|--------|--------|
| `mismatch_family` | `imu_bias`, `motor_asymmetry` |
| `severity` | `0`, `small`, `medium`, `large` |
| `adaptation` | `A0`, `A1`, `A2` |
| `seed` | debug `{0,1,2}`; decision `{0,1,2,3,4}` |

## 2. Severity map

| Label | `imu_bias` (deg/s) | `motor_asymmetry` δ |
|-------|--------------------:|--------------------:|
| 0 | 0.0 | 0.0 |
| small | 0.5 | 0.015 |
| medium | 1.5 | 0.03 |
| large | 3.0 | 0.045 |

When `mismatch_family=imu_bias`, motor δ=0.  
When `mismatch_family=motor_asymmetry`, IMU bias=0.

## 3. Primary cells

Full factorial for decision seeds:

`2 families × 4 severities × 3 loci × 5 seeds = 120 adaptation runs`

Each run: 2000 online steps + periodic eval snapshots.

## 4. Reference cells

| Name | Definition |
|------|------------|
| Nominal | family irrelevant, severity 0, A0 |
| Shifted-unadapted | each (family, severity), A0 |

## 5. Transfer cells

For each family and seed:

1. Adapt at `medium` with A1 and with A2 (reuse primary medium runs).
2. Freeze adapted parameters.
3. Evaluate at `small` and `large` (no further adaptation), 4 episodes each.

## 6. Debug vs decision

| Stage | Seeds | Command intent |
|-------|-------|----------------|
| Unit tests | n/a | `python -m pytest tests/test_adaptation_locus_phase0.py -q` |
| Smoke | `{0}` | `--stage smoke` (subset severities medium only) |
| Debug | `{0,1,2}` | `--stage debug` |
| Decision | `{0,1,2,3,4}` | `--stage decision` |

## 7. One-command entry

```bash
python -m research.adaptation_locus.run_phase0 --stage decision --config research/adaptation_locus/configs/phase0.yaml
```

## 8. Required plots (decision stage)

Written under `results/adaptation_locus_phase0/plots/`:

1. `severity_vs_performance.png`
2. `samples_vs_recovery.png`
3. `severity_vs_control_magnitude.png`
4. `mismatch_x_locus_interaction.png`
