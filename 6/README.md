# Assignment 6 – DPA on serialized AES

**Student:** Omidizadeh

This solution performs a *vectorized* Differential Power Analysis (DPA) using a two–group *t-test* style score on the **transition** between two serialized S-box outputs:
\[ \mathrm{SB}(p_5\oplus k_5) \to \mathrm{SB}(p_{10}\oplus k_{10}) \].
Because the register is serialized and shifts the S-box outputs across cycles, the leakage model uses a bitwise **XOR transition** between these two values. We recover **both bytes simultaneously** over the 2^16 key space as requested.

## Files produced
- `plot_over_kg.png` — score for **all 2^16 key guesses** (max |t|-like over bits & POIs).
- `plot_over_traces.png` — |t|-curves over **time** for the recovered key and a few strong impostors.
- `keys.txt` — recovered `k5` and `k10` in dec & hex.
- `description.txt` — traces needed (empirical) + brief notes.
- `src_dpa_assignment6.py` — fully commented, reproducible script (NumPy + Matplotlib).
- `README.md` — this file.

> **Data policy**: Provided traces/plaintexts are never re-packaged. Plots are clear enough to understand results.

## Method (t-test & grouping)
For each key pair (k5, k10) and each bit b∈[0..7], traces are grouped by the predicted **bit transition**
\[ g_i = ((\mathrm{SB}(p_{5,i}\oplus k_5)\!\gg\! b)\&1)\oplus((\mathrm{SB}(p_{10,i}\oplus k_{10})\!\gg\! b)\&1). \]
The Welch-style t separation at automatically selected **Points of Interest (POIs)** is computed fully **vectorized**:
numerators via \(\mathbf{X}\,\mathrm{diag}(\mathbf{v})\,\mathbf{Y}^T\) and denominators via group sizes.

## Plots
- **Over keys:** the correct pair dominates with the largest peak.
- **Over time:** |t|-curve highlights serialized S-box activity; impostors stay flatter.

## Reproducibility
- Deterministic NumPy pipeline.
- Only `numpy` and `matplotlib`; one chart per figure; no seaborn; no explicit colors.
