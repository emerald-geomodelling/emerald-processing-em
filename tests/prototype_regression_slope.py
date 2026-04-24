"""
Prototype: Regression-based slope & curvature vs wide-window finite difference.

Loads real HeliTEM data (lines 50010 and 10040), computes slopes/curvatures
using three approaches:
  1. Current wide-window finite difference
  2. Regression (positive-only, as first attempt)
  3. Regression v2: adaptive window + all |values| + Theil-Sen robust fit

Usage:
    python tests/prototype_regression_slope.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace
from pathlib import Path

# ─── Data Loading ──────────────────────────────────────────────────────────

DATA_FILE = Path(__file__).resolve().parents[1] / \
    ".." / "emerald-helitem-converter" / "data_helitem_example" / \
    "Ecorodovias_Data" / "2501013_prelim_beryl.asc"

# Gate center times from .gex (seconds)
GATE_TIMES = np.array([
    0.00520020, 0.00521650, 0.00523270, 0.00525310, 0.00527750,
    0.00530600, 0.00534260, 0.00538740, 0.00544430, 0.00551760,
    0.00560710, 0.00572100, 0.00586340, 0.00603840, 0.00625810,
    0.00653480, 0.00688070, 0.00731200, 0.00785320, 0.00853270,
    0.00938310, 0.01044510, 0.01177160, 0.01343180, 0.01551110,
])

SCALEFACTOR = 1e-12
OUTPUT_DIR = Path(__file__).resolve().parent / "prototype_output"


def load_helitem_line(line_number):
    """Load Z-component gate data for a specific flight line."""
    with open(DATA_FILE) as f:
        header = f.readline().split()
    if header[0] == '/':
        header = header[1:]

    df = pd.read_csv(DATA_FILE, sep=r'\s+', comment='/', header=None,
                     names=header, na_values='*')

    line_data = df[df['line'] == line_number].copy()
    if len(line_data) == 0:
        raise ValueError(f"Line {line_number} not found. Available: "
                         f"{sorted(df['line'].unique())}")

    gate_cols = [f'emz_db_final[{i}]' for i in range(25)]
    gate_data = line_data[gate_cols].values
    print(f"Line {line_number}: {gate_data.shape[0]} soundings, {gate_data.shape[1]} gates")
    return pd.DataFrame(gate_data)


# ─── Shared helpers ────────────────────────────────────────────────────────

L10_TIMES = np.log10(GATE_TIMES)


def _build_adaptive_windows(l10_times, min_gates=5, max_half_decades=0.15):
    """Build per-gate windows: at least min_gates, up to max_half_decades.

    For tightly-spaced HeliTEM early gates, the decade-based window captures
    many gates. For widely-spaced late gates, we guarantee at least min_gates.
    """
    n = len(l10_times)
    windows = []
    for k in range(n):
        # Start with decade-based window
        decade_mask = np.abs(l10_times - l10_times[k]) <= max_half_decades
        indices = np.where(decade_mask)[0]

        # Ensure at least min_gates (expand symmetrically from k)
        if len(indices) < min_gates:
            left = k
            right = k
            while (right - left + 1) < min_gates:
                if left > 0:
                    left -= 1
                if (right - left + 1) >= min_gates:
                    break
                if right < n - 1:
                    right += 1
            indices = np.arange(left, right + 1)

        windows.append(indices)
    return windows


# ─── Method 1: Current wide-window ────────────────────────────────────────

def _find_wide_window_pairs(l10_times_1d, min_span=0.01):
    n = len(l10_times_1d)
    bwd = np.full(n, -1, dtype=int)
    fwd = np.full(n, -1, dtype=int)
    for k in range(n):
        if np.isnan(l10_times_1d[k]):
            continue
        best_j = -1
        for j in range(k - 1, -1, -1):
            if np.isnan(l10_times_1d[j]):
                continue
            best_j = j
            if l10_times_1d[k] - l10_times_1d[j] >= min_span:
                break
        bwd[k] = best_j
    for k in range(n):
        if np.isnan(l10_times_1d[k]):
            continue
        best_m = -1
        for m in range(k + 1, n):
            if np.isnan(l10_times_1d[m]):
                continue
            best_m = m
            if l10_times_1d[m] - l10_times_1d[k] >= min_span:
                break
        fwd[k] = best_m
    return bwd, fwd


def widewindow_slopes(gate_df):
    """Current production implementation."""
    dBdt = np.abs(gate_df) * SCALEFACTOR
    l10_dBdt = np.log10(dBdt)
    bwd, _ = _find_wide_window_pairs(L10_TIMES)

    slope = pd.DataFrame(np.nan, index=gate_df.index, columns=gate_df.columns)
    for k in range(len(L10_TIMES)):
        j = bwd[k]
        if j == -1:
            continue
        slope.iloc[:, k] = (l10_dBdt.iloc[:, k] - l10_dBdt.iloc[:, j]) / \
                            (L10_TIMES[k] - L10_TIMES[j])

    # Sign change guard
    for k in range(len(L10_TIMES)):
        j = bwd[k]
        if j == -1:
            continue
        bad = ((gate_df.iloc[:, k] * gate_df.iloc[:, j] < 0) |
               (gate_df.iloc[:, k] == 0) | (gate_df.iloc[:, j] == 0))
        slope.loc[bad, slope.columns[k]] = np.nan
    return slope


def widewindow_curvatures(gate_df):
    dBdt = np.abs(gate_df) * SCALEFACTOR
    l10_dBdt = np.log10(dBdt)
    bwd, fwd = _find_wide_window_pairs(L10_TIMES)

    curv = pd.DataFrame(np.nan, index=gate_df.index, columns=gate_df.columns)
    for k in range(len(L10_TIMES)):
        j, m = bwd[k], fwd[k]
        if j == -1 or m == -1:
            continue
        t_span = L10_TIMES[m] - L10_TIMES[j]
        curv.iloc[:, k] = (l10_dBdt.iloc[:, m] - 2 * l10_dBdt.iloc[:, k] +
                            l10_dBdt.iloc[:, j]) / (t_span ** 2)
    for k in range(len(L10_TIMES)):
        j, m = bwd[k], fwd[k]
        if j == -1 or m == -1:
            continue
        bad = ((gate_df.iloc[:, j] <= 0) | (gate_df.iloc[:, k] <= 0) |
               (gate_df.iloc[:, m] <= 0))
        curv.loc[bad, curv.columns[k]] = np.nan
    return curv


# ─── Method 2: Regression v2 (robust, adaptive, all values) ───────────────

def regression_slopes_v2(gate_df, min_gates=5, max_half_decades=0.15,
                         min_points=3):
    """Robust regression slopes using Theil-Sen estimator.

    Key differences from v1:
    - Uses ALL |dBdt| values (not just positive originals)
    - Theil-Sen estimator is robust to outlier artifacts from abs(negative)
    - Adaptive window: at least min_gates, up to max_half_decades
    - Only excludes gates where |dBdt| == 0 (which gives -inf in log space)
    """
    dBdt_abs = np.abs(gate_df.values) * SCALEFACTOR
    l10_dBdt = np.log10(dBdt_abs)  # sign changes → V-shaped dips (outliers)

    n_soundings, n_gates = gate_df.shape
    slope_arr = np.full((n_soundings, n_gates), np.nan)

    windows = _build_adaptive_windows(L10_TIMES, min_gates, max_half_decades)

    for k in range(n_gates):
        indices = windows[k]
        t_local = L10_TIMES[indices]
        y_all = l10_dBdt[:, indices]  # (n_soundings, n_window)

        for i in range(n_soundings):
            y_row = y_all[i]
            # Only exclude truly invalid values (inf, nan from zero data)
            valid = np.isfinite(y_row)
            if valid.sum() < min_points:
                continue

            t_fit = t_local[valid]
            y_fit = y_row[valid]

            # Theil-Sen: robust to up to ~29% outliers in the window
            try:
                result = stats.theilslopes(y_fit, t_fit)
                slope_arr[i, k] = result.slope
            except Exception:
                continue

    return pd.DataFrame(slope_arr, index=gate_df.index, columns=gate_df.columns)


def regression_curvatures_v2(gate_df, min_gates=7, max_half_decades=0.15,
                             min_points=5):
    """Robust regression curvatures via quadratic fit with outlier rejection.

    Uses iteratively reweighted least squares (IRLS) with Huber weights
    to fit a quadratic to log10(|dBdt|) vs log10(t). Curvature = 2*a
    where a is the quadratic coefficient.
    """
    dBdt_abs = np.abs(gate_df.values) * SCALEFACTOR
    l10_dBdt = np.log10(dBdt_abs)

    n_soundings, n_gates = gate_df.shape
    curv_arr = np.full((n_soundings, n_gates), np.nan)

    windows = _build_adaptive_windows(L10_TIMES, min_gates, max_half_decades)

    for k in range(n_gates):
        indices = windows[k]
        t_local = L10_TIMES[indices]
        y_all = l10_dBdt[:, indices]

        for i in range(n_soundings):
            y_row = y_all[i]
            valid = np.isfinite(y_row)
            if valid.sum() < min_points:
                continue

            t_fit = t_local[valid]
            y_fit = y_row[valid]

            # Robust quadratic fit using IRLS with Huber weights
            try:
                coeffs = _robust_polyfit(t_fit, y_fit, deg=2, n_iter=3)
                curv_arr[i, k] = 2 * coeffs[0]  # Second derivative
            except Exception:
                continue

    return pd.DataFrame(curv_arr, index=gate_df.index, columns=gate_df.columns)


def _robust_polyfit(x, y, deg=2, n_iter=3):
    """Iteratively reweighted least squares polynomial fit.

    Uses Huber weights to downweight outliers (like sign-change artifacts).
    """
    weights = np.ones(len(x))
    for _ in range(n_iter):
        coeffs = np.polyfit(x, y, deg, w=weights)
        residuals = y - np.polyval(coeffs, x)
        mad = np.median(np.abs(residuals))
        if mad < 1e-10:
            break
        # Huber weights: downweight points > 1.5 MAD from the fit
        scale = mad * 1.4826  # Convert MAD to approximate std
        threshold = 1.5 * scale
        weights = np.where(np.abs(residuals) <= threshold,
                          1.0,
                          threshold / np.abs(residuals))
    return coeffs


# ─── Comparison & Visualization ────────────────────────────────────────────

def compare_methods(gate_df, line_number, output_dir):
    print(f"\n{'='*60}")
    print(f"Line {line_number}: {len(gate_df)} soundings")
    print(f"{'='*60}")

    # Compute all methods
    print("  [1/4] Wide-window slopes & curvatures...")
    s_ww = widewindow_slopes(gate_df)
    c_ww = widewindow_curvatures(gate_df)

    print("  [2/4] Regression v2 slopes (Theil-Sen, robust, all |values|)...")
    s_reg2 = regression_slopes_v2(gate_df)

    print("  [3/4] Regression v2 curvatures (IRLS quadratic)...")
    c_reg2 = regression_curvatures_v2(gate_df)

    print("  [4/4] Generating plots...")

    # --- Summary stats ---
    print(f"\n  SLOPE COMPARISON (5th-95th percentile):")
    print(f"  {'Gate':>6}  {'WW mean':>10} {'WW std':>10} {'WW n':>6}  "
          f"{'REG mean':>10} {'REG std':>10} {'REG n':>6}")
    for k in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
        ww = s_ww.iloc[:, k].dropna()
        r2 = s_reg2.iloc[:, k].dropna()
        ww_str = f"{ww.mean():10.1f} {ww.std():10.1f} {len(ww):6d}" if len(ww) else "         -          -      0"
        r2_str = f"{r2.mean():10.1f} {r2.std():10.1f} {len(r2):6d}" if len(r2) else "         -          -      0"
        print(f"  {k:>6}  {ww_str}  {r2_str}")

    print(f"\n  CURVATURE COMPARISON:")
    print(f"  {'Gate':>6}  {'WW mean':>10} {'WW std':>10} {'WW n':>6}  "
          f"{'REG mean':>10} {'REG std':>10} {'REG n':>6}")
    for k in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
        ww = c_ww.iloc[:, k].dropna()
        r2 = c_reg2.iloc[:, k].dropna()
        ww_str = f"{ww.mean():10.1f} {ww.std():10.1f} {len(ww):6d}" if len(ww) else "         -          -      0"
        r2_str = f"{r2.mean():10.1f} {r2.std():10.1f} {len(r2):6d}" if len(r2) else "         -          -      0"
        print(f"  {k:>6}  {ww_str}  {r2_str}")

    # --- PLOT 1: Heatmap comparison of slopes ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'Line {line_number}: Slope Values (clipped to [-10, 2])',
                 fontsize=14, fontweight='bold')

    vmin, vmax = -10, 2
    for ax, data, title in [
        (axes[0, 0], s_ww, 'Wide-Window Slope'),
        (axes[0, 1], s_reg2, 'Regression v2 Slope (Theil-Sen)'),
    ]:
        im = ax.imshow(data.values, aspect='auto', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(title)
        ax.set_xlabel('Gate')
        ax.set_ylabel('Sounding')
        plt.colorbar(im, ax=ax)

    # Filter comparison: slope > -0.5 (noise-like = too shallow)
    thresh = -0.5
    for ax, data, title in [
        (axes[1, 0], s_ww, f'WW: culled by slope > {thresh}'),
        (axes[1, 1], s_reg2, f'REG v2: culled by slope > {thresh}'),
    ]:
        cull = (data > thresh).fillna(False).astype(int)
        ax.imshow(cull.values, aspect='auto', cmap='Reds', vmin=0, vmax=1,
                  interpolation='none')
        pct = cull.sum().sum() / cull.size * 100
        ax.set_title(f'{title} ({pct:.1f}% culled)')
        ax.set_xlabel('Gate')
        ax.set_ylabel('Sounding')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / f'line_{line_number}_slope_comparison.png', dpi=150)
    print(f"  Saved: line_{line_number}_slope_comparison.png")

    # --- PLOT 2: Heatmap comparison of curvatures ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'Line {line_number}: Curvature Values (clipped to [-50, 50])',
                 fontsize=14, fontweight='bold')

    vmin_c, vmax_c = -50, 50
    for ax, data, title in [
        (axes[0, 0], c_ww, 'Wide-Window Curvature'),
        (axes[0, 1], c_reg2, 'Regression v2 Curvature (IRLS)'),
    ]:
        im = ax.imshow(data.values, aspect='auto', cmap='RdBu_r',
                       vmin=vmin_c, vmax=vmax_c, interpolation='none')
        ax.set_title(title)
        ax.set_xlabel('Gate')
        ax.set_ylabel('Sounding')
        plt.colorbar(im, ax=ax)

    c_thresh = 10
    for ax, data, title in [
        (axes[1, 0], c_ww, f'WW: culled by |curvature| > {c_thresh}'),
        (axes[1, 1], c_reg2, f'REG v2: culled by |curvature| > {c_thresh}'),
    ]:
        cull = (data.abs() > c_thresh).fillna(False).astype(int)
        ax.imshow(cull.values, aspect='auto', cmap='Reds', vmin=0, vmax=1,
                  interpolation='none')
        pct = cull.sum().sum() / cull.size * 100
        ax.set_title(f'{title} ({pct:.1f}% culled)')
        ax.set_xlabel('Gate')
        ax.set_ylabel('Sounding')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / f'line_{line_number}_curvature_comparison.png', dpi=150)
    print(f"  Saved: line_{line_number}_curvature_comparison.png")

    # --- PLOT 3: Individual sounding traces (6 representative) ---
    n = len(gate_df)
    pos_frac = (gate_df > 0).sum(axis=1) / gate_df.shape[1]
    indices_to_plot = [
        (pos_frac.idxmax(), 'Cleanest'),
        ((pos_frac - 0.8).abs().idxmin(), '~80% positive'),
        ((pos_frac - pos_frac.median()).abs().idxmin(), 'Median'),
        ((pos_frac - 0.4).abs().idxmin(), '~40% positive'),
        (pos_frac.idxmin(), 'Noisiest'),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(20, 22))
    fig.suptitle(f'Line {line_number}: Individual Sounding Traces', fontsize=14,
                 fontweight='bold')

    for row, (idx, label) in enumerate(indices_to_plot):
        raw = gate_df.loc[idx].values
        n_pos = (raw > 0).sum()

        # Decay curve
        ax = axes[row, 0]
        ax.semilogy(range(25), np.abs(raw), 'k.-', label='|dBdt|', ms=4)
        neg = raw < 0
        if neg.any():
            ax.semilogy(np.where(neg)[0], np.abs(raw[neg]), 'rv', ms=6,
                        label='negative')
        ax.set_title(f'{label} (#{idx}, {n_pos}/25 pos)')
        ax.set_xlabel('Gate')
        ax.set_ylabel('|dBdt|')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Slopes
        ax = axes[row, 1]
        ax.plot(range(25), s_ww.loc[idx].values, 'b.-', label='Wide-window',
                alpha=0.7, ms=4)
        ax.plot(range(25), s_reg2.loc[idx].values, 'r.-', label='Reg v2',
                alpha=0.7, ms=4)
        ax.axhline(-0.5, color='gray', ls='--', alpha=0.5, label='thresh -0.5')
        ax.axhline(-2.5, color='green', ls=':', alpha=0.5, label='halfspace -2.5')
        ax.set_title('Slopes')
        ax.set_xlabel('Gate')
        ax.set_ylabel('Slope')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-15, 5)

        # Curvatures
        ax = axes[row, 2]
        ax.plot(range(25), c_ww.loc[idx].values, 'b.-', label='Wide-window',
                alpha=0.7, ms=4)
        ax.plot(range(25), c_reg2.loc[idx].values, 'r.-', label='Reg v2',
                alpha=0.7, ms=4)
        ax.axhline(10, color='gray', ls='--', alpha=0.5, label='thresh ±10')
        ax.axhline(-10, color='gray', ls='--', alpha=0.5)
        ax.set_title('Curvatures')
        ax.set_xlabel('Gate')
        ax.set_ylabel('Curvature')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-80, 80)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / f'line_{line_number}_traces_v2.png', dpi=150)
    print(f"  Saved: line_{line_number}_traces_v2.png")

    # --- PLOT 4: Coverage and consistency ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Line {line_number}: Coverage & Consistency', fontsize=14,
                 fontweight='bold')

    # Coverage
    valid_ww = s_ww.notna().sum() / len(s_ww) * 100
    valid_reg2 = s_reg2.notna().sum() / len(s_reg2) * 100
    x = np.arange(25)
    axes[0].bar(x - 0.2, valid_ww.values, 0.4, label='Wide-window', color='steelblue')
    axes[0].bar(x + 0.2, valid_reg2.values, 0.4, label='Reg v2', color='coral')
    axes[0].set_xlabel('Gate')
    axes[0].set_ylabel('% valid slopes')
    axes[0].set_title('Slope Coverage')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gate-to-gate consistency: stdev of slopes within each sounding
    # (lower = more consistent across adjacent gates)
    ww_consistency = s_ww.diff(axis=1).abs().mean(axis=0)
    r2_consistency = s_reg2.diff(axis=1).abs().mean(axis=0)
    axes[1].bar(x[1:] - 0.2, ww_consistency.values[1:], 0.4, label='Wide-window',
                color='steelblue')
    axes[1].bar(x[1:] + 0.2, r2_consistency.values[1:], 0.4, label='Reg v2',
                color='coral')
    axes[1].set_xlabel('Gate')
    axes[1].set_ylabel('Mean |Δslope| between adjacent gates')
    axes[1].set_title('Gate-to-Gate Slope Smoothness (lower = better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Sounding-to-sounding consistency (for each gate, how much do
    # adjacent soundings differ?)
    ww_spatial = s_ww.diff(axis=0).abs().mean(axis=0)
    r2_spatial = s_reg2.diff(axis=0).abs().mean(axis=0)
    axes[2].bar(x - 0.2, ww_spatial.values, 0.4, label='Wide-window',
                color='steelblue')
    axes[2].bar(x + 0.2, r2_spatial.values, 0.4, label='Reg v2', color='coral')
    axes[2].set_xlabel('Gate')
    axes[2].set_ylabel('Mean |Δslope| between adjacent soundings')
    axes[2].set_title('Spatial Slope Smoothness (lower = better)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f'line_{line_number}_consistency.png', dpi=150)
    print(f"  Saved: line_{line_number}_consistency.png")

    # --- PLOT 5: Gate spacing context (once) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('HeliTEM Gate Context', fontsize=14, fontweight='bold')

    diffs = np.diff(L10_TIMES)
    axes[0].bar(range(1, 25), diffs)
    axes[0].axhline(0.01, color='r', ls='--', label='0.01 threshold')
    axes[0].set_xlabel('Gate')
    axes[0].set_ylabel('Δlog₁₀(t)')
    axes[0].set_title('Adjacent Gate Spacing')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    windows = _build_adaptive_windows(L10_TIMES)
    win_sizes = [len(w) for w in windows]
    axes[1].bar(range(25), win_sizes, color='steelblue')
    axes[1].set_xlabel('Gate')
    axes[1].set_ylabel('Gates in adaptive window')
    axes[1].set_title('Adaptive Window Sizes (min=5, max=0.15 dec)')
    axes[1].grid(True, alpha=0.3)

    # Window span in decades
    win_spans = [L10_TIMES[w[-1]] - L10_TIMES[w[0]] for w in windows]
    axes[2].bar(range(25), win_spans, color='coral')
    axes[2].set_xlabel('Gate')
    axes[2].set_ylabel('Window span (decades)')
    axes[2].set_title('Decade Span per Window')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f'gate_context_v2.png', dpi=150)
    print(f"  Saved: gate_context_v2.png")

    plt.close('all')
    return {'s_ww': s_ww, 's_reg2': s_reg2, 'c_ww': c_ww, 'c_reg2': c_reg2}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")

    for line_num in [50010, 10040]:
        gate_df = load_helitem_line(line_num)
        compare_methods(gate_df, line_num, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
