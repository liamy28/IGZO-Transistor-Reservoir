"""Prepare balanced STARS light curves for reservoir_network_testing.py.

The output NPZ contains balanced real light curves from CEP, EB, RRAB and RRC,
phase-folded to 100 values in [0, 1], phase-aligned to a common minimum, plus
deterministic stratified train/test indices.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import lombscargle


# Different (4 types) of variable star brightness time series from telescope
# CEP = Cepheid variables
# EB = Eclipsing Binaries
# RRAB = RR Lyrae type AB (fundamental mode)
# RRC = RR Lyrae type C (first overtone)
CLASS_DIRS = {
    "CEP": Path("CEP_LCs"),
    "EB": Path("EB_LCs"),
    "RRAB": Path("RR_LCs") / "RRAB",
    "RRC": Path("RR_LCs") / "RRC",
}


def phase_fold(time: np.ndarray, flux: np.ndarray, length: int) -> np.ndarray:
    """Estimate the dominant period and resample one circular phase cycle."""

    # 1. Finds the dominant period using Lomb-Scargle periodogram
    # 2. Wraps the light curve to phase [0, 1)
    # 3. Bins and resamples to 'length' evenly spaced points
    # 4. Returns one clean phase-folded light curve

    baseline = time[-1] - time[0]

    # Center flux for period detection.
    centered = flux - np.mean(flux)
    if baseline <= 0 or np.std(centered) <= 0:
        raise ValueError("constant light curve")

    # Period range covers short RR Lyrae periods through long Cepheid/EB periods.
    min_period = 0.05
    max_period = min(200.0, baseline / 2.0)

    if max_period <= min_period:
        raise ValueError("time span too short for period estimation")

    # Log-spaced frequency grid gives useful coverage over a wide period range.
    frequencies = np.geomspace(
        1.0 / max_period,
        1.0 / min_period,
        5000,
    )

    power = lombscargle(
        time - time[0],
        centered,
        2 * np.pi * frequencies,
        precenter=False,
        normalize=True,
    )

    # Dominant Lomb-Scargle period.
    period = 1.0 / frequencies[int(np.nanargmax(power))]

    # Fold observations onto one cycle.
    phase = np.mod(time - time[0], period) / period

    # Sort by phase.
    order = np.argsort(phase)
    phase = phase[order]
    flux = flux[order]

    # Bin before interpolation to reduce noise and duplicate phases.
    n_bins = max(length, 200)
    bin_id = np.minimum((phase * n_bins).astype(int), n_bins - 1)

    counts = np.bincount(bin_id, minlength=n_bins)
    sums = np.bincount(bin_id, weights=flux, minlength=n_bins)

    occupied = counts > 0
    phase_binned = (np.flatnonzero(occupied) + 0.5) / n_bins
    flux_binned = sums[occupied] / counts[occupied]

    if len(phase_binned) < 20:
        raise ValueError("insufficient phase coverage")

    # Circular extension prevents an artificial discontinuity at phase 0/1.
    phase_ext = np.concatenate(
        (
            phase_binned[-1:] - 1,
            phase_binned,
            phase_binned[:1] + 1,
        )
    )
    flux_ext = np.concatenate(
        (
            flux_binned[-1:],
            flux_binned,
            flux_binned[:1],
        )
    )

    # Resample to exactly 'length' evenly spaced phase points.
    phase_target = np.arange(length) / length
    return np.interp(phase_target, phase_ext, flux_ext)


def phase_align_minimum(
    curve: np.ndarray,
    smooth_window: int = 5,
) -> np.ndarray:
    """Circularly shift a folded curve so its dominant minimum is at index 0.

    The anchor is found using a lightly smoothed circular copy of the curve,
    which reduces sensitivity to a single noisy point. The original unsmoothed
    curve is shifted, so the curve values themselves are not altered.
    """

    curve = np.asarray(curve, dtype=float)

    if curve.ndim != 1 or len(curve) == 0:
        raise ValueError("invalid curve for phase alignment")

    if smooth_window <= 1:
        anchor = int(np.argmin(curve))
    else:
        # Use an odd smoothing window so the filter is centered.
        if smooth_window % 2 == 0:
            smooth_window += 1

        # Avoid a smoothing window larger than the curve.
        smooth_window = min(smooth_window, len(curve))

        if smooth_window % 2 == 0:
            smooth_window -= 1

        if smooth_window <= 1:
            anchor = int(np.argmin(curve))
        else:
            half = smooth_window // 2

            # Wrap padding is appropriate because phase 0 and phase 1
            # are adjacent points on the folded light curve.
            padded = np.pad(curve, (half, half), mode="wrap")

            kernel = np.ones(smooth_window, dtype=float) / smooth_window
            smooth = np.convolve(padded, kernel, mode="valid")

            anchor = int(np.argmin(smooth))

    # Put the dominant minimum at index 0.
    return np.roll(curve, -anchor)


def load_light_curve(path: Path, length: int) -> np.ndarray:
    """Load, clean, phase-fold, phase-align and robustly scale one light curve."""

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
    )

    if data.size == 0 or not {"time", "flux"}.issubset(data.dtype.names or ()):
        raise ValueError("missing time/flux data")

    time = np.atleast_1d(data["time"])
    flux = np.atleast_1d(data["flux"])

    good = np.isfinite(time) & np.isfinite(flux)
    time = time[good]
    flux = flux[good]

    if len(time) < 20:
        raise ValueError("fewer than 20 finite observations")

    # Sort observations chronologically.
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    # Average duplicate timestamps.
    unique_time, inverse = np.unique(time, return_inverse=True)

    if len(unique_time) != len(time):
        sums = np.bincount(inverse, weights=flux)
        counts = np.bincount(inverse)

        flux = sums / counts
        time = unique_time

    # Remove extreme measurement failures before phase folding.
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))

    if mad > 0:
        # 1.4826 converts MAD to an estimate of standard deviation for
        # normally distributed data. 8 sigma is intentionally conservative.
        keep = np.abs(flux - median) <= 8.0 * 1.4826 * mad
        time = time[keep]
        flux = flux[keep]

    if len(time) < 20 or time[-1] <= time[0]:
        raise ValueError("insufficient time span after cleaning")

    # ------------------------------------------------------------------
    # PHASE FOLD
    # ------------------------------------------------------------------
    curve = phase_fold(time, flux, length)

    # ------------------------------------------------------------------
    # PHASE ALIGNMENT
    #
    # Previously, phase zero was determined by the first observation time.
    # That means two stars with the same shape could appear as circularly
    # shifted versions of one another. Aligning the dominant minimum gives
    # the classifier and reservoir a more consistent phase reference.
    # ------------------------------------------------------------------
    curve = phase_align_minimum(curve, smooth_window=5)

    # ------------------------------------------------------------------
    # NORMALIZE TO [0, 1]
    # ------------------------------------------------------------------
    lo, hi = np.percentile(curve, [1, 99])

    if not np.isfinite(lo + hi) or hi <= lo:
        raise ValueError("constant or invalid flux")

    curve = np.clip(
        (curve - lo) / (hi - lo),
        0.0,
        1.0,
    )

    return curve.astype(np.float32)


def prepare(
    stars_dir: Path,
    per_class: int,
    length: int,
    seed: int,
):
    """Prepare a balanced phase-folded dataset and train/test split."""

    rng = np.random.default_rng(seed)

    curves = []
    labels = []
    source_ids = []
    rejected = {}

    for label, (class_name, relative_dir) in enumerate(CLASS_DIRS.items()):
        files = sorted(
            (stars_dir / relative_dir).glob("*.csv")
        )

        rng.shuffle(files)

        accepted = 0
        failures = 0

        for path in files:
            try:
                curve = load_light_curve(path, length)

            except (OSError, ValueError, IndexError):
                failures += 1
                continue

            curves.append(curve)
            labels.append(label)
            source_ids.append(path.stem)

            accepted += 1

            if accepted == per_class:
                break

        if accepted < per_class:
            raise RuntimeError(
                f"{class_name}: only {accepted} valid files; "
                f"need {per_class}"
            )

        rejected[class_name] = failures

    labels = np.asarray(labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # STRATIFIED TRAIN/TEST SPLIT
    # ------------------------------------------------------------------
    train_idx = []
    test_idx = []

    for label in range(len(CLASS_DIRS)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)

        n_test = max(
            1,
            int(round(0.1 * len(indices))),
        )

        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    return {
        "curves": np.asarray(curves, dtype=np.float32),
        "labels": labels,
        "class_names": np.asarray(list(CLASS_DIRS)),
        "source_ids": np.asarray(source_ids),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
    }, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--stars-dir",
        type=Path,
        default=Path(__file__).parent / "STARS",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "stars_reservoir_dataset.npz",
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--length",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    dataset, rejected = prepare(
        args.stars_dir,
        args.per_class,
        args.length,
        args.seed,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        **dataset,
    )

    # Save a companion CSV showing the source star and split for every row.
    with args.output.with_suffix(".csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "row",
                "source_id",
                "class_id",
                "class_name",
                "split",
            ]
        )

        test_rows = set(
            dataset["test_idx"].tolist()
        )

        for row, (source_id, label) in enumerate(
            zip(
                dataset["source_ids"],
                dataset["labels"],
            )
        ):
            writer.writerow(
                [
                    row,
                    source_id,
                    label,
                    dataset["class_names"][label],
                    "test" if row in test_rows else "train",
                ]
            )

    print(f"Saved {args.output}")

    print(
        f"curves={dataset['curves'].shape}, "
        f"train={len(dataset['train_idx'])}, "
        f"test={len(dataset['test_idx'])}"
    )

    print(
        "classes:",
        ", ".join(dataset["class_names"]),
    )

    print(
        "rejected before quotas:",
        rejected,
    )


if __name__ == "__main__":
    main()
