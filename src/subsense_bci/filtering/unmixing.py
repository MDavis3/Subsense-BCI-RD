"""
Source Unmixing via PCA/ICA - Phase 3 Inverse Problem

Recovers the original neural source waveforms from the mixed 10,000-sensor
recording using dimensionality reduction (PCA) followed by Independent
Component Analysis (FastICA).

Pipeline:
1. Load noisy sensor recording X(t) and ground truth sources S(t)
2. PCA: Reduce 10,000 sensors to k components (99.9% variance explained)
3. FastICA: Extract exactly 3 independent components from PCA space
4. Signal Matching: Correlate recovered ICs with ground truth, reorder & sign-flip

Mathematical Model:
    X = L @ S + N    (forward model)
    S_hat = W @ X    (inverse via ICA, where W ≈ L^(-1))
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


@dataclass
class UnmixingResult:
    """Container for unmixing pipeline results."""

    # Recovered signals
    recovered_sources: np.ndarray  # (n_sources, n_samples)
    matched_sources: np.ndarray  # (n_sources, n_samples) - reordered & sign-flipped

    # Correlation metrics
    correlation_matrix: np.ndarray  # (n_recovered, n_ground_truth)
    matched_correlations: np.ndarray  # (n_sources,) - diagonal after matching
    source_order: np.ndarray  # Permutation applied to match ground truth
    sign_flips: np.ndarray  # Sign corrections applied

    # PCA statistics
    n_pca_components: int
    variance_explained: float
    pca_singular_values: np.ndarray

    # ICA statistics
    ica_n_iter: int


def load_phase2_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Phase 2 simulation data.

    Returns
    -------
    recording : np.ndarray
        Noisy sensor recording, shape (n_sensors, n_samples).
    ground_truth : np.ndarray
        Original source waveforms, shape (n_sources, n_samples).
    time_vector : np.ndarray
        Time vector in seconds, shape (n_samples,).
    """
    data_dir = get_project_root() / "data" / "raw"

    recording = np.load(data_dir / "recording_simulation.npy")
    ground_truth = np.load(data_dir / "source_waveforms.npy")
    time_vector = np.load(data_dir / "time_vector.npy")

    return recording, ground_truth, time_vector


def pca_denoise(
    data: np.ndarray,
    variance_threshold: float = 0.999,
) -> tuple[np.ndarray, PCA]:
    """
    Reduce dimensionality via PCA while preserving target variance.

    Parameters
    ----------
    data : np.ndarray
        Sensor data, shape (n_sensors, n_samples).
    variance_threshold : float
        Fraction of variance to preserve (default: 99.9%).

    Returns
    -------
    reduced_data : np.ndarray
        PCA-transformed data, shape (n_components, n_samples).
    pca : PCA
        Fitted PCA object for inspection.
    """
    # Transpose: sklearn expects (n_samples, n_features)
    X = data.T  # (n_samples, n_sensors)

    # Center the data (required for PCA)
    scaler = StandardScaler(with_std=False)  # Only center, don't scale
    X_centered = scaler.fit_transform(X)

    # Fit PCA with enough components to explain variance_threshold
    # First fit with all components to determine how many we need
    pca_full = PCA(n_components=min(X.shape))
    pca_full.fit(X_centered)

    # Find number of components for target variance
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cumsum, variance_threshold) + 1
    n_components = max(3, n_components)  # At least 3 for our 3 sources

    print(f"  PCA: {n_components} components explain {cumsum[n_components-1]*100:.2f}% variance")

    # Refit with exact number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_centered)

    # Transpose back: (n_samples, n_components) -> (n_components, n_samples)
    return X_pca.T, pca


def run_fastica(
    data: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
    max_iter: int = 1000,
) -> tuple[np.ndarray, FastICA]:
    """
    Extract independent components using FastICA.

    Parameters
    ----------
    data : np.ndarray
        PCA-reduced data, shape (n_pca_components, n_samples).
    n_components : int
        Number of independent components to extract.
    random_state : int
        Random seed for reproducibility.
    max_iter : int
        Maximum iterations for FastICA convergence.

    Returns
    -------
    sources : np.ndarray
        Recovered independent components, shape (n_components, n_samples).
    ica : FastICA
        Fitted ICA object.
    """
    # Transpose: sklearn expects (n_samples, n_features)
    X = data.T  # (n_samples, n_pca_components)

    ica = FastICA(
        n_components=n_components,
        algorithm="parallel",
        whiten="unit-variance",  # Whiten to unit variance
        fun="logcosh",  # Robust for super-Gaussian sources
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-6,
    )

    sources = ica.fit_transform(X)

    # Transpose back: (n_samples, n_components) -> (n_components, n_samples)
    return sources.T, ica


def match_sources(
    recovered: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match recovered sources to ground truth via correlation.

    Uses the Hungarian algorithm to find optimal assignment that maximizes
    total absolute correlation. Also handles sign ambiguity (ICA can flip signs).

    Parameters
    ----------
    recovered : np.ndarray
        Recovered sources, shape (n_components, n_samples).
    ground_truth : np.ndarray
        Ground truth sources, shape (n_sources, n_samples).

    Returns
    -------
    matched : np.ndarray
        Reordered and sign-corrected recovered sources.
    correlation_matrix : np.ndarray
        Full correlation matrix (n_recovered, n_ground_truth).
    order : np.ndarray
        Permutation indices applied.
    signs : np.ndarray
        Sign corrections applied (+1 or -1).
    """
    n_recovered = recovered.shape[0]
    n_truth = ground_truth.shape[0]

    # Compute Pearson correlation matrix
    corr_matrix = np.zeros((n_recovered, n_truth))
    for i in range(n_recovered):
        for j in range(n_truth):
            corr, _ = stats.pearsonr(recovered[i], ground_truth[j])
            corr_matrix[i, j] = corr

    # Use Hungarian algorithm on absolute correlations (maximize)
    # linear_sum_assignment minimizes, so we negate
    cost_matrix = -np.abs(corr_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder recovered sources to match ground truth order
    order = col_ind  # Maps recovered index to ground truth index
    # Create inverse mapping: for each ground truth source, which recovered source?
    inverse_order = np.argsort(col_ind)

    # Determine sign flips based on correlation sign
    signs = np.sign(corr_matrix[row_ind, col_ind])

    # Apply reordering and sign correction
    matched = np.zeros_like(ground_truth)
    for i, (rec_idx, gt_idx) in enumerate(zip(row_ind, col_ind)):
        matched[gt_idx] = recovered[rec_idx] * signs[i]

    # Reorder signs to match ground truth order
    signs_ordered = signs[inverse_order]

    return matched, corr_matrix, order, signs_ordered


def unmix_sources(
    recording: np.ndarray,
    ground_truth: np.ndarray,
    variance_threshold: float = 0.999,
    n_sources: int = 3,
    random_state: int = 42,
) -> UnmixingResult:
    """
    Full PCA + ICA unmixing pipeline.

    Parameters
    ----------
    recording : np.ndarray
        Noisy sensor recording, shape (n_sensors, n_samples).
    ground_truth : np.ndarray
        Ground truth sources for matching, shape (n_sources, n_samples).
    variance_threshold : float
        PCA variance preservation threshold.
    n_sources : int
        Number of sources to recover.
    random_state : int
        Random seed.

    Returns
    -------
    UnmixingResult
        Complete results including recovered sources and metrics.
    """
    print("\n[1/3] PCA Dimensionality Reduction...")
    pca_data, pca = pca_denoise(recording, variance_threshold)

    print(f"\n[2/3] FastICA Source Separation...")
    print(f"  Extracting {n_sources} independent components...")
    recovered, ica = run_fastica(pca_data, n_components=n_sources, random_state=random_state)
    print(f"  ICA converged in {ica.n_iter_} iterations")

    print(f"\n[3/3] Source Matching...")
    matched, corr_matrix, order, signs = match_sources(recovered, ground_truth)

    # Extract matched correlations (diagonal after optimal assignment)
    matched_correlations = np.array([
        corr_matrix[i, order[i]] * signs[i]
        for i in range(n_sources)
    ])
    # Actually we want absolute correlations for the matched pairs
    matched_correlations = np.abs(np.diag(
        corr_matrix[np.arange(n_sources), :][:, order]
    ))

    return UnmixingResult(
        recovered_sources=recovered,
        matched_sources=matched,
        correlation_matrix=corr_matrix,
        matched_correlations=matched_correlations,
        source_order=order,
        sign_flips=signs,
        n_pca_components=pca.n_components_,
        variance_explained=np.sum(pca.explained_variance_ratio_),
        pca_singular_values=pca.singular_values_,
        ica_n_iter=ica.n_iter_,
    )


def save_unmixing_results(
    result: UnmixingResult,
    output_dir: Path | str = None,
) -> dict[str, Path]:
    """
    Save unmixing results to disk.

    Parameters
    ----------
    result : UnmixingResult
        Unmixing pipeline results.
    output_dir : Path, optional
        Output directory. Defaults to data/processed.

    Returns
    -------
    dict
        Paths to saved files.
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save recovered sources (matched to ground truth)
    path_sources = output_dir / "recovered_sources.npy"
    np.save(path_sources, result.matched_sources)
    paths["recovered_sources"] = path_sources

    # Save correlation matrix
    path_corr = output_dir / "correlation_matrix.npy"
    np.save(path_corr, result.correlation_matrix)
    paths["correlation_matrix"] = path_corr

    return paths


def main() -> None:
    """Run Phase 3 source unmixing pipeline."""
    print("=" * 60)
    print("  PHASE 3: Source Unmixing via PCA/ICA")
    print("=" * 60)

    # Load data
    print("\n[0/3] Loading Phase 2 data...")
    recording, ground_truth, time_vector = load_phase2_data()
    print(f"  Recording: {recording.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    print(f"  Time vector: {time_vector.shape}")

    # Run unmixing
    result = unmix_sources(recording, ground_truth)

    # Report results
    print("\n" + "=" * 60)
    print("  UNMIXING RESULTS")
    print("=" * 60)

    print(f"\nPCA Summary:")
    print(f"  Components retained: {result.n_pca_components}")
    print(f"  Variance explained: {result.variance_explained * 100:.2f}%")

    print(f"\nICA Summary:")
    print(f"  Iterations to converge: {result.ica_n_iter}")

    print(f"\nSource Recovery Correlations:")
    source_names = ["A (10Hz Alpha)", "B (20Hz Beta)", "C (Pink Noise)"]
    for i, (name, corr) in enumerate(zip(source_names, result.matched_correlations)):
        quality = "Excellent" if corr > 0.95 else "Good" if corr > 0.85 else "Fair" if corr > 0.70 else "Poor"
        print(f"  Source {name}: r = {corr:.4f} [{quality}]")

    print(f"\nSign corrections applied: {result.sign_flips}")
    print(f"Source permutation: {result.source_order}")

    # Save results
    print("\nSaving results...")
    paths = save_unmixing_results(result)
    for name, path in paths.items():
        print(f"  {name}: {path.name}")

    print("\n" + "=" * 60)
    print("  Phase 3 unmixing complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
