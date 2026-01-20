"""Evaluation metrics for protein structure prediction."""

from typing import Dict, List, Optional, Any
import numpy as np


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root mean squared error."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(predictions - targets))


def pearson_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if len(predictions) < 2:
        return 0.0
    
    pred_mean = np.mean(predictions)
    target_mean = np.mean(targets)
    
    numerator = np.sum((predictions - pred_mean) * (targets - target_mean))
    denominator = np.sqrt(
        np.sum((predictions - pred_mean) ** 2) * 
        np.sum((targets - target_mean) ** 2)
    )
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def r2_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - ss_res / ss_tot


def spearman_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Spearman rank correlation."""
    from scipy import stats
    correlation, _ = stats.spearmanr(predictions, targets)
    return float(correlation) if not np.isnan(correlation) else 0.0


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: List[str] = None,
) -> Dict[str, float]:
    """Compute multiple metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        metrics: List of metric names to compute
    
    Returns:
        Dictionary of metric values
    """
    metrics = metrics or ["rmse", "mae", "pearson", "r2"]
    
    metric_functions = {
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson_correlation,
        "r2": r2_score,
        "spearman": spearman_correlation,
    }
    
    results = {}
    for metric_name in metrics:
        if metric_name in metric_functions:
            results[metric_name] = metric_functions[metric_name](predictions, targets)
    
    return results


def compute_lddt(
    predicted_coords: np.ndarray,
    true_coords: np.ndarray,
    mask: Optional[np.ndarray] = None,
    cutoff: float = 15.0,
) -> np.ndarray:
    """Compute local distance difference test (lDDT).
    
    Args:
        predicted_coords: Predicted coordinates [N, 3]
        true_coords: True coordinates [N, 3]
        mask: Optional mask for valid residues [N]
        cutoff: Distance cutoff in Angstroms
    
    Returns:
        Per-residue lDDT scores [N]
    """
    N = predicted_coords.shape[0]
    
    # Compute pairwise distances
    pred_dist = np.sqrt(np.sum(
        (predicted_coords[:, None] - predicted_coords[None, :]) ** 2,
        axis=-1
    ))
    true_dist = np.sqrt(np.sum(
        (true_coords[:, None] - true_coords[None, :]) ** 2,
        axis=-1
    ))
    
    # Thresholds for lDDT
    thresholds = [0.5, 1.0, 2.0, 4.0]
    
    lddt_scores = np.zeros(N)
    
    for i in range(N):
        if mask is not None and not mask[i]:
            continue
        
        # Find residues within cutoff
        within_cutoff = true_dist[i] < cutoff
        within_cutoff[i] = False  # Exclude self
        
        if mask is not None:
            within_cutoff = within_cutoff & mask
        
        if not np.any(within_cutoff):
            lddt_scores[i] = 0.0
            continue
        
        # Compute distance differences
        dist_diff = np.abs(pred_dist[i, within_cutoff] - true_dist[i, within_cutoff])
        
        # Check against thresholds
        preserved = 0.0
        for threshold in thresholds:
            preserved += np.mean(dist_diff < threshold)
        
        lddt_scores[i] = preserved / len(thresholds)
    
    return lddt_scores


def compute_tm_score(
    predicted_coords: np.ndarray,
    true_coords: np.ndarray,
    d0: Optional[float] = None,
) -> float:
    """Compute TM-score (simplified version without alignment).
    
    For accurate TM-score, use TMalign.
    
    Args:
        predicted_coords: Predicted CA coordinates [N, 3]
        true_coords: True CA coordinates [N, 3]
        d0: Distance scaling parameter
    
    Returns:
        TM-score
    """
    N = len(predicted_coords)
    
    if d0 is None:
        # Standard d0 formula
        d0 = 1.24 * (N - 15) ** (1/3) - 1.8
        d0 = max(d0, 0.5)
    
    # Compute distances (assuming structures are aligned)
    distances = np.sqrt(np.sum((predicted_coords - true_coords) ** 2, axis=-1))
    
    # TM-score formula
    tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / N
    
    return tm_score


def evaluate_model(
    model: Any,
    dataloader: Any,
    metrics: List[str] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        metrics: List of metrics to compute
        device: Device to run on
    
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        return {}
    
    if not TORCH_AVAILABLE:
        return {}
    
    model.eval()
    device = torch.device(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = model(**batch)
            
            if "predictions" in outputs:
                all_predictions.append(outputs["predictions"].cpu().numpy())
            
            if "targets" in batch:
                all_targets.append(batch["targets"].cpu().numpy())
    
    if not all_predictions or not all_targets:
        return {}
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    return compute_metrics(predictions, targets, metrics)
