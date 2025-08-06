# losses/metrics.py - Memory-friendly loss function implementation (Corrected version)
import torch
import torch.nn.functional as F


def _ensure_time_dim(x):
    """Ensure input has time dimension [B, T, 2, H, W]"""
    return x.unsqueeze(1) if x.dim() == 4 else x


def nmse_loss(y_hat, y_true, eps=1e-12):
    """
    Normalized Mean Square Error loss - supports 4D and 5D inputs
    Args:
        y_hat: [B, 2, H, W] or [B, T, 2, H, W] - predicted values
        y_true: [B, 2, H, W] or [B, T, 2, H, W] - ground truth values
    """
    # Ensure both are 5D tensors
    y_hat = _ensure_time_dim(y_hat)
    y_true = _ensure_time_dim(y_true)

    # Flatten last three dimensions
    y_hat_flat = y_hat.flatten(2)  # [B, T, 2*H*W]
    y_true_flat = y_true.flatten(2)  # [B, T, 2*H*W]

    # Calculate power and MSE (compute separately for each batch dimension to save memory)
    power = torch.sum(y_true_flat * y_true_flat, dim=2)  # [B, T]
    mse = torch.sum((y_hat_flat - y_true_flat) ** 2, dim=2)  # [B, T]

    # Avoid division by zero
    power = torch.clamp(power, min=eps)

    # Calculate NMSE (use linear version for training to avoid gradient issues)
    nmse_linear = mse / power
    return torch.mean(nmse_linear)


def nmse_loss_db(y_hat, y_true, eps=1e-12):
    """dB version of NMSE, for evaluation and logging"""
    nmse_linear = nmse_loss(y_hat, y_true, eps)
    return 10 * torch.log10(nmse_linear + eps)


def cosine_similarity_loss(y_hat, y_true, eps=1e-12):
    """
    Cosine similarity loss - supports 4D and 5D inputs
    """
    # Ensure both are 5D tensors
    y_hat = _ensure_time_dim(y_hat)
    y_true = _ensure_time_dim(y_true)

    # Flatten spatial dimensions
    y_hat_flat = y_hat.flatten(2)  # [B, T, features]
    y_true_flat = y_true.flatten(2)  # [B, T, features]

    # Calculate cosine similarity
    dot_product = torch.sum(y_hat_flat * y_true_flat, dim=2)  # [B, T]
    norm_pred = torch.norm(y_hat_flat, dim=2) + eps  # [B, T]
    norm_true = torch.norm(y_true_flat, dim=2) + eps  # [B, T]

    cosine_sim = dot_product / (norm_pred * norm_true)

    # Return 1 minus average cosine similarity as loss
    return 1.0 - torch.mean(cosine_sim)


def temporal_smooth_loss(y_hat, eps=1e-12):
    """
    Temporal smoothness loss - automatically handles 4D and 5D inputs
    """
    y_hat = _ensure_time_dim(y_hat)

    if y_hat.size(1) <= 1:
        return torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)

    # Calculate adjacent frame differences
    diff = y_hat[:, 1:] - y_hat[:, :-1]  # [B, T-1, 2, H, W]

    # Calculate average squared difference
    smooth_loss = torch.mean(diff ** 2)

    return smooth_loss


def compute_rho(y_hat, y_true, eps=1e-12):
    """
    Calculate average cosine similarity ρ (for evaluation)
    Uses safer complex number computation method
    """
    # Ensure both are 5D tensors
    y_hat = _ensure_time_dim(y_hat)
    y_true = _ensure_time_dim(y_true)

    # Safer complex conversion method
    # Stack real and imaginary parts to last dimension, then use view_as_complex
    y_hat_complex_input = torch.stack([y_hat[:, :, 0], y_hat[:, :, 1]], dim=-1)  # [B, T, H, W, 2]
    y_true_complex_input = torch.stack([y_true[:, :, 0], y_true[:, :, 1]], dim=-1)  # [B, T, H, W, 2]

    y_hat_complex = torch.view_as_complex(y_hat_complex_input)  # [B, T, H, W]
    y_true_complex = torch.view_as_complex(y_true_complex_input)  # [B, T, H, W]

    # Flatten spatial dimensions
    y_hat_flat = y_hat_complex.flatten(2)  # [B, T, H*W]
    y_true_flat = y_true_complex.flatten(2)  # [B, T, H*W]

    # Calculate cosine similarity for each subcarrier
    numerator = torch.abs(torch.sum(torch.conj(y_hat_flat) * y_true_flat, dim=2))  # [B, T]
    denominator = (torch.norm(y_hat_flat, dim=2) * torch.norm(y_true_flat, dim=2) + eps)  # [B, T]

    rho = numerator / denominator
    return torch.mean(rho).item()


# Helper functions for evaluation
def compute_metrics(y_hat, y_true):
    """
    Calculate all evaluation metrics
    Returns:
        dict: {'nmse_db': float, 'rho': float, 'mse': float}
    """
    with torch.no_grad():
        # NMSE (dB) - Fixed: use correct dB version
        nmse_db = nmse_loss_db(y_hat, y_true).item()

        # Cosine similarity ρ
        rho = compute_rho(y_hat, y_true)

        # Raw MSE
        mse = F.mse_loss(y_hat, y_true).item()

        return {
            'nmse_db': nmse_db,
            'rho': rho,
            'mse': mse
        }


# Data loader creation function
def create_dummy_data(cfg):
    """
    Create dummy data for testing - adapted for GTX 1650 Ti memory constraints
    """
    batch_size = cfg.training.batch_size
    T = cfg.model.sequence.T

    # Generate simulated COST2100 data
    H = torch.randn(batch_size * 10, T, 2, 32, 32)  # Data for 10 batches

    # Simple dataset class
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = DummyDataset(H)

    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Ensure using parameters from configuration
    num_workers = cfg.training.get('num_workers', 0)
    pin_memory = cfg.training.get('pin_memory', False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader