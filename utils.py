## utils.py
from pathlib import Path
from filelock import FileLock
import mlx.core as mx # Import mx if using it directly here, or pass numpy/list

def ensure_directory(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def get_lock(path: Path) -> FileLock:
    """Return a FileLock object for the given path's directory."""
    # Lock should ideally be on the directory level if multiple files change
    lock_path = path / ".store.lock" # Lock file within the store directory
    return FileLock(str(lock_path))

def validate_vector_shape(tensor, expected_dim: int) -> None:
    """
    Raise ValueError if tensor is not 2D or has wrong dimensionality if expected_dim > 0.
    Accepts numpy or mlx arrays.
    """
    # Get shape, works for both numpy and mlx tensors
    shape = tensor.shape

    # Prüfe, ob der Tensor überhaupt 2 Dimensionen hat
    if len(shape) != 2:
         raise ValueError(f"Expected 2D tensor (n, d), got {len(shape)}D shape {shape}")

    # Prüfe die letzte Dimension nur, wenn eine positive Erwartung gegeben ist
    if expected_dim > 0 and shape[-1] != expected_dim:
        raise ValueError(f"Expected vector dimension {expected_dim}, got {shape[-1]}")

    # Optional: Check for empty tensor (0 rows) if needed, although often allowed
    # if shape[0] == 0:
    #     print("Warning: Validating shape of an empty tensor (0 rows).")