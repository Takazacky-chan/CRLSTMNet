# datasets/cost2100.py - Enhanced version supporting 32×64 data with physically reasonable downsampling
import os
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import glob
from pathlib import Path
import logging


class COST2100Dataset(Dataset):
    """
    COST2100 dataset loader - Enhanced version
    Supports 32×32 and 32×64 inputs with physically reasonable frequency domain downsampling
    """

    def __init__(self, data_path, sequence_length=10, normalize='standardize',
                 split='train', dataset_spec=None, minmax_file=None, stride=None,
                 downsample_method='smart_avg'):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length  # 🔧 Fix: correctly set stride
        self.normalize = normalize
        self.split = split
        self.dataset_spec = dataset_spec
        self.minmax_file = minmax_file
        self.downsample_method = downsample_method  # 'smart_avg', 'lowpass_decimate', 'learnable'

        print(f"Initializing COST2100Dataset:")
        print(f"  data_path: {self.data_path}")
        print(f"  sequence_length: {self.sequence_length}")
        print(f"  stride: {self.stride}")
        print(f"  normalize: {self.normalize}")
        print(f"  split: {self.split}")
        print(f"  downsample_method: {self.downsample_method}")

        # If minmax file exists, load normalization parameters
        self.norm_params = None
        if minmax_file and Path(minmax_file).exists():
            self._load_norm_params()

        # Find all .mat files
        self.data_files = self._find_data_files()

        if len(self.data_files) == 0:
            raise RuntimeError(f"No .mat files found in {data_path}")

        print(f"Found {len(self.data_files)} .mat files")

        # Load and preprocess data
        self.data = self._load_all_data()

        # Split dataset by proportion
        if dataset_spec and 'train_split' in dataset_spec:
            self._split_dataset(dataset_spec['train_split'])

        print(f"Final dataset size: {len(self.data)} sequences")

    def _find_data_files(self):
        """Find data files - supports CSINet-LSTM format"""
        mat_files = []

        if hasattr(self, 'dataset_spec') and self.dataset_spec:
            prefix = self.dataset_spec.get('prefix', 'H_user_t')
            suffix = self.dataset_spec.get('suffix', '32all.mat')
            pattern = f"{prefix}*{suffix}"
            mat_files.extend(glob.glob(str(self.data_path / pattern)))
        else:
            for pattern in ['*.mat', '**/*.mat']:
                mat_files.extend(glob.glob(str(self.data_path / pattern), recursive=True))

        mat_files.sort()
        logging.info(f"Found {len(mat_files)} .mat files with pattern")
        return mat_files

    def _load_all_data(self):
        """Load all data files"""
        all_sequences = []

        print(f"Loading data from {len(self.data_files)} files...")

        for file_idx, file_path in enumerate(self.data_files):
            print(f"\nProcessing file {file_idx + 1}/{len(self.data_files)}: {Path(file_path).name}")

            try:
                mat_data = sio.loadmat(file_path)
                available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                print(f"Available keys: {available_keys}")

                # Prioritize variable name specified in dataset_spec
                h_matrix = None
                if hasattr(self, 'dataset_spec') and self.dataset_spec:
                    var_name = self.dataset_spec.get('variable_name', 'Hur_down_t1')
                    if var_name in mat_data:
                        h_matrix = mat_data[var_name]
                        print(
                            f"Using variable '{var_name}' with shape: {h_matrix.shape if h_matrix is not None else 'None'}")

                # If specified variable not found, try other possible key names
                if h_matrix is None:
                    possible_keys = ['H', 'Hur_down_t1', 'channel', 'CSI', 'data', 'H_sequences', 'H_user']
                    for key in possible_keys:
                        if key in mat_data and not key.startswith('__'):
                            h_matrix = mat_data[key]
                            if h_matrix is not None:
                                print(f"Using fallback key '{key}' with shape: {h_matrix.shape}")
                                break

                if h_matrix is None:
                    if available_keys:
                        h_matrix = mat_data[available_keys[0]]
                        if h_matrix is not None:
                            print(f"Using first available key '{available_keys[0]}' with shape: {h_matrix.shape}")

                if h_matrix is None:
                    print(f"No valid data found in {file_path}")
                    continue

                if hasattr(h_matrix, 'shape'):
                    print(f"Data shape: {h_matrix.shape}, dtype: {h_matrix.dtype}")
                else:
                    print(f"Data is not array-like: {type(h_matrix)}")
                    continue

                # Process data format
                sequences = self._process_matrix(h_matrix, file_path)
                print(f"_process_matrix returned: {len(sequences) if sequences else 0} sequences")

                if sequences:
                    all_sequences.extend(sequences)
                    print(f"Added {len(sequences)} sequences from {file_path}")
                    print(f"Total sequences so far: {len(all_sequences)}")
                else:
                    print(f"No sequences extracted from {file_path}")

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Final total sequences loaded: {len(all_sequences)}")
        return all_sequences

    def _downsample_64_to_32(self, complex_data):
        """
        Downsample 32×64 complex data to 32×32
        Args:
            complex_data: [time, 32, 64] complex array
        Returns:
            downsampled: [time, 32, 32] complex array
        """
        if self.downsample_method == 'smart_avg':
            # Method 1: Smart averaging + energy normalization
            # Adjacent pair averaging while preserving energy: (x1+x2)/√2
            reshaped = complex_data.reshape(complex_data.shape[0], 32, 32, 2)  # [time, 32, 32, 2]
            averaged = np.mean(reshaped, axis=-1)  # [time, 32, 32]
            # Energy normalization: multiply by √2 to preserve power
            return averaged * np.sqrt(2.0)

        elif self.downsample_method == 'lowpass_decimate':
            # Method 2: Low-pass filtering + decimation
            from scipy import signal

            # Design simple low-pass filter (3-point moving average)
            b = np.array([0.25, 0.5, 0.25])  # Low-pass filter coefficients

            downsampled = np.zeros((complex_data.shape[0], 32, 32), dtype=complex_data.dtype)

            for t in range(complex_data.shape[0]):
                for ant in range(32):
                    # Low-pass filtering
                    filtered = signal.filtfilt(b, [1.0], complex_data[t, ant, :])
                    # Decimate by 2 (take even indices)
                    downsampled[t, ant, :] = filtered[::2]

            return downsampled

        elif self.downsample_method == 'center_crop':
            # Method 3: Center cropping (take middle 32 subcarriers)
            start_idx = (64 - 32) // 2  # 16
            return complex_data[:, :, start_idx:start_idx + 32]

        else:
            # Default: Simple averaging (maintain backward compatibility)
            reshaped = complex_data.reshape(complex_data.shape[0], 32, 32, 2)
            return np.mean(reshaped, axis=-1)

    def _process_matrix(self, h_matrix, file_path):
        """
        Process single matrix data, return list of [N_seq, T, 2, 32, 32]
        Supports 32×32 and 32×64 inputs with physically reasonable downsampling
        """
        sequences = []

        if torch.is_tensor(h_matrix):
            h_matrix = h_matrix.numpy()

        logging.info(f"Processing {file_path}: shape {getattr(h_matrix, 'shape', None)}")

        T = self.sequence_length
        stride = self.stride

        try:
            # ---- Case 1: [time, 32, 32] complex ----
            if h_matrix.ndim == 3 and h_matrix.shape[-2:] == (32, 32) and \
                    h_matrix.dtype in (np.complex64, np.complex128):
                print(f"Processing 32×32 complex data")
                real = np.real(h_matrix)
                imag = np.imag(h_matrix)
                h = np.stack([real, imag], axis=1)  # [time, 2, 32, 32]
                total_t = h.shape[0]

                for s in range(0, total_t - T + 1, stride):
                    sequences.append(h[s:s + T].astype(np.float32))

                print(f"Extracted {len(sequences)} sequences from 32×32 data")

            # ---- Case 2: [time, 32, 64] complex → physically reasonable downsampling to 32×32 ----
            elif h_matrix.ndim == 3 and h_matrix.shape[-2:] == (32, 64) and \
                    h_matrix.dtype in (np.complex64, np.complex128):
                print(f"Processing 32×64 complex data with method: {self.downsample_method}")

                # Use physically reasonable downsampling method
                h_downsampled = self._downsample_64_to_32(h_matrix)  # [time, 32, 32]

                # Verify energy preservation (optional debug info)
                orig_power = np.mean(np.abs(h_matrix) ** 2)
                down_power = np.mean(np.abs(h_downsampled) ** 2)
                power_ratio_db = 10 * np.log10(down_power / orig_power) if orig_power > 0 else 0
                print(f"Power change: {power_ratio_db:.2f} dB (target: ~0 dB)")

                # Convert to real/imaginary separated format
                real = np.real(h_downsampled)
                imag = np.imag(h_downsampled)
                h = np.stack([real, imag], axis=1)  # [time, 2, 32, 32]
                total_t = h.shape[0]

                for s in range(0, total_t - T + 1, stride):
                    sequences.append(h[s:s + T].astype(np.float32))

                print(f"Extracted {len(sequences)} sequences from 32×64→32×32 data")

            # ---- Case 3: [batch, time, 2, 32, 32] ----
            elif h_matrix.ndim == 5 and h_matrix.shape[-3] == 2 and h_matrix.shape[-2:] == (32, 32):
                print(f"Processing pre-formatted [B, T, 2, 32, 32] data")
                B, total_t = h_matrix.shape[0], h_matrix.shape[1]
                for b in range(B):
                    for s in range(0, total_t - T + 1, stride):
                        sequences.append(h_matrix[b, s:s + T].astype(np.float32))

                print(f"Extracted {len(sequences)} sequences from pre-formatted data")

            # ---- Case 4: [batch, time, 32, 32] real ----
            elif h_matrix.ndim == 4 and h_matrix.shape[-2:] == (32, 32):
                print(f"Processing real-valued [B, T, 32, 32] data")
                if h_matrix.shape[1] >= T:  # [batch, time, 32, 32]
                    B, total_t = h_matrix.shape[0], h_matrix.shape[1]
                    # Assume real part = data, imaginary part = 0
                    h_matrix = np.stack([h_matrix, np.zeros_like(h_matrix)], axis=2)  # [B, T, 2, 32, 32]
                    for b in range(B):
                        for s in range(0, total_t - T + 1, stride):
                            sequences.append(h_matrix[b, s:s + T].astype(np.float32))
                elif h_matrix.shape[0] >= T:  # [time, batch, 32, 32]
                    total_t, B = h_matrix.shape[0], h_matrix.shape[1]
                    h_matrix = np.stack([h_matrix, np.zeros_like(h_matrix)], axis=2)  # [T, B, 2, 32, 32]
                    for b in range(B):
                        for s in range(0, total_t - T + 1, stride):
                            sequences.append(h_matrix[s:s + T, b].astype(np.float32))

                print(f"Extracted {len(sequences)} sequences from real-valued data")

            else:
                print(f"⚠️ Unsupported shape {h_matrix.shape} for {file_path}")
                logging.warning(f"Unsupported shape {h_matrix.shape} in {file_path}")

        except Exception as e:
            logging.exception(f"Failed to process {file_path}: {e}")

        return sequences

    def _validate_energy_preservation(self, original, downsampled, method_name):
        """Validate energy preservation (for debugging)"""
        orig_power = np.mean(np.abs(original) ** 2)
        down_power = np.mean(np.abs(downsampled) ** 2)

        if orig_power > 1e-12:
            power_ratio = down_power / orig_power
            power_ratio_db = 10 * np.log10(power_ratio)

            print(f"Energy validation ({method_name}):")
            print(f"  Original power: {orig_power:.6f}")
            print(f"  Downsampled power: {down_power:.6f}")
            print(f"  Power ratio: {power_ratio:.4f} ({power_ratio_db:.2f} dB)")

            # Check if within reasonable range
            if abs(power_ratio_db) < 1.0:  # Within ±1dB
                print(f"  ✅ Energy well preserved")
            elif abs(power_ratio_db) < 3.0:  # Within ±3dB
                print(f"  ⚠️ Energy moderately preserved")
            else:
                print(f"  ❌ Significant energy change detected")

    def _load_norm_params(self):
        """Load normalization parameters"""
        try:
            import pandas as pd
            df = pd.read_csv(self.minmax_file)
            self.norm_params = {
                'min_val': df['min'].values[0],
                'max_val': df['max'].values[0]
            }
            logging.info(
                f"Loaded normalization params: min={self.norm_params['min_val']}, max={self.norm_params['max_val']}")
        except Exception as e:
            logging.warning(f"Failed to load normalization params: {e}")
            self.norm_params = None

    def _split_dataset(self, train_split):
        """Split dataset by proportion"""
        total_samples = len(self.data)
        train_size = int(total_samples * train_split)

        if self.split == 'train':
            self.data = self.data[:train_size]
        else:  # val
            self.data = self.data[train_size:]

        logging.info(f"Split dataset: {len(self.data)} samples for {self.split}")

    def _normalize_data(self, data):
        """Normalize data - supports multiple normalization methods"""
        if self.normalize == 'none' or self.normalize is False:
            return data
        elif self.normalize == 'per_batch':
            # 🔧 New: per-batch dynamic normalization
            normalized_data = np.zeros_like(data)
            for i in range(data.shape[0]):  # Normalize each sequence separately
                seq_power = np.mean(data[i] ** 2)
                if seq_power > 1e-12:
                    normalized_data[i] = data[i] / np.sqrt(seq_power)
                else:
                    normalized_data[i] = data[i]
            return normalized_data
        elif self.normalize == 'minmax':
            # Min-Max normalization
            if self.norm_params:
                min_val, max_val = self.norm_params['min_val'], self.norm_params['max_val']
                data = (data - min_val) / (max_val - min_val)
            else:
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        else:  # 'standardize' or True
            # Standard normalization (global power)
            power = np.mean(data ** 2)
            if power > 1e-12:
                data = data / np.sqrt(power)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get sequence data
        sequence = self.data[idx].copy()  # [T, 2, 32, 32]

        # Ensure data type
        if sequence.dtype != np.float32:
            sequence = sequence.astype(np.float32)

        # Normalization
        sequence = self._normalize_data(sequence)

        # Convert to tensor
        sequence = torch.from_numpy(sequence)

        return sequence


def create_dataloaders(cfg):
    """
    Create training and validation data loaders - supports stride configuration
    """
    # Build data path
    indoor_path = Path(cfg.data.root) / cfg.data.indoor_path

    # Check if path exists
    if not indoor_path.exists():
        raise RuntimeError(f"Data path not found: {indoor_path}")

    logging.info(f"Using data from {indoor_path}")

    # Prepare dataset configuration
    dataset_spec = cfg.data.get('dataset_spec', None)
    minmax_file = cfg.data.get('minmax_file', None)
    normalize_method = cfg.data.get('normalize', 'standardize')
    stride = cfg.data.get('stride', cfg.model.sequence.T)  # 🔧 Fix: support stride configuration
    downsample_method = cfg.data.get('downsample_method', 'smart_avg')  # 🔧 New: downsampling method configuration

    # Create training dataset
    try:
        train_dataset = COST2100Dataset(
            data_path=indoor_path,
            sequence_length=cfg.model.sequence.T,
            normalize=normalize_method,
            split='train',
            dataset_spec=dataset_spec,
            minmax_file=minmax_file,
            stride=stride,  # 🔧 Pass stride
            downsample_method=downsample_method  # 🔧 Pass downsampling method
        )

        val_dataset = COST2100Dataset(
            data_path=indoor_path,
            sequence_length=cfg.model.sequence.T,
            normalize=normalize_method,
            split='val',
            dataset_spec=dataset_spec,
            minmax_file=minmax_file,
            stride=stride,  # 🔧 Pass stride
            downsample_method=downsample_method  # 🔧 Pass downsampling method
        )
    except Exception as e:
        logging.error(f"Failed to create dataset: {e}")
        raise

    # If no dataset_spec, use traditional random_split
    if not dataset_spec or 'train_split' not in dataset_spec:
        full_dataset = COST2100Dataset(
            data_path=indoor_path,
            sequence_length=cfg.model.sequence.T,
            normalize=normalize_method,
            split='full',
            dataset_spec=None,
            minmax_file=minmax_file,
            stride=stride,
            downsample_method=downsample_method
        )

        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed)
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.get('pin_memory', False),
        drop_last=cfg.training.get('drop_last', False)  # 🔧 Fix: allow partial batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.get('pin_memory', False),
        drop_last=False
    )

    logging.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def test_dataset(data_path):
    """Test dataset loading"""
    print(f"Testing dataset loading from {data_path}")

    try:
        dataset = COST2100Dataset(data_path, sequence_length=10, stride=5)
        print(f"Successfully loaded {len(dataset)} sequences")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample shape: {sample.shape}")
            print(f"Sample range: [{sample.min():.4f}, {sample.max():.4f}]")
            print(f"Sample mean: {sample.mean():.4f}, std: {sample.std():.4f}")

        return True
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return False


if __name__ == '__main__':
    # Test data loading
    indoor_path = "F:/CRLSTMNet/data/cost2100/indoor_20slots"

    print("Testing enhanced data loading...")
    test_dataset(indoor_path)