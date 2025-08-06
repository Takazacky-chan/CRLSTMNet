# train_stages.py - Three-stage training strategy (supports intelligent file naming)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm.auto import tqdm
import json
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf


class ThreeStageTrainer:
    """
    Implement three-stage training strategy for CR-LSTM-Net - supports intelligent file naming
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

        # Ensure model is completely on the correct device
        self.model.to(self.device)

        # Ensure all submodules are on the correct device
        for module in self.model.modules():
            module.to(self.device)

        # Set random seeds
        if hasattr(cfg, 'seed'):
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cfg.seed)

        # Create directory structure with naming
        self.base_save_dir = Path(cfg.logging.save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)

        # Generate model filename prefix
        self.model_prefix = self._generate_model_prefix()

        # Create latent cache directory
        self.cache_dir = self.base_save_dir / "latent_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Print compression rate information (confirm configuration is effective)
        self._print_compression_info()

    def _generate_model_prefix(self):
        """Generate model filename prefix"""
        naming_cfg = self.cfg.get('model_naming', {})
        dataset_type = naming_cfg.get('dataset_type', self.cfg.data.get('split', 'unknown'))
        cr_str = naming_cfg.get('compression_ratio', self._parse_cr_from_config())

        prefix = f"{dataset_type}_cr{cr_str}"

        if naming_cfg.get('include_timestamp', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            prefix += f"_{timestamp}"

        return prefix

    def _parse_cr_from_config(self):
        """Parse compression ratio string from configuration"""
        cr_value = self.cfg.data.get('cr', '1/32')
        if isinstance(cr_value, str):
            if '/' in cr_value:
                # "1/32" -> "1_32"
                return cr_value.replace('/', '_')
            else:
                return str(cr_value)
        else:
            # Numeric type, convert to string
            if cr_value < 1:
                # 0.03125 -> "1_32" (approximate)
                approx_denom = int(1 / cr_value)
                return f"1_{approx_denom}"
            else:
                return str(int(cr_value))

    def _print_compression_info(self):
        """Print compression rate information to confirm configuration is effective"""
        cr = self.cfg.data.get('cr', 'unknown')
        cr_rate = self.cfg.data.get('cr_rate', 'unknown')
        cr_num = self.cfg.data.get('cr_num', 'unknown')

        print(f"📊 Compression Settings:")
        print(f"  CR = {cr}")
        print(f"  CR Rate = {cr_rate}")
        print(f"  M (measurements) = {cr_num}")
        print(f"  N (original) = 1024 (32×32)")
        print(f"  Model prefix: {self.model_prefix}")
        print()

    def _get_model_path(self, stage, suffix="best"):
        """Generate model save path"""
        filename = f"{self.model_prefix}_{stage}_{suffix}.pth"
        return self.base_save_dir / filename

    def _compute_loss(self, y_hat, y_true, stage='full'):
        """Unified loss function computation"""
        from losses.metrics import nmse_loss, cosine_similarity_loss, temporal_smooth_loss

        loss_nmse = nmse_loss(y_hat, y_true)

        if stage == 'stage0':
            return loss_nmse
        else:
            loss_cos = cosine_similarity_loss(y_hat, y_true)
            loss_smooth = temporal_smooth_loss(y_hat)

            total_loss = (self.cfg.loss.nmse * loss_nmse +
                          self.cfg.loss.cos * loss_cos +
                          self.cfg.loss.tsmooth * loss_smooth)

            return total_loss

    def stage0_spatial_pretraining(self, train_loader, val_loader):
        """Stage 0: CRNet spatial pre-training (single frame)"""
        print("=== Stage 0: Spatial Pre-training ===")

        # Step 1: Train Low-CR branch
        print("Training Low-CR branch...")
        low_cr_model = SingleFrameCRNet(self.cfg, cr_type='low').to(self.device)
        best_low_loss = self._train_single_branch(low_cr_model, train_loader, val_loader, 'low')

        # Step 2: Train High-CR branch
        print("Training High-CR branch...")
        high_cr_model = SingleFrameCRNet(self.cfg, cr_type='high').to(self.device)
        best_high_loss = self._train_single_branch(high_cr_model, train_loader, val_loader, 'high')

        # Step 3: Load weights to main model
        self._load_pretrained_weights_to_main_model()
        print("Stage 0 completed: Both High/Low-CR weights loaded.")

        return {'low_loss': best_low_loss, 'high_loss': best_high_loss}

    def _train_single_branch(self, model, train_loader, val_loader, branch_type):
        """Train single branch"""
        epochs = self.cfg.stage0.epochs
        lr = self.cfg.stage0.lr
        validate_every = self.cfg.stage0.get('validate_every', 10)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Improved learning rate scheduling
        if self.cfg.stage0.get('cosine', False):
            warmup_epochs = self.cfg.stage0.get('warmup_epochs', 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
        else:
            scheduler = None
            warmup_scheduler = None

        best_loss = float('inf')
        early_stop_config = self.cfg.stage0.get('early_stop', {})
        patience = early_stop_config.get('patience', 15)
        min_delta = early_stop_config.get('min_delta', 0.001)
        patience_counter = 0

        use_amp = self.cfg.training.get('amp', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"[Stage0-{branch_type.upper()}] Epoch {epoch}")

            for batch_idx, H in enumerate(pbar):
                H = H.to(self.device, non_blocking=True)
                t = np.random.randint(0, H.size(1))
                H_single = H[:, t]

                optimizer.zero_grad()

                if use_amp and scaler:
                    with torch.cuda.amp.autocast():
                        H_hat = model(H_single)
                        loss = self._compute_loss(H_hat, H_single, stage='stage0')

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    H_hat = model(H_single)
                    loss = self._compute_loss(H_hat, H_single, stage='stage0')

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.grad_clip)
                    optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            # Learning rate scheduling
            if epoch < warmup_epochs and warmup_scheduler:
                warmup_scheduler.step()
            elif scheduler:
                scheduler.step()

            # Validation
            if epoch % validate_every == 0 or epoch == epochs - 1:
                val_loss = self._validate_single_frame(model, val_loader)
                avg_train_loss = train_loss / len(train_loader)

                print(f"{branch_type.upper()}-CR Epoch {epoch}: "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    patience_counter = 0

                    # Save using new naming convention
                    model_path = self._get_model_path(f"stage0_{branch_type}", "best")
                    torch.save(model.state_dict(), model_path)
                    print(f"New best {branch_type}-CR model saved to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered for {branch_type}-CR at epoch {epoch}")
                        break

        return best_loss

    def _precompute_latents(self, train_loader, val_loader):
        """Precompute and cache latent features"""
        train_cache_file = self.cache_dir / f"train_latents_{self.model_prefix}.pt"
        val_cache_file = self.cache_dir / f"val_latents_{self.model_prefix}.pt"

        # If cache exists, load directly
        if train_cache_file.exists() and val_cache_file.exists():
            print("Loading cached latents...")
            train_data = torch.load(train_cache_file, map_location='cpu')
            val_data = torch.load(val_cache_file, map_location='cpu')
            return train_data, val_data

        print("Precomputing latents for Stage 1...")

        def compute_latents(loader, name):
            latents = []
            targets = []

            self.model.eval()
            with torch.no_grad():
                pbar = tqdm(loader, desc=f"Computing {name} latents")
                for H in pbar:
                    H = H.to(self.device, non_blocking=True)
                    B, T = H.shape[:2]

                    z_raw_list = []
                    for t in range(T):
                        H_frame = H[:, t].float()
                        if t == 0:
                            z = self.model.enc_hi(H_frame)
                            z = self.model.proj_hi2lo(z)
                        else:
                            z = self.model.enc_lo(H_frame)
                        z_raw_list.append(z.unsqueeze(1))

                    z_raw_seq = torch.cat(z_raw_list, dim=1)
                    latents.append(z_raw_seq.cpu())
                    targets.append(H.cpu())

            return {'latents': torch.cat(latents, dim=0), 'targets': torch.cat(targets, dim=0)}

        train_data = compute_latents(train_loader, "train")
        val_data = compute_latents(val_loader, "val")

        # Save cache
        torch.save(train_data, train_cache_file)
        torch.save(val_data, val_cache_file)
        print(f"Latents cached to {self.cache_dir}")

        return train_data, val_data

    def stage1_temporal_training(self, train_loader, val_loader):
        """Stage 1: Temporal training"""
        print("=== Stage 1: Temporal Training (Optimized) ===")

        self.model.to(self.device)
        self._freeze_spatial_modules()

        # Precompute latent features
        train_data, val_data = self._precompute_latents(train_loader, val_loader)

        # Create DataLoader for latent data
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(train_data['latents'], train_data['targets'])
        val_dataset = TensorDataset(val_data['latents'], val_data['targets'])

        train_latent_loader = DataLoader(
            train_dataset, batch_size=self.cfg.training.batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        val_latent_loader = DataLoader(
            val_dataset, batch_size=self.cfg.training.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )

        # Only optimize temporal-related parameters
        temporal_params = list(self.model.temporal.parameters()) + \
                          list(self.model.fusion_gate.parameters())

        epochs = self.cfg.stage1.epochs
        lr = self.cfg.stage1.lr
        validate_every = self.cfg.stage1.get('validate_every', 5)

        optimizer = optim.Adam(temporal_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, verbose=True
        )

        best_loss = float('inf')
        early_stop_config = self.cfg.stage1.get('early_stop', {})
        patience = early_stop_config.get('patience', 10)
        min_delta = early_stop_config.get('min_delta', 0.001)
        patience_counter = 0

        use_amp = self.cfg.training.get('amp', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_latent_loader, desc=f"[Stage1] Epoch {epoch}")

            for z_raw_seq, H in pbar:
                z_raw_seq = z_raw_seq.to(self.device, non_blocking=True)
                H = H.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                if use_amp and scaler:
                    with torch.cuda.amp.autocast():
                        z_temporal = self.model.temporal(z_raw_seq)

                        outputs = []
                        for t in range(H.size(1)):
                            z_raw = z_raw_seq[:, t]
                            z_temp = z_temporal[:, t]

                            fusion_input = torch.cat([z_raw, z_temp], dim=1)
                            alpha = self.model.fusion_gate(fusion_input)
                            z_fused = alpha * z_raw + (1 - alpha) * z_temp

                            H_hat_t = self.model.decoder(z_fused)
                            outputs.append(H_hat_t.unsqueeze(1))

                        H_hat = torch.cat(outputs, dim=1)

                        recon_loss = self._compute_loss(H_hat, H)
                        latent_consistency = torch.nn.functional.mse_loss(z_temporal, z_raw_seq.detach())
                        loss = recon_loss + 0.15 * latent_consistency

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(temporal_params, self.cfg.training.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z_temporal = self.model.temporal(z_raw_seq)

                    outputs = []
                    for t in range(H.size(1)):
                        z_raw = z_raw_seq[:, t]
                        z_temp = z_temporal[:, t]

                        fusion_input = torch.cat([z_raw, z_temp], dim=1)
                        alpha = self.model.fusion_gate(fusion_input)
                        z_fused = alpha * z_raw + (1 - alpha) * z_temp

                        H_hat_t = self.model.decoder(z_fused)
                        outputs.append(H_hat_t.unsqueeze(1))

                    H_hat = torch.cat(outputs, dim=1)

                    recon_loss = self._compute_loss(H_hat, H)
                    latent_consistency = torch.nn.functional.mse_loss(z_temporal, z_raw_seq.detach())
                    loss = recon_loss + 0.15 * latent_consistency

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(temporal_params, self.cfg.training.grad_clip)
                    optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{recon_loss.item():.4f}",
                    'lat_cons': f"{latent_consistency.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            # Validation
            if epoch % validate_every == 0 or epoch == epochs - 1:
                val_loss = self._validate_latent_model(val_latent_loader)
                avg_train_loss = train_loss / len(train_latent_loader)

                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

                scheduler.step(val_loss)

                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    patience_counter = 0

                    model_path = self._get_model_path("stage1", "best")
                    torch.save(self.model.state_dict(), model_path)
                    print(f"New best model saved to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        print("Stage 1 completed: Temporal modules trained.")
        return best_loss

    def stage2_end_to_end_finetuning(self, train_loader, val_loader):
        """Stage 2: End-to-end fine-tuning"""
        print("=== Stage 2: End-to-End Fine-tuning ===")

        self._unfreeze_all_modules()

        epochs = self.cfg.stage2.epochs
        lr = self.cfg.stage2.lr
        validate_every = self.cfg.stage2.get('validate_every', 10)

        optimizer = optim.AdamW(self.model.parameters(),
                                lr=lr,
                                weight_decay=self.cfg.training.get('weight_decay', 1e-4))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        best_loss = float('inf')
        early_stop_config = self.cfg.stage2.get('early_stop', {})
        patience = early_stop_config.get('patience', 15)
        min_delta = early_stop_config.get('min_delta', 0.001)
        patience_counter = 0

        use_amp = self.cfg.training.get('amp', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"[Stage2] Epoch {epoch}")

            for H in pbar:
                H = H.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                if use_amp and scaler:
                    with torch.cuda.amp.autocast():
                        H_hat = self.model(H)
                        loss = self._compute_loss(H_hat, H)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    H_hat = self.model(H)
                    loss = self._compute_loss(H_hat, H)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                    optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            scheduler.step()

            # Validation and saving
            if epoch % validate_every == 0 or epoch == epochs - 1:
                val_loss = self._validate_full_model(val_loader)
                avg_train_loss = train_loss / len(train_loader)

                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    patience_counter = 0

                    model_path = self._get_model_path("stage2", "best")
                    torch.save(self.model.state_dict(), model_path)
                    print(f"New best model saved to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        print("Stage 2 completed: End-to-end training finished.")
        return best_loss

    def _freeze_spatial_modules(self):
        """Freeze spatial processing modules"""
        for param in self.model.enc_hi.parameters():
            param.requires_grad = False
        for param in self.model.enc_lo.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False

    def _unfreeze_all_modules(self):
        """Unfreeze all modules"""
        for param in self.model.parameters():
            param.requires_grad = True

    def _validate_single_frame(self, model, val_loader):
        """Single frame model validation"""
        model.eval()
        total_loss = 0.0
        all_metrics = []
        num_batches = 0

        with torch.no_grad():
            for H in val_loader:
                H = H.to(self.device, non_blocking=True)
                t = np.random.randint(0, H.size(1))
                H_single = H[:, t]

                H_hat = model(H_single)

                from losses.metrics import nmse_loss, compute_metrics
                loss = nmse_loss(H_hat, H_single)
                metrics = compute_metrics(H_hat, H_single)
                all_metrics.append(metrics)

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= 20:
                    break

        avg_loss = total_loss / num_batches
        avg_nmse_db = np.mean([m['nmse_db'] for m in all_metrics])
        avg_rho = np.mean([m['rho'] for m in all_metrics])

        print(f"Val Loss: {avg_loss:.6f} | NMSE(dB): {avg_nmse_db:.2f} | ρ: {avg_rho:.4f}")
        return avg_loss

    def _validate_latent_model(self, val_latent_loader):
        """Validate latent model"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        num_batches = 0

        with torch.no_grad():
            for z_raw_seq, H in val_latent_loader:
                z_raw_seq = z_raw_seq.to(self.device, non_blocking=True)
                H = H.to(self.device, non_blocking=True)

                z_temporal = self.model.temporal(z_raw_seq)

                outputs = []
                for t in range(H.size(1)):
                    z_raw = z_raw_seq[:, t]
                    z_temp = z_temporal[:, t]

                    fusion_input = torch.cat([z_raw, z_temp], dim=1)
                    alpha = self.model.fusion_gate(fusion_input)
                    z_fused = alpha * z_raw + (1 - alpha) * z_temp

                    H_hat_t = self.model.decoder(z_fused)
                    outputs.append(H_hat_t.unsqueeze(1))

                H_hat = torch.cat(outputs, dim=1)
                loss = self._compute_loss(H_hat, H)

                from losses.metrics import compute_metrics
                metrics = compute_metrics(H_hat, H)
                all_metrics.append(metrics)

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= 50:
                    break

        avg_loss = total_loss / num_batches
        avg_nmse_db = np.mean([m['nmse_db'] for m in all_metrics])
        avg_rho = np.mean([m['rho'] for m in all_metrics])

        print(f"Val Loss: {avg_loss:.6f} | NMSE(dB): {avg_nmse_db:.2f} | ρ: {avg_rho:.4f}")
        return avg_loss

    def _validate_full_model(self, val_loader):
        """Full model validation"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        num_batches = 0

        with torch.no_grad():
            for H in val_loader:
                H = H.to(self.device, non_blocking=True)
                H_hat = self.model(H)
                loss = self._compute_loss(H_hat, H)

                from losses.metrics import compute_metrics
                metrics = compute_metrics(H_hat, H)
                all_metrics.append(metrics)

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= 50:
                    break

        avg_loss = total_loss / num_batches
        avg_nmse_db = np.mean([m['nmse_db'] for m in all_metrics])
        avg_rho = np.mean([m['rho'] for m in all_metrics])

        print(f"Val Loss: {avg_loss:.6f} | NMSE(dB): {avg_nmse_db:.2f} | ρ: {avg_rho:.4f}")
        return avg_loss

    def _load_pretrained_weights_to_main_model(self):
        """Load pretrained weights to main model"""
        print("Loading pretrained weights to main model...")

        # Find weight files using new naming convention
        low_pth = self._get_model_path("stage0_low", "best")
        high_pth = self._get_model_path("stage0_high", "best")

        if not low_pth.exists() or not high_pth.exists():
            print(f"Warning: Stage 0 checkpoints not found")
            print(f"  Looking for: {low_pth}")
            print(f"  Looking for: {high_pth}")
            return

        try:
            self.model.to(self.device)

            low_weights = torch.load(low_pth, map_location=self.device)
            high_weights = torch.load(high_pth, map_location=self.device)

            # Load weights
            enc_lo_weights = {k.replace('encoder.', ''): v for k, v in low_weights.items() if k.startswith('encoder.')}
            if enc_lo_weights:
                self.model.enc_lo.load_state_dict(enc_lo_weights, strict=False)

            decoder_weights = {k.replace('decoder.', ''): v for k, v in low_weights.items() if k.startswith('decoder.')}
            if decoder_weights:
                self.model.decoder.load_state_dict(decoder_weights, strict=False)

            enc_hi_weights = {k.replace('encoder.', ''): v for k, v in high_weights.items() if k.startswith('encoder.')}
            if enc_hi_weights:
                self.model.enc_hi.load_state_dict(enc_hi_weights, strict=False)

            print("Successfully loaded pretrained weights from Stage 0")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")

    def save_final_model(self, additional_info=None):
        """Save final model"""
        final_model_path = self._get_model_path("final", "model")

        # Prepare content to save
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': OmegaConf.to_container(self.cfg, resolve=True) if hasattr(self.cfg, '_content') else dict(
                self.cfg),
            'model_prefix': self.model_prefix,
            'compression_info': {
                'cr': self.cfg.data.get('cr'),
                'cr_rate': self.cfg.data.get('cr_rate'),
                'cr_num': self.cfg.data.get('cr_num'),
                'dataset_type': self.cfg.data.get('split')
            },
            'timestamp': datetime.now().isoformat()
        }

        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, final_model_path)
        print(f"Final model saved to {final_model_path}")
        return final_model_path


class SingleFrameCRNet(nn.Module):
    """Single frame CRNet model for Stage 0"""

    def __init__(self, cfg, cr_type='low'):
        super().__init__()

        if cr_type == 'high':
            latent_dim = cfg.model.spatial.latent_dim_high
            res_blocks = cfg.model.spatial.res_blocks + 2
        else:
            latent_dim = cfg.model.spatial.latent_dim_low
            res_blocks = cfg.model.spatial.res_blocks

        from models.crlstmnet import CRNetEncoder, CRNetDecoder

        self.encoder = CRNetEncoder(
            latent_dim=latent_dim,
            res_blocks=res_blocks,
            channels=cfg.model.spatial.channels,
            negative_slope=cfg.model.spatial.negative_slope
        )
        self.decoder = CRNetDecoder(
            latent_dim=latent_dim,
            channels=cfg.model.spatial.channels,
            negative_slope=cfg.model.spatial.negative_slope,
            output_activation=cfg.model.spatial.get('output_activation', 'linear')
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)