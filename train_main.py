# train_main.py Fixed Version - Correctly integrated complexity analysis
import os
import torch
import torch.multiprocessing as mp
import numpy as np
import argparse
from omegaconf import OmegaConf
import logging
from pathlib import Path
import json
import traceback
from datetime import datetime


def main():
    """Main training function"""
    # Command line arguments
    parser = argparse.ArgumentParser(description='Train CR-LSTM-Net')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')
    parser.add_argument('--allow_dummy', action='store_true',
                        help='Allow using dummy data for testing')
    parser.add_argument('--stage', type=int, choices=[0, 1, 2], default=None,
                        help='Start from specific stage (0=all stages)')
    parser.add_argument('--cr', type=str, default=None,
                        help='Override compression ratio (e.g., "1/64")')
    parser.add_argument('--analyze_complexity', action='store_true',
                        help='Analyze model complexity at startup')
    args = parser.parse_args()

    # Load configuration
    try:
        cfg = OmegaConf.load(args.config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        return

    # Override compression ratio configuration (if specified via command line)
    if args.cr:
        cfg.data.cr = args.cr
        # Automatically calculate corresponding cr_rate and cr_num
        if args.cr == "1/64":
            cfg.data.cr_rate = 0.015625
            cfg.data.cr_num = 16
        elif args.cr == "1/32":
            cfg.data.cr_rate = 0.03125
            cfg.data.cr_num = 32
        elif args.cr == "1/16":
            cfg.data.cr_rate = 0.0625
            cfg.data.cr_num = 64
        print(f"Compression ratio override: {args.cr}")

    # Read complexity analysis option from command line arguments or config file
    analyze_complexity = args.analyze_complexity or cfg.get('analyze_complexity', False)

    # Set random seeds
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

    # Set deterministic behavior
    if cfg.get('deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Ensure output directory exists
    save_dir = cfg.get('logging', {}).get('save_dir', 'checkpoints')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Set log filename
    log_filename = f"training_{cfg.data.get('split', 'unknown')}_cr{cfg.data.get('cr', '').replace('/', '_')}.log"

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{save_dir}/{log_filename}"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting training on {cfg.get('device', 'auto')}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Compression ratio: {cfg.data.get('cr')}")
    logger.info(f"Dataset type: {cfg.data.get('split')}")

    # Display hardware information
    gpu_name = "CPU"
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")

        # Set GPU memory optimization
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_allocated'):
            logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    else:
        logger.info("Running on CPU")

    # Check data path
    use_dummy_data = False
    allow_dummy = args.allow_dummy or cfg.get('data', {}).get('allow_dummy_data', False)
    data_root = cfg.get('data', {}).get('root', '')

    if not os.path.exists(data_root):
        if allow_dummy:
            logger.warning(f"Data path not found: {data_root}")
            logger.info("Using dummy data for testing...")
            use_dummy_data = True
        else:
            logger.error(f"Dataset not found at {data_root}. "
                         f"Use --allow_dummy flag to use dummy data.")
            return

    # Create data loaders
    train_loader = None
    val_loader = None

    if use_dummy_data:
        try:
            from losses.metrics import create_dummy_data
            train_loader, val_loader = create_dummy_data(cfg)
            logger.info("Created dummy data loaders for testing")
        except ImportError as e:
            logger.error(f"Cannot import dummy data creator: {e}")
            return
    else:
        try:
            from datasets.cost2100 import create_dataloaders
            train_loader, val_loader = create_dataloaders(cfg)
            logger.info(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
        except ImportError:
            if allow_dummy:
                logger.warning("datasets.cost2100 not available, falling back to dummy data")
                try:
                    from losses.metrics import create_dummy_data
                    train_loader, val_loader = create_dummy_data(cfg)
                    logger.info("Created dummy data loaders as fallback")
                    use_dummy_data = True
                except ImportError as e:
                    logger.error(f"Cannot import dummy data creator: {e}")
                    return
            else:
                logger.error("Real dataset module not available and dummy data not allowed")
                return

    if train_loader is None or val_loader is None:
        logger.error("Failed to create data loaders")
        return

    # Import and initialize trainer
    try:
        from models.crlstmnet import CRLSTMNet
        from train_stages import ThreeStageTrainer
    except ImportError as e:
        logger.error(f"Cannot import required modules: {e}")
        return

    # Create model
    try:
        model = CRLSTMNet(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")

        # 🔧 Fix: Perform complexity analysis here
        complexity_info = None
        if analyze_complexity:
            complexity_info = analyze_model_complexity_safe(model, cfg, logger, save_dir)

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return

    # Create trainer
    try:
        trainer = ThreeStageTrainer(model, cfg)
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return

    # Move summary_results initialization to correct position
    summary_results = {
        'config_path': args.config,
        'compression_ratio': cfg.data.get('cr'),
        'dataset_type': cfg.data.get('split'),
        'use_dummy_data': use_dummy_data,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gpu_info': {
            'name': gpu_name,
            'memory_gb': gpu_memory
        },
        'start_time': datetime.now().isoformat(),
        'analyze_complexity': analyze_complexity
    }

    # 🔧 Fix: Add complexity information to summary_results
    if complexity_info:
        summary_results['model_complexity'] = complexity_info

    # Execute training
    try:
        # Decide which stage to start from based on args.stage
        if args.stage is None or args.stage == 0:
            logger.info("=== Starting Stage 0: Spatial Pretraining ===")
            stage0_result = trainer.stage0_spatial_pretraining(train_loader, val_loader)
            summary_results['stage0'] = {
                'completed': True,
                'low_loss': float(stage0_result['low_loss']),
                'high_loss': float(stage0_result['high_loss'])
            }
        else:
            # If skipping Stage 0, need to load pretrained weights
            logger.info("Skipping Stage 0, loading pretrained weights...")
            trainer._load_pretrained_weights_to_main_model()
            summary_results['stage0'] = {'completed': True, 'loaded_from_checkpoint': True}

        if args.stage is None or args.stage <= 1:
            logger.info("=== Starting Stage 1: Temporal Training ===")
            stage1_loss = trainer.stage1_temporal_training(train_loader, val_loader)
            summary_results['stage1'] = {
                'completed': True,
                'best_loss': float(stage1_loss)
            }

        if args.stage is None or args.stage <= 2:
            logger.info("=== Starting Stage 2: End-to-End Fine-tuning ===")
            stage2_loss = trainer.stage2_end_to_end_finetuning(train_loader, val_loader)
            summary_results['stage2'] = {
                'completed': True,
                'best_loss': float(stage2_loss)
            }

        logger.info("Training completed successfully!")

        # Save final model
        summary_results['end_time'] = datetime.now().isoformat()
        final_model_path = trainer.save_final_model(summary_results)
        logger.info(f"Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        summary_results['interrupted'] = True
        summary_results['end_time'] = datetime.now().isoformat()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        summary_results['error'] = str(e)
        summary_results['end_time'] = datetime.now().isoformat()
        logger.error(traceback.format_exc())
    finally:
        # Save training summary
        summary_filename = f"training_summary_{cfg.data.get('split', 'unknown')}_cr{cfg.data.get('cr', '').replace('/', '_')}.json"
        summary_path = f"{save_dir}/{summary_filename}"

        try:
            # Ensure all values are JSON serializable
            serializable_results = make_json_serializable(summary_results)

            with open(summary_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Training summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")


def analyze_model_complexity_safe(model, cfg, logger, save_dir):
    """Safe model complexity analysis function"""
    try:
        from utils.complexity import analyze_model_complexity

        # Get input shape
        T = cfg.model.sequence.get('T', 32)
        input_shape = (T, 2, 32, 32)

        logger.info("Analyzing model complexity...")
        complexity_results = analyze_model_complexity(
            model,
            input_shape,
            device=cfg.get('device', 'cuda'),
            print_summary=True,
            print_table=False  # Don't print detailed table during training
        )

        # Save complexity analysis to training directory
        complexity_file = Path(
            save_dir) / f"complexity_analysis_{cfg.data.get('split', 'unknown')}_cr{cfg.data.get('cr', '').replace('/', '_')}.json"

        # Ensure results are JSON serializable
        serializable_results = make_json_serializable(complexity_results)

        with open(complexity_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Complexity analysis saved to {complexity_file}")

        # Extract key information for summary
        complexity_summary = {}
        if 'parameters' in complexity_results:
            complexity_summary['total_params_M'] = complexity_results['parameters']['total_params_M']
            complexity_summary['trainable_params_M'] = complexity_results['parameters']['trainable_params_M']

        if 'flops_main' in complexity_results and 'error' not in complexity_results['flops_main']:
            complexity_summary['total_flops_G'] = complexity_results['flops_main']['total_flops_G']
            complexity_summary['flops_method'] = complexity_results['flops_main']['method']

        if 'efficiency' in complexity_results:
            complexity_summary['flops_per_param'] = complexity_results['efficiency']['flops_per_parameter']

        if 'memory' in complexity_results and 'error' not in complexity_results['memory']:
            complexity_summary['peak_memory_MB'] = complexity_results['memory']['peak_memory_allocated_MB']

        return complexity_summary

    except ImportError as e:
        logger.warning(f"⚠Cannot perform complexity analysis: {e}")
        logger.info("Install required packages: pip install fvcore thop ptflops")
        return None
    except Exception as e:
        logger.error(f"Complexity analysis failed: {e}")
        return None


def make_json_serializable(obj):
    """Convert object to JSON serializable format"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        if hasattr(obj, 'numel') and obj.numel() < 100:
            return obj.tolist()
        else:
            return f"Tensor/Array of shape {obj.shape}"
    elif isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, 'item'):  # Handle single element tensors
        return obj.item()
    else:
        return str(obj)


if __name__ == "__main__":
    # Windows multiprocessing compatibility
    mp.set_start_method('spawn', force=True)

    # Set CUDA optimizations
    if torch.cuda.is_available():
        # Basic optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # GTX 1650 Ti specific optimizations
        torch.set_float32_matmul_precision("medium")

        # Memory management
        torch.cuda.empty_cache()

        # Set memory allocator
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Start main function
    main()