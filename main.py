"""
Dynamic3DGS - Main Entry Point

This is the main script for training, evaluating, and running Dynamic3DGS.
It provides command-line interface for all major operations.
"""

import os
import sys
import argparse
import yaml
import torch
from typing import Dict, Optional

from training.trainer import Dynamic3DGSTrainer, TrainerConfig
from evaluation.metrics import ComprehensiveEvaluator


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(args):
    """Setup experiment based on arguments."""
    # Load base configuration
    if args.config:
        base_config = load_config(args.config)
    else:
        base_config = {}

    # Override with command line arguments
    if args.experiment_name:
        base_config['experiment'] = base_config.get('experiment', {})
        base_config['experiment']['name'] = args.experiment_name

    if args.data_path:
        base_config['data'] = base_config.get('data', {})
        base_config['data']['path'] = args.data_path

    if args.checkpoint_dir:
        base_config['training'] = base_config.get('training', {})
        base_config['training']['checkpoint_dir'] = args.checkpoint_dir

    if args.log_dir:
        base_config['logging'] = base_config.get('logging', {})
        base_config['logging']['log_dir'] = args.log_dir

    return base_config


def train_model(args):
    """Train the Dynamic3DGS model."""
    print("🚀 Starting Dynamic3DGS Training")
    print("=" * 50)

    # Setup experiment
    config_dict = setup_experiment(args)

    # Extract configurations
    trainer_config = TrainerConfig(
        experiment_name=config_dict.get('experiment', {}).get('name', 'dynamic_3dgs'),
        device=args.device,
        mixed_precision=True,
        max_epochs=args.max_epochs or 1000,
        batch_size=args.batch_size or 4,
        learning_rate=args.learning_rate or 0.001
    )

    model_config = config_dict.get('model', {})
    dataset_config = config_dict.get('dataset', {})

    # Create trainer
    trainer = Dynamic3DGSTrainer(
        config=trainer_config,
        model_config=model_config,
        dataset_config=dataset_config,
        experiment_dir=args.checkpoint_dir or "./experiments"
    )

    # Resume from checkpoint if specified
    if args.resume:
        try:
            trainer._load_checkpoint(args.resume)
            print(f"✅ Resumed training from checkpoint: {args.resume}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("Starting training from scratch...")

    # Debug mode
    if args.debug_mode:
        print("🐛 Running in debug mode...")
        trainer.config.max_epochs = 2
        trainer.config.batch_size = 2
        trainer.config.log_interval = 1

    # Start training
    trainer.train()

    print("\n🎉 Training completed!")
    print(f"Checkpoints saved to: {trainer.exp_dir}")


def evaluate_model(args):
    """Evaluate the trained model."""
    print("📊 Starting Dynamic3DGS Evaluation")
    print("=" * 50)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Setup experiment directory
    exp_dir = args.experiment_dir or "./experiments"
    os.makedirs(exp_dir, exist_ok=True)

    # Load dataset configuration
    if args.dataset_config:
        dataset_config = load_config(args.dataset_config)
    else:
        # Default dataset config
        dataset_config = {
            'name': args.dataset,
            'path': args.data_path
        }

    # Create evaluator
    evaluator = ComprehensiveEvaluator()

    # Load model (simplified - would need full model loading in practice)
    print("⚠️  Model loading from checkpoint not implemented yet")
    print("Please implement proper model loading in the future.")

    # Run evaluation
    results = evaluator.evaluate(
        model=None,  # Would be loaded model
        renderer=None,  # Would be loaded renderer
        dataloader=None,  # Would be created dataloader
        device=torch.device(args.device),
        num_batches=args.num_batches
    )

    # Save results
    output_file = os.path.join(args.output_dir, "metrics.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📈 Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    print(f"\n💾 Results saved to: {output_file}")


def visualize_results(args):
    """Visualize training/evaluation results."""
    print("🖼️  Starting Dynamic3DGS Visualization")
    print("=" * 50)

    # This would use the visualization tools
    print("Visualization functionality coming soon!")
    print("Implement visualization scripts based on your needs.")


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Dynamic3DGS - Deformable 3D Gaussian Splatting for Dynamic Scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python main.py --mode train

  # Train with custom configuration
  python main.py --mode train --config configs/experiments/final_model.yaml

  # Evaluate a trained model
  python main.py --mode evaluate --checkpoint ./checkpoints/latest.pth

  # Resume training from checkpoint
  python main.py --mode train --resume ./checkpoints/checkpoint.pth

  # Debug mode (fast training)
  python main.py --mode train --debug-mode
        """
    )

    # Main mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str,
                             help='Path to training configuration file')
    train_parser.add_argument('--data-path', type=str, default='./data/nvidia_dynamic',
                             help='Path to dataset')
    train_parser.add_argument('--experiment-name', type=str,
                             help='Name for this experiment')
    train_parser.add_argument('--checkpoint-dir', type=str,
                             help='Directory to save checkpoints')
    train_parser.add_argument('--log-dir', type=str,
                             help='Directory to save logs')
    train_parser.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'cpu'],
                             help='Device to run on')
    train_parser.add_argument('--batch-size', type=int, default=4,
                             help='Training batch size')
    train_parser.add_argument('--max-epochs', type=int,
                             help='Maximum number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--resume', type=str,
                             help='Resume training from checkpoint')
    train_parser.add_argument('--debug-mode', action='store_true',
                             help='Run in debug mode with reduced settings')

    # Evaluate mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--dataset', type=str, default='nvidia_dynamic',
                            choices=['nvidia_dynamic', 'hypernerf', 'custom'],
                            help='Dataset name')
    eval_parser.add_argument('--data-path', type=str,
                            help='Path to dataset')
    eval_parser.add_argument('--dataset-config', type=str,
                            help='Path to dataset configuration file')
    eval_parser.add_argument('--output-dir', type=str, default='./results',
                            help='Output directory for results')
    eval_parser.add_argument('--experiment-dir', type=str, default='./experiments',
                            help='Experiment directory')
    eval_parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'],
                            help='Device to run on')
    eval_parser.add_argument('--num-batches', type=int, default=20,
                            help='Number of batches to evaluate')

    # Visualize mode
    vis_parser = subparsers.add_parser('visualize', help='Visualize results')
    vis_parser.add_argument('--input-dir', type=str,
                             help='Input directory containing results')
    vis_parser.add_argument('--output-dir', type=str,
                             help='Output directory for visualizations')
    vis_parser.add_argument('--format', type=str, default='png',
                             choices=['png', 'mp4', 'gif'],
                             help='Output format')

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    try:
        if args.mode == 'train':
            train_model(args)
        elif args.mode == 'evaluate':
            evaluate_model(args)
        elif args.mode == 'visualize':
            visualize_results(args)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()