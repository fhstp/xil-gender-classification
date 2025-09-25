"""
Comprehensive experimental pipeline for XIL gender classification research.

This script runs all 28 experiments described in the paper:
- 6 baseline models (no XIL)
- 6 CAIPI experiments (uncertain/confident x k={1,3,5})
- 4 RRR experiments (uncertain/confident x GradCAM/BLA) 
- 12 Hybrid experiments (uncertain/confident x k={1,3,5} x GradCAM/BLA)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
from datetime import datetime
import argparse

from src.data.dataset import GenderDataset
from src.models.cnn_models import create_model
from src.explainability.gradcam import GradCAMWrapper
from src.explainability.bla import create_bla_model
from src.training.rrr_trainer import RRRTrainer
from src.training.hybrid_trainer import HybridXILTrainer
from src.evaluation.bias_metrics import evaluate_model_bias
from scripts.train_caipi import train_caipi_model


class ComprehensiveExperimentRunner:
    """
    Runs all experiments from the XIL gender classification paper.
    """
    
    def __init__(self, data_dir: str, results_dir: str, device: str = 'cpu'):
        """
        Initialize experiment runner.
        
        Args:
            data_dir: Path to gender dataset
            results_dir: Directory to save all results
            device: Device to run experiments on
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.device = device
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load datasets
        print("Loading datasets...")
        self.train_dataset = GenderDataset(data_dir, split='train')
        self.val_dataset = GenderDataset(data_dir, split='val')
        self.test_dataset = GenderDataset(data_dir, split='test')
        
        print(f"Dataset sizes - Train: {len(self.train_dataset)}, "
              f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        # Model architectures to test
        self.model_architectures = [
            'densenet121', 'efficientnet_b0', 'googlenet', 
            'mobilenet_v2', 'resnet50', 'vgg16'
        ]
        
        # Results storage
        self.all_results = {}
        
    def train_baseline_model(self, model_name: str) -> dict:
        """
        Train baseline model without XIL steering.
        
        Args:
            model_name: Name of model architecture
            
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*60}")
        print(f"Training Baseline Model: {model_name}")
        print(f"{'='*60}")
        
        # Create model
        model = create_model(model_name, num_classes=2, pretrained=True)
        model = model.to(self.device)
        
        # Train baseline model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        print("Training baseline model...")
        for epoch in range(20):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            if (epoch + 1) % 5 == 0:
                accuracy = 100 * correct / total
                print(f"Epoch {epoch+1}: Loss {total_loss/len(self.train_loader):.4f}, "
                      f"Accuracy {accuracy:.2f}%")
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        
        # Evaluate bias metrics with GradCAM
        gradcam = GradCAMWrapper(model, target_layer_name='features')
        bias_metrics = evaluate_model_bias(
            model=model,
            dataloader=self.test_loader,
            explainer=gradcam,
            device=self.device
        )
        
        results = {
            'model': model_name,
            'method': 'baseline',
            'test_accuracy': test_accuracy,
            **bias_metrics
        }
        
        # Save model
        save_path = os.path.join(self.results_dir, f'baseline_{model_name}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'results': results
        }, save_path)
        
        print(f"Baseline {model_name} - Accuracy: {test_accuracy:.2f}%")
        print(f"FFP: {bias_metrics['FFP']:.3f}, BFP: {bias_metrics['BFP']:.3f}, "
              f"BSR: {bias_metrics['BSR']:.3f}, DICE: {bias_metrics['DICE']:.3f}")
        
        return results
    
    def run_caipi_experiments(self) -> list:
        """
        Run all CAIPI experiments (6 total).
        
        Returns:
            List of experiment results
        """
        print(f"\n{'='*60}")
        print("Running CAIPI Experiments")
        print(f"{'='*60}")
        
        results = []
        
        # Use EfficientNet-B0 as main model (best performing baseline)
        configurations = [
            {'sampling_strategy': 'uncertainty', 'k': 1},
            {'sampling_strategy': 'uncertainty', 'k': 3},
            {'sampling_strategy': 'uncertainty', 'k': 5},
            {'sampling_strategy': 'high_confidence', 'k': 1},
            {'sampling_strategy': 'high_confidence', 'k': 3},
            {'sampling_strategy': 'high_confidence', 'k': 5},
        ]
        
        for config in configurations:
            print(f"\nCAIPI: {config['sampling_strategy']} k={config['k']}")
            
            # Create fresh model
            model = create_model('efficientnet_b0', num_classes=2, pretrained=True)
            gradcam = GradCAMWrapper(model, target_layer_name='features')
            
            # Train CAIPI model
            trained_model, final_metrics = train_caipi_model(
                model=model,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                explainer=gradcam,
                k=config['k'],
                num_samples=50,
                sampling_strategy=config['sampling_strategy'],
                num_epochs=20,
                batch_size=32,
                device=self.device,
                save_path=os.path.join(
                    self.results_dir, 
                    f"caipi_{config['sampling_strategy']}_k{config['k']}.pth"
                )
            )
            
            # Test accuracy
            trained_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = trained_model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            test_accuracy = 100 * correct / total
            
            result = {
                'method': 'CAIPI',
                'configuration': f"{config['sampling_strategy']} k={config['k']}",
                'test_accuracy': test_accuracy,
                **final_metrics
            }
            
            results.append(result)
            print(f"CAIPI {config['sampling_strategy']} k={config['k']} - "
                  f"Accuracy: {test_accuracy:.2f}%")
        
        return results
    
    def run_rrr_experiments(self) -> list:
        """
        Run RRR experiments (4 total).
        
        Returns:
            List of experiment results
        """
        print(f"\n{'='*60}")
        print("Running RRR Experiments")
        print(f"{'='*60}")
        
        results = []
        
        # RRR configurations
        configurations = [
            {'sampling_strategy': 'uncertainty', 'explainer': 'gradcam'},
            {'sampling_strategy': 'high_confidence', 'explainer': 'gradcam'},
            {'sampling_strategy': 'uncertainty', 'explainer': 'bla'},
            {'sampling_strategy': 'high_confidence', 'explainer': 'bla'},
        ]
        
        for config in configurations:
            print(f"\nRRR: {config['sampling_strategy']} + {config['explainer']}")
            
            # Create model and explainer
            if config['explainer'] == 'bla':
                model = create_bla_model('efficientnet_b0', num_classes=2)
                explainer = model.get_explanation_wrapper()
            else:
                model = create_model('efficientnet_b0', num_classes=2, pretrained=True)
                explainer = GradCAMWrapper(model, target_layer_name='features')
            
            # Create RRR trainer
            rrr_trainer = RRRTrainer(
                model=model,
                lambda_reg=10.0,
                learning_rate=1e-4,
                device=self.device
            )
            
            # Train with RRR
            history = rrr_trainer.train(
                train_dataloader=self.train_loader,
                val_dataloader=self.val_loader,
                num_epochs=20
            )
            
            # Evaluate
            final_metrics = evaluate_model_bias(
                model=model,
                dataloader=self.test_loader,
                explainer=explainer,
                device=self.device
            )
            
            # Test accuracy
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    if config['explainer'] == 'bla':
                        outputs, _ = model(images)
                    else:
                        outputs = model(images)
                        
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            test_accuracy = 100 * correct / total
            
            result = {
                'method': 'RRR',
                'configuration': f"{config['sampling_strategy']} + {config['explainer']}",
                'test_accuracy': test_accuracy,
                **final_metrics
            }
            
            results.append(result)
            
            # Save model
            save_path = os.path.join(
                self.results_dir, 
                f"rrr_{config['sampling_strategy']}_{config['explainer']}.pth"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'history': history,
                'results': result
            }, save_path)
            
            print(f"RRR {config['sampling_strategy']} + {config['explainer']} - "
                  f"Accuracy: {test_accuracy:.2f}%")
        
        return results
    
    def run_hybrid_experiments(self) -> list:
        """
        Run Hybrid experiments (12 total).
        
        Returns:
            List of experiment results
        """
        print(f"\n{'='*60}")
        print("Running Hybrid Experiments")
        print(f"{'='*60}")
        
        results = []
        
        # Hybrid configurations
        explainers = ['gradcam', 'bla']
        sampling_strategies = ['uncertainty', 'high_confidence']
        k_values = [1, 3, 5]
        
        for explainer_name in explainers:
            for sampling_strategy in sampling_strategies:
                for k in k_values:
                    config_name = f"{sampling_strategy}_k{k}_{explainer_name}"
                    print(f"\nHybrid: {config_name}")
                    
                    # Create model and explainer
                    if explainer_name == 'bla':
                        model = create_bla_model('efficientnet_b0', num_classes=2)
                        explainer = model.get_explanation_wrapper()
                    else:
                        model = create_model('efficientnet_b0', num_classes=2, pretrained=True)
                        explainer = GradCAMWrapper(model, target_layer_name='features')
                    
                    # Create hybrid trainer
                    hybrid_trainer = HybridXILTrainer(
                        model=model,
                        base_dataset=self.train_dataset,
                        explainer=explainer,
                        device=self.device,
                        caipi_k=k,
                        sampling_strategy=sampling_strategy
                    )
                    
                    # Train hybrid model
                    history = hybrid_trainer.train(
                        num_epochs=20,
                        batch_size=32,
                        eval_dataloader=self.val_loader,
                        save_path=os.path.join(self.results_dir, f"hybrid_{config_name}"),
                        evaluate_every=5
                    )
                    
                    # Final evaluation
                    final_metrics = evaluate_model_bias(
                        model=model,
                        dataloader=self.test_loader,
                        explainer=explainer,
                        device=self.device
                    )
                    
                    # Test accuracy
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for images, labels in self.test_loader:
                            images, labels = images.to(self.device), labels.to(self.device)
                            
                            if explainer_name == 'bla':
                                outputs, _ = model(images)
                            else:
                                outputs = model(images)
                                
                            _, predicted = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    
                    test_accuracy = 100 * correct / total
                    
                    result = {
                        'method': 'Hybrid',
                        'configuration': config_name,
                        'test_accuracy': test_accuracy,
                        **final_metrics
                    }
                    
                    results.append(result)
                    print(f"Hybrid {config_name} - Accuracy: {test_accuracy:.2f}%")
        
        return results
    
    def run_all_experiments(self):
        """
        Run all 28 experiments from the paper.
        """
        print(f"Starting comprehensive XIL experiments")
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Device: {self.device}")
        
        start_time = datetime.now()
        
        # 1. Baseline experiments (6 models)
        print(f"\n{'='*80}")
        print("PHASE 1: BASELINE EXPERIMENTS (6 models)")
        print(f"{'='*80}")
        
        baseline_results = []
        for model_name in self.model_architectures:
            result = self.train_baseline_model(model_name)
            baseline_results.append(result)
        
        self.all_results['baseline'] = baseline_results
        
        # 2. CAIPI experiments (6 configurations)
        print(f"\n{'='*80}")
        print("PHASE 2: CAIPI EXPERIMENTS (6 configurations)")
        print(f"{'='*80}")
        
        caipi_results = self.run_caipi_experiments()
        self.all_results['caipi'] = caipi_results
        
        # 3. RRR experiments (4 configurations)
        print(f"\n{'='*80}")
        print("PHASE 3: RRR EXPERIMENTS (4 configurations)")
        print(f"{'='*80}")
        
        rrr_results = self.run_rrr_experiments()
        self.all_results['rrr'] = rrr_results
        
        # 4. Hybrid experiments (12 configurations)
        print(f"\n{'='*80}")
        print("PHASE 4: HYBRID EXPERIMENTS (12 configurations)")
        print(f"{'='*80}")
        
        hybrid_results = self.run_hybrid_experiments()
        self.all_results['hybrid'] = hybrid_results
        
        # Save consolidated results
        self.save_consolidated_results()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print("ALL EXPERIMENTS COMPLETED!")
        print(f"{'='*80}")
        print(f"Total experiments: {len(baseline_results) + len(caipi_results) + len(rrr_results) + len(hybrid_results)}")
        print(f"Duration: {duration}")
        print(f"Results saved to: {self.results_dir}")
        
        # Print summary table
        self.print_results_summary()
    
    def save_consolidated_results(self):
        """Save all results in multiple formats."""
        
        # Save as JSON
        json_path = os.path.join(self.results_dir, 'all_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Save as PyTorch file
        torch_path = os.path.join(self.results_dir, 'all_results.pth')
        torch.save(self.all_results, torch_path)
        
        # Create CSV summary
        all_experiments = []
        
        for method_type, results_list in self.all_results.items():
            for result in results_list:
                all_experiments.append(result)
        
        df = pd.DataFrame(all_experiments)
        csv_path = os.path.join(self.results_dir, 'results_summary.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved:")
        print(f"  - JSON: {json_path}")
        print(f"  - PyTorch: {torch_path}")
        print(f"  - CSV: {csv_path}")
    
    def print_results_summary(self):
        """Print summary table of all results."""
        print(f"\n{'='*100}")
        print("RESULTS SUMMARY")
        print(f"{'='*100}")
        
        # Create summary DataFrame
        all_results = []
        for method_type, results_list in self.all_results.items():
            for result in results_list:
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        
        # Print key metrics
        if not df.empty:
            print(f"{'Method':<15} {'Configuration':<25} {'Accuracy':<10} {'FFP':<8} {'BFP':<8} {'BSR':<8} {'DICE':<8}")
            print("-" * 100)
            
            for _, row in df.iterrows():
                method = row.get('method', 'N/A')
                config = row.get('configuration', row.get('model', 'N/A'))
                accuracy = f"{row.get('test_accuracy', 0):.2f}%"
                ffp = f"{row.get('FFP', 0):.3f}"
                bfp = f"{row.get('BFP', 0):.3f}"
                bsr = f"{row.get('BSR', 0):.3f}"
                dice = f"{row.get('DICE', 0):.3f}"
                
                print(f"{method:<15} {config:<25} {accuracy:<10} {ffp:<8} {bfp:<8} {bsr:<8} {dice:<8}")
        
        print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive XIL experiments')
    parser.add_argument('--data_dir', type=str, default='gender_dataset',
                       help='Path to gender dataset')
    parser.add_argument('--results_dir', type=str, 
                       default=f'xil_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize and run experiments
    runner = ComprehensiveExperimentRunner(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=args.device
    )
    
    runner.run_all_experiments()


if __name__ == '__main__':
    main()