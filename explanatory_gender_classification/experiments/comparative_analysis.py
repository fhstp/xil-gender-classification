"""
Comparative analysis across different models and training methods.

This script compares baseline models vs RRR models across different architectures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from matplotlib.ticker import MaxNLocator


class ModelComparator:
    """Class for comparing different model experiments."""
    
    def __init__(self, results_dir):
        """
        Initialize model comparator.
        
        Args:
            results_dir: Directory containing experimental results
        """
        self.results_dir = results_dir
        self.experiments = {}
        
    def load_experiments(self):
        """Load all experimental results from the results directory."""
        
        # Find all result CSV files
        pattern = os.path.join(self.results_dir, '**', '*_results.csv')
        result_files = glob.glob(pattern, recursive=True)
        
        print(f"Found {len(result_files)} result files")
        
        for file_path in result_files:
            # Extract experiment info from path
            rel_path = os.path.relpath(file_path, self.results_dir)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 2:
                experiment_type = path_parts[0]  # e.g., 'baseline', 'rrr'
                experiment_name = path_parts[1]  # e.g., 'efficientnet_b0_baseline'
                
                # Load results
                try:
                    results_df = pd.read_csv(file_path)
                    
                    # Extract model architecture and method
                    if 'baseline' in experiment_name:
                        architecture = experiment_name.replace('_baseline', '')
                        method = 'baseline'
                    elif 'rrr' in experiment_name:
                        # Extract architecture from RRR experiment name
                        parts = experiment_name.split('_')
                        architecture = parts[0]
                        method = 'rrr'
                    else:
                        architecture = 'unknown'
                        method = 'unknown'
                    
                    key = f"{architecture}_{method}"
                    self.experiments[key] = {
                        'architecture': architecture,
                        'method': method,
                        'results': results_df,
                        'file_path': file_path
                    }
                    
                    print(f"Loaded {key}: {len(results_df)} epochs")
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    def get_final_metrics(self):
        """Get final metrics for all experiments."""
        
        metrics_data = []
        
        for exp_key, exp_data in self.experiments.items():
            results_df = exp_data['results']
            final_epoch = results_df.iloc[-1]
            
            metrics = {
                'Architecture': exp_data['architecture'],
                'Method': exp_data['method'],
                'Experiment': exp_key,
                'Final_Train_Acc': final_epoch.get('Train Accuracy', 0),
                'Final_Val_Acc': final_epoch.get('Validation Accuracy', 0),
                'Final_Test_Acc': final_epoch.get('Test Accuracy', 0),
                'Best_Val_Acc': results_df['Validation Accuracy'].max(),
                'Best_Test_Acc': results_df['Test Accuracy'].max(),
                'Epochs_Trained': len(results_df)
            }
            
            # Add RRR specific metrics if available
            if 'Train Answer Loss' in results_df.columns:
                metrics['Final_Answer_Loss'] = final_epoch.get('Train Answer Loss', 0)
                metrics['Final_Reason_Loss'] = final_epoch.get('Train Reason Loss', 0)
            
            metrics_data.append(metrics)
        
        return pd.DataFrame(metrics_data)
    
    def plot_training_curves_comparison(self, save_path=None):
        """Plot training curves comparison across all experiments."""
        
        # Organize experiments by architecture
        architectures = list(set([exp['architecture'] for exp in self.experiments.values()]))
        architectures.sort()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for exp_key, exp_data in self.experiments.items():
            results_df = exp_data['results']
            architecture = exp_data['architecture']
            method = exp_data['method']
            
            color = colors[architectures.index(architecture) % len(colors)]
            linestyle = '-' if method == 'baseline' else '--'
            label = f"{architecture}_{method}"
            
            # Training Loss
            axes[0].plot(results_df['Epoch'], results_df['Train Loss'], 
                        color=color, linestyle=linestyle, label=label, alpha=0.8)
            
            # Validation Loss
            axes[1].plot(results_df['Epoch'], results_df['Validation Loss'], 
                        color=color, linestyle=linestyle, label=label, alpha=0.8)
            
            # Training Accuracy
            axes[2].plot(results_df['Epoch'], results_df['Train Accuracy'], 
                        color=color, linestyle=linestyle, label=label, alpha=0.8)
            
            # Validation Accuracy
            axes[3].plot(results_df['Epoch'], results_df['Validation Accuracy'], 
                        color=color, linestyle=linestyle, label=label, alpha=0.8)
        
        # Configure subplots
        titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']
        y_labels = ['Loss', 'Loss', 'Accuracy', 'Accuracy']
        
        for i, (ax, title, ylabel) in enumerate(zip(axes, titles, y_labels)):
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            if i >= 2:  # Accuracy plots
                ax.set_ylim(0, 1)
        
        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves comparison saved to {save_path}")
        
        plt.show()
    
    def plot_final_metrics_comparison(self, save_path=None):
        """Plot final metrics comparison."""
        
        metrics_df = self.get_final_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Test Accuracy by Architecture and Method
        pivot_test = metrics_df.pivot(index='Architecture', columns='Method', values='Final_Test_Acc')
        pivot_test.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Final Test Accuracy by Architecture and Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Architecture')
        axes[0, 0].legend(title='Method')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Best Validation Accuracy
        pivot_val = metrics_df.pivot(index='Architecture', columns='Method', values='Best_Val_Acc')
        pivot_val.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Best Validation Accuracy by Architecture and Method')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xlabel('Architecture')
        axes[0, 1].legend(title='Method')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training Time (Epochs to Completion)
        pivot_epochs = metrics_df.pivot(index='Architecture', columns='Method', values='Epochs_Trained')
        pivot_epochs.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Training Duration (Epochs) by Architecture and Method')
        axes[1, 0].set_ylabel('Number of Epochs')
        axes[1, 0].set_xlabel('Architecture')
        axes[1, 0].legend(title='Method')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance Improvement (RRR vs Baseline)
        baseline_metrics = metrics_df[metrics_df['Method'] == 'baseline'].set_index('Architecture')
        rrr_metrics = metrics_df[metrics_df['Method'] == 'rrr'].set_index('Architecture')
        
        # Calculate improvement
        common_architectures = set(baseline_metrics.index) & set(rrr_metrics.index)
        improvements = []
        
        for arch in common_architectures:
            baseline_acc = baseline_metrics.loc[arch, 'Final_Test_Acc']
            rrr_acc = rrr_metrics.loc[arch, 'Final_Test_Acc']
            improvement = rrr_acc - baseline_acc
            improvements.append({'Architecture': arch, 'Improvement': improvement})
        
        if improvements:
            improvement_df = pd.DataFrame(improvements)
            bars = axes[1, 1].bar(improvement_df['Architecture'], improvement_df['Improvement'])
            axes[1, 1].set_title('Test Accuracy Improvement (RRR vs Baseline)')
            axes[1, 1].set_ylabel('Accuracy Improvement')
            axes[1, 1].set_xlabel('Architecture')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Color bars based on positive/negative improvement
            for bar, improvement in zip(bars, improvement_df['Improvement']):
                if improvement > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Final metrics comparison saved to {save_path}")
        
        plt.show()
    
    def generate_summary_table(self, save_path=None):
        """Generate a summary table of all experiments."""
        
        metrics_df = self.get_final_metrics()
        
        # Round numerical columns
        numerical_cols = ['Final_Train_Acc', 'Final_Val_Acc', 'Final_Test_Acc', 
                         'Best_Val_Acc', 'Best_Test_Acc']
        
        for col in numerical_cols:
            if col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].round(4)
        
        # Sort by architecture and method
        metrics_df = metrics_df.sort_values(['Architecture', 'Method'])
        
        print("Summary of All Experiments:")
        print("=" * 100)
        print(metrics_df.to_string(index=False))
        
        if save_path:
            metrics_df.to_csv(save_path, index=False)
            print(f"\nSummary table saved to {save_path}")
        
        return metrics_df
    
    def analyze_rrr_components(self, save_path=None):
        """Analyze RRR loss components if available."""
        
        rrr_experiments = {k: v for k, v in self.experiments.items() if v['method'] == 'rrr'}
        
        if not rrr_experiments:
            print("No RRR experiments found for component analysis")
            return
        
        fig, axes = plt.subplots(len(rrr_experiments), 2, figsize=(15, 5 * len(rrr_experiments)))
        
        if len(rrr_experiments) == 1:
            axes = [axes]
        
        for i, (exp_key, exp_data) in enumerate(rrr_experiments.items()):
            results_df = exp_data['results']
            
            if 'Train Answer Loss' not in results_df.columns:
                continue
            
            # Plot answer vs reason loss
            axes[i][0].plot(results_df['Epoch'], results_df['Train Answer Loss'], 
                           label='Answer Loss', color='blue')
            axes[i][0].plot(results_df['Epoch'], results_df['Train Reason Loss'], 
                           label='Reason Loss', color='red')
            axes[i][0].set_title(f'{exp_key}: RRR Loss Components')
            axes[i][0].set_xlabel('Epoch')
            axes[i][0].set_ylabel('Loss')
            axes[i][0].legend()
            axes[i][0].grid(True, alpha=0.3)
            
            # Plot loss ratio
            ratio = results_df['Train Reason Loss'] / results_df['Train Answer Loss']
            axes[i][1].plot(results_df['Epoch'], ratio, color='green')
            axes[i][1].set_title(f'{exp_key}: Reason/Answer Loss Ratio')
            axes[i][1].set_xlabel('Epoch')
            axes[i][1].set_ylabel('Ratio')
            axes[i][1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RRR component analysis saved to {save_path}")
        
        plt.show()


def main():
    """Main function for comparative analysis."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Comparative analysis of gender classification experiments')
    parser.add_argument('--results_dir', type=str, default='experiments',
                       help='Directory containing experimental results')
    parser.add_argument('--output_dir', type=str, default='comparative_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting comparative analysis...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize comparator
    comparator = ModelComparator(args.results_dir)
    
    # Load all experiments
    print("\nLoading experiments...")
    comparator.load_experiments()
    
    if not comparator.experiments:
        print("No experiments found! Check the results directory.")
        return
    
    print(f"\nLoaded {len(comparator.experiments)} experiments:")
    for exp_key in comparator.experiments.keys():
        print(f"  - {exp_key}")
    
    # Generate analyses
    print("\nGenerating comparative analysis...")
    
    # 1. Summary table
    summary_path = os.path.join(args.output_dir, 'experiment_summary.csv')
    summary_df = comparator.generate_summary_table(save_path=summary_path)
    
    # 2. Training curves comparison
    curves_path = os.path.join(args.output_dir, 'training_curves_comparison.png')
    comparator.plot_training_curves_comparison(save_path=curves_path)
    
    # 3. Final metrics comparison
    metrics_path = os.path.join(args.output_dir, 'final_metrics_comparison.png')
    comparator.plot_final_metrics_comparison(save_path=metrics_path)
    
    # 4. RRR component analysis (if available)
    rrr_path = os.path.join(args.output_dir, 'rrr_component_analysis.png')
    comparator.analyze_rrr_components(save_path=rrr_path)
    
    # 5. Statistical analysis
    print("\nStatistical Analysis:")
    print("=" * 50)
    
    baseline_accs = summary_df[summary_df['Method'] == 'baseline']['Final_Test_Acc']
    rrr_accs = summary_df[summary_df['Method'] == 'rrr']['Final_Test_Acc']
    
    if len(baseline_accs) > 0 and len(rrr_accs) > 0:
        print(f"Baseline Test Accuracy - Mean: {baseline_accs.mean():.4f}, Std: {baseline_accs.std():.4f}")
        print(f"RRR Test Accuracy - Mean: {rrr_accs.mean():.4f}, Std: {rrr_accs.std():.4f}")
        
        # Statistical significance test (if scipy is available)
        try:
            from scipy import stats
            stat, p_value = stats.ttest_ind(rrr_accs, baseline_accs)
            print(f"T-test p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("Difference is statistically significant (p < 0.05)")
            else:
                print("Difference is not statistically significant (p >= 0.05)")
        except ImportError:
            print("Scipy not available for statistical testing")
    
    # 6. Best performing models
    print("\nBest Performing Models:")
    print("=" * 30)
    
    best_overall = summary_df.loc[summary_df['Final_Test_Acc'].idxmax()]
    print(f"Overall Best: {best_overall['Experiment']} - {best_overall['Final_Test_Acc']:.4f}")
    
    best_baseline = summary_df[summary_df['Method'] == 'baseline'].loc[
        summary_df[summary_df['Method'] == 'baseline']['Final_Test_Acc'].idxmax()
    ]
    print(f"Best Baseline: {best_baseline['Experiment']} - {best_baseline['Final_Test_Acc']:.4f}")
    
    if len(summary_df[summary_df['Method'] == 'rrr']) > 0:
        best_rrr = summary_df[summary_df['Method'] == 'rrr'].loc[
            summary_df[summary_df['Method'] == 'rrr']['Final_Test_Acc'].idxmax()
        ]
        print(f"Best RRR: {best_rrr['Experiment']} - {best_rrr['Final_Test_Acc']:.4f}")
    
    print(f"\nComparative analysis completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()