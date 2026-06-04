"""Analyse distractor regions in model explanations."""
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['baseline', 'best_caipi_bbox', 'best_hybrid_bbox'])
    parser.add_argument('--data_dir', type=str, default='gender_dataset')
    parser.add_argument('--object_categories', nargs='+', default=['handbag', 'tie', 'oven', 'microwave', 'skateboard'])
    parser.add_argument('--n_images_per_class', type=int, default=50)
    parser.add_argument('--results_dir', type=str, default='results/journal_extension/distractor')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print(f"Running distractor analysis with args: {args}")
    os.makedirs(args.results_dir, exist_ok=True)
    
if __name__ == '__main__':
    main()
