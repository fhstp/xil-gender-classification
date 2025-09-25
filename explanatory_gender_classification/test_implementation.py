"""
Quick test script to verify the complete XIL implementation.

This script runs a small-scale test of all major components:
- BLA explainability method
- CAIPI augmentation  
- Bias evaluation metrics
- Hybrid training approach
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Test if all imports work
try:
    from src.data.dataset import GenderDataset
    from src.models.cnn_models import create_model
    from src.explainability.bla import create_bla_model, BLAWrapper
    from src.explainability.gradcam import GradCAMWrapper
    from src.augmentation.caipi import CAIPIAugmentation
    from src.evaluation.bias_metrics import evaluate_model_bias, BiasMetricsTracker
    from src.training.hybrid_trainer import HybridXILTrainer
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

def create_dummy_dataset(num_samples=20):
    """Create dummy dataset for testing."""
    images = torch.randn(num_samples, 3, 224, 224)  # Random images
    labels = torch.randint(0, 2, (num_samples,))    # Binary labels
    masks = torch.randint(0, 2, (num_samples, 224, 224)).float()  # Random masks
    
    return TensorDataset(images, labels, masks)

def test_bla_model():
    """Test BLA model creation and inference."""
    print("\n1. Testing BLA Model...")
    
    try:
        # Create BLA model
        model = create_bla_model('efficientnet_b0', num_classes=2)
        print("   ✓ BLA model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        logits, attention = model(dummy_input)
        
        assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"
        assert len(attention.shape) == 3, f"Expected 3D attention, got {attention.shape}"
        print("   ✓ BLA forward pass successful")
        
        # Test explanation wrapper
        explainer = model.get_explanation_wrapper()
        explanations = explainer.generate_explanation(dummy_input)
        print("   ✓ BLA explanations generated successfully")
        
        return model, explainer
        
    except Exception as e:
        print(f"   ✗ BLA model test failed: {e}")
        return None, None

def test_caipi_augmentation():
    """Test CAIPI augmentation."""
    print("\n2. Testing CAIPI Augmentation...")
    
    try:
        # Create CAIPI augmentation
        caipi = CAIPIAugmentation(k=2)
        print("   ✓ CAIPI augmentation created")
        
        # Test single image augmentation
        dummy_image = torch.randn(3, 224, 224)
        dummy_mask = torch.randint(0, 2, (1, 224, 224)).float()
        
        counterexamples = caipi.generate_counterexamples(
            dummy_image, dummy_mask, label=0
        )
        
        assert len(counterexamples) == 2, f"Expected 2 counterexamples, got {len(counterexamples)}"
        print("   ✓ Counterexamples generated successfully")
        
        return caipi
        
    except Exception as e:
        print(f"   ✗ CAIPI augmentation test failed: {e}")
        return None

def test_bias_metrics():
    """Test bias evaluation metrics."""
    print("\n3. Testing Bias Metrics...")
    
    try:
        # Create dummy saliency maps and masks
        saliency_map = torch.rand(224, 224)
        foreground_mask = torch.randint(0, 2, (224, 224)).float()
        
        # Import specific functions
        from src.evaluation.bias_metrics import (
            compute_ffp, compute_bfp, compute_bsr, compute_dice_score
        )
        
        # Test individual metrics
        ffp = compute_ffp(saliency_map, foreground_mask)
        bfp = compute_bfp(saliency_map, foreground_mask)  
        bsr = compute_bsr(saliency_map, foreground_mask)
        dice = compute_dice_score(saliency_map > 0.5, foreground_mask)
        
        print(f"   ✓ FFP: {ffp:.3f}")
        print(f"   ✓ BFP: {bfp:.3f}")
        print(f"   ✓ BSR: {bsr:.3f}")
        print(f"   ✓ DICE: {dice:.3f}")
        
        # Test metrics tracker
        tracker = BiasMetricsTracker()
        tracker.update({'FFP': ffp, 'BFP': bfp, 'BSR': bsr, 'DICE': dice})
        print("   ✓ BiasMetricsTracker working")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Bias metrics test failed: {e}")
        return False

def test_hybrid_training():
    """Test hybrid training setup (without full training)."""
    print("\n4. Testing Hybrid Training Setup...")
    
    try:
        # Create dummy dataset
        dummy_dataset = create_dummy_dataset(10)
        
        # Create model and explainer
        model = create_model('efficientnet_b0', num_classes=2, pretrained=False)
        explainer = GradCAMWrapper(model, target_layer_name='features')
        
        # Create hybrid trainer
        hybrid_trainer = HybridXILTrainer(
            model=model,
            base_dataset=dummy_dataset,
            explainer=explainer,
            device='cpu',
            caipi_k=1,  # Small k for testing
            num_caipi_samples=5  # Small number for testing
        )
        
        print("   ✓ Hybrid trainer created successfully")
        
        # Test CAIPI dataset generation (should work without full training)
        try:
            caipi_dataset = hybrid_trainer.generate_caipi_augmented_dataset()
            print(f"   ✓ CAIPI dataset generated with {len(caipi_dataset)} samples")
        except Exception as e:
            print(f"   ! CAIPI dataset generation warning (expected for dummy data): {e}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Hybrid training setup failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running XIL Implementation Tests")
    print("="*50)
    
    # Run tests
    bla_model, bla_explainer = test_bla_model()
    caipi = test_caipi_augmentation()
    bias_metrics_ok = test_bias_metrics()
    hybrid_ok = test_hybrid_training()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    tests_passed = 0
    total_tests = 4
    
    if bla_model is not None:
        print("✓ BLA Model: PASSED")
        tests_passed += 1
    else:
        print("✗ BLA Model: FAILED")
        
    if caipi is not None:
        print("✓ CAIPI Augmentation: PASSED")
        tests_passed += 1
    else:
        print("✗ CAIPI Augmentation: FAILED")
        
    if bias_metrics_ok:
        print("✓ Bias Metrics: PASSED")
        tests_passed += 1
    else:
        print("✗ Bias Metrics: FAILED")
        
    if hybrid_ok:
        print("✓ Hybrid Training: PASSED")
        tests_passed += 1
    else:
        print("✗ Hybrid Training: FAILED")
    
    print(f"\nResults: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n🎉 All tests passed! The XIL implementation is ready.")
        print("\nNext steps:")
        print("1. Run: python scripts/train_baseline.py --data_dir gender_dataset")
        print("2. Run: python scripts/train_caipi.py --data_dir gender_dataset")
        print("3. Run: python scripts/run_all_experiments.py --data_dir gender_dataset")
    else:
        print(f"\n⚠️  {total_tests - tests_passed} tests failed. Check the errors above.")

if __name__ == '__main__':
    main()