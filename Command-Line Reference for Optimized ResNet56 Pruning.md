# Command-Line Reference for Optimized ResNet56 Pruning

**Quick Reference Guide**  
**Author**: Manus AI  
**Date**: June 30, 2025

## Essential Commands

### 1. Basic Optimized Training

**Standard optimized pruning with knowledge distillation:**

```bash
python train_optimized_resnet56.py --enable-pruning --use-distillation
```

**With custom output directory:**

```bash
python train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --output-dir ./results/experiment_1 \
    --checkpoint-dir ./checkpoints/experiment_1
```

### 2. Conservative Pruning (High Accuracy)

**For maximum accuracy retention:**

```bash
python train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --stage1-ratio 0.10 \
    --stage2-ratio 0.25 \
    --stage3-ratio 0.45 \
    --phase1-epochs 15 \
    --phase2-epochs 20 \
    --phase3-epochs 25 \
    --distillation-weight 0.4
```

**Expected Results:**
- Compression: ~45%
- Accuracy retention: ~98%

### 3. Aggressive Pruning (Maximum Compression)

**For resource-constrained deployment:**

```bash
python train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --stage1-ratio 0.20 \
    --stage2-ratio 0.55 \
    --stage3-ratio 0.75 \
    --temperature 5.0 \
    --phase1-epochs 8 \
    --phase2-epochs 12 \
    --phase3-epochs 15
```

**Expected Results:**
- Compression: ~75%
- Accuracy retention: ~92%

### 4. Fine-Tuning Pretrained Model

**Start from existing checkpoint:**

```bash
python train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --skip-initial-training \
    --pretrained-path ./checkpoints/baseline/best_model.pth \
    --phase1-epochs 12 \
    --phase2-epochs 18 \
    --phase3-epochs 22
```

### 5. Baseline Training (No Pruning)

**For comparison purposes:**

```bash
python train_optimized_resnet56.py \
    --initial-epochs 200 \
    --initial-lr 0.1 \
    --output-dir ./results/baseline
```

## Parameter Quick Reference

### Pruning Ratios
```bash
--stage1-ratio FLOAT    # Early layers (default: 0.15, range: 0.05-0.25)
--stage2-ratio FLOAT    # Middle layers (default: 0.40, range: 0.20-0.60)  
--stage3-ratio FLOAT    # Late layers (default: 0.60, range: 0.40-0.80)
```

### Progressive Schedule
```bash
--phase1-epochs INT     # Phase 1 fine-tuning (default: 10)
--phase2-epochs INT     # Phase 2 fine-tuning (default: 15)
--phase3-epochs INT     # Phase 3 fine-tuning (default: 20)
--phase1-lr FLOAT      # Phase 1 learning rate (default: 0.01)
--phase2-lr FLOAT      # Phase 2 learning rate (default: 0.005)
--phase3-lr FLOAT      # Phase 3 learning rate (default: 0.001)
```

### Knowledge Distillation
```bash
--use-distillation         # Enable knowledge distillation
--distillation-weight FLOAT # Loss weight (default: 0.3, range: 0.1-0.5)
--temperature FLOAT        # Softmax temperature (default: 4.0, range: 2.0-8.0)
```

### Training Control
```bash
--batch-size INT           # Batch size (default: 128)
--initial-epochs INT       # Initial training epochs (default: 200)
--initial-lr FLOAT        # Initial learning rate (default: 0.1)
--skip-initial-training   # Skip initial training phase
```

### Output Control
```bash
--checkpoint-dir PATH     # Checkpoint directory (default: ./checkpoints)
--output-dir PATH        # Results directory (default: ./results)
--pretrained-path PATH   # Pretrained model path
```

## Common Use Cases

### Research and Development

**Experiment with different configurations:**

```bash
# Conservative experiment
python train_optimized_resnet56.py \
    --enable-pruning --use-distillation \
    --stage1-ratio 0.10 --stage2-ratio 0.30 --stage3-ratio 0.50 \
    --output-dir ./results/conservative

# Standard experiment  
python train_optimized_resnet56.py \
    --enable-pruning --use-distillation \
    --output-dir ./results/standard

# Aggressive experiment
python train_optimized_resnet56.py \
    --enable-pruning --use-distillation \
    --stage1-ratio 0.20 --stage2-ratio 0.50 --stage3-ratio 0.70 \
    --output-dir ./results/aggressive
```

### Production Deployment

**Mobile/Edge deployment:**

```bash
python train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --stage1-ratio 0.15 \
    --stage2-ratio 0.45 \
    --stage3-ratio 0.65 \
    --batch-size 64 \
    --temperature 4.5 \
    --output-dir ./results/mobile_deployment
```

**Server deployment (balanced):**

```bash
python train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --stage1-ratio 0.12 \
    --stage2-ratio 0.35 \
    --stage3-ratio 0.55 \
    --phase1-epochs 12 \
    --phase2-epochs 18 \
    --phase3-epochs 24 \
    --output-dir ./results/server_deployment
```

### Hyperparameter Tuning

**Grid search example:**

```bash
#!/bin/bash
# Grid search script

for stage1 in 0.10 0.15 0.20; do
    for stage2 in 0.30 0.40 0.50; do
        for stage3 in 0.50 0.60 0.70; do
            echo "Testing: stage1=$stage1, stage2=$stage2, stage3=$stage3"
            python train_optimized_resnet56.py \
                --enable-pruning \
                --use-distillation \
                --stage1-ratio $stage1 \
                --stage2-ratio $stage2 \
                --stage3-ratio $stage3 \
                --output-dir ./results/grid_${stage1}_${stage2}_${stage3}
        done
    done
done
```

## Performance Monitoring

### Real-time Monitoring

**Monitor training progress:**

```bash
# Run training in background and monitor logs
python train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --output-dir ./results/monitored 2>&1 | tee training.log &

# Monitor in real-time
tail -f training.log
```

### Resource Usage

**Monitor GPU usage:**

```bash
# In separate terminal
watch -n 1 nvidia-smi

# Or use gpustat if available
watch -n 1 gpustat
```

**Monitor system resources:**

```bash
# Monitor CPU and memory
htop

# Monitor disk usage
df -h
du -sh ./checkpoints ./results
```

## Troubleshooting Commands

### Memory Issues

**Reduce memory usage:**

```bash
python train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --batch-size 64 \
    --phase1-epochs 8 \
    --phase2-epochs 12 \
    --phase3-epochs 16
```

### Quick Testing

**Fast test run:**

```bash
python train_optimized_resnet56.py \
    --enable-pruning \
    --skip-initial-training \
    --phase1-epochs 2 \
    --phase2-epochs 2 \
    --phase3-epochs 2 \
    --output-dir ./results/test
```

### Debug Mode

**Enable detailed logging:**

```bash
python -u train_optimized_resnet56.py \
    --enable-pruning \
    --use-distillation \
    --output-dir ./results/debug 2>&1 | tee debug.log
```

## Output Files Reference

### Checkpoint Directory Structure
```
checkpoints/
├── latest_checkpoint.pth    # Most recent model state
├── best_model.pth          # Best performing model
└── training_log.txt        # Training progress log
```

### Results Directory Structure
```
results/
├── final_model.pth         # Final pruned model
├── training_results.json   # Comprehensive statistics
└── pruning_analysis.json   # Detailed pruning information
```

### Results File Contents

**training_results.json:**
```json
{
  "final_statistics": {
    "test_accuracy": 93.45,
    "compression_ratio": 0.67,
    "original_parameters": 853018,
    "pruned_parameters": 281506,
    "model_size_mb": 1.13
  },
  "training_history": [...],
  "arguments": {...},
  "pruning_configuration": {...}
}
```

## Integration Examples

### With Existing Scripts

**Use with original repository structure:**

```bash
# Copy files to original repository
cp optimized_prune_resnet56.py /path/to/implementation-of-pruning-filters/
cp train_optimized_resnet56.py /path/to/implementation-of-pruning-filters/

# Run from original repository directory
cd /path/to/implementation-of-pruning-filters/
python train_optimized_resnet56.py --enable-pruning --use-distillation
```

### Batch Processing

**Process multiple configurations:**

```bash
#!/bin/bash
# Batch processing script

configs=(
    "0.10 0.25 0.45 conservative"
    "0.15 0.40 0.60 standard"  
    "0.20 0.55 0.75 aggressive"
)

for config in "${configs[@]}"; do
    read -r stage1 stage2 stage3 name <<< "$config"
    echo "Running $name configuration..."
    
    python train_optimized_resnet56.py \
        --enable-pruning \
        --use-distillation \
        --stage1-ratio $stage1 \
        --stage2-ratio $stage2 \
        --stage3-ratio $stage3 \
        --output-dir ./results/$name \
        --checkpoint-dir ./checkpoints/$name
done
```

## Performance Expectations

### Training Time Estimates

| Configuration | GPU | Estimated Time | Memory Usage |
|---------------|-----|----------------|--------------|
| Conservative | RTX 3080 | 5-6 hours | 4-6 GB |
| Standard | RTX 3080 | 6-7 hours | 4-6 GB |
| Aggressive | RTX 3080 | 4-5 hours | 3-5 GB |
| Conservative | V100 | 3-4 hours | 6-8 GB |
| Standard | V100 | 4-5 hours | 6-8 GB |
| Aggressive | V100 | 3-4 hours | 5-7 GB |

### Expected Results Summary

| Pruning Level | Compression | Accuracy | Use Case |
|---------------|-------------|----------|----------|
| Conservative | 40-50% | 97-98% | High-accuracy applications |
| Standard | 60-70% | 95-97% | Balanced deployment |
| Aggressive | 70-80% | 90-95% | Resource-constrained edge |

This reference guide provides the essential commands and configurations needed to effectively use the optimized ResNet56 pruning implementation. For detailed explanations and advanced usage, refer to the complete README documentation.

