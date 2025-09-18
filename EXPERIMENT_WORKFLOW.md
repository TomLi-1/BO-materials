# Experimental Code Workflow

## Safe Experimentation Protocol

### Before Making Changes

1. **Create Experiment Branch**:
   ```bash
   git checkout -b experiment-[description]
   # Example: git checkout -b experiment-element-filtering
   ```

2. **Document Baseline**:
   ```bash
   python run.py > baseline_results.txt 2>&1
   # Save current results before modifications
   ```

### During Development

3. **Frequent Commits**:
   ```bash
   git add -A
   git commit -m "Work in progress: [description]"
   # Commit every significant change
   ```

4. **Test Changes**:
   ```bash
   python run_bandgap_targeting_quick.py  # Quick test
   python run.py                          # Full test
   ```

### After Completion

5. **Compare Results**:
   ```bash
   python run.py > experiment_results.txt 2>&1
   diff baseline_results.txt experiment_results.txt
   ```

6. **Decision Point**:
   - **Keep Changes**: `git checkout main && git merge experiment-[name]`
   - **Discard Changes**: `git checkout main && git branch -D experiment-[name]`

### Emergency Revert

7. **Quick Revert to Last Good State**:
   ```bash
   git checkout main
   git reset --hard HEAD
   # This discards ALL uncommitted changes
   ```

## Current Backup Status

- ✅ **Element filtering implementation**: Backed up on `element-filtering-backup` branch
- ✅ **Working baseline**: Current `main` branch (commit `6f32a6a`)

## Recovery Commands

```bash
# Restore element filtering (if needed later)
git checkout element-filtering-backup

# Always return to stable baseline
git checkout main

# See all experimental branches
git branch -a
```

## Results Comparison Template

When testing experiments, always compare these metrics:

1. **Formation Energy Optimization**:
   - Best found value vs. true global minimum
   - Gap to optimum (should be < 1.0 eV)
   - Global ranking (lower is better)
   - MAE on validation set (should be < 1.0)

2. **Bandgap Targeting**:
   - Success rate finding target materials
   - Distance from target bandgap
   - Materials in target range count

3. **Performance**:
   - Runtime for featurization
   - BO loop completion time
   - Memory usage during optimization