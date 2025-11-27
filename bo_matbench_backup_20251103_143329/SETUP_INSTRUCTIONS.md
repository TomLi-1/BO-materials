# Setup Instructions for Bandgap Targeting

## Prerequisites

The bandgap targeting framework requires additional dependencies beyond the formation energy experiment.

## Installation

### 1. Install Required Packages

```bash
# Core materials science packages
pip install matminer>=0.9.0
pip install pymatgen>=2023.0.0

# Configuration and visualization
pip install pyyaml>=6.0

# If you encounter issues, try installing with conda:
conda install -c conda-forge matminer pymatgen
```

### 2. Verify Installation

```bash
python -c "import matminer; print('✅ matminer installed')"
python -c "import pymatgen; print('✅ pymatgen installed')"
```

### 3. Test Data Loading

```bash
cd bo_matbench
python debug_bandgap_data.py
```

## Expected Output

You should see:
```
Testing bandgap data loading...
✅ Import successful

🔍 Loading dataset...
Dataset shape: (4604, X)
Columns: ['composition', 'gap expt', ...]
Converting composition strings to Composition objects...
Successfully converted XXXX compositions
✅ Dataset loaded: XXXX total materials

🧬 Testing featurization...
✅ Transformer created
🔬 Testing on first 3 samples...
✅ Featurization successful! Shape: (3, 132)
```

## Common Issues

### Issue 1: matminer not found
```bash
# Solution: Install matminer
pip install matminer
```

### Issue 2: pymatgen version conflicts
```bash
# Solution: Update pymatgen
pip install --upgrade pymatgen
```

### Issue 3: Featurization takes too long
This is normal for the first run. Matminer caches results, so subsequent runs are faster.

### Issue 4: Memory issues with large datasets
Reduce the dataset size in the config or use a machine with more RAM (>8GB recommended).

## Running Experiments

### Formation Energy (Original)
```bash
python run.py
```

### Bandgap Targeting (New)
```bash
python run_bandgap_targeting.py
```

## Configuration

Edit `src/multi_objective_config.yaml` to customize:
- Target bandgap value (default: 1.5 eV)
- Tolerance (default: ±0.2 eV)
- BO budget and parameters
- Enable/disable classification tasks

## Troubleshooting

If you encounter any issues:
1. Check the error messages in the terminal
2. Verify all dependencies are installed
3. Try running the debug script: `python debug_bandgap_data.py`
4. Check available memory (featurization can be memory-intensive)

## Performance Notes

- First run downloads ~100MB of data and may take 5-10 minutes
- Featurization of 6,354 materials takes ~2-5 minutes 
- BO with 200 iterations takes ~10-15 minutes
- Results are automatically saved and cached for faster subsequent runs