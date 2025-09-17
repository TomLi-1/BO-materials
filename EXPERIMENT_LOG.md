# BO-Materials Experiment Log

## Experiment 1: Formation Energy Minimization

**Date**: 2025-09-06  
**Objective**: Find thermodynamically stable materials (minimize formation energy)  
**Dataset**: Matbench MP Formation Energy (`matbench_mp_e_form`)  

### Experimental Setup

#### Data Configuration
- **Total samples**: Combined train + test split from Matbench
- **Target variable**: Formation energy per atom (eV/atom)
- **Optimization goal**: **MINIMIZE** formation energy (find stable materials)
- **Ground truth range**: [min, max] eV/atom (to be filled from run output)

#### Featurization
- **Method**: Matminer ElementProperty with Magpie preset
- **Feature set**: Magpie elemental property descriptors
- **Total features**: ~132 dimensional feature space
- **Preprocessing**: MinMaxScaler applied before GP fitting

#### Bayesian Optimization Parameters
```yaml
BO Configuration:
  initialization_budget: 10      # Random samples to start
  total_sample_budget: 200       # Total BO iterations
  acquisition_function: "EI"     # Expected Improvement
  minimize: true                 # Target stable materials

Feature Selection:
  method: "MI"                   # Mutual Information
  num_sparsity_feats: 30         # Selected features per iteration
  
GP Model:
  kernel: RBF (default BoTorch)
  input_transform: Normalize + MinMaxScaler
  outcome_transform: Standardize
```

#### Model Architecture
- **Surrogate Model**: Single-task Gaussian Process (BoTorch)
- **Acquisition**: Expected Improvement with `maximize=False`
- **Feature Selection**: Mutual Information-based selection at each BO iteration
- **Optimization**: Dynamic feature selection on growing dataset

### Results Summary

#### Performance Metrics
*(To be filled after run completion)*
```
BO Performance:
- Most stable material found: _____ eV/atom
- True global minimum: _____ eV/atom  
- Gap to global optimum: _____ eV/atom
- Global ranking: #___ out of _____ materials

Model Evaluation:
- Reduced features (30) MAE: _____ 
- Full Magpie features MAE: _____
- R² score: _____
```

#### Key Findings
*(To be documented)*
1. Feature selection effectiveness
2. Convergence behavior  
3. Stability of discovered materials
4. Model prediction quality

### Technical Implementation

#### Code Structure
```
bo_matbench/
├── src/
│   ├── data_loader.py     # Matbench dataset loading with caching
│   ├── featurizer.py      # Matminer wrapper for Magpie features  
│   ├── surrogate.py       # GP model + acquisition functions
│   ├── bo_loop.py         # Main BO iteration logic
│   └── config.yaml        # Configuration (not actively used)
├── run.py                 # Experiment orchestration
└── requirements.txt       # Dependencies
```

#### Key Dependencies
- `botorch`: GP models and acquisition functions
- `matminer`: Materials featurization  
- `scikit-learn`: Feature selection and metrics
- `torch`: Tensor operations
- `matplotlib`: Visualization

### Modifications Made
1. **Physics-Correct Targeting**: Changed from maximization to minimization
2. **Acquisition Function**: Added `minimize=True` flag for stable material discovery
3. **Evaluation Metrics**: Updated to report stability rankings and gaps to global minimum
4. **Visualization**: Dual plots comparing reduced vs full feature performance

### Next Steps
- **Bandgap Optimization**: Adapt framework for electronic property targets
- **Multi-objective**: Potentially combine stability + bandgap criteria  
- **Feature Engineering**: Explore additional descriptors for electronic properties
- **Hyperparameter Tuning**: Optimize BO parameters based on formation energy results

---

---

## Experiment 2: Bandgap Target Optimization

**Date**: 2025-09-06  
**Objective**: Find materials with target bandgap for semiconductor applications  
**Dataset**: Matbench Experimental Bandgap (`matbench_expt_gap`)  

### Experimental Setup

#### Data Configuration
- **Dataset**: 6,354 experimental bandgap measurements from Matbench
- **Target variable**: Experimental bandgap (eV)
- **Optimization goal**: **TARGET** specific bandgap value (1.5 ± 0.2 eV)
- **Application**: Optimal for single-junction solar cells

#### Target Specification
```yaml
Target Configuration:
  bandgap_target: 1.5 eV        # Optimal for solar applications
  tolerance: ±0.2 eV            # Acceptable range [1.3, 1.7 eV]  
  semiconductor_type: any       # Direct/indirect both acceptable
```

#### Featurization  
- **Method**: Matminer ElementProperty with Magpie preset (same as Exp 1)
- **Feature set**: 132-dimensional Magpie elemental descriptors
- **Selection**: Mutual Information with 40 features (increased from 30)

#### Multi-Objective BO Parameters
```yaml
BO Configuration:
  experiment_type: "bandgap_target"
  acquisition_function: "EI_target"    # Custom target-based EI
  total_sample_budget: 200
  initialization_budget: 15
  
Target-Based Acquisition:
  method: "TargetBasedAcquisition"     # P(in_range) × uncertainty / distance  
  combines: [probability_in_range, exploration, target_proximity]

Optional Classification:
  enabled: true/false
  task: spacegroup_classification      # 230 space groups
  weight: 0.3                         # Secondary to bandgap targeting
```

#### Success Metrics
- **Success Rate**: % materials found within target range
- **Target Detection F1**: Precision/recall for identifying target materials  
- **Distance Minimization**: Closest approach to 1.5 eV target
- **Convergence**: Progressive improvement toward target

### Results Summary
*(To be filled after execution)*
```
Target Achievement:
- Success rate: ____%
- Materials in target range: ___/200  
- Best material found: ____ eV
- Distance from target: ____ eV

Model Performance:
- Standard regression MAE: ____
- Target detection F1: ____
- R² score: ____
```

#### Key Technical Features
1. **Target-Based Acquisition**: Custom function optimizing P(in_range) × σ / distance
2. **Multi-Objective Ready**: Framework supports bandgap + spacegroup classification
3. **Semiconductor Focus**: Dataset filtered for reasonable bandgap ranges (0.1-6.0 eV)
4. **Application-Driven**: Target chosen for solar cell efficiency optimization

---

## Future Experiments

### Planned: Multi-Objective Optimization
- **Primary**: Bandgap targeting (1.5 eV)  
- **Secondary**: Spacegroup classification (prioritize specific crystal structures)
- **Weight**: 70% bandgap, 30% classification
- **Applications**: Solar cells with preferred crystal symmetries

---

*Log maintained for reproducibility and method tracking*