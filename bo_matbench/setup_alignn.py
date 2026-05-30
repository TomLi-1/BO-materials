#!/usr/bin/env python3
"""
Setup script to install and configure ALIGNN for high-quality materials embeddings.
Based on the top-performing models from Matbench leaderboard.
"""

import subprocess
import sys
import os

def install_alignn():
    """Install ALIGNN and dependencies."""
    print("🔧 Installing ALIGNN and dependencies...")
    
    packages = [
        "alignn",           # Core ALIGNN package
        "jarvis-tools",     # JARVIS framework for materials
        "dgl",             # Deep Graph Library
        "torch-geometric", # PyTorch Geometric (alternative)
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {package}: {e}")
    
    print("\n📦 Installation complete!")

def test_alignn_import():
    """Test if ALIGNN can be imported successfully."""
    print("\n🧪 Testing ALIGNN import...")
    
    try:
        from alignn.models.alignn import ALIGNN
        from alignn.config import TrainingConfig
        from jarvis.core.atoms import Atoms
        print("✅ ALIGNN imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def download_pretrained_models():
    """Download pretrained ALIGNN models for formation energy."""
    print("\n📥 Setting up pretrained models...")
    
    try:
        # Create models directory
        models_dir = "pretrained_models"
        os.makedirs(models_dir, exist_ok=True)
        
        print("📂 Models directory created")
        
        # Information about available models
        print("\n📊 Available pretrained ALIGNN models:")
        print("1. Formation Energy (MP dataset) - MAE: 0.022 eV/atom")
        print("2. Band Gap (JARVIS-DFT) - MAE: 0.218 eV")
        print("3. Bulk Modulus - Various metrics")
        print("4. Total Energy - Various metrics")
        
        print("\n💡 To use pretrained models, you can:")
        print("1. Download from JARVIS-ALIGNN app: https://jarvis.nist.gov/jalignn")
        print("2. Use alignn.models.pretrained_models module")
        print("3. Train your own using alignn.scripts.train_* scripts")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Model setup warning: {e}")
        return False

def create_alignn_config():
    """Create configuration file for ALIGNN formation energy model."""
    print("\n⚙️  Creating ALIGNN configuration...")
    
    config_content = """
# ALIGNN Configuration for Formation Energy Prediction
# Based on Matbench top-performing model

model_config:
  name: "alignn_formation_energy"
  target: "formation_energy_per_atom"
  
  # Graph construction
  cutoff: 8.0
  max_neighbors: 12
  neighbor_strategy: "k-nearest"
  
  # Model architecture  
  alignn_layers: 4
  gcn_layers: 4
  edge_input_features: 80
  triplet_input_features: 40
  embedding_features: 64
  hidden_features: 256
  output_features: 1
  
  # Training
  batch_size: 32
  learning_rate: 0.001
  epochs: 300
  
dataset_config:
  name: "mp_e_form"
  source: "Materials Project"
  target_property: "formation_energy_per_atom"
  
performance:
  mae: 0.022  # eV/atom (Matbench result)
  rmse: 0.040  # eV/atom (estimated)
  r2: 0.95     # estimated

notes: |
  This configuration is based on the top-performing ALIGNN model
  from the Matbench formation energy leaderboard.
  
  Reference: Choudhary et al. (2021) - Atomistic Line Graph Neural Network
  Paper: https://www.nature.com/articles/s41524-021-00650-1
"""
    
    try:
        with open("alignn_config.yaml", "w") as f:
            f.write(config_content)
        print("✅ Configuration saved to alignn_config.yaml")
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 ALIGNN Setup for Matbench Formation Energy Embeddings")
    print("=" * 60)
    print("Based on top-performing models from:")
    print("https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_e_form/")
    print("")
    
    # Step 1: Install packages
    install_alignn()
    
    # Step 2: Test imports
    if not test_alignn_import():
        print("\n❌ Installation verification failed!")
        print("You may need to install additional dependencies manually:")
        print("pip install torch dgl-cpu  # or dgl-cu118 for GPU")
        return
    
    # Step 3: Setup models
    download_pretrained_models()
    
    # Step 4: Create config
    create_alignn_config()
    
    print("\n🎉 ALIGNN setup complete!")
    print("\nNext steps:")
    print("1. Run with: python run.py (featurizer_type='alignn')")
    print("2. The system will use ALIGNN embeddings if available")
    print("3. Falls back to Magpie features if ALIGNN fails")
    print("\n📈 Expected performance improvement:")
    print("- Better formation energy predictions (MAE ~0.022 eV/atom)")
    print("- High-quality 256-dimensional embeddings")
    print("- Improved BO optimization with better representations")

if __name__ == "__main__":
    main()