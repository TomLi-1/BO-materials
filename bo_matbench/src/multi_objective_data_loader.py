import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matminer.datasets import load_dataset
from pymatgen.core import Structure

CACHE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")

def load_bandgap_dataset(dataset_name="matbench_expt_gap", test_size=0.2, random_state=42):
    """
    Load bandgap datasets from matminer/matbench
    
    Args:
        dataset_name: "matbench_expt_gap", "matbench_mp_gap", etc.
        test_size: Train/test split ratio
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test: Compositions and bandgap values
        metadata: Additional information (spacegroups, etc.)
    """
    cache_path = os.path.join(CACHE_DIR, f"{dataset_name}.pkl")
    
    if os.path.exists(cache_path):
        print(f"Loading cached dataset: {cache_path}")
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
    else:
        print(f"Downloading dataset: {dataset_name}")
        df = load_dataset(dataset_name)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        print(f"Dataset cached at: {cache_path}")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Debug: Show sample data
    print(f"First few rows:")
    print(df.head())
    
    # Extract compositions and bandgap values
    from pymatgen.core import Composition
    
    compositions = None
    if "composition" in df.columns:
        # Check if compositions are strings or Composition objects
        sample_comp = df["composition"].iloc[0]
        if isinstance(sample_comp, str):
            print(f"Converting composition strings to Composition objects...")
            try:
                compositions = []
                for i, comp_str in enumerate(df["composition"]):
                    try:
                        compositions.append(Composition(comp_str))
                    except Exception as e:
                        print(f"Warning: Could not parse composition '{comp_str}' at index {i}: {e}")
                        continue
                print(f"Successfully converted {len(compositions)} compositions")
            except Exception as e:
                print(f"Error during composition conversion: {e}")
                raise
        else:
            compositions = df["composition"].tolist()
    elif "formula" in df.columns:
        # Convert formula strings to Composition objects
        print(f"Converting formula strings to Composition objects...")
        compositions = [Composition(formula) for formula in df["formula"]]
    else:
        # Check for other possible composition columns
        possible_cols = [col for col in df.columns if 'comp' in col.lower() or 'formula' in col.lower()]
        if possible_cols:
            comp_col = possible_cols[0]
            print(f"Using column '{comp_col}' for compositions...")
            sample_comp = df[comp_col].iloc[0]
            if isinstance(sample_comp, str):
                compositions = [Composition(comp_str) for comp_str in df[comp_col]]
            else:
                compositions = df[comp_col].tolist()
        else:
            raise ValueError(f"No composition column found in {dataset_name}. Available columns: {list(df.columns)}")
    
    if compositions is None:
        raise ValueError(f"Failed to extract compositions from {dataset_name}")
    
    # Extract target values (bandgap)
    bandgap_columns = ["gap expt", "gap", "bandgap", "band_gap"]
    bandgap_col = None
    for col in bandgap_columns:
        if col in df.columns:
            bandgap_col = col
            break
    
    if bandgap_col is None:
        raise ValueError(f"No bandgap column found. Available columns: {df.columns}")
    
    bandgaps = df[bandgap_col]
    
    # Extract metadata for classification tasks
    metadata = {}
    if "structure" in df.columns:
        metadata["structures"] = df["structure"].tolist()
        # Extract spacegroups from structures
        try:
            spacegroups = []
            crystal_systems = []
            for struct in metadata["structures"]:
                if isinstance(struct, Structure):
                    analyzer = struct.get_space_group_info()
                    spacegroups.append(analyzer[1])  # Space group number
                    crystal_systems.append(analyzer[0][:3])  # Crystal system (first 3 chars)
                else:
                    spacegroups.append(None)
                    crystal_systems.append(None)
            metadata["spacegroups"] = spacegroups
            metadata["crystal_systems"] = crystal_systems
        except Exception as e:
            print(f"Warning: Could not extract spacegroups: {e}")
            metadata["spacegroups"] = [None] * len(compositions)
            metadata["crystal_systems"] = [None] * len(compositions)
    
    # Split data
    if len(metadata) > 0:
        # Include metadata in split
        train_indices, test_indices = train_test_split(
            range(len(compositions)), test_size=test_size, random_state=random_state
        )
        
        X_train = [compositions[i] for i in train_indices]
        X_test = [compositions[i] for i in test_indices]
        y_train = bandgaps.iloc[train_indices].reset_index(drop=True)
        y_test = bandgaps.iloc[test_indices].reset_index(drop=True)
        
        # Split metadata
        train_metadata = {}
        test_metadata = {}
        for key, values in metadata.items():
            train_metadata[key] = [values[i] for i in train_indices]
            test_metadata[key] = [values[i] for i in test_indices]
        
        return X_train, X_test, y_train, y_test, train_metadata, test_metadata
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            compositions, bandgaps, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, {}, {}


def get_semiconductor_target_materials(bandgap_target=1.5, tolerance=0.2):
    """
    Filter materials within semiconductor bandgap range
    
    Args:
        bandgap_target: Target bandgap in eV
        tolerance: Acceptable range ±tolerance
    
    Returns:
        Filtered dataset focusing on semiconductor range
    """
    min_gap = bandgap_target - tolerance
    max_gap = bandgap_target + tolerance
    
    return min_gap, max_gap


def create_classification_targets(metadata, task_type="spacegroup"):
    """
    Create classification targets from metadata
    
    Args:
        metadata: Dictionary containing structural information
        task_type: "spacegroup", "crystal_system", or "both"
    
    Returns:
        Classification targets and label mappings
    """
    if task_type == "spacegroup":
        if "spacegroups" not in metadata:
            raise ValueError("Spacegroup information not available")
        
        # Convert spacegroup numbers to classification labels
        spacegroups = metadata["spacegroups"]
        # Remove None values and create mapping
        valid_sg = [sg for sg in spacegroups if sg is not None]
        unique_sg = sorted(list(set(valid_sg)))
        sg_to_label = {sg: i for i, sg in enumerate(unique_sg)}
        
        labels = [sg_to_label.get(sg, -1) for sg in spacegroups]  # -1 for missing
        
        return labels, {"spacegroup_mapping": sg_to_label, "num_classes": len(unique_sg)}
        
    elif task_type == "crystal_system":
        if "crystal_systems" not in metadata:
            raise ValueError("Crystal system information not available")
        
        crystal_systems = metadata["crystal_systems"] 
        # Standard crystal systems
        cs_mapping = {
            "cub": 0, "tet": 1, "ort": 2, "hex": 3, 
            "tri": 4, "mon": 5, "rho": 6
        }
        
        labels = [cs_mapping.get(cs, -1) for cs in crystal_systems]
        
        return labels, {"crystal_system_mapping": cs_mapping, "num_classes": 7}
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


if __name__ == "__main__":
    # Test the data loader
    print("Testing bandgap dataset loading...")
    try:
        X_train, X_test, y_train, y_test, train_meta, test_meta = load_bandgap_dataset()
        print(f"✅ Successfully loaded data:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test: {len(X_test)} samples") 
        print(f"   Bandgap range: [{y_train.min():.3f}, {y_train.max():.3f}] eV")
        print(f"   Metadata keys: {list(train_meta.keys())}")
        
        if "spacegroups" in train_meta:
            labels, mapping = create_classification_targets(train_meta, "spacegroup")
            print(f"   Spacegroups: {mapping['num_classes']} unique classes")
            
        # Test composition types
        print(f"   Composition types: {type(X_train[0]) if X_train else 'Empty'}")
        if X_train:
            print(f"   Sample composition: {X_train[0]}")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")