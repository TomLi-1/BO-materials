"""
ALIGNN-based featurizer that extracts graph embeddings from pretrained ALIGNN
models. Falls back to Matminer elemental features when ALIGNN is unavailable.
"""

from __future__ import annotations

import importlib
import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import dgl
import numpy as np
import torch
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure
from sklearn.impute import SimpleImputer

# Lazy ALIGNN imports (avoid crashing environments without working DGL builds)
ALIGNN_AVAILABLE = False
ALIGNN_IMPORT_ERROR: Optional[Exception] = None

ALIGNN = None  # type: ignore
ALIGNNConfig = None  # type: ignore
Graph = None  # type: ignore
pmg_to_atoms = None  # type: ignore
get_figshare_model = None  # type: ignore


def _lazy_import_alignn() -> bool:
    """Import ALIGNN and supporting packages on demand."""
    global ALIGNN_AVAILABLE, ALIGNN_IMPORT_ERROR
    global ALIGNN, ALIGNNConfig, Graph, pmg_to_atoms, get_figshare_model

    if ALIGNN_AVAILABLE:
        return True
    if ALIGNN_IMPORT_ERROR is not None:
        return False

    try:
        models_module = importlib.import_module("alignn.models.alignn")
        ALIGNN = getattr(models_module, "ALIGNN")
        ALIGNNConfig = getattr(models_module, "ALIGNNConfig")

        graphs_module = importlib.import_module("alignn.graphs")
        Graph = getattr(graphs_module, "Graph")

        pretrained_module = importlib.import_module("alignn.pretrained")
        get_figshare_model = getattr(pretrained_module, "get_figshare_model", None)

        atoms_module = importlib.import_module("jarvis.core.atoms")
        pmg_to_atoms = getattr(atoms_module, "pmg_to_atoms")

        ALIGNN_AVAILABLE = True
        return True
    except Exception as exc:  # pragma: no cover - depends on optional deps
        ALIGNN_IMPORT_ERROR = exc
        warnings.warn(
            f"ALIGNN import failed. Falling back to matminer features. "
            f"Reason: {exc}"
        )
        return False


@dataclass
class ALIGNNGraphOptions:
    """Options controlling graph construction for ALIGNN."""

    neighbor_strategy: str = "k-nearest"
    cutoff: float = 8.0
    max_neighbors: int = 12
    atom_features: str = "cgcnn"
    use_canonize: bool = True
    dtype: str = "float32"


class ALIGNNTransformer:
    """
    Featurizer that converts pymatgen Structures into ALIGNN graph embeddings.

    If ALIGNN cannot be loaded, the transformer falls back to Magpie
    elemental property vectors so downstream pipelines continue to function.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_pretrained: bool = True,
        requested_embedding_dim: Optional[int] = None,
        alignn_model_name: str = "mp_e_form_alignn",
        fallback_to_matminer: bool = True,
        graph_options: Optional[ALIGNNGraphOptions] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.use_pretrained = use_pretrained
        self.alignn_model_name = alignn_model_name
        self.fallback_to_matminer = fallback_to_matminer
        self.graph_options = graph_options or ALIGNNGraphOptions()
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = max(1, batch_size)

        self.model: Optional[torch.nn.Module] = None
        self.embedding_dim: Optional[int] = requested_embedding_dim
        self._fc_hook_handle = None
        self._last_embedding: Optional[torch.Tensor] = None
        self._last_prediction: Optional[torch.Tensor] = None
        self.alignn_loaded: bool = False

        self.fallback_featurizer = None
        self._setup_fallback()

        if _lazy_import_alignn():
            self._load_alignn_model()

    def _setup_fallback(self) -> None:
        """Initialize Magpie fallback featurizer."""
        featurizer = ElementProperty.from_preset("magpie")
        featurizer.set_n_jobs(1)
        self.fallback_featurizer = featurizer

    # --------------------------------------------------------------------- #
    # ALIGNN loading utilities
    # --------------------------------------------------------------------- #
    def _load_alignn_model(self) -> None:
        """Load a pretrained ALIGNN model (local checkpoint or Figshare)."""
        try:
            if self.model_path:
                self.model = self._load_from_local_checkpoint(self.model_path, self.config_path)
            elif self.use_pretrained and get_figshare_model is not None:
                self.model = get_figshare_model(self.alignn_model_name)
            else:
                warnings.warn(
                    "ALIGNN requested but neither model_path nor pretrained weights "
                    "were available. Falling back to matminer features."
                )
                self.model = None

            if self.model is not None:
                self.model = self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = int(self.model.config.hidden_features)
                self._register_embedding_hook()
                self.alignn_loaded = True
        except Exception as exc:
            warnings.warn(
                f"Failed to load ALIGNN model ({exc}). Will use fallback features instead."
            )
            self.model = None
            self.alignn_loaded = False

    def _load_from_local_checkpoint(
        self,
        model_path: str,
        config_path: Optional[str],
    ) -> torch.nn.Module:
        """
        Load ALIGNN model from a local checkpoint directory or file.

        Args:
            model_path: Path to `best_model.pt` or a directory containing it.
            config_path: Optional path to `config.json` describing the model.
        """
        resolved_model_path = model_path
        resolved_config_path = config_path

        if os.path.isdir(model_path):
            resolved_model_path = os.path.join(model_path, "best_model.pt")
            if resolved_config_path is None:
                resolved_config_path = os.path.join(model_path, "config.json")

        if resolved_config_path is None or not os.path.isfile(resolved_config_path):
            raise FileNotFoundError(
                "ALIGNN config.json not found. Specify config_path or provide a "
                "directory containing both best_model.pt and config.json."
            )

        if not os.path.isfile(resolved_model_path):
            raise FileNotFoundError(
                f"ALIGNN checkpoint not found at {resolved_model_path}"
            )

        with open(resolved_config_path, "r") as handle:
            config_dict = json.load(handle)

        model_cfg = config_dict.get("model", {})
        config_obj = ALIGNNConfig(**model_cfg)
        model = ALIGNN(config_obj)

        checkpoint = torch.load(resolved_model_path, map_location=self.device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model

    def _register_embedding_hook(self) -> None:
        """Capture the latent graph embedding before the final linear head."""
        if self.model is None or self._fc_hook_handle is not None:
            return

        def _capture_embedding(module, inputs, _output):
            self._last_embedding = inputs[0].detach()

        self._fc_hook_handle = self.model.fc.register_forward_hook(_capture_embedding)

    # --------------------------------------------------------------------- #
    # ALIGNN feature extraction
    # --------------------------------------------------------------------- #
    def _structure_to_graph(self, structure: Structure):
        """Convert pymatgen Structure to ALIGNN DGL graphs."""
        if pmg_to_atoms is None or Graph is None:
            raise RuntimeError("ALIGNN graph utilities unavailable.")

        jarvis_atoms = pmg_to_atoms(structure)
        options = self.graph_options
        g, lg = Graph.atom_dgl_multigraph(
            jarvis_atoms,
            neighbor_strategy=options.neighbor_strategy,
            cutoff=options.cutoff,
            max_neighbors=options.max_neighbors,
            atom_features=options.atom_features,
            use_canonize=options.use_canonize,
            dtype=options.dtype,
        )
        lattice = torch.tensor(jarvis_atoms.lattice_mat, dtype=torch.get_default_dtype())
        return g, lg, lattice

    def _run_alignn_forward(
        self,
        structures: List[Structure],
        capture_embeddings: bool = True,
        capture_predictions: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Run ALIGNN forward pass, returning embeddings and predictions."""
        if self.model is None:
            return None

        embeddings: List[np.ndarray] = []
        predictions: List[np.ndarray] = []

        batch_graphs: List[dgl.DGLGraph] = []
        batch_line_graphs: List[dgl.DGLGraph] = []
        batch_lattices: List[torch.Tensor] = []

        def _flush_batch() -> bool:
            if not batch_graphs:
                return True
            try:
                g_batch = dgl.batch(batch_graphs).to(self.device)
                lg_batch = dgl.batch(batch_line_graphs).to(self.device)
                lattice_batch = torch.stack(batch_lattices, dim=0).to(self.device)

                with torch.no_grad():
                    self._last_embedding = None
                    batch_output = self.model((g_batch, lg_batch, lattice_batch))

                if capture_embeddings:
                    if self._last_embedding is None:
                        raise RuntimeError("Failed to capture ALIGNN embedding from forward hook.")
                    latent = self._last_embedding.detach().cpu().numpy().astype(np.float32)
                    latent = np.atleast_2d(latent)
                    embeddings.append(latent)

                if capture_predictions:
                    batch_np = np.atleast_1d(batch_output.detach().cpu().numpy()).astype(np.float32)
                    predictions.append(batch_np.reshape(-1))

            except Exception as exc:
                warnings.warn(f"ALIGNN failed during batched processing: {exc}")
                return False
            finally:
                batch_graphs.clear()
                batch_line_graphs.clear()
                batch_lattices.clear()
            return True

        for idx, structure in enumerate(structures):
            try:
                g, lg, lattice = self._structure_to_graph(structure)
                batch_graphs.append(g)
                batch_line_graphs.append(lg)
                batch_lattices.append(lattice)

                if len(batch_graphs) >= self.batch_size:
                    if not _flush_batch():
                        return None
                elif (idx + 1) % 500 == 0:
                    print(f"   ALIGNN processed {idx + 1}/{len(structures)} structures")
            except Exception as exc:
                warnings.warn(f"ALIGNN failed on structure index {idx}: {exc}")
                return None

        if not _flush_batch():
            return None

        emb_array = (
            np.vstack(embeddings) if capture_embeddings and embeddings else np.empty((0, 0), dtype=np.float32)
        )
        pred_array = (
            np.concatenate(predictions) if capture_predictions and predictions else np.empty((0,), dtype=np.float32)
        )
        return emb_array, pred_array

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def transform(self, structures: List[Structure]) -> np.ndarray:
        """
        Transform a list of pymatgen Structures into feature vectors.

        Returns ALIGNN embeddings when available, otherwise Magpie features.
        """
        if not structures:
            return np.empty((0, self.embedding_dim or 0), dtype=np.float32)

        if self.model is not None:
            result = self._run_alignn_forward(structures, capture_embeddings=True, capture_predictions=False)
            if result is not None:
                embeddings, _ = result
                if embeddings.size > 0:
                    return embeddings
            warnings.warn("ALIGNN embedding extraction failed; using fallback features.")

        if not self.fallback_to_matminer:
            raise RuntimeError("ALIGNN embeddings unavailable and fallback disabled.")

        compositions = [struct.composition for struct in structures]
        features = np.array(
            self.fallback_featurizer.featurize_many(compositions, ignore_errors=True),
            dtype=np.float32,
        )

        if np.isnan(features).any():
            imputer = SimpleImputer(strategy="median")
            features = imputer.fit_transform(features)

        return features

    def predict(self, structures: List[Structure]) -> np.ndarray:
        """
        Predict formation energy using the pretrained ALIGNN model.

        Raises RuntimeError if ALIGNN is unavailable.
        """
        if not structures:
            return np.empty((0,), dtype=np.float32)

        if self.model is None:
            raise RuntimeError(
                "ALIGNN predictor not available. Install optional ALIGNN dependencies "
                "or provide a local checkpoint."
            )

        result = self._run_alignn_forward(structures, capture_embeddings=False, capture_predictions=True)
        if result is None:
            raise RuntimeError("ALIGNN prediction failed.")
        _, predictions = result
        return predictions

    def is_alignn_active(self) -> bool:
        """Return True if a pretrained ALIGNN model was loaded successfully."""
        return self.alignn_loaded and self.model is not None


def make_alignn_featurizer(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    use_pretrained: bool = True,
    embedding_dim: Optional[int] = None,
    alignn_model_name: str = "mp_e_form_alignn",
    fallback_to_matminer: bool = True,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> ALIGNNTransformer:
    """Factory helper mirroring existing featurizer pattern."""
    return ALIGNNTransformer(
        model_path=model_path,
        config_path=config_path,
        use_pretrained=use_pretrained,
        requested_embedding_dim=embedding_dim,
        alignn_model_name=alignn_model_name,
        fallback_to_matminer=fallback_to_matminer,
        device=device,
        batch_size=batch_size,
    )


def print_installation_instructions() -> None:
    """Prominent instructions for setting up ALIGNN dependencies."""
    print("\nTo enable ALIGNN embeddings install the optional packages:")
    print("  pip install alignn jarvis-tools dgl")
    print("Ensure your PyTorch/DGL build matches the local CUDA or CPU setup.")


if __name__ == "__main__":
    print("ALIGNN featurizer diagnostics")
    if _lazy_import_alignn():
        transformer = make_alignn_featurizer()
        dim = transformer.embedding_dim or "(not loaded)"
        print(f"ALIGNN successfully imported (embedding_dim={dim})")
    else:
        print_installation_instructions()
