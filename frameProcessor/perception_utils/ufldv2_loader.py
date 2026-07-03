import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .lane_detection import segment_lanes_ufldv2


DEFAULT_UFLDV2_REPO_NAMES = [
    "Ultra-Fast-Lane-Detection-v2",
    "Ultra-Fast-Lane-Detection-V2",
    "ufldv2",
]


def _get_utility_file_directory() -> Path:
    return Path(__file__).resolve().parent


def _find_first_checkpoint(models_folder: Optional[Path] = None) -> Path:
    env_model_path = os.environ.get("UFLDV2_MODEL")

    if env_model_path:
        model_path = Path(env_model_path).expanduser().resolve()
        if model_path.exists():
            return model_path
        raise FileNotFoundError(
            f"UFLDV2_MODEL was set, but the file does not exist: {model_path}"
        )

    if models_folder is None:
        # Default expected project layout:
        # project/
        #   util/
        #   models/ufldv2/*.pth
        models_folder = _get_utility_file_directory().parent / "models" / "ufldv2"
    else:
        models_folder = Path(models_folder).expanduser().resolve()

    if not models_folder.exists():
        raise FileNotFoundError(
            "Could not find UFLDv2 model checkpoint folder.\n"
            f"Expected folder: {models_folder}\n\n"
            "Fix options:\n"
            "  1. Put your checkpoint in models/ufldv2/\n"
            "  2. Set UFLDV2_MODEL=/absolute/path/to/checkpoint.pth\n"
            "  3. Pass checkpoint_path=... to load_ufldv2_lane_detector()"
        )

    checkpoint_paths = []
    for extension in ("*.pth", "*.pt", "*.ckpt"):
        checkpoint_paths.extend(models_folder.glob(extension))

    checkpoint_paths = sorted(checkpoint_paths)

    if not checkpoint_paths:
        raise FileNotFoundError(
            "No UFLDv2 checkpoint found.\n"
            f"Searched folder: {models_folder}\n"
            "Expected a .pth, .pt, or .ckpt file."
        )

    return checkpoint_paths[0]


def _find_ufldv2_repo_path(repo_path: Optional[Path] = None) -> Path:
    candidates = []

    if repo_path is not None:
        candidates.append(Path(repo_path).expanduser())

    env_repo_path = os.environ.get("UFLDV2_REPO")
    if env_repo_path:
        candidates.append(Path(env_repo_path).expanduser())

    project_dir = _get_utility_file_directory().parent
    current_dir = Path.cwd()

    for repo_name in DEFAULT_UFLDV2_REPO_NAMES:
        candidates.append(project_dir / "external" / repo_name)
        candidates.append(project_dir / repo_name)
        candidates.append(current_dir / "external" / repo_name)
        candidates.append(current_dir / repo_name)

    for candidate in candidates:
        candidate = candidate.resolve()
        model_file = candidate / "model" / "model_culane.py"
        if model_file.exists():
            return candidate

    searched_paths = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the official Ultra-Fast-Lane-Detection-v2 repo.\n\n"
        "Expected to find: model/model_culane.py\n\n"
        f"Searched:\n{searched_paths}\n\n"
        "Fix options:\n"
        "  1. Put the repo in external/Ultra-Fast-Lane-Detection-v2/\n"
        "  2. Set UFLDV2_REPO=/absolute/path/to/Ultra-Fast-Lane-Detection-v2\n"
        "  3. Pass repo_path=... to load_ufldv2_lane_detector()"
    )


def _choose_ufldv2_config_path(
    repo_path: Path,
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
) -> Path:
    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()
        if config_path.exists():
            return config_path
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    env_config_path = os.environ.get("UFLDV2_CONFIG")
    if env_config_path:
        config_path = Path(env_config_path).expanduser().resolve()
        if config_path.exists():
            return config_path
        raise FileNotFoundError(
            f"UFLDV2_CONFIG was set, but the file does not exist: {config_path}"
        )

    checkpoint_name = checkpoint_path.name.lower()

    if "tusimple" in checkpoint_name:
        dataset_name = "tusimple"
    elif "curvelanes" in checkpoint_name or "curve" in checkpoint_name:
        dataset_name = "curvelanes"
    else:
        dataset_name = "culane"

    if "res34" in checkpoint_name or "resnet34" in checkpoint_name:
        backbone_name = "res34"
    else:
        backbone_name = "res18"

    guessed_config_path = repo_path / "configs" / f"{dataset_name}_{backbone_name}.py"

    if not guessed_config_path.exists():
        raise FileNotFoundError(
            "Could not find UFLDv2 config file.\n"
            f"Guessed config: {guessed_config_path}\n\n"
            "Fix options:\n"
            "  1. Rename your checkpoint to include dataset/backbone, e.g. culane_res18.pth\n"
            "  2. Set UFLDV2_CONFIG=/absolute/path/to/config.py\n"
            "  3. Pass config_path=... to load_ufldv2_lane_detector()"
        )

    return guessed_config_path


def _load_python_config(config_path: Path) -> types.SimpleNamespace:
    config_globals: Dict[str, Any] = {}
    config_text = config_path.read_text(encoding="utf-8")
    exec(compile(config_text, str(config_path), "exec"), config_globals)
    public_values = {
        key: value for key, value in config_globals.items() if not key.startswith("__")
    }
    return types.SimpleNamespace(**public_values)


def _add_ufldv2_anchors_to_config(config: types.SimpleNamespace) -> types.SimpleNamespace:
    dataset_name = str(config.dataset)

    if dataset_name == "CULane":
        config.row_anchor = np.linspace(0.42, 1.0, config.num_row)
        config.col_anchor = np.linspace(0.0, 1.0, config.num_col)
    elif dataset_name == "Tusimple":
        config.row_anchor = np.linspace(160, 710, config.num_row) / 720.0
        config.col_anchor = np.linspace(0.0, 1.0, config.num_col)
    elif dataset_name == "CurveLanes":
        config.row_anchor = np.linspace(0.4, 1.0, config.num_row)
        config.col_anchor = np.linspace(0.0, 1.0, config.num_col)
    else:
        raise ValueError(
            f"Unsupported UFLDv2 dataset: {dataset_name}. "
            "Expected CULane, Tusimple, or CurveLanes."
        )

    return config


def _initialize_ufldv2_weights(*models: torch.nn.Module) -> None:
    for model in models:
        _initialize_single_ufldv2_module(model)


def _initialize_single_ufldv2_module(module: torch.nn.Module) -> None:
    if isinstance(module, (list, tuple)):
        for child in module:
            _initialize_single_ufldv2_module(child)
        return

    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(0.0, std=0.01)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Module):
        for child in module.children():
            _initialize_single_ufldv2_module(child)


def _install_lightweight_ufldv2_common_module(repo_path: Path) -> None:
    if "utils.common" in sys.modules:
        return

    utils_module = types.ModuleType("utils")
    utils_module.__path__ = [str(repo_path / "utils")]
    sys.modules.setdefault("utils", utils_module)

    common_module = types.ModuleType("utils.common")
    common_module.initialize_weights = _initialize_ufldv2_weights
    sys.modules["utils.common"] = common_module


def _clean_ufldv2_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_key = key[len("module.") :]
        else:
            cleaned_key = key
        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict


class UFLDv2LaneDetector:
    """
    Plug-and-play UFLDv2 lane detector.

    Basic use:
        lane_detector = load_ufldv2_lane_detector()
        lane_mask = lane_detector.segment(image)
    """

    def __init__(
        self,
        network: torch.nn.Module,
        config: types.SimpleNamespace,
        device: torch.device,
    ) -> None:
        self.network = network
        self.config = config
        self.device = device
        self.input_width = int(config.train_width)
        self.input_height = int(config.train_height)
        self.crop_ratio = float(config.crop_ratio)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_height_before_crop = int(self.input_height / self.crop_ratio)
        resized_rgb = cv2.resize(
            frame_rgb,
            (self.input_width, resized_height_before_crop),
            interpolation=cv2.INTER_LINEAR,
        )
        cropped_rgb = resized_rgb[-self.input_height :, :, :]
        image_float = cropped_rgb.astype(np.float32) / 255.0
        image_float = (image_float - self.mean) / self.std
        tensor = np.transpose(image_float, (2, 0, 1))
        tensor = np.ascontiguousarray(tensor)
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    def predict(self, frame_bgr: np.ndarray) -> List[List[Tuple[int, int]]]:
        original_height, original_width = frame_bgr.shape[:2]
        input_tensor = self.preprocess_frame(frame_bgr)

        with torch.no_grad():
            predictions = self.network(input_tensor)

        return self.predictions_to_coordinates(
            predictions,
            original_image_width=original_width,
            original_image_height=original_height,
        )

    def segment(self, image: np.ndarray, thickness: int = 6, return_points: bool = False):
        return segment_lanes_ufldv2(
            lane_model=self,
            image=image,
            thickness=thickness,
            return_points=return_points,
        )

    def predictions_to_coordinates(
        self,
        predictions: Dict[str, torch.Tensor],
        original_image_width: int,
        original_image_height: int,
        local_width: int = 1,
    ) -> List[List[Tuple[int, int]]]:
        loc_row = predictions["loc_row"].detach().cpu()
        loc_col = predictions["loc_col"].detach().cpu()
        exist_row = predictions["exist_row"].detach().cpu()
        exist_col = predictions["exist_col"].detach().cpu()

        _, num_grid_row, num_cls_row, num_lane_row = loc_row.shape
        _, num_grid_col, num_cls_col, num_lane_col = loc_col.shape

        max_indices_row = loc_row.argmax(1)
        valid_row = exist_row.argmax(1)
        max_indices_col = loc_col.argmax(1)
        valid_col = exist_col.argmax(1)

        detected_lanes = []
        row_lane_indices = [1, 2]
        col_lane_indices = [0, 3]

        for lane_index in row_lane_indices:
            if lane_index >= num_lane_row:
                continue

            lane_points = []
            valid_count = int(valid_row[0, :, lane_index].sum().item())

            if valid_count > num_cls_row / 2:
                for anchor_index in range(num_cls_row):
                    lane_exists_here = int(valid_row[0, anchor_index, lane_index].item()) == 1
                    if not lane_exists_here:
                        continue

                    center_index = int(max_indices_row[0, anchor_index, lane_index].item())
                    start_index = max(0, center_index - local_width)
                    end_index = min(num_grid_row - 1, center_index + local_width)
                    nearby_indices = torch.arange(start_index, end_index + 1)
                    nearby_logits = loc_row[0, nearby_indices, anchor_index, lane_index]
                    refined_index = (
                        torch.softmax(nearby_logits, dim=0) * nearby_indices.float()
                    ).sum().item() + 0.5
                    x = refined_index / (num_grid_row - 1) * original_image_width
                    y = self.config.row_anchor[anchor_index] * original_image_height
                    point = (int(round(x)), int(round(y)))

                    if self._point_is_inside_image(point, original_image_width, original_image_height):
                        lane_points.append(point)

            if lane_points:
                detected_lanes.append(lane_points)

        for lane_index in col_lane_indices:
            if lane_index >= num_lane_col:
                continue

            lane_points = []
            valid_count = int(valid_col[0, :, lane_index].sum().item())

            if valid_count > num_cls_col / 4:
                for anchor_index in range(num_cls_col):
                    lane_exists_here = int(valid_col[0, anchor_index, lane_index].item()) == 1
                    if not lane_exists_here:
                        continue

                    center_index = int(max_indices_col[0, anchor_index, lane_index].item())
                    start_index = max(0, center_index - local_width)
                    end_index = min(num_grid_col - 1, center_index + local_width)
                    nearby_indices = torch.arange(start_index, end_index + 1)
                    nearby_logits = loc_col[0, nearby_indices, anchor_index, lane_index]
                    refined_index = (
                        torch.softmax(nearby_logits, dim=0) * nearby_indices.float()
                    ).sum().item() + 0.5
                    x = self.config.col_anchor[anchor_index] * original_image_width
                    y = refined_index / (num_grid_col - 1) * original_image_height
                    point = (int(round(x)), int(round(y)))

                    if self._point_is_inside_image(point, original_image_width, original_image_height):
                        lane_points.append(point)

            if lane_points:
                detected_lanes.append(lane_points)

        return detected_lanes

    @staticmethod
    def _point_is_inside_image(point: Tuple[int, int], image_width: int, image_height: int) -> bool:
        x, y = point
        return 0 <= x < image_width and 0 <= y < image_height


def load_ufldv2_lane_detector(
    repo_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    models_folder: Optional[Path] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> UFLDv2LaneDetector:
    """
    Load UFLDv2 as a plug-and-play lane detector.

    Expected default layout:
        project/
        ├── util/
        ├── models/ufldv2/culane_res18.pth
        └── external/Ultra-Fast-Lane-Detection-v2/
            ├── model/model_culane.py
            └── configs/culane_res18.py

    Optional environment variables:
        UFLDV2_REPO=/path/to/Ultra-Fast-Lane-Detection-v2
        UFLDV2_MODEL=/path/to/checkpoint.pth
        UFLDV2_CONFIG=/path/to/config.py
    """
    repo_path = _find_ufldv2_repo_path(repo_path)

    if checkpoint_path is None:
        checkpoint_path = _find_first_checkpoint(models_folder)
    else:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    config_path = _choose_ufldv2_config_path(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    if verbose:
        print(f"Using UFLDv2 repository: {repo_path}")
        print(f"Using UFLDv2 checkpoint: {checkpoint_path}")
        print(f"Using UFLDv2 config: {config_path}")

    config = _load_python_config(config_path)
    config = _add_ufldv2_anchors_to_config(config)

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    _install_lightweight_ufldv2_common_module(repo_path)

    try:
        from model.model_culane import parsingNet
    except Exception as error:
        raise RuntimeError(
            "Failed to import the official UFLDv2 model architecture.\n"
            f"Repo path: {repo_path}\n"
            "Expected import: from model.model_culane import parsingNet"
        ) from error

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if verbose:
        print(f"Using UFLDv2 device: {torch_device}")

    network = parsingNet(
        pretrained=False,
        backbone=str(config.backbone),
        num_grid_row=int(config.num_cell_row),
        num_cls_row=int(config.num_row),
        num_grid_col=int(config.num_cell_col),
        num_cls_col=int(config.num_col),
        num_lane_on_row=int(config.num_lanes),
        num_lane_on_col=int(config.num_lanes),
        use_aux=bool(config.use_aux),
        input_height=int(config.train_height),
        input_width=int(config.train_width),
        fc_norm=bool(getattr(config, "fc_norm", False)),
    ).to(torch_device)

    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    except Exception as error:
        raise RuntimeError(f"Failed to load UFLDv2 checkpoint: {checkpoint_path}") from error

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            "Unsupported UFLDv2 checkpoint format. Expected a state dict or a dict containing 'model' or 'state_dict'."
        )

    state_dict = _clean_ufldv2_state_dict_keys(state_dict)
    missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)

    if verbose and missing_keys:
        print(f"Warning: missing UFLDv2 model keys: {len(missing_keys)}")
    if verbose and unexpected_keys:
        print(f"Warning: unexpected UFLDv2 checkpoint keys: {len(unexpected_keys)}")

    network.eval()

    return UFLDv2LaneDetector(network=network, config=config, device=torch_device)


load_lane_model = load_ufldv2_lane_detector
