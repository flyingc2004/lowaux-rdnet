from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


class ModelAdapter:
    behavior: dict[str, object]

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _pad_to_multiple(input_tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = input_tensor.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    return F.pad(input_tensor, (0, pad_width, 0, pad_height), mode="replicate"), height, width


def _load_external_module(source_root: Path, module_name: str):
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    top_level = module_name.split(".", 1)[0]
    loaded = sys.modules.get(top_level)
    loaded_path = Path(getattr(loaded, "__file__", "") or "").resolve()
    if loaded is not None and source_root not in loaded_path.parents:
        raise RuntimeError(
            f"Cannot load {module_name} from {source_root}: {top_level} is already loaded "
            f"from {loaded_path}. Public benchmark workers must load one model each."
        )

    sys.path.insert(0, str(source_root))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(source_root))


class ERRNetAdapter(ModelAdapter):
    behavior = {
        "input": "shared RGB float32 [0,1]",
        "model_padding": "none",
        "output_alignment": "crop/pad to input size with replicate padding if required",
    }

    def __init__(self, checkpoint: Path, device: torch.device):
        from models.errnet_model import ERRNetModel

        gpu_ids = [device.index or 0] if device.type == "cuda" else []
        opt = SimpleNamespace(
            checkpoints_dir="./checkpoints",
            freeze_backbone=False,
            gpu_ids=gpu_ids,
            hyper=True,
            icnn_path=str(checkpoint),
            inet="errnet",
            init_type="edsr",
            isTrain=False,
            name="public_benchmark_errnet",
            no_verbose=True,
            output_mode=None,
            refiner_channels=None,
            refiner_dilations=None,
            refiner_mode=None,
            refiner_res_scale=None,
            reset_output_layer=False,
            resblock_dilations=None,
            resume=True,
            resume_epoch=None,
        )
        self.model = ERRNetModel()
        self.model.initialize(opt)
        self.model._eval()
        self.device = device

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        from models.errnet_model import _match_spatial_size

        input_tensor = input_tensor.to(self.device)
        self.model.input = input_tensor
        with torch.inference_mode():
            prediction = self.model.forward()
        return _match_spatial_size(prediction, input_tensor).detach().cpu()


class ERRNetFusionAdapter(ModelAdapter):
    behavior = {
        "input": "shared RGB float32 [0,1]",
        "model_padding": "none",
        "inference": "ERRNet predictions fused in float32 before clipping and metrics; identical VGG hypercolumns are shared",
        "output_alignment": "each branch matches input size before weighted fusion",
    }

    def __init__(
        self,
        first_checkpoint: Path,
        second_checkpoint: Path,
        alpha: float,
        device: torch.device,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"ERRNet fusion alpha must be in [0,1], got {alpha}")
        self.first = ERRNetAdapter(checkpoint=first_checkpoint, device=device)
        self.second = ERRNetAdapter(checkpoint=second_checkpoint, device=device)
        self.alpha = float(alpha)

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        first_model = self.first.model
        second_model = self.second.model
        can_share_hypercolumn = (
            first_model.vgg is not None
            and second_model.vgg is not None
            and first_model.opt.output_mode == "direct"
            and second_model.opt.output_mode == "direct"
            and first_model.refiner is None
            and second_model.refiner is None
        )
        if can_share_hypercolumn:
            from models.errnet_model import _match_spatial_size

            device_input = input_tensor.to(self.first.device)
            with torch.inference_mode():
                hypercolumn = first_model.vgg(device_input)
                height, width = device_input.shape[-2:]
                features = [
                    F.interpolate(
                        feature.detach(),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    for feature in hypercolumn
                ]
                model_input = torch.cat([device_input, *features], dim=1)
                first = _match_spatial_size(first_model.net_i(model_input), device_input)
                second = _match_spatial_size(second_model.net_i(model_input), device_input)
                return first.mul(self.alpha).add(second, alpha=1.0 - self.alpha).detach().cpu()

        first = self.first.predict(input_tensor)
        second = self.second.predict(input_tensor)
        return first.mul(self.alpha).add(second, alpha=1.0 - self.alpha)


class RDNetAdapter(ModelAdapter):
    behavior = {
        "input": "shared RGB float32 [0,1]",
        "model_padding": "replicate pad right/bottom to a multiple of 32",
        "output_alignment": "remove model padding and return input size",
    }

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        xreflection_root: Path,
        cls_model: Path,
        focal_model: Path,
    ):
        from run_rdnet_sweep import load_rdnet_network

        self.net = load_rdnet_network(
            xreflection_root=xreflection_root,
            checkpoint=checkpoint,
            cls_model=cls_model,
            focal_model=focal_model,
            device=device,
        )
        self.device = device

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        from run_rdnet_sweep import pad_to_multiple

        padded, height, width = pad_to_multiple(input_tensor, multiple=32)
        with torch.inference_mode():
            _, image_outputs = self.net(padded.to(self.device))
            prediction = image_outputs[-1][:, :3, :height, :width]
        return prediction.detach().cpu()


class DSRNetAdapter(ModelAdapter):
    behavior = {
        "input": "shared RGB float32 [0,1]",
        "model_source": "official DSRNet dsrnet_l_nature (Setting II)",
        "model_padding": "replicate pad right/bottom to a multiple of 32",
        "output": "transmission stream",
        "output_alignment": "remove model padding and return input size",
    }

    def __init__(self, checkpoint: Path, device: torch.device, source_root: Path):
        arch = _load_external_module(source_root, "models.arch")
        vgg = _load_external_module(source_root, "models.vgg")

        self.net = arch.dsrnet_l_nature().to(device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()

        self.vgg = vgg.Vgg19(requires_grad=False).to(device).eval()
        self.device = device

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        padded, height, width = _pad_to_multiple(input_tensor, multiple=32)
        padded = padded.to(self.device)
        with torch.inference_mode():
            transmission, _, _ = self.net(padded, self.vgg(padded))
        return transmission[:, :, :height, :width].detach().cpu()


class DSITAdapter(ModelAdapter):
    behavior = {
        "input": "shared RGB float32 [0,1]",
        "model_source": "official DSIT Large Setting II epoch 66",
        "model_padding": "replicate pad right/bottom to a multiple of 32",
        "inference": "official dsit_large with frozen Swin-Large O365 prior",
        "output": "transmission stream",
        "output_alignment": "remove model padding and return input size",
    }

    def __init__(
        self,
        checkpoint: Path,
        backbone_checkpoint: Path,
        device: torch.device,
        source_root: Path,
    ):
        arch = _load_external_module(source_root, "models.arch")
        args = SimpleNamespace(backbone_weight_path=str(backbone_checkpoint))
        self.net = arch.dsit_large(args).to(device)

        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        self.net.load_state_dict(checkpoint_data["weights"], strict=True)
        self.net.eval()
        self.checkpoint_epoch = int(checkpoint_data["epoch"])
        self.checkpoint_iterations = int(checkpoint_data["iterations"])
        self.device = device

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        padded, height, width = _pad_to_multiple(input_tensor, multiple=32)
        with torch.inference_mode():
            transmission, _, _ = self.net(padded.to(self.device))
        return transmission[:, :, :height, :width].detach().cpu()


class IBCLNAdapter(ModelAdapter):
    behavior = {
        "input": "shared RGB float32 [0,1]",
        "model_source": "official IBCLN Generator_drop pair",
        "model_padding": "replicate pad right/bottom to a multiple of 4",
        "inference": "three cascaded transmission/reflection iterations; T0=input, R0=0.1",
        "output": "final transmission estimate",
        "output_alignment": "remove model padding and return input size",
    }

    def __init__(
        self,
        transmission_checkpoint: Path,
        reflection_checkpoint: Path,
        device: torch.device,
        source_root: Path,
    ):
        networks = _load_external_module(source_root, "models.networks")
        self.net_t = networks.Generator_drop(9, 3, 64).to(device)
        self.net_r = networks.Generator_drop(9, 3, 64).to(device)
        self.net_t.load_state_dict(torch.load(transmission_checkpoint, map_location="cpu"), strict=True)
        self.net_r.load_state_dict(torch.load(reflection_checkpoint, map_location="cpu"), strict=True)
        self.net_t.eval()
        self.net_r.eval()
        self.device = device

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        padded, height, width = _pad_to_multiple(input_tensor, multiple=4)
        image = padded.to(self.device)
        batch, _, padded_height, padded_width = image.shape
        state_shape = (batch, 256, padded_height // 4, padded_width // 4)

        t_hidden = torch.zeros(state_shape, device=self.device, dtype=image.dtype)
        t_cell = torch.zeros_like(t_hidden)
        r_hidden = torch.zeros_like(t_hidden)
        r_cell = torch.zeros_like(t_hidden)
        transmission = image
        reflection = torch.full_like(image, 0.1)

        with torch.inference_mode():
            for _ in range(3):
                transmission, t_hidden, t_cell, _, _ = self.net_t(
                    torch.cat((image, transmission, reflection), dim=1),
                    t_hidden,
                    t_cell,
                )
                reflection, r_hidden, r_cell, _, _ = self.net_r(
                    torch.cat((image, transmission, reflection), dim=1),
                    r_hidden,
                    r_cell,
                )
        return transmission[:, :, :height, :width].detach().cpu()


def _required_file(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def build_adapter(model_spec: dict, device: torch.device, runtime: dict) -> ModelAdapter:
    adapter_type = model_spec["adapter"]

    if adapter_type == "errnet":
        return ERRNetAdapter(checkpoint=_required_file(model_spec["checkpoint"]), device=device)
    if adapter_type == "errnet_fusion":
        checkpoints = model_spec["checkpoints"]
        return ERRNetFusionAdapter(
            first_checkpoint=_required_file(checkpoints["first"]),
            second_checkpoint=_required_file(checkpoints["second"]),
            alpha=float(model_spec["alpha"]),
            device=device,
        )
    if adapter_type == "rdnet":
        return RDNetAdapter(
            checkpoint=_required_file(model_spec["checkpoint"]),
            device=device,
            xreflection_root=Path(runtime["xreflection_root"]).resolve(),
            cls_model=Path(runtime["cls_model"]).resolve(),
            focal_model=Path(runtime["focal_model"]).resolve(),
        )
    if adapter_type == "dsrnet":
        return DSRNetAdapter(
            checkpoint=_required_file(model_spec["checkpoint"]),
            device=device,
            source_root=Path(runtime["dsrnet_root"]),
        )
    if adapter_type == "dsit":
        checkpoints = model_spec["checkpoints"]
        return DSITAdapter(
            checkpoint=_required_file(checkpoints["model"]),
            backbone_checkpoint=_required_file(checkpoints["backbone"]),
            device=device,
            source_root=Path(runtime["dsit_root"]),
        )
    if adapter_type == "ibcln":
        checkpoints = model_spec["checkpoints"]
        return IBCLNAdapter(
            transmission_checkpoint=_required_file(checkpoints["transmission"]),
            reflection_checkpoint=_required_file(checkpoints["reflection"]),
            device=device,
            source_root=Path(runtime["ibcln_root"]),
        )
    raise ValueError(f"Unknown adapter type: {adapter_type}")
