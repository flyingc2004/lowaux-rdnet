from collections import namedtuple
import os
from pathlib import Path

import torch
from torchvision import models


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _resolve_local_vgg19_weights() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    env_path = os.environ.get("ERRNET_VGG19_PATH")

    candidates = [
        Path(env_path).expanduser() if env_path else None,
        root / "checkpoints" / "vgg19-dcbb9e9d.pth",
        root / "checkpoints" / "vgg19.pth",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "vgg19-dcbb9e9d.pth",
    ]

    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return None


def _build_vgg19_features():
    local_weights = _resolve_local_vgg19_weights()
    if local_weights is not None:
        try:
            try:
                model = models.vgg19(weights=None)
            except TypeError:
                model = models.vgg19(pretrained=False)

            state = torch.load(local_weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = _strip_module_prefix(state)
            model.load_state_dict(state, strict=True)
            print(f"[VGG19] loaded local weights: {local_weights}")
            return model.features
        except Exception as exc:
            print(f"[VGG19] failed to load local weights {local_weights}: {exc}")
            print("[VGG19] fallback to torchvision pretrained download/cache")

    try:
        return models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    except AttributeError:
        return models.vgg19(pretrained=True).features


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        try:
            vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except AttributeError:
            vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = _build_vgg19_features()
        # self.slice1 = torch.nn.Sequential()
        # self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        # for x in range(2):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(2, 7):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        #indices = sorted(indices)
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in indices:
                out.append(X)
        
        return out

        # h_relu1 = self.slice1(X)
        # h_relu2 = self.slice2(h_relu1)
        # h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        # return out


if __name__ == '__main__':
    vgg = Vgg19()
    import ipdb; ipdb.set_trace()
