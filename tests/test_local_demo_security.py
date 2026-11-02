"""CPU regression checks of extracted production code; no GPU pipeline imports.

Run: python -m pytest -q tests/test_local_demo_security.py
Set HUNYUAN3D_TEST_EMBEDDINGS to a trusted directory containing uc_text_emb.pt
and uc_text_emb_2.pt to also check published embeddings.
Only GPU model initialization/rendering is stubbed. Filesystem copies and torch
serialization use the real libraries. Extraction avoids changing the demo's
existing eager model initialization solely to make it importable in tests.
"""
import argparse
import ast
import copy
import os
import pickle
import shutil
import sys
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Union
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def tree(path):
    return ast.parse((ROOT / path).read_text(), filename=str(ROOT / path))


def execute(nodes, namespace):
    code = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(code, "<production code>", "exec"), namespace)


def method_class(path, name, method, base, namespace):
    cls = next(n for n in tree(path).body if isinstance(n, ast.ClassDef) and n.name == name)
    cls = copy.deepcopy(cls)
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method]
    namespace = dict(namespace, Base=base)
    execute([cls], namespace)
    return namespace[name]


@pytest.mark.parametrize("argv,expected", [([], "127.0.0.1"), (["--server_name", "0.0.0.0"], "0.0.0.0"), (["--server_name", "192.0.2.1", "--use_lite"], "192.0.2.1")])
def test_launch_address(argv, expected):
    module = tree("app.py")
    start = next(i for i, n in enumerate(module.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser" for t in n.targets))
    end = next(i for i, n in enumerate(module.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets))
    namespace = {"argparse": argparse, "demo": Mock()}
    with patch.object(sys, "argv", ["app.py", *argv]):
        execute(module.body[start:end + 1], namespace)
    launch = next(n for n in ast.walk(module) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "demo" and n.func.attr == "launch")
    execute([ast.Expr(value=launch)], namespace)
    namespace["demo"].launch.assert_called_once_with(server_name=expected, server_port=8080)


def bake_output(tmp_path, material_name):
    # Execute the real output block, after GPU texture computation completes.
    func = next(n for n in tree("third_party/gen_baking.py").body if isinstance(n, ast.FunctionDef) and n.name == "back_projection")
    start = next(i for i, n in enumerate(func.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bake_dir" for t in n.targets))
    source = tmp_path / "模型 with spaces ; & $() ' directory"
    source.mkdir()
    obj = source / "mesh.obj"
    obj.write_text("test obj")
    if material_name:
        (source / material_name).write_text("test material")
    dest = tmp_path / "custom output ; & $()"
    namespace = dict(os=os, shutil=shutil, Image=Image, np=np, trimesh=Mock(), obj_file=str(obj), dst_dir=str(dest), init_texture=np.zeros((2, 2, 3)), init_texture_file="texture.png")
    with patch.object(os, "system", side_effect=AssertionError("shell must not be invoked")):
        execute(func.body[start:], namespace)
    assert (dest / "bake" / "mesh.obj").read_text() == "test obj"
    assert (dest / "bake" / material_name).read_text() == "test material"
    assert (dest / "bake" / "texture.png").exists()
    namespace["trimesh"].load_mesh.assert_called_once_with(str(dest / "bake" / "mesh.obj"))


@pytest.mark.parametrize("material_name", ["mesh.mtl", "material.mtl", "texture.mtl"])
def test_bake_paths_and_material_fallback(tmp_path, material_name):
    bake_output(tmp_path, material_name)


def test_bake_missing_material_is_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No material file found"):
        bake_output(tmp_path, None)


class BasePipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return SimpleNamespace()


def pipeline_class(torch_module=torch):
    return method_class("mvd/hunyuan3d_mvd_std_pipeline.py", "HunYuan3D_MVD_Std_Pipeline", "from_pretrained", BasePipeline, dict(torch=torch_module, os=os))


class RejectedObject:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        # A benign marker would be written if unrestricted unpickling were used.
        return eval, (f"open({self.marker!r}, 'w').write('unexpected unpickling')",)


def test_pipeline_tensor_roundtrip(tmp_path):
    expected = torch.arange(6).reshape(2, 3).float()
    torch.save(expected, tmp_path / "uc_text_emb.pt")
    torch.save(expected + 1, tmp_path / "uc_text_emb_2.pt")
    result = pipeline_class().from_pretrained(str(tmp_path))
    torch.testing.assert_close(result.uc_text_emb, expected)
    torch.testing.assert_close(result.uc_text_emb_2, expected + 1)


def test_official_embeddings():
    # Published embeddings retain CUDA device tags; remap only for CPU testing.
    directory = os.environ.get("HUNYUAN3D_TEST_EMBEDDINGS")
    if not directory:
        pytest.skip("Set HUNYUAN3D_TEST_EMBEDDINGS to check trusted published embeddings")
    source = Path(directory)
    cpu_torch = SimpleNamespace(load=lambda *args, **kwargs: torch.load(*args, map_location="cpu", **kwargs))
    result = pipeline_class(cpu_torch).from_pretrained(str(source))
    assert isinstance(result.uc_text_emb, torch.Tensor)
    assert isinstance(result.uc_text_emb_2, torch.Tensor)
    assert torch.isfinite(result.uc_text_emb).all()
    assert torch.isfinite(result.uc_text_emb_2).all()


@pytest.mark.parametrize("bad_file", ["uc_text_emb.pt", "uc_text_emb_2.pt"])
def test_pipeline_rejects_pickle_objects(tmp_path, bad_file):
    marker = tmp_path / "marker"
    for name in ["uc_text_emb.pt", "uc_text_emb_2.pt"]:
        torch.save(RejectedObject(marker) if name == bad_file else torch.ones(1), tmp_path / name)
    with pytest.raises(pickle.UnpicklingError):
        pipeline_class().from_pretrained(str(tmp_path))
    assert not marker.exists()


def dino_encoder():
    cls = method_class("svrm/ldm/modules/encoders/dinov2_mod.py", "FrozenDinoV2ImageEmbedder", "load_pretrained", object, dict(torch=torch, OrderedDict=OrderedDict))
    encoder = cls()
    encoder.model = torch.nn.Linear(2, 2)
    return encoder


def test_dino_state_dict_roundtrip(tmp_path):
    encoder = dino_encoder()
    expected = OrderedDict(weight=torch.ones(2, 2), bias=torch.zeros(2))
    weights = expected
    path = tmp_path / "encoder.pt"
    torch.save(weights, path)
    encoder.load_pretrained(str(path))
    for name, tensor in encoder.model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])


def test_dino_rejects_pickle_objects(tmp_path):
    marker = tmp_path / "marker"
    path = tmp_path / "bad.pt"
    torch.save(RejectedObject(marker), path)
    with pytest.raises(pickle.UnpicklingError):
        dino_encoder().load_pretrained(str(path))
    assert not marker.exists()


@pytest.mark.parametrize("malicious", [False, True])
def test_dino_download_loading(tmp_path, malicious):
    module = tree("svrm/ldm/modules/encoders/dinov2/hub/backbones.py")
    nodes = [n for n in module.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ["Weights", "_make_dinov2_model"]]
    model = torch.nn.Linear(2, 2)
    expected = OrderedDict(weight=torch.ones(2, 2), bias=torch.zeros(2))
    weights_dir = tmp_path / "model"
    weights_dir.mkdir()
    marker = tmp_path / "marker"
    torch.save(RejectedObject(marker) if malicious else expected, weights_dir / "model_pretrain.pth")
    vits = ModuleType("isolated_dino.models.vision_transformer")
    vits.vit_large = lambda **kwargs: model
    models = ModuleType("isolated_dino.models")
    models.vision_transformer = vits
    namespace = dict(__package__="isolated_dino.hub", torch=torch, Enum=Enum, Union=Union, _DINOV2_BASE_URL=tmp_path.as_uri(), _make_dinov2_model_name=lambda *args: "model")
    execute(nodes, namespace)
    with patch.dict(sys.modules, {"isolated_dino.models": models}), patch.object(torch.hub, "get_dir", return_value=str(tmp_path / "cache")):
        if malicious:
            with pytest.raises(pickle.UnpicklingError):
                namespace["_make_dinov2_model"]()
            assert not marker.exists()
        else:
            result = namespace["_make_dinov2_model"]()
            for name, tensor in result.state_dict().items():
                torch.testing.assert_close(tensor, expected[name])
