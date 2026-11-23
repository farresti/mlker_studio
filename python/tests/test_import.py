import importlib.util
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
    from torch import nn
except ImportError:
    raise unittest.SkipTest("PyTorch is optional")
from mlker_graph import load_model, save_model, validate_model
from mlker_graph.pytorch import compile_model
from mlker_import import import_model
from mlker_import.sources import browse


class Sequence(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)
        self.second.weight = self.projection.weight

    def forward(self, x):
        return self.second(self.projection(x)) + self.projection(x)


class Dynamic(nn.Module):
    def forward(self, x):
        return x if x.sum() > 0 else -x


class Partial(nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamic = Dynamic()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.dynamic(x))


class Specializes(nn.Module):
    def forward(self, x):
        return x if isinstance(x, torch.fx.Proxy) else x + 1


class Unsupported(nn.Module):
    def forward(self, x):
        return torch.sin(torch.relu(x))


def load(cls, **extra):
    return import_model(dict(source="pytorch", identity=f"{cls.__module__}:{cls.__name__}", trusted=True,
                             inputs=[{"shape": [2, 4]}], verify=True) | extra)


def report(doc):
    return doc.data["model"]["provenance"]


class ImportTests(unittest.TestCase):
    def test_sequential_and_residual(self):
        self.assertEqual(report(load(Sequence))["verification"]["status"], "passed")
        source = ROOT.parent / "examples/import/residual.py"
        original = source.read_bytes()
        doc = load(Sequence, identity=str(source) + ":Residual")
        self.assertEqual(report(doc)["quality"], "fully editable")
        self.assertEqual(report(doc)["verification"]["seeds"], [17, 29])
        self.assertEqual(report(doc)["source_identity"], str(source) + ":Residual")
        self.assertEqual(source.read_bytes(), original)

    def test_checkpoint_round_trip_and_sharing(self):
        source = Shared().eval()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "original.pt"
            torch.save(source.state_dict(), checkpoint)
            before = checkpoint.read_bytes()
            doc = load(Shared, weights_dir=tmp, checkpoint=str(checkpoint))
            save_model(doc, root / "import.yaml")
            reopened = load_model(root / "import.yaml")
            self.assertEqual(doc.data, reopened.data)
            project = compile_model(reopened, root / "export")
            generated = runpy.run_path(str(project / "model.py"))["Model"]().eval()
            weights = report(doc)["weights"]
            self.assertEqual(weights["format"], "generated state_dict")
            generated.load_state_dict(torch.load(weights["path"], weights_only=True))
            x = torch.randn(3, 4)
            torch.testing.assert_close(generated(x), source(x))
            self.assertIs(generated.layer_projection.weight, generated.layer_second.weight)
            self.assertEqual(len(list(generated.children())), 2)
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertNotIn("!!binary", (root / "import.yaml").read_text())
            self.assertEqual(json.loads((project / "config.json").read_text())["import_provenance"], report(doc))

    def test_data_dependent_child_and_trace_failure(self):
        partial = load(Partial)
        self.assertEqual(report(partial)["strategy"], "fx_module_boundaries")
        self.assertEqual(report(partial)["quality"], "partially editable")
        self.assertEqual(report(partial)["opaque"], ["dynamic"])
        self.assertIn("FX tracing failed", report(partial)["warnings"][0])
        failed = load(Dynamic)
        self.assertEqual(report(failed)["quality"], "inspectable only")
        self.assertEqual(report(failed)["observed_outputs"], [{"dtype": "float32", "shape": [2, 4]}])
        self.assertFalse([d for d in validate_model(failed) if d.severity == "error"])

    def test_unsupported_function_and_composite(self):
        self.assertEqual(report(load(Unsupported))["opaque"], ["sin"])
        composite = load(Sequence, strategy="composite")
        self.assertEqual(report(composite)["strategy"], "module_tree")
        self.assertGreater(len(report(composite)["hierarchy"]), 1)
        self.assertEqual(report(composite)["verification"]["status"], "unverified")

    def test_mismatch_falls_back_and_preserves_evidence(self):
        doc = load(Specializes)
        self.assertEqual(report(doc)["quality"], "inspectable only")
        self.assertEqual(report(doc)["verification"]["status"], "failed")

    def test_permissions_and_state_dict_only(self):
        with self.assertRaises(PermissionError), patch("mlker_import.sources.importlib.import_module") as imp:
            load(Sequence, trusted=False)
        imp.assert_not_called()
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "model.pt"
            torch.save(Sequence(), file)
            import pickle
            with self.assertRaises(pickle.UnpicklingError):
                load(Sequence, checkpoint=str(file))

    @unittest.skipUnless(importlib.util.find_spec("timm"), "timm is optional")
    def test_real_timm_resnet_discovery_and_conversion_without_download(self):
        import timm
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with patch.object(timm, "create_model", wraps=timm.create_model) as make:
            rows = browse("resnet18", True)
            self.assertTrue(any(item["name"] == "resnet18" for item in rows["models"]))
            make.assert_not_called()
            doc = import_model({"source":"timm", "identity":"resnet18", "config":{"num_classes":3},
                                "inputs":[{"shape":[1,3,64,64]}], "verify":True})
            make.assert_called_once_with("resnet18", pretrained=False, num_classes=3)
        self.assertEqual(report(doc)["quality"], "fully editable")
        self.assertEqual(report(doc)["verification"]["status"], "passed")
        self.assertEqual(report(doc)["verification"]["outputs"][0]["shape"], [1,3])
