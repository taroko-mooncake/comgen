"""
Minimal example: query ONNX model for stable/metastable/unstable formation energy.
Model expects normed composition array as input.
"""

from pathlib import Path

from comgen import IonicComposition, SpeciesCollection
import onnx

model_path = Path(__file__).resolve().parent / "ehull_1040_bn.onnx"
onnx_model = onnx.load(str(model_path))

sps = SpeciesCollection.for_elements()
query = IonicComposition(sps)
query.category_prediction(onnx_model, 0)

print(query.get_next())
