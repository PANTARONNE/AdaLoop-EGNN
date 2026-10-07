from dataclasses import replace

import pytest
import torch

from eqcompute.contracts import PropertyContext
from eqcompute.models.iterative import build_egnn
from eqcompute.models.trace import TraceSpec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_all_step_backward_and_symmetry(config, batch):
    model = build_egnn(config.model, "mu").cuda()
    graph = batch.graph.to("cuda")
    context = PropertyContext("mu")
    spec = TraceSpec(predictions=True)
    out = model(graph, context, trace=spec)
    loss = torch.stack([p.square().mean() for p in out.predictions_by_depth.values()]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], device="cuda")
    changed = replace(graph, positions=graph.positions @ rotation.T + 2.5)
    with torch.inference_mode():
        transformed = model(changed, context, trace=spec)
    for k in out.predictions_by_depth:
        torch.testing.assert_close(
            out.predictions_by_depth[k], transformed.predictions_by_depth[k], atol=1e-5, rtol=1e-5
        )
