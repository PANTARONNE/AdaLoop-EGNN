from torch import Tensor


def aggregate(values: Tensor, index: Tensor, size: int, reduction="sum") -> Tensor:
    result = values.new_zeros((size, values.size(-1))).index_add(0, index, values)
    if reduction == "mean":
        counts = values.new_zeros(size).index_add(0, index, values.new_ones(index.numel()))
        result = result / counts.clamp_min(1).unsqueeze(-1)
    elif reduction != "sum":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return result
