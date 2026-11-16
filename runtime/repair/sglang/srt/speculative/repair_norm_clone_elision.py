"""Optional fast path for ContextRepair's CUDA, no-residual RMSNorm query."""

import torch
from torch.nn.modules import module as nn_module

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.srt.layers.layernorm import RMSNorm, is_batch_invariant_mode_enabled


def can_use_original(norm, hidden):
    if type(norm) is not RMSNorm or type(hidden) is not torch.Tensor:
        return False
    forward = norm._forward_method
    if (
        getattr(forward, "__func__", None) is not RMSNorm.forward_cuda
        or getattr(forward, "__self__", None) is not norm
        or getattr(norm.forward, "__func__", None) is not RMSNorm.forward
        or get_fused_op_backend() is not None
        or norm.is_torch_compile
        or norm._compiled_call_impl is not None
        or torch.compiler.is_compiling()
        or (torch.is_grad_enabled() and hidden.requires_grad)
        or is_batch_invariant_mode_enabled()
        or norm.variance_size_override is not None
        or norm.cast_x_before_out_mul
        or norm._forward_pre_hooks
        or norm._forward_hooks
        or nn_module._global_forward_pre_hooks
        or nn_module._global_forward_hooks
    ):
        return False
    weight = norm.weight
    return (
        hidden.is_cuda
        and hidden.ndim == 2
        and hidden.shape[0] > 0
        and hidden.shape[1] == 4096
        and hidden.stride() == (4096, 1)
        and hidden.dtype == torch.bfloat16
        and hidden.data_ptr() % 256 == 0
        and weight.device == hidden.device
        and weight.dtype == hidden.dtype
        and weight.shape == (4096,)
        and weight.is_contiguous()
    )


def normalize_query(norm, hidden, audit):
    """Keep the same RMSNorm dispatch and allocate a distinct output as usual."""
    if can_use_original(norm, hidden):
        audit["clone_elided_calls"] += 1
        return norm(hidden)
    audit["clone_retained_calls"] += 1
    return norm(hidden.clone())
