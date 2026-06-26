import pytest
import torch
import torch.nn.functional as F

from ml_kernel_lab.kernel import triton_kernel


def torch_swiglu(x, gate):
    return F.silu(gate) * x


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape', [(16, 128), (32, 4, 1024), (2048, 512)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_match_torch_results(shape, dtype):
    x = torch.randn(*shape, dtype=dtype, device='cuda')
    gate = torch.randn(*shape, dtype=dtype, device='cuda')

    actual = triton_kernel.swiglu_fwd(x, gate)
    expected = torch_swiglu(x, gate)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
