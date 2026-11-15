"""Check the real reduced input length and gradients across CPU/MPS."""
import torch
from model_final import SpectralWavelengthPredictor


if __name__ == '__main__':
    assert torch.backends.mps.is_available(), 'Run on a Mac with MPS available'
    torch.manual_seed(44)
    model = SpectralWavelengthPredictor(num_labels=220, mode='classification').eval()
    x_cpu = torch.randn(2, 13931, requires_grad=True)
    y_cpu = model(x_cpu)
    y_cpu.sum().backward()
    x_mps = x_cpu.detach().to('mps').requires_grad_()
    model.zero_grad(set_to_none=True)
    y_mps = model.to('mps')(x_mps)
    y_mps.sum().backward()
    torch.testing.assert_close(y_mps.cpu(), y_cpu, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-5)
    print('PASS: CPU/MPS predictions and input gradients agree at length 13931')
