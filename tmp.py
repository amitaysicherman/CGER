import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBFLayer(nn.Module):
    """Radial Basis Function Layer for activation functions"""

    def __init__(self, num_rbf=31, max_val=150):
        super(RBFLayer, self).__init__()
        # Create centers for RBFs evenly spaced
        centers = torch.linspace(-max_val, max_val, num_rbf)
        self.register_buffer('centers', centers)

        # Sigma controls the width of RBFs
        sigma = 2 * max_val / (num_rbf - 1)
        self.register_buffer('sigma', torch.tensor(sigma))

        # Weights for the RBF combination - these are learnable
        self.weights = nn.Parameter(torch.zeros(num_rbf))

    def forward(self, x):
        # Compute RBF values for each center
        rbf_values = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.sigma) ** 2)

        # Weighted sum of RBF values
        return torch.matmul(rbf_values, self.weights)


class GradientStep(nn.Module):
    """One step/layer of the variational network"""

    def __init__(self, num_filters=48, filter_size=11, num_rbf=31, max_val=150):
        super(GradientStep, self).__init__()

        # Create filters for real and imaginary parts
        self.filters_real = nn.Parameter(torch.randn(num_filters, 1, filter_size, filter_size))
        self.filters_imag = nn.Parameter(torch.randn(num_filters, 1, filter_size, filter_size))

        # Create RBF activation functions
        self.activations = nn.ModuleList([RBFLayer(num_rbf, max_val) for _ in range(num_filters)])

        # Data term weight
        self.lambda_data = nn.Parameter(torch.tensor(0.1))

        # Initialize filters to be zero-mean and unit-norm
        self._init_filters()

    def _init_filters(self):
        # Zero-mean constraint
        with torch.no_grad():
            self.filters_real.data = self.filters_real.data - self.filters_real.data.mean(dim=(1, 2, 3), keepdim=True)
            self.filters_imag.data = self.filters_imag.data - self.filters_imag.data.mean(dim=(1, 2, 3), keepdim=True)

            # Unit-norm constraint
            norm = torch.sqrt(
                torch.sum(self.filters_real.data ** 2 + self.filters_imag.data ** 2, dim=(1, 2, 3), keepdim=True))
            self.filters_real.data = self.filters_real.data / norm
            self.filters_imag.data = self.filters_imag.data / norm

    def forward(self, u_real, u_imag, A_op, AT_op, f):
        """
        Single gradient step

        Args:
            u_real, u_imag: Current image estimate (real and imaginary parts)
            A_op: Forward operator (undersampling, FFT, coil sensitivities)
            AT_op: Adjoint operator
            f: Measured k-space data
        """
        # Data consistency term: lambda * A*(Au - f)
        Au = A_op(u_real, u_imag)
        residual = [Au[i] - f[i] for i in range(len(Au))]
        data_grad_real, data_grad_imag = AT_op(residual)

        # Apply lambda
        data_grad_real = self.lambda_data * data_grad_real
        data_grad_imag = self.lambda_data * data_grad_imag

        # Regularization term
        reg_grad_real = torch.zeros_like(u_real)
        reg_grad_imag = torch.zeros_like(u_imag)

        # Apply filters and activations
        for i in range(len(self.activations)):
            # Apply filters (convolutions)
            f_real = F.conv2d(u_real, self.filters_real[i:i + 1], padding=self.filters_real.shape[2] // 2)
            f_imag = F.conv2d(u_imag, self.filters_imag[i:i + 1], padding=self.filters_imag.shape[2] // 2)

            # Compute filter response magnitude
            f_mag = torch.sqrt(f_real ** 2 + f_imag ** 2 + 1e-10)

            # Apply activation function
            act_out = self.activations[i](f_mag)

            # Compute gradient contribution
            weight = act_out / f_mag

            # Apply transposed convolution (adjoint of the filter)
            filters_real_rot = torch.rot90(self.filters_real[i:i + 1], 2, dims=[2, 3])
            filters_imag_rot = torch.rot90(self.filters_imag[i:i + 1], 2, dims=[2, 3])

            reg_grad_real += F.conv2d(weight * f_real, filters_real_rot, padding=self.filters_real.shape[2] // 2)
            reg_grad_imag += F.conv2d(weight * f_imag, filters_imag_rot, padding=self.filters_imag.shape[2] // 2)

        # Update step
        u_real_new = u_real - (reg_grad_real + data_grad_real)
        u_imag_new = u_imag - (reg_grad_imag + data_grad_imag)

        return u_real_new, u_imag_new


class VariationalNetwork(nn.Module):
    """Full Variational Network for MRI Reconstruction"""

    def __init__(self, num_steps=10, num_filters=48, filter_size=11, num_rbf=31, max_val=150):
        super(VariationalNetwork, self).__init__()

        # Create gradient steps
        self.steps = nn.ModuleList([
            GradientStep(num_filters, filter_size, num_rbf, max_val)
            for _ in range(num_steps)
        ])

    def forward(self, kspace_data, mask, sensitivity_maps):
        """
        Args:
            kspace_data: Undersampled k-space data [batch, coils, height, width, 2]
            mask: Sampling mask [batch, 1, height, width, 1]
            sensitivity_maps: Coil sensitivity maps [batch, coils, height, width, 2]
        """
        # Create forward and adjoint operators
        A_op = lambda ur, ui: self._forward_op(ur, ui, sensitivity_maps, mask)
        AT_op = lambda r: self._adjoint_op(r, sensitivity_maps, mask)

        # Get initial estimate (zero-filled reconstruction)
        u_real, u_imag = self._initial_estimate(kspace_data, sensitivity_maps, mask)

        # Apply gradient steps
        for step in self.steps:
            u_real, u_imag = step(u_real, u_imag, A_op, AT_op, kspace_data)

        return u_real, u_imag

    def _forward_op(self, u_real, u_imag, sensitivity_maps, mask):
        """Forward operator A: image -> k-space"""
        batch_size, num_coils, h, w, _ = sensitivity_maps.shape

        # Apply sensitivity maps
        coil_images_real = u_real * sensitivity_maps[..., 0] - u_imag * sensitivity_maps[..., 1]
        coil_images_imag = u_real * sensitivity_maps[..., 1] + u_imag * sensitivity_maps[..., 0]

        # Reshape for FFT
        coil_images_complex = torch.complex(coil_images_real, coil_images_imag)

        # Apply FFT
        kspace_complex = torch.fft.fft2(coil_images_complex, dim=(-2, -1))

        # Apply mask
        kspace_complex = kspace_complex * mask

        # Convert back to real/imag pairs
        kspace_real = kspace_complex.real
        kspace_imag = kspace_complex.imag

        return [torch.stack([kspace_real[:, i], kspace_imag[:, i]], dim=-1) for i in range(num_coils)]

    def _adjoint_op(self, kspace_residuals, sensitivity_maps, mask):
        """Adjoint operator A*: k-space -> image"""
        batch_size, num_coils, h, w, _ = sensitivity_maps.shape

        # Separate real and imaginary parts
        kspace_residuals_complex = [torch.complex(kr[..., 0], kr[..., 1]) for kr in kspace_residuals]
        kspace_residuals_complex = torch.stack(kspace_residuals_complex, dim=1)

        # Apply mask
        kspace_residuals_complex = kspace_residuals_complex * mask

        # Apply inverse FFT
        image_residuals_complex = torch.fft.ifft2(kspace_residuals_complex, dim=(-2, -1))

        # Apply sensitivity maps (conjugate)
        image_real_sum = torch.sum(
            image_residuals_complex.real * sensitivity_maps[..., 0] +
            image_residuals_complex.imag * sensitivity_maps[..., 1],
            dim=1
        )
        image_imag_sum = torch.sum(
            image_residuals_complex.imag * sensitivity_maps[..., 0] -
            image_residuals_complex.real * sensitivity_maps[..., 1],
            dim=1
        )

        return image_real_sum, image_imag_sum

    def _initial_estimate(self, kspace_data, sensitivity_maps, mask):
        """Zero-filled reconstruction"""
        # Combine k-space data from all coils
        kspace_complex = [torch.complex(kd[..., 0], kd[..., 1]) for kd in kspace_data]
        kspace_complex = torch.stack(kspace_complex, dim=1)

        # Apply inverse FFT
        image_complex = torch.fft.ifft2(kspace_complex, dim=(-2, -1))

        # Apply sensitivity maps (conjugate)
        image_real_sum = torch.sum(
            image_complex.real * sensitivity_maps[..., 0] +
            image_complex.imag * sensitivity_maps[..., 1],
            dim=1, keepdim=True
        )
        image_imag_sum = torch.sum(
            image_complex.imag * sensitivity_maps[..., 0] -
            image_complex.real * sensitivity_maps[..., 1],
            dim=1, keepdim=True
        )

        return image_real_sum, image_imag_sum


# Example of how to use the model
def main():
    # Create model
    model = VariationalNetwork()

    # Example data (would normally be loaded from MRI files)
    batch_size = 1
    num_coils = 15  # Example: 15-channel knee coil
    height, width = 320, 320  # Example image dimensions

    # Create dummy input data
    kspace_data = [torch.randn(batch_size, num_coils, height, width, 2) for _ in range(num_coils)]
    mask = torch.zeros(batch_size, 1, height, width, 1)
    mask[..., ::4, :, :] = 1  # Example: regularly undersampled by factor of 4
    sensitivity_maps = torch.randn(batch_size, num_coils, height, width, 2)

    # Forward pass
    u_real, u_imag = model(kspace_data, mask, sensitivity_maps)

    # Compute magnitude image
    magnitude = torch.sqrt(u_real ** 2 + u_imag ** 2)

    print(f"Output shape: {magnitude.shape}")


if __name__ == "__main__":
    main()