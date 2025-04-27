import time
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        # Split into real and imaginary parts
        real, imag = x.real, x.imag

        # Perform convolution
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)

        # Return complex output
        return torch.complex(real_out, imag_out)


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued 2D transposed convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_transpose_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_transpose_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        # Perform transposed convolution
        real_out = self.conv_transpose_real(x)
        imag_out = self.conv_transpose_imag(x)

        # Return complex output
        return torch.complex(real_out, imag_out)


def fftc2d(x):
    """Centered 2D FFT"""
    x_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))
    return x_fft


def ifftc2d(x):
    """Centered 2D IFFT"""
    x_ifft = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))
    return x_ifft


def rbf_activation(x, w, vmin=-150, vmax=150):
    """RBF activation function"""
    # Create centers evenly spaced between vmin and vmax
    num_centers = w.shape[1]
    centers = torch.linspace(vmin, vmax, num_centers, device=x.device)

    # Calculate RBF activations
    x_expanded = x.unsqueeze(-1)
    centers_expanded = centers.unsqueeze(0)

    # Calculate RBF values
    rbf_values = torch.exp(-(x_expanded - centers_expanded) ** 2 / 2)

    # Apply weights
    output = torch.matmul(rbf_values, w.t())
    return output


def ssim(output, target, L=None):
    """Calculate SSIM metric"""
    if L is None:
        L = torch.max(target) - torch.min(target)

    # Constants for stability
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # Calculate mean
    mu_x = F.avg_pool2d(output, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)

    # Calculate variance and covariance
    sigma_x = F.avg_pool2d(output ** 2, kernel_size=11, stride=1, padding=5) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, kernel_size=11, stride=1, padding=5) - mu_y ** 2
    sigma_xy = F.avg_pool2d(output * target, kernel_size=11, stride=1, padding=5) - mu_x * mu_y

    # Calculate SSIM
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return torch.mean(ssim_map)


class VnMriReconstructionCell(nn.Module):
    def __init__(self, params, options, num_filters):
        super(VnMriReconstructionCell, self).__init__()

        self.options = options
        self.num_filters = num_filters

        # Create parameters
        self.lambda_params = nn.Parameter(torch.ones(options['num_stages']) * params['lambda_init'])

        # Convolution kernels
        kernel_size = params['filter_size']
        self.k_real = nn.ParameterList([
            nn.Parameter(torch.randn(num_filters, 1, kernel_size, kernel_size) * 0.01)
            for _ in range(options['num_stages'])
        ])
        self.k_imag = nn.ParameterList([
            nn.Parameter(torch.randn(num_filters, 1, kernel_size, kernel_size) * 0.01)
            for _ in range(options['num_stages'])
        ])

        # Activation function weights
        num_centers = params['num_centers']
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(num_filters, num_centers) * 0.01)
            for _ in range(options['num_stages'])
        ])

        self.vmin = options['vmin']
        self.vmax = options['vmax']
        self.pad = options['pad']

    def mriForwardOpWithOS(self, u, coil_sens, sampling_mask):
        """MRI forward operator with oversampling"""
        # Add frequency encoding oversampling
        batch_size, height, width = u.shape
        pad_u = int(width * 0.25 + 1)
        pad_l = int(width * 0.25 - 1)

        u_pad = F.pad(u, (0, 0, pad_u, pad_l))
        u_pad = u_pad.unsqueeze(1)

        # Apply sensitivities
        coil_imgs = u_pad * coil_sens

        # Centered Fourier transform
        Fu = fftc2d(coil_imgs)

        # Apply sampling mask
        mask = sampling_mask.unsqueeze(1)
        kspace = Fu * mask

        return kspace

    def mriAdjointOpWithOS(self, f, coil_sens, sampling_mask):
        """MRI adjoint operator with oversampling"""
        # Variables to remove frequency encoding oversampling
        batch_size, coils, height, width = f.shape
        pad_u = int(width * 0.25 + 1)
        pad_l = int(width * 0.25 - 1)

        # Apply mask and perform inverse centered Fourier transform
        mask = sampling_mask.unsqueeze(1)
        Finv = ifftc2d(f * mask)

        # Multiply coil images with sensitivities and sum up over channels
        img = torch.sum(Finv * torch.conj(coil_sens), dim=1)[:, pad_u:-pad_l, :]

        return img

    def mriForwardOp(self, u, coil_sens, sampling_mask):
        """MRI forward operator"""
        # Apply sensitivities
        coil_imgs = u.unsqueeze(1) * coil_sens

        # Centered Fourier transform
        Fu = fftc2d(coil_imgs)

        # Apply sampling mask
        mask = sampling_mask.unsqueeze(1)
        kspace = Fu * mask

        return kspace

    def mriAdjointOp(self, f, coil_sens, sampling_mask):
        """MRI adjoint operator"""
        # Apply mask and perform inverse centered Fourier transform
        mask = sampling_mask.unsqueeze(1)
        Finv = ifftc2d(f * mask)

        # Multiply coil images with sensitivities and sum up over channels
        img = torch.sum(Finv * torch.conj(coil_sens), dim=1)

        return img

    def forward(self, u_list, f, coil_sens, sampling_mask, t):
        """Forward pass through one VN cell"""
        # Get the current image
        u_t_1 = u_list[t]

        # Get parameters for current step
        lambdaa = self.lambda_params[t]
        w = self.w[t]
        k_real = self.k_real[t]
        k_imag = self.k_imag[t]

        # Pad the image to avoid problems at the border
        batch_size, height, width = u_t_1.shape
        u_p = F.pad(u_t_1.unsqueeze(-1), (0, 0, self.pad, self.pad, self.pad, self.pad), mode='reflect')

        # Split into real and imaginary parts and perform convolution
        u_p_real = u_p.real
        u_p_imag = u_p.imag

        u_k_real = F.conv2d(u_p_real, k_real, padding='same')
        u_k_imag = F.conv2d(u_p_imag, k_imag, padding='same')

        # Add up the convolution results
        u_k = u_k_real + u_k_imag

        # Apply activation function
        shape = u_k.shape
        u_k_reshaped = u_k.view(shape[0], shape[1], -1).transpose(1, 2)

        # Apply RBF activation
        f_u_k = rbf_activation(u_k_reshaped, w, vmin=self.vmin, vmax=self.vmax)
        f_u_k = f_u_k.transpose(1, 2).view(shape)

        # Perform transpose convolution
        u_k_T_real = F.conv_transpose2d(f_u_k, k_real, padding=self.pad)
        u_k_T_imag = F.conv_transpose2d(f_u_k, k_imag, padding=self.pad)

        # Rebuild complex image
        u_k_T = torch.complex(u_k_T_real, u_k_T_imag)

        # Remove padding
        Ru = u_k_T[:, :, self.pad:-self.pad, self.pad:-self.pad].squeeze(1)

        # Normalize regularizer by number of filters
        Ru = Ru / self.num_filters

        # Define dataterm operators according to sampling pattern
        if self.options.get('sampling_pattern') == 'cartesian':
            forwardOp = self.mriForwardOp
            adjointOp = self.mriAdjointOp
        else:  # default to cartesian with oversampling
            forwardOp = self.mriForwardOpWithOS
            adjointOp = self.mriAdjointOpWithOS

        # Build dataterm
        Au = forwardOp(u_t_1, coil_sens, sampling_mask)
        At_Au_f = adjointOp(Au - f, coil_sens, sampling_mask)
        Du = At_Au_f * lambdaa

        # Gradient step
        u_t = u_t_1 - Ru - Du

        return u_t


class VariationalNetwork(nn.Module):
    def __init__(self, cell, num_stages):
        super(VariationalNetwork, self).__init__()
        self.cell = cell
        self.num_stages = num_stages

    def forward(self, u_0, f, coil_sens, sampling_mask):
        """Forward pass through the entire variational network"""
        # Initialize list of outputs
        u_all = [u_0]

        # Loop through stages
        for t in range(self.num_stages):
            u_t = self.cell(u_all, f, coil_sens, sampling_mask, t)
            u_all.append(u_t)

        return u_all


class MriDataset(Dataset):
    def __init__(self, data_config, split='train'):
        self.config = data_config
        self.split = split

        # Load file paths
        self.file_paths = self._load_file_paths()

    def _load_file_paths(self):
        # This would need to be implemented according to your data structure
        # For now, we'll just return a placeholder
        return []

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load and preprocess data from file_paths[idx]
        # This would need to be implemented according to your data format

        # Placeholder return
        # In a real implementation, load your actual MRI data
        u = torch.randn(256, 256, dtype=torch.complex64)
        target = torch.randn(256, 256, dtype=torch.complex64)
        coil_sens = torch.randn(8, 256, 256, dtype=torch.complex64)
        sampling_mask = torch.ones(256, 256)
        f = torch.randn(8, 256, 256, dtype=torch.complex64)

        return {
            'u': u,
            'target': target,
            'coil_sens': coil_sens,
            'sampling_mask': sampling_mask,
            'f': f
        }


def load_yaml(config_path, keys):
    """Load YAML configuration file and extract specified keys"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    result = {}
    for key in keys:
        if key in config:
            result[key] = config[key]
        else:
            result[key] = {}

    if len(keys) == 1:
        return result[keys[0]]
    return [result[key] for key in keys]


def main():
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_config', type=str, default='./configs/training.yaml')
    parser.add_argument('--network_config', type=str, default='./configs/mri_vn.yaml')
    parser.add_argument('--data_config', type=str, default='./configs/data.yaml')
    parser.add_argument('--global_config', type=str, default='./configs/global.yaml')

    args = parser.parse_args()

    # Load the configs
    network_config, reg_config = load_yaml(args.network_config, ['network', 'reg'])
    checkpoint_config, optimizer_config = load_yaml(args.training_config, ['checkpoint_config', 'optimizer_config'])
    data_config = load_yaml(args.data_config, ['data_config'])
    global_config = load_yaml(args.global_config, ['global_config'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the output locations
    base_name = os.path.basename(args.network_config).split('.')[0]
    suffix = base_name + '_' + time.strftime('%Y-%m-%d--%H-%M-%S')
    log_dir = os.path.join(checkpoint_config['log_dir'], suffix)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create dataset and dataloader
    train_dataset = MriDataset(data_config, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=global_config.get('batch_size', 1),
        shuffle=True,
        num_workers=global_config.get('data_num_threads', 4)
    )

    # Setup network
    filter_params = {
        'filter_size': reg_config['filter1']['filter_size'],
        'lambda_init': network_config.get('lambda_init', 0.1),
        'num_centers': reg_config['activation1']['num_centers']
    }

    network_options = {
        'num_stages': network_config['num_stages'],
        'vmin': network_config.get('vmin', -150),
        'vmax': network_config.get('vmax', 150),
        'pad': network_config.get('pad', 4),
        'sampling_pattern': data_config.get('sampling_pattern', 'cartesian_with_os')
    }

    # Create model
    vn_cell = VnMriReconstructionCell(
        filter_params,
        network_options,
        num_filters=reg_config['filter1']['num_filters']
    )

    vn_network = VariationalNetwork(
        cell=vn_cell,
        num_stages=network_config['num_stages']
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        vn_network.parameters(),
        lr=optimizer_config.get('learning_rate', 1e-4)
    )

    # Create tensorboard writer
    writer = SummaryWriter(log_dir)

    # Training loop
    iter_per_epoch = len(train_loader)
    start_time = time.time()

    for epoch in range(optimizer_config.get('max_iter', 100) + 1):
        epoch_loss = 0
        epoch_rmse = 0
        epoch_ssim = 0

        for batch_idx, data in enumerate(train_loader):
            # Get batch data
            u = data['u'].to(device)
            target = data['target'].to(device)
            coil_sens = data['coil_sens'].to(device)
            sampling_mask = data['sampling_mask'].to(device)
            f = data['f'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            u_all = vn_network(u, f, coil_sens, sampling_mask)
            u_T = u_all[-1]

            # Calculate loss
            target_abs = torch.abs(target)
            output_abs = torch.abs(u_T)

            # MSE loss
            loss = F.mse_loss(output_abs, target_abs)

            # Calculate RMSE
            denominator = torch.sum(torch.abs(target) ** 2, dim=(1, 2))
            nominator = torch.sum(torch.abs(u_T - target) ** 2, dim=(1, 2))
            rmse = torch.mean(torch.sqrt(nominator / denominator))

            # Calculate SSIM
            output_abs_expanded = output_abs.unsqueeze(1)
            target_abs_expanded = target_abs.unsqueeze(1)
            L = torch.max(target_abs_expanded) - torch.min(target_abs_expanded)
            ssim_val = ssim(output_abs_expanded, target_abs_expanded, L=L)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_rmse += rmse.item()
            epoch_ssim += ssim_val.item()

        # Average metrics
        epoch_loss /= len(train_loader)
        epoch_rmse /= len(train_loader)
        epoch_ssim /= len(train_loader)

        # Log metrics
        print(
            f"Epoch {epoch}/{optimizer_config.get('max_iter')}, Loss: {epoch_loss:.6f}, RMSE: {epoch_rmse:.6f}, SSIM: {epoch_ssim:.6f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('RMSE/train', epoch_rmse, epoch)
        writer.add_scalar('SSIM/train', epoch_ssim, epoch)

        # Save checkpoints
        if (epoch % checkpoint_config.get('save_modulo', 10) == 0) or epoch == optimizer_config.get('max_iter'):
            torch.save({
                'epoch': epoch,
                'model_state_dict': vn_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt'))

            # Log images
            if len(u_all) > 0:
                for i in range(min(3, u.size(0))):  # Log up to 3 images
                    writer.add_image(f'Input/{i}', torch.abs(u[i]).unsqueeze(0), epoch)
                    writer.add_image(f'Output/{i}', torch.abs(u_T[i]).unsqueeze(0), epoch)
                    writer.add_image(f'Target/{i}', torch.abs(target[i]).unsqueeze(0), epoch)

    print(f'Elapsed training time: {time.time() - start_time:.2f}s')
    writer.close()


if __name__ == '__main__':
    main()