import torch
import numpy as np
import torchkbnufft as tkbn
from typing import OrderedDict
import yaml
# import wandb

def fft2c_np(data, dim=(-2, -1)):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data, axes=dim), norm='ortho', axes=dim), axes=dim)

def ifft2c_np(data, dim=(-2, -1)):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data, axes=dim), norm='ortho', axes=dim), axes=dim)

def fft2c_torch(data, dim=(-2, -1)):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(data, dim), norm='ortho', dim=dim), dim=dim)
    #return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(data, dim), dim=dim), dim=dim)

def ifft2c_torch(data, dim=(-2, -1)):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data, dim), norm='ortho', dim=dim), dim=dim)
    #return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data, dim), dim=dim), dim=dim)
    
def inufft2c_torch(data, ktraj, sens, dim):
    # JBM implemented
    inufft_ob = tkbn.KbNufftAdjoint(im_size=(dim,dim),)
    image = inufft_ob(data, ktraj, smaps=sens, norm='ortho')
#     print('inufft Not tested')
    return image
    #return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data, dim), dim=dim), dim=dim)

def nufft2c_torch(data, ktraj, sens, dim, device):
    # JBM implemented
    nufft_ob = tkbn.KbNufft(im_size=(dim,dim), device=device)
    kdata = nufft_ob(data, ktraj, smaps=sens, norm='ortho')
#     print('nufft Not tested')
    return kdata
    #return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data, dim), dim=dim), dim=dim)
    
def inufft3c_torch(data, ktraj, sens, dim, nslice, device):
    # JBM implemented
    inufft_ob = tkbn.KbNufftAdjoint(im_size=(dim,dim, nslice),device=device)
    image = inufft_ob(data, ktraj, smaps=sens, norm='ortho')
#     print('inufft Not tested')
    return image
    #return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data, dim), dim=dim), dim=dim)

def nufft3c_torch(data, ktraj, sens, dim, nslice, device):
    # JBM implemented
    nufft_ob = tkbn.KbNufft(im_size=(dim,dim, nslice),device=device)
    kdata = nufft_ob(data, ktraj, smaps=sens, norm='ortho')
#     print('nufft Not tested')
    return kdata
    #return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data, dim), dim=dim), dim=dim)


def fft1c_np(data, dim=-1):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data, axes=dim), norm='ortho', axis=dim), axes=dim)

def ifft1c_np(data, dim=-1):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(data, axes=dim), norm='ortho', axis=dim), axes=dim)

def fft1c_torch(data, dim=-1, norm='ortho'):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(data, dim=dim), norm=norm, dim=dim), dim=dim)

def ifft1c_torch(data, dim=-1, norm='ortho'):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(data, dim=dim), norm=norm, dim=dim), dim=dim)

def plot_coordinates(coodinates, title):
    # coordinates must have shape (2, ...)
    import matplotlib.pyplot as plt
    # vectorize the coordinates
    coodinates = np.reshape(coodinates, (2, -1)).T  # (npoints, 2)
    plt.figure()
    # keep the aspect ratio
    plt.axis('equal')
    plt.plot(coodinates[:, 0], coodinates[:, 1], 'x')
    plt.title(title)
    plt.show()


def compute_dcf(ktraj, im_size):
    ktraj = torch.from_numpy(ktraj).to('cuda') if not isinstance(ktraj, torch.Tensor) else ktraj
    if len(ktraj.shape) == 3:
        ktraj = torch.reshape(ktraj, (2, -1))
    else:
        ktraj = torch.reshape(ktraj, (ktraj.shape[0], 2, -1))
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size, num_iterations=5)
    return dcomp



class NUFFT(torch.nn.Module):
    r"""
    Base class for NUFFT and NUFFTAdjoint
    Attributes:
        config: dictionary, configuration parameters
            - im_size: list, image size
            - osf: float, oversampling factor
        nufft_op: torchkbnufft.KbNufft, NUFFT operator
    """
    def __init__(self, config):
        super(NUFFT, self).__init__()
        from warnings import filterwarnings
        filterwarnings("ignore") # ignores floor divide warnings

        im_size = [config['img_dim'], config['img_dim']]
        grid_size = [int(sz * config['osf']) for sz in im_size]
        self.nufft_op = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, n_shift=[i//2 for i in im_size])

    def forward(self, input, smaps, ktraj, dcomp):
        r"""
        Forward operation of NUFFT
        Args:
            input: torch tensor, input image data, (..., nc, nx, ny)
            smaps: torch tensor, sensitivity maps, (..., nc, nx, ny)
            ktraj: torch tensor, k-space trajectory with range [-pi, pi], (..., 2, nspokes*nFE)
            dcomp: torch tensor, density compensation, (..., 1, nspokes*nFE)
        Returns:
            out: torch tensor, output data, (..., nc, nspokes*nFE)
        """
        # get the first few dimensions of the input tensor and treat them as the batch dimensions
        batch_dims = input.shape[:-3]
        input = input.view(-1, *input.shape[-3:])   # (nbatch, nc, nx, ny)
        smaps = smaps.view(-1, *smaps.shape[-3:])   # (nbatch, nc, nx, ny)
        ktraj = ktraj.view(-1, *ktraj.shape[-2:])   # (nbatch, 2, nspokes*nFE)
        dcomp = dcomp.view(-1, *dcomp.shape[-2:])   # (nbatch, 1, nspokes*nFE)
        
        # The torchkbnufft operator takes batched 2D image data as input:
        # - input: image data, (nbatch, nc, nx, ny)
        # - ktraj: k-space trajectory, (nbatch, 2, nspokes*nFE)
        # - smaps: sensitivity maps, (nbatch, nc, nx, ny), nbatch can be 1 if the sensitivity maps are the same for all slices
        # - dcomp: density compensation, (nbatch, 1, nspokes*nFE)
        # The output is the k-space data, (nbatch, nc, nspokes*nFE)
        out = self.nufft_op(input, ktraj, smaps=smaps, norm='ortho') * torch.sqrt(dcomp)

        # reshape the output to the original shape
        out = out.view(*batch_dims, *out.shape[-2:])    # (..., nc, nspokes*nFE)
        return out


class NUFFTAdjoint(torch.nn.Module):
    r"""
    Base class for NUFFT and NUFFTAdjoint
    Attributes:
        config: dictionary, configuration parameters
            - im_size: list, image size
            - osf: float, oversampling factor
        adjnufft_op: torchkbnufft.KbNufftAdjoint, NUFFT adjoint operator
    """
    def __init__(self, config):
        super(NUFFTAdjoint, self).__init__()
        from warnings import filterwarnings
        filterwarnings("ignore")

        im_size = [config['img_dim'], config['img_dim']]
        grid_size = [int(sz * config['osf']) for sz in im_size]
        self.adjnufft_op = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, n_shift=[i//2 for i in im_size])
    
    def forward(self, input, smaps, ktraj, dcomp):
        r"""
        Adjoint operation of NUFFT
        Args:
            input: torch tensor, input kspace data, (..., nc, nspokes*nFE)
            smaps: torch tensor, sensitivity maps, (..., nc, nx, ny), should has same dimensions as input
            ktraj: torch tensor, k-space trajectory, (..., 2, nspokes*nFE)
            dcomp: torch tensor, density compensation, (..., 1, nspokes*nFE)
        Returns:
            out: torch tensor, output image data, (..., 1, nx, ny)
        """
        # get the first few dimensions of the input tensor and treat them as the batch dimensions
        batch_dims = input.shape[:-2]
        print('batch dims =', batch_dims)
        input = input.view(-1, *input.shape[-2:])   # (nbatch, nc, nSpokes*nFE)
        smaps = smaps.view(-1, *smaps.shape[-3:])   # (nbatch, nc, nx, ny)
        ktraj = ktraj.view(-1, *ktraj.shape[-2:])   # (nbatch, 2, nspokes*nFE)
        dcomp = dcomp.view(-1, *dcomp.shape[-2:])   # (nbatch, 1, nspokes*nFE)

        # the torchkbnufft operator takes batched 2D k-space data as input:
        # - input: k-space data, (nbatch, nc, nSpokes*nFE)
        # - ktraj: k-space trajectory, (nbatch, 2, nspokes*nFE)
        # - smaps: sensitivity maps, (nbatch, nc, nx, ny), nbatch can be 1 if the sensitivity maps are the same for all slices
        # - dcomp: density compensation, (nbatch, 1, nspokes*nFE)
        # the output is the image data, (nbatch, 1, nx, ny), 1 is the number of coils (already combined)
#         out = self.adjnufft_op(input * torch.sqrt(dcomp), ktraj, smaps=smaps, norm='ortho') # (nbatch, 1, nx, ny)
        out = self.adjnufft_op(input * dcomp, ktraj, smaps=smaps, norm='ortho') # (nbatch, 1, nx, ny)

        # reshape the output to the original shape
        out = out.view(*batch_dims, *out.shape[-3:])    # (..., 1, nx, ny)
        return out
    

def read_yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# def ndarray2video(data, range=(0,1), fps=10, format='gif'):
#     r"""
#     Convert a 3D numpy array to a wandb video object
#     Args:
#         data (np.ndarray): 3D numpy array, (nt, nx, ny) or (nt, nc, nx, ny) for color images
#         range (tuple): the normalized range of the data, (min, max)
#         fps (int): frame per second
#         format (str): the format of the video, 'gif' or 'mp4'
#     Returns:
#         video: wandb.Video object
#     """
#     if isinstance(data, torch.Tensor):
#         data = data.detach().cpu().numpy()
#     if data.dtype == np.complex64 or data.dtype == np.complex128:
#         data = np.abs(data)
#     if len(data.shape) == 3:
#         data = np.expand_dims(data, axis=1)
    
#     # normalize the data
#     data /= data.max()
#     data = (data - range[0]) / (range[1] - range[0])
#     data = np.clip(data, 0, 1)

#     # convert the data to uint8
#     data = (data * 255).astype(np.uint8)

#     # expand the color channel when it's 1
#     if data.shape[1] == 1:
#         data = np.repeat(data, 3, axis=1)

#     # create a video
#     video = wandb.Video(data, fps=fps, format=format)
#     return video



    
    

    