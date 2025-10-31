from typing import List
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import numpy as np
import torch.nn as nn
import tinycudann as tcnn
from utils import fft2c_torch, ifft2c_torch, fft1c_torch, ifft1c_torch, nufft2c_torch


class WeightedKspaceMSELoss(nn.Module):
    def __init__(self, im_size, device='cuda'):
        super(WeightedKspaceMSELoss, self).__init__()
        # Calculate distance from the center for each point in the Cartesian grid
        y = torch.linspace(-1, 1, im_size[0], device=device)
        z = torch.linspace(-1, 1, im_size[1], device=device)
        yy, zz = torch.meshgrid(y, z)
        dist = torch.sqrt(yy**2 + zz**2)
        self.weight = (dist + 1).clone().detach()  # Avoid division by zero
        self.mse = nn.MSELoss()

    def forward(self, img_pred, kdata, csm, undersample_mask):
        
        coil_img = img_pred[None]*csm.squeeze()[:,:,:,None]
        coil_kspace = fft2c_torch(coil_img.permute(3,0,1,2))
        coil_kspace = coil_kspace*undersample_mask.squeeze()
        kdata = (kdata*undersample_mask).squeeze()

        self.weight.expand_as(coil_kspace)
        coil_kspace = torch.mul(coil_kspace,self.weight)
        kdata = torch.mul(kdata,self.weight)

        loss = self.mse(torch.view_as_real(coil_kspace), torch.view_as_real(kdata))

        return loss

    
class UnWeightedKspaceMSELossNonCart(nn.Module):
    def __init__(self, im_size, device='cuda'):
        super(UnWeightedKspaceMSELossNonCart, self).__init__()
        # Calculate distance from the center for each point in the Cartesian grid
        y = torch.linspace(-1, 1, im_size[0], device=device)
        z = torch.linspace(-1, 1, im_size[1], device=device)
        yy, zz = torch.meshgrid(y, z)
        dist = torch.sqrt(yy**2 + zz**2)
        self.weight = (dist + 1).clone().detach()  # Avoid division by zero
        self.mse = nn.MSELoss()

    def forward(self, img_pred, kdata, all_omega, all_dcomp):
        
        # TODO: get images across contrast. perform FORWARD NUFFT.
        # Compute k-space loss with known kdata. 
        
        # JBM issue 1: this is very inefficient looping but just trying to be quick & dirty: 
        # JBM issue 2: that density compensation factoring in feels incorrect. 
        dim = 100 # should programmatically get this 
#         print('img_pred device: ', img_pred.get_device())
#         print('kdata device: ', kdata.get_device())
#         print('all_omega device: ', all_omega.get_device())
#         print('all_dcomp device: ', all_dcomp.get_device())
        weight = torch.sqrt((all_omega[:,0,:]*2*torch.pi)**2 + (all_omega[:,1,:]*2*torch.pi)**2)+1
        print('weight shape = ', np.shape(weight))
        loss = 0
        device = img_pred.get_device()
        for ii in range(10):
            contrast_img = torch.squeeze(img_pred[:,:,ii])[None,None,:,:]/1000 # JBM try this neext??#
#             contrast_img = torch.squeeze(img_pred[:,:,ii]) # JBM try this neext??#

            kspace_nufft_contrast = nufft2c_torch(contrast_img, 
                                             all_omega[ii,:,:].squeeze(), None, 
                                                  dim,device=device)#*torch.sqrt(all_dcomp[None,ii,:,:])/2/np.pi
            weight_contrast = weight[ii,:].squeeze()
            #/ torch.sqrt(all_dcomp[None,ii,:,:])*torch.pi

        # Weight the k-space loss (do last 
        
#         coil_img = img_pred[None]*csm.squeeze()[:,:,:,None]
#         coil_kspace = fft2c_torch(coil_img.permute(3,0,1,2))
#         coil_kspace = coil_kspace*undersample_mask.squeeze()
#         kdata = (kdata*undersample_mask).squeeze()

#         self.weight.expand_as(coil_kspace)
#         coil_kspace = torch.mul(coil_kspace,self.weight)
#         kdata = torch.mul(kdata,self.weight)
            
            kspace_nufft_contrast = kspace_nufft_contrast.cdouble()
            kdata = kdata.cdouble() # this is stupid and needs to move 

            loss += self.mse(torch.view_as_real(weight_contrast*kspace_nufft_contrast.squeeze()), 
                             torch.view_as_real(weight_contrast*kdata[ii,:].squeeze()))
#             loss += self.mse(torch.view_as_real(kspace_nufft_contrast.squeeze()), 
#                              torch.view_as_real(kdata[ii,:].squeeze()))

#         loss = loss.double()
        loss.double()
        return loss


class NoneEncoder(nn.Module):
    def __init__(self, in_dim):
        super(NoneEncoder, self).__init__()
        self.output_dim = in_dim

    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim*2))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        x = self.model(x)
        return torch.view_as_complex(x.reshape(x.shape[:-1] + (-1, 2)))

class tcnnHashgridEncoder(nn.Module):
    def __init__(self, dim, encoding_config):
        super().__init__()
        self.pseudo_1d = False
        if dim == 1:
            self.pseudo_1d = True
            dim = 2
        self.encoder = tcnn.Encoding(dim, encoding_config, dtype=torch.float32)
        self.output_dim = self.encoder.n_output_dims
    
    def forward(self, x):
        if self.pseudo_1d:
            x = torch.cat([x, torch.zeros_like(x)], dim=-1) # (..., 1) -> (..., 2)
            x = (x + 1) / 2  # change range from [-1, 1] to [0, 1] for t
            x = self.encoder(x)
        else:
            x = (x + 1) / 2  # change range from [-1, 1] to [0, 1]
            x = self.encoder(x) #ELLIPSE?
        return x



def get_model(in_dim, out_dim, encoding, model_type, hidden_dims):
    if encoding == 'hashgrid':
        #if in_dim == 2
        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 20,
            "base_resolution": 16,
            "per_level_scale": 1.26, #1.26, #1.22, #1.32, #1.26 #1.5  16*1.26^15 ~= 512
            "interpolation": "linear"
        }
        encoder = tcnnHashgridEncoder(in_dim, encoding_config)
    elif encoding == 'none':
        encoder = NoneEncoder(in_dim)
    else:
        raise ValueError(f"Encoding type {encoding} is not supported.")
    enc_dim = encoder.output_dim

    if model_type == 'mlp':
        model = MLP(enc_dim, hidden_dims, out_dim)
    else:
        raise ValueError(f"MLP type {model_type} is not supported.")
    
    return nn.Sequential(encoder, model)


class DirectINRReconstructor():
    def __init__(self, data, cfg, test=False):
        r"""
        Args:
            data (dict): input data, must have keys 'kdata', 'csm'
            cfg (OmegaConf): configuration object
        """
        self.kdata = data['kdata']   # (nTI, nc, Vy, Vz), np.complex64
        self.csm = data['csm']       # (nc, Vy, Vz), np.complex64
        self.reference_img = data['reference_img']
        self.undersample_mask = data['undersample_mask']
        self.im_size = data['im_size']
        self.out_dim = data['out_dim']
        self.slice_x = data['slice_x']
        self.brain_mask = data['brain_mask']

        self.config = cfg
        self.device = torch.device('cuda')

        self.model = self._build_model(cfg.network)
        self.loss = self._get_loss(cfg.training.train.loss)
        self.optimizer = self._get_optimizer(
            cfg.training.train.optimizer.type, self.model,
            lr=cfg.training.train.optimizer.lr)


    def _build_model(self, cfg):
        r"""
        Build the model for the reconstruction.
        Args:
            cfg (OmegaConf): model configuration, must have in_dim, n_components, encoding, model_type, hidden_dims.
        Returns:
            nn.Module: model object
        """
        return get_model(cfg.in_dim, self.out_dim, cfg.encoding, cfg.model_type, cfg.hidden_dims).to(self.device)
    
    def _get_loss(self, loss_cfg):
        r"""
        Get loss function for the model.
        Args:
            loss_cfg (OmegaConf): loss configuration, must have name.
        Returns:
            Loss: loss object.
        """
        if loss_cfg.type == 'weighted_k_mse':
            return WeightedKspaceMSELoss(self.im_size)
        else:
            raise ValueError(f"Loss {loss_cfg.type} is not supported.")

        
    def _get_optimizer(self, opt_type, model, lr):
        r"""
        Get optimizer for the model.
        Args:
            type (str): optimizer type.
            model (nn.Module or list of nn.Module): model to optimize.
            lr (float or list of float): learning rate.
        Returns:
            torch.optim.Optimizer: optimizer.
        """
        if type(lr) == list:
            assert len(lr) == len(model) # different lr for spatial and temporal
            params = [{'params': m.parameters(), 'lr': lr} for m, lr in zip(model, lr)]
        elif type(model) == List:
            params = [{'params': m.parameters(), 'lr': lr} for m in model]
        else:
            params = [{'params': model.parameters(), 'lr': lr}]

        if opt_type == 'adam':
            return torch.optim.Adam(params)
        elif opt_type == 'adamw':
            return torch.optim.AdamW(params)
        elif opt_type == 'sgd':
            return torch.optim.SGD(params)
        elif opt_type == 'adamax':
            return torch.optim.Adamax(params)
        else:
            raise ValueError(f"Optimizer {opt_type} is not supported.")

    def _get_img(self, grid, nTI):
        grid_batch_size = grid.shape[:-1]

        # Grid shuffling
        flattened_grid = grid.reshape(-1, 2)
        shuffled_indices = torch.randperm(flattened_grid.size(0))
        shuffled_grid = flattened_grid[shuffled_indices]

        img = self.model(shuffled_grid)

        inverse_indices = torch.argsort(shuffled_indices)
        img = img[inverse_indices,:]
        img = img.reshape(grid_batch_size + (nTI,))

        return img


    def run(self, train_cfg):

        self.kdata = self.kdata.clone().detach().to(device=self.device, dtype=torch.complex64)
        self.csm = torch.tensor(self.csm, device=self.device, dtype=torch.complex64)
        self.undersample_mask = torch.tensor(self.undersample_mask, device=self.device, dtype=torch.int)

        n_epochs = train_cfg.n_epochs
        batch_size = train_cfg.batch_size
        val_im_size = self.im_size

        batch_dim, nTI, nc, Vy, Vz = self.kdata.shape
        n_iter_per_epoch = int(np.ceil(batch_size / batch_size))
        scale = 1

        grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, int(Vy*scale)+1, device=self.kdata.device)[:-1], 
                torch.linspace(-1, 1, int(Vz*scale)+1, device=self.kdata.device)[:-1], indexing='ij'), dim=-1)  # (nx, Vy, 2)
        

        for epoch in range(n_epochs):
            loss_epoch = 0
            dc_epoch = 0


            for i in range(n_iter_per_epoch):
                self.optimizer.zero_grad()
                start=0
                end=1
                
                kdata = self.kdata[start:end].clone().detach().clone()
                csm = self.csm.unsqueeze(0).repeat(end-start,1,1,1)

                if end - start < batch_size:
                    # concatenate the last batch with the first batch to make a full batch
                    kdata = torch.cat([kdata, self.kdata[:batch_size-(end-start)]], dim=0)
                    csm = torch.cat([csm, self.csm[:batch_size-(end-start)]], dim=0)
                print('JBM shape of grid input to net: ', np.shape(grid))
                img_pred = self._get_img(grid, nTI)
                print('shape of predicted image: ', np.shape(img_pred))
                print('in training loop: shape of kdata: ', np.shape(kdata))
                print('in training loop: shape of csm: ', np.shape(csm))


                loss_dc = self.loss(img_pred, kdata, csm, self.undersample_mask)
                loss = loss_dc

                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.item()
                dc_epoch += loss_dc.item()

                del loss, loss_dc
            
            loss_epoch /= n_iter_per_epoch
            dc_epoch /= n_iter_per_epoch

            print(f"Epoch {epoch}, loss: {loss_epoch}")
            
            if epoch == n_epochs-1:
                img = self.validate(val_im_size, nTI)
                return img.squeeze().cpu().clone().detach().numpy()



    def validate(self, im_size,nTI, osf=1):
        with torch.no_grad():
            Vy, Vz = im_size
            grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, Vy+1, device=self.device)[:-1]/osf, 
                torch.linspace(-1, 1, Vz+1, device=self.device)[:-1]/osf, indexing='ij'), dim=-1)  # (nx, Vy, 2)
            
        img = self._get_img(grid[None],nTI)

        return img

    
    
class DirectINRReconstructorNonCart():
    def __init__(self, data, cfg, test=False):
        r"""
        Args:
            data (dict): input data, must have keys 'kdata', 'csm'
            cfg (OmegaConf): configuration object
        """
        self.kdata = data['kdata']   # (nTI, nc, Vy, Vz), np.complex64
        self.csm = data['csm']       # (nc, Vy, Vz), np.complex64
        self.dcomp = data['dcomp']
        self.omega = data['omega']
        self.reference_img = data['reference_img']
        self.undersample_mask = data['undersample_mask']
        self.im_size = data['im_size']
        self.out_dim = data['out_dim']
        self.slice_x = data['slice_x']
        self.brain_mask = data['brain_mask']
        self.kspace_norm_fact = data['kspace_norm_fact']

        self.config = cfg
        self.device = torch.device('cuda')

        self.model = self._build_model(cfg.network)
        self.loss = self._get_loss(cfg.training.train.loss)
        self.optimizer = self._get_optimizer(
            cfg.training.train.optimizer.type, self.model,
            lr=cfg.training.train.optimizer.lr)


    def _build_model(self, cfg):
        r"""
        Build the model for the reconstruction.
        Args:
            cfg (OmegaConf): model configuration, must have in_dim, n_components, encoding, model_type, hidden_dims.
        Returns:
            nn.Module: model object
        """
        return get_model(cfg.in_dim, self.out_dim, cfg.encoding, cfg.model_type, cfg.hidden_dims).to(self.device)
    
    def _get_loss(self, loss_cfg):
        r"""
        Get loss function for the model.
        Args:
            loss_cfg (OmegaConf): loss configuration, must have name.
        Returns:
            Loss: loss object.
        """
        if loss_cfg.type == 'weighted_k_mse':
            return UnWeightedKspaceMSELossNonCart(self.im_size)
        else:
            raise ValueError(f"Loss {loss_cfg.type} is not supported.")

        
    def _get_optimizer(self, opt_type, model, lr):
        r"""
        Get optimizer for the model.
        Args:
            type (str): optimizer type.
            model (nn.Module or list of nn.Module): model to optimize.
            lr (float or list of float): learning rate.
        Returns:
            torch.optim.Optimizer: optimizer.
        """
        if type(lr) == list:
            assert len(lr) == len(model) # different lr for spatial and temporal
            params = [{'params': m.parameters(), 'lr': lr} for m, lr in zip(model, lr)]
        elif type(model) == List:
            params = [{'params': m.parameters(), 'lr': lr} for m in model]
        else:
            params = [{'params': model.parameters(), 'lr': lr}]

        if opt_type == 'adam':
            return torch.optim.Adam(params)
        elif opt_type == 'adamw':
            return torch.optim.AdamW(params)
        elif opt_type == 'sgd':
            return torch.optim.SGD(params)
        elif opt_type == 'adamax':
            return torch.optim.Adamax(params)
        else:
            raise ValueError(f"Optimizer {opt_type} is not supported.")

    def _get_img(self, grid, nTI):
        grid_batch_size = grid.shape[:-1]

        # Grid shuffling
        flattened_grid = grid.reshape(-1, 2)
        shuffled_indices = torch.randperm(flattened_grid.size(0))
        shuffled_grid = flattened_grid[shuffled_indices]

        img = self.model(shuffled_grid)

        inverse_indices = torch.argsort(shuffled_indices)
        img = img[inverse_indices,:]
        img = img.reshape(grid_batch_size + (nTI,))

        return img


    def run(self, train_cfg):

        self.kdata = self.kdata.clone().detach().to(device=self.device, dtype=torch.complex64)
        self.omega = self.omega.clone().detach().to(device=self.device, dtype=torch.float64)
        self.dcomp = self.dcomp.clone().detach().to(device=self.device, dtype=torch.complex64)

        # don't need either of the below
#         self.csm = torch.tensor(self.csm, device=self.device, dtype=torch.complex64)
#         self.undersample_mask = torch.tensor(self.undersample_mask, device=self.device, dtype=torch.int)

        n_epochs = train_cfg.n_epochs
        batch_size = train_cfg.batch_size
        val_im_size = self.im_size

#         batch_dim, nTI, nc, Vy, Vz = self.kdata.shape
        batch_dim, nTI, nc, Vy, Vz = 1, 10, 1, 100, 100 # JBM hardcoded
        n_iter_per_epoch = int(np.ceil(batch_size / batch_size))
        scale = 1

        # JBM an image-space grid. Not sure why 2? But fixed. 
        grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, int(Vy*scale)+1, device=self.kdata.device)[:-1], 
                torch.linspace(-1, 1, int(Vz*scale)+1, device=self.kdata.device)[:-1], indexing='ij'), dim=-1)  # (nx, Vy, 2)
        

        for epoch in range(n_epochs):
            loss_epoch = 0
            dc_epoch = 0


            for i in range(n_iter_per_epoch):
                self.optimizer.zero_grad()
                # JBM don't need below
#                 start=0
#                 end=1
                
#                 kdata = self.kdata[start:end].clone().detach().clone()
#                 csm = self.csm.unsqueeze(0).repeat(end-start,1,1,1)

#                 if end - start < batch_size:
#                     # concatenate the last batch with the first batch to make a full batch
#                     kdata = torch.cat([kdata, self.kdata[:batch_size-(end-start)]], dim=0)
#                     csm = torch.cat([csm, self.csm[:batch_size-(end-start)]], dim=0)
                
                # grid is just image space! won't need to change this. 
                img_pred = self._get_img(grid, nTI)
#                 print('shape of predicted image: ', np.shape(img_pred))

                # ok, input to my loss function
#                 kdata = self.kdata.detach().clone()
#                 omega = self.omega.detach().clone()
#                 dcomp = self.dcomp.detach().clone()
                loss_dc = self.loss(img_pred, self.kdata, self.omega, self.dcomp)
                loss = loss_dc

                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.item()
                dc_epoch += loss_dc.item()

                del loss, loss_dc
            
            loss_epoch /= n_iter_per_epoch
            dc_epoch /= n_iter_per_epoch

            print(f"Epoch {epoch}, loss: {loss_epoch}")
            
            if epoch == n_epochs-1:
                img = self.validate(val_im_size, nTI)
                return img.squeeze().cpu().clone().detach().numpy()



    def validate(self, im_size,nTI, osf=1):
        with torch.no_grad():
            Vy, Vz = im_size
            grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, Vy+1, device=self.device)[:-1]/osf, 
                torch.linspace(-1, 1, Vz+1, device=self.device)[:-1]/osf, indexing='ij'), dim=-1)  # (nx, Vy, 2)
            
        img = self._get_img(grid[None],nTI)

        return img
