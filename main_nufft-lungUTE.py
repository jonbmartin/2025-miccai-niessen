import os
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import numpy as np
import hydra
from models.direct_inr import DirectINRReconstructor, DirectINRReconstructorNonCart
from models.pics import PICSReconstructor
import sigpy.mri
import time
import nibabel
import scipy.io as sio
import torchkbnufft as tkbn


@hydra.main(version_base=None, config_path='configs', config_name='config_direct_inr')

# Uncomment the following to run the comparison method PICS
#@hydra.main(version_base=None, config_path='configs', config_name='config_pics')

def main(cfg):

    # Load your data in the following format
    #JBM NOTE: assumes that the in-plane sampling direction is the y-z direction 
    # TODO: 1) need weighting to be corrected - just based on the trajectory location 
    # 2) 
    '''
    Vx,Vy,Vz: spatial dimensions
    C: coil dimension
    N: # of contrasts
    Nshot: # of shots in kspace trajectory
    Ktraj: duration of kspace trajectory in samples 

    # cartesian inputs are: 
    csm : (Vx, Vy, Vz, C)
    brain_mask : (Vx, Vy, Vz)
    kspace_loaded: (Vx, Vy, Vz, C, N)
    reference_img: (Vx, Vy, Vz, N)
    omega: (2, Nshot, Ktraj, N) 
    
    
    For NuFFT input, going to want: 
    input: torch tensor, input kspace data, (..., nc, nspokes*nFE)
    smaps: torch tensor, sensitivity maps, (..., nc, nx, ny), should has same dimensions as input
    ktraj: torch tensor, k-space trajectory, (..., 2, nspokes*nFE)
    dcomp: torch tensor, density compensation, (..., 1, nspokes*nFE)
    '''


    # ---------------- INR training and inference for all slices ----------------
    
    img_allSlices = None
    img_allSlicesCplx = None
    time0 = time.time()

    for slice in range(1):
        data = load_sliceDataNonCart("lungUTE")
        # create output containers once we know dimensions
        if img_allSlices is None:
            Nslices = 1
            Ncontrasts = data['out_dim']
            print('Ncontrasts = ', Ncontrasts)
            Nx, Ny = data['im_size']
            img_allSlices = np.zeros((Nslices, Ncontrasts, Nx, Ny))
            img_allSlicesCplx = np.zeros((Nslices, Ncontrasts, Nx, Ny), dtype=complex)
        time1 = time.time()
        if cfg.method == 'pics':
            reconstructor = PICSReconstructor(data,cfg)
            img_slice = reconstructor.run(cfg)
        else:
            reconstructor = DirectINRReconstructorNonCart(data, cfg)
            img_slice = reconstructor.run(cfg.training.train)
        time2 = time.time()

        print('output image slice = ', np.shape(img_slice))
        img_allSlices[slice,:,:,:] = np.transpose(np.abs(img_slice), axes=(2,0,1))
        img_allSlicesCplx[slice,:,:,:] = np.transpose(img_slice, axes=(2,0,1)) # JBM just doing extra saving


    print("Time for one slice:", (time2 - time1))
    total_time = time2 - time0
    print("Time for all slices:", total_time)
    save_dir = f"{cfg.data.save_dir}/{cfg.method}"
    os.makedirs(save_dir, exist_ok=True)

    # use a valid affine (avoid passing None)
    nifti_file = nibabel.Nifti1Image(np.abs(img_allSlices), affine=np.eye(4))
    nibabel.save(nifti_file, save_dir + '/' + data['runname'] + '.nii.gz')
    # JBM also just saving it the way I prefer, MATLAB file: 
    sio.savemat(save_dir + '/' + data['runname'] + '.mat', {'cplx_img':img_allSlicesCplx})


def fullysampled_kcenter(undersample_mask,kdim,downsample):

    k_center_y = int(kdim[-2]/2)
    k_center_z = int(kdim[-1]/2)
    kc_y = int(k_center_y/downsample)
    kc_z = int(k_center_z/downsample)
    start_idx_y = int(kdim[-2]/2 - kc_y)
    end_idx_y = int(kdim[-2]/2 + kc_y)
    start_idx_z = int(kdim[-1]/2 - kc_z)
    end_idx_z = int(kdim[-1]/2 + kc_z) + 1
    undersample_mask[:,:,:,start_idx_y:end_idx_y,start_idx_z:end_idx_z] = 1

    return undersample_mask


def load_sliceData(cfg,csm,reference_img,kspace_FTx_echos,R_accel,undersample_mask,slice_x, brain_mask):
    csm = csm[slice_x,:,:,:]
    csm = np.transpose(csm,(2,0,1))
    reference_img = reference_img[slice_x,:,:,:]
    runname = create_runname(cfg,R_accel,slice_x)

    kspace_oneXslice = kspace_FTx_echos[slice_x,:,:,:,:]
    kdata = np.transpose(kspace_oneXslice, (3,2,0,1))
    kdata = kdata[None,:,:,:,:] / np.abs(kspace_FTx_echos).max()*100 # k-space normalization (same for all slices)
    kdim = kspace_FTx_echos.shape
    
    print('shape of mask in load_sliceData():', np.shape(undersample_mask))
    kdata = kdata * undersample_mask # (1,num_TIs,num_coils,Vy,Vz)

    Xslice_size = [kdim[1],kdim[2]]
    out_dim = kdim[-1]

    data = {
        'kdata': kdata,
        'csm': csm,
        'brain_mask': brain_mask,
        'reference_img': reference_img,
        'undersample_mask': undersample_mask,
        'runname': runname,
        'slice_x': slice_x,
        'im_size': Xslice_size,
        'out_dim': out_dim,
    }

    return data


def load_sliceDataNonCart(dataset_name='tubes'):
    # my custom-defined bins
    if dataset_name == 'ferret':
        rosdat = sio.loadmat('data/rosette_ferret_data_NOVD_7T.mat')
        bin_bots = [12, 80, 144, 208, 270, 334, 398, 462, 526, 594] 
        Ro_len = 68 
        Nshots = 40
        csm = None
        dim = 100
    elif dataset_name == 'tubes':
        rosdat = sio.loadmat('data/rosette_raw_data_tubes_reformatted.mat')
        bin_bots = [37, 371, 704, 1037, 1365, 1700, 2029, 2362, 2677] 
        Ro_len = 333 
        Nshots = 98
        csm = None
        shot_skip = 7 # for debugging: use every nth shot
        dim = 100
    elif dataset_name == 'lungUTE':
        mydat = sio.loadmat('data/INR_SISO_data_for_niessen.mat')
        bin_bots = [0] # this dataset, we are just reconstructing a single contrast
        Ro_len = 501 
        Nshots = 120
        shot_skip = 1 # for debugging: use every nth shot
        dim = 300
        # JBM: defining a dictionary similar to rosdat for compatibility
        rosdat = {}
        # JBM know that theese coordinates are not-normalized, so normalizing here
        # JBM: experimentally realized that I also needed a factor of 2 division. was too small
        rosdat['kxI'] = mydat['kx_in']/150/2 # (Ro_len, Nshots)
        rosdat['kyI'] = mydat['ky_in']/150/2 # (Ro_len, Nshots)
        slice_idx = 65 # middle slice
        nslices = 0
        coil_idx = 1 # JBM first coil. will need to extend to multi-coil later
        csm = mydat['sens_map'][:,:,slice_idx,0,coil_idx] # (Nx, Ny), coil 1, slice 65
        csm = torch.from_numpy(csm)
        csm = None
        rosdat['sigI'] = np.squeeze(mydat['kspace_3d_in'][:,:,slice_idx,:,coil_idx]) # given as (Ro_len, Nshots, Nslice, 1, Ncoils)
        # rosdat['csm'] = mydat['sens_map'][:,:,slice_idx,:,:] # given as (Nx, Ny, Nslice, 1, Ncoils)
    else:
        print('Invalid dataset provided to load_sliceDataNonCart()')
        return
    
    shot_idx = np.arange(0, Nshots, max(1, int(shot_skip)), dtype=np.int64)
    nshots_sel = shot_idx.size
    print(f"Shot subsampling: original={Nshots}, shot_skip={shot_skip}, selected={nshots_sel}")

    bin_tops = [x+Ro_len for x in bin_bots]
    Ncontrast = len(bin_bots)
    
    all_sigI = torch.zeros(Ncontrast,Ro_len*nshots_sel, dtype=torch.cdouble)
    all_omega = torch.zeros(Ncontrast,2,Ro_len*nshots_sel)
    all_dcomp = torch.zeros(Ncontrast,1,Ro_len*nshots_sel, dtype=torch.cdouble)

    
    for ii in range(len(bin_bots)):
        print('Calc. density comp for segment: ', ii+1)
        tbot,ttop = bin_bots[ii],bin_tops[ii]

        kx = torch.from_numpy(rosdat['kxI'])[tbot:ttop,shot_idx]
        ky = torch.from_numpy(rosdat['kyI'])[tbot:ttop,shot_idx]
        sigI = torch.from_numpy(rosdat['sigI'])[tbot:ttop,shot_idx]
        omega = torch.stack((ky.flatten(),kx.flatten()),dim=0)
        omega = omega*2*torch.pi
        sigI = sigI.flatten()[None,None,:].type(torch.cdouble)

        dcomp = tkbn.calc_density_compensation_function(omega, (dim, dim),num_iterations=15)

        all_sigI[ii,:] = sigI
        all_omega[ii,:,:] = omega
        all_dcomp[ii,:,:] = dcomp
        
        # JBM below is just a recon test
#         img = inufft2c_torch(sigI*dcomp, omega, None,dim)
#         img = np.squeeze(img)
#         plt.figure()
#         plt.imshow(np.squeeze(np.abs(img.numpy())), cmap='gray')
    
    
    # below is not me
#     csm = csm[slice_x,:,:,:]
#     csm = np.transpose(csm,(2,0,1))
#     reference_img = reference_img[slice_x,:,:,:]
#     runname = create_runname(cfg,R_accel,slice_x)

#     kspace_oneXslice = kspace_FTx_echos[slice_x,:,:,:,:]
#     kdata = np.transpose(kspace_oneXslice, (3,2,0,1))
#JBM  a note of warning: the normalization factor above does really impact performance...
    kspace_norm_fact = torch.max(torch.abs(all_sigI))/10#*100
    all_sigI = all_sigI / kspace_norm_fact # k-space normalization (same for all slices)
#     kdim = kspace_FTx_echos.shape
    
#     print('shape of mask in load_sliceData():', np.shape(undersample_mask))
#     kdata = kdata * undersample_mask # (1,num_TIs,num_coils,Vy,Vz)

#     Xslice_size = [kdim[1],kdim[2]]
    out_dim = Ncontrast # output is the number of "contrasts" or time bins

    data = {
        'kdata': all_sigI,
        'csm': csm,
        'omega': all_omega,
        'dcomp':all_dcomp,
        'brain_mask': None, # don't need
        'reference_img': None, # don't need
        'undersample_mask': None, # delete
        'runname': 'JBM_testing_noncart_7skip_LARGE_REG',
        'slice_x': 0,
        'im_size': [dim,dim],
        'out_dim': out_dim,
        'kspace_norm_fact': kspace_norm_fact #JBM: maybe need to be careful about how we normalize/denormalize kspace...
    }
    print('Data successfully loaded in load_sliceDataNonCart()')

    return data


def create_runname(cfg,R_accel,slice_x) -> str:
    
    if cfg.method == 'direct_inr':
        # Extract relevant values from the configuration
        learn_rate = cfg.training.train_lr
        hidden_dims = cfg.network.hidden_dims
        undersample_type = cfg.data.undersampling.type
        loss_type = cfg.training.train.loss.type

        runname = f"INR_{cfg.data.dataset_name}_{loss_type}_{undersample_type}_R{R_accel}_lr{learn_rate}_hdim{hidden_dims}_sliceX{slice_x}"
    
    elif cfg.method == 'pics':
        undersample_type = cfg.data.undersampling.type
        regularizer = cfg.bart_settings.regularizer
        r_param = cfg.bart_settings.r_param
        niter = cfg.bart_settings.niter

        runname = f"pics_{cfg.data.dataset_name}_{undersample_type}_R{R_accel}_reg-{regularizer}_r{r_param}_niter{niter}_sliceX{slice_x}"


    print(f"Name of current run: {runname}")
    return runname

if __name__ == '__main__':
    main()
