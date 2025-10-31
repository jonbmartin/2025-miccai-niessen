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
    
    # -----------JBM: my code for loading in the required  data ------
    # Note: some weird axis expansions due to me realizing late that y-z is in-plane
#     dat = sio.loadmat('testdata_INR_multicontrast.mat')
#     csm = np.repeat(np.repeat(dat['csm'][None,:,:,None],2,axis=3),2,axis=0)
#     csm = csm.astype(complex)
#     brain_mask = np.repeat(dat['brain_mask'][None,:,:],2,axis=0)
#     kspace_loaded = np.repeat(np.repeat(dat['ksp'][None,:,:,None,:],2,axis=3),2,axis=0)
#     reference_img = np.repeat(np.squeeze(dat['reference_img'])[None,:,:,:],2,axis=0)
#     print('csm shape = ', np.shape(csm))
#     print('brain_mask shape = ', np.shape(brain_mask))
#     print('kspace shape = ', np.shape(kspace_loaded))
#     print('reference img shape = ', np.shape(reference_img))
    

# #     # ------------- Perform 1D FFT in kx direction ------------------
#     kspace_loaded_torch = torch.tensor(kspace_loaded, dtype=torch.complex64)
#     kdim = kspace_loaded_torch.shape
#     kspace_FTx_echos = torch.zeros(kdim, dtype=torch.complex64)
#     num_echos = kdim[-1]
#     print('num echos = ', num_echos)

#     for echo in range(num_echos):
#         x = torch.fft.ifftshift(kspace_loaded_torch[:,:,:,:,echo], dim=0)
#         x = torch.fft.ifft(x, dim=0, norm='ortho')
#         kspace_FTx_echos[:,:,:,:,echo] = torch.fft.fftshift(x, dim=0)

#     # Get kdim used for individual slice training to create the undersampling mask
#     kdim_all = kspace_FTx_echos.shape
#     print('kdim_all =', print(kdim_all))
#     slice_x = 0 # JBM was 75
#     kspace_oneXslice = kspace_FTx_echos[slice_x,:,:,:,:]
#     # Reorder k-space data for training
#     kdata = np.transpose(kspace_oneXslice, (3,2,0,1))
#     kdata = kdata[None,:,:,:,:]
# #     print('new shape of kdim = ', np.shape(kdata))
# #     print(kdata)

#     kdim = kdata.shape # kDIM IS JUST DIMENSIONS, NOT DATA!! 
    
#     # JBM: overwriting the previous. None of the above is necessary
# #     kdata = np.transpose(kspace_loaded,axes=(0,4,3,1,2))
# #     kdata = kdata[0,:,:,:,:]
# #     kdim = kdata.shape
# #     kdim_all = kdim


    # ------------- Create complementary undersampling mask ------------------------------

#     # previously calculated undersampling masks are saved and loaded for comparability between methods
#     os.makedirs(f"{cfg.data.save_dir}/undersample_masks", exist_ok=True)
#     mask_name = cfg.data.undersampling.type + '_center1:' +str(cfg.data.undersampling.full_kcenter) + '_accel' + str(cfg.data.undersampling.accel)
#     mask_file = cfg.data.save_dir + '/undersample_masks/'+ mask_name

#     if os.path.isfile(mask_file):
#         undersample_mask = np.load(mask_file)
#     else:
#         undersample_mask = np.zeros(kdim, dtype=int)
#         fullcenter_mask = fullysampled_kcenter(undersample_mask,kdim,downsample=cfg.data.undersampling.full_kcenter)

#         if cfg.data.undersampling.type == 'poisson':
#             print('UNDERSAMPLE MASK SHAPE = ', undersample_mask.shape) # JBM debug
#             print('kdim= ', kdim) # JBM debug
#             other_mask = undersampling_PoissonDisk(undersample_mask.shape, kdim, accel=cfg.data.undersampling.accel)
#         elif cfg.data.undersampling.type == 'rosette':
#             other_mask = undersampling_Rosette(undersample_mask.shape, kdim)
#         elif cfg.data.undersampling.type == 'fullysampled':
#             other_mask = np.ones(kdim,dtype=int)
#         else:
#             print('WRONG')

#         undersample_mask = np.maximum(fullcenter_mask, other_mask)

#     full_kspace = kdim[-1]*kdim[-2]
#     undersampling_samples = np.sum(undersample_mask[0,0,0,:,:] != 0)
#     R_accel = full_kspace / undersampling_samples
#     R_accel = round(R_accel,2)
#     np.save(mask_file, undersample_mask)


    # ---------------- INR training and inference for all slices ----------------
    
    # THIS IS THE CORE FUNCTION. DONT NEED ANYTHING ABOVE IT 
#     img_allSlices = np.zeros((kdim_all[0],kdim_all[1],kdim_all[2],kdim_all[4]))
    img_allSlices = np.zeros((1,10,100,100)) # Nslices, Ncontrasts, Nx, Ny
    time0 = time.time()

    for slice in range(1):
        data = load_sliceDataNonCart()
        time1 = time.time()
        if cfg.method == 'pics':
            reconstructor = PICSReconstructor(data,cfg)
            img_slice = reconstructor.run(cfg)
        else:
            reconstructor = DirectINRReconstructorNonCart(data, cfg)
            img_slice = reconstructor.run(cfg.training.train)
        time2 = time.time()

        img_allSlices[slice,:,:,:] = np.transpose(np.abs(img_slice), axes=(2,0,1))


    print("Time for one slice:", (time2 - time1))
    total_time = time2 - time0
    print("Time for all slices:", total_time)
    save_dir = f"{cfg.data.save_dir}/{cfg.method}"
    os.makedirs(save_dir, exist_ok=True)

    nifti_file = nibabel.Nifti1Image(np.abs(img_allSlices), affine=None)
    nibabel.save(nifti_file, save_dir + '/' + data['runname'] + '.nii.gz')


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


def load_sliceDataNonCart():
    # my custom-defined bins
    rosdat = sio.loadmat('data/rosette_ferret_data_NOVD_7T.mat')

    bin_bots = [12, 80, 144, 208, 270, 334, 398, 462, 526, 594] 
    Ro_len = 68 
    Nshots = 40
    bin_tops = [x+Ro_len for x in bin_bots]
    dim = 100
    Ncontrast = len(bin_bots)
    
    all_sigI = torch.zeros(Ncontrast,Ro_len*Nshots, dtype=torch.cdouble)
    all_omega = torch.zeros(Ncontrast,2,Ro_len*Nshots)
    all_dcomp = torch.zeros(Ncontrast,1,Ro_len*Nshots, dtype=torch.cdouble)

    
    for ii in range(len(bin_bots)):
        print('Calc. density comp for segment: ', ii+1)
        tbot,ttop = bin_bots[ii],bin_tops[ii]

        kx = torch.from_numpy(rosdat['kxI'])[tbot:ttop,:]
        ky = torch.from_numpy(rosdat['kyI'])[tbot:ttop,:]
        sigI = torch.from_numpy(rosdat['sigI'])[tbot:ttop,:]
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
    kspace_norm_fact = torch.max(torch.abs(all_sigI))*100
    all_sigI = all_sigI / kspace_norm_fact # k-space normalization (same for all slices)
#     kdim = kspace_FTx_echos.shape
    
#     print('shape of mask in load_sliceData():', np.shape(undersample_mask))
#     kdata = kdata * undersample_mask # (1,num_TIs,num_coils,Vy,Vz)

#     Xslice_size = [kdim[1],kdim[2]]
    out_dim = Ncontrast # output is the number of "contrasts" or time bins

    data = {
        'kdata': all_sigI,
        'csm': None,
        'omega': all_omega,
        'dcomp':all_dcomp,
        'brain_mask': None, # don't need
        'reference_img': None, # don't need
        'undersample_mask': None, # delete
        'runname': 'JBM_testing_noncart',
        'slice_x': 0,
        'im_size': [100,100],
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
