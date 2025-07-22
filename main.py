import os
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import numpy as np
import hydra
from models.direct_inr import DirectINRReconstructor
from models.pics import PICSReconstructor
import sigpy.mri
import time
import nibabel


@hydra.main(version_base=None, config_path='configs', config_name='config_direct_inr')

# Uncomment the following to run the comparison method PICS
# @hydra.main(version_base=None, config_path='configs', config_name='config_pics')

def main(cfg):

    # Load your data in the following format
    '''
    Vx,Vy,Vz: spatial dimensions
    C: coil dimension
    N: contrast dimension

    csm : (Vx, Vy, Vz, C)
    brain_mask : (Vx, Vy, Vz)
    kspace_loaded: (Vx, Vy, Vz, C, N)
    reference_img: (Vx, Vy, Vz, N)
    '''

    csm = [...]
    brain_mask = [...]
    kspace_loaded = [...]
    reference_img = [...]


    # ------------- Perform 1D FFT in kx direction ------------------
    kspace_loaded_torch = torch.tensor(kspace_loaded, dtype=torch.complex64)
    kdim = kspace_loaded_torch.shape
    kspace_FTx_echos = torch.zeros(kdim, dtype=torch.complex64)
    num_echos = kdim[-1]

    for echo in range(num_echos):
        x = torch.fft.ifftshift(kspace_loaded_torch[:,:,:,:,echo], dim=0)
        x = torch.fft.ifft(x, dim=0, norm='ortho')
        kspace_FTx_echos[:,:,:,:,echo] = torch.fft.fftshift(x, dim=0)


    # Get kdim used for individual slice training to create the undersampling mask
    kdim_all = kspace_FTx_echos.shape
    slice_x = 75
    kspace_oneXslice = kspace_FTx_echos[slice_x,:,:,:,:]
    # Reorder k-space data for training
    kdata = np.transpose(kspace_oneXslice, (3,2,0,1))
    kdata = kdata[None,:,:,:,:]
    kdim = kdata.shape

    # ------------- Create complementary undersampling mask ------------------------------

    # previously calculated undersampling masks are saved and loaded for comparability between methods
    os.makedirs(f"{cfg.data.save_dir}/undersample_masks", exist_ok=True)
    mask_name = cfg.data.undersampling.type + '_center1:' +str(cfg.data.undersampling.full_kcenter) + '_accel' + str(cfg.data.undersampling.accel)
    mask_file = cfg.data.save_dir + '/undersample_masks/'+ mask_name

    if os.path.isfile(mask_file):
        undersample_mask = np.load(mask_file)
    else:
        undersample_mask = np.zeros(kdim, dtype=int)
        fullcenter_mask = fullysampled_kcenter(undersample_mask,kdim,downsample=cfg.data.undersampling.full_kcenter)

        if cfg.data.undersampling.type == 'poisson':
            other_mask = undersampling_PoissonDisk(undersample_mask.shape, kdim, accel=cfg.data.undersampling.accel)
        elif cfg.data.undersampling.type == 'fullysampled':
            other_mask = np.ones(kdim,dtype=int)
        else:
            print('WRONG')

        undersample_mask = np.maximum(fullcenter_mask, other_mask)

    full_kspace = kdim[-1]*kdim[-2]
    undersampling_samples = np.sum(undersample_mask[0,0,0,:,:] != 0)
    R_accel = full_kspace / undersampling_samples
    R_accel = round(R_accel,2)
    np.save(mask_file, undersample_mask)


    # ---------------- INR training and inference for all slices ----------------

    img_allSlices = np.zeros((kdim_all[0],kdim_all[1],kdim_all[2],kdim_all[4]))
    time0 = time.time()

    for slice in range(kspace_FTx_echos.shape[0]):
        data = load_sliceData(cfg,csm,reference_img,kspace_FTx_echos,R_accel,undersample_mask,slice, brain_mask)
        time1 = time.time()
        if cfg.method == 'pics':
            reconstructor = PICSReconstructor(data,cfg)
            img_slice = reconstructor.run(cfg)
        else:
            reconstructor = DirectINRReconstructor(data, cfg)
            img_slice = reconstructor.run(cfg.training.train)
        time2 = time.time()

        img_allSlices[slice,:,:,:] = np.abs(img_slice)


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


def undersampling_PoissonDisk(mask_dim, kdim, accel):
    mask = np.zeros(mask_dim)
    for i in range(kdim[1]):
        poisson = sigpy.mri.poisson((kdim[-2],kdim[-1]), accel=accel, calib=(0, 0), seed=i, dtype=int, crop_corner=True)
        broadcasted_poisson = np.broadcast_to(poisson, (mask_dim[0],mask_dim[2],mask_dim[3],mask_dim[4]))
        mask[:,i,:,:,:] = np.copy(broadcasted_poisson)

    return mask
    


def load_sliceData(cfg,csm,reference_img,kspace_FTx_echos,R_accel,undersample_mask,slice_x, brain_mask):
    csm = csm[slice_x,:,:,:]
    csm = np.transpose(csm,(2,0,1))
    reference_img = reference_img[slice_x,:,:,:]
    runname = create_runname(cfg,R_accel,slice_x)

    kspace_oneXslice = kspace_FTx_echos[slice_x,:,:,:,:]
    kdata = np.transpose(kspace_oneXslice, (3,2,0,1))
    kdata = kdata[None,:,:,:,:] / np.abs(kspace_FTx_echos).max()*100 # k-space normalization (same for all slices)
    kdim = kspace_FTx_echos.shape

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