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
import scipy.io as sio

@hydra.main(version_base=None, config_path='configs', config_name='config_direct_inr')

# Uncomment the following to run the comparison method PICS
#@hydra.main(version_base=None, config_path='configs', config_name='config_pics')

def main(cfg):

    # Load your data in the following format
    #JBM NOTE: assumes that the in-plane sampling direction is the y-z direction 
    '''
    Vx,Vy,Vz: spatial dimensions
    C: coil dimension

    csm : (Vx, Vy, Vz, C)
    brain_mask : (Vx, Vy, Vz)
    kspace_loaded: (Vx, Vy, Vz, C, N)
    reference_img: (Vx, Vy, Vz, N)
    '''
    
    # -----------JBM: my code for loading in the required  data ------
    # Note: some weird axis expansions due to me realizing late that y-z is in-plane
    dat = sio.loadmat('testdata_INR_multicontrast.mat')
    csm = np.repeat(np.repeat(dat['csm'][None,:,:,None],2,axis=3),2,axis=0)
    csm = csm.astype(complex)
    brain_mask = np.repeat(dat['brain_mask'][None,:,:],2,axis=0)
    kspace_loaded = np.repeat(np.repeat(dat['ksp'][None,:,:,None,:],2,axis=3),2,axis=0)
    reference_img = np.repeat(np.squeeze(dat['reference_img'])[None,:,:,:],2,axis=0)
    print('csm shape = ', np.shape(csm))
    print('brain_mask shape = ', np.shape(brain_mask))
    print('kspace shape = ', np.shape(kspace_loaded))
    print('reference img shape = ', np.shape(reference_img))
    

    # ------------- Perform 1D FFT in kx direction ------------------
    kspace_loaded_torch = torch.tensor(kspace_loaded, dtype=torch.complex64)
    kdim = kspace_loaded_torch.shape
    kspace_FTx_echos = torch.zeros(kdim, dtype=torch.complex64)
    num_echos = kdim[-1]
    print('num echos = ', num_echos)

    for echo in range(num_echos):
        x = torch.fft.ifftshift(kspace_loaded_torch[:,:,:,:,echo], dim=0)
        x = torch.fft.ifft(x, dim=0, norm='ortho')
        kspace_FTx_echos[:,:,:,:,echo] = torch.fft.fftshift(x, dim=0)


    # Get kdim used for individual slice training to create the undersampling mask
    kdim_all = kspace_FTx_echos.shape
    slice_x = 0 # JBM was 75
    print('shape of kdim_all = ', np.shape(kdim_all))
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
            print('UNDERSAMPLE MASK SHAPE = ', undersample_mask.shape) # JBM debug
            print('kdim= ', kdim) # JBM debug
            other_mask = undersampling_PoissonDisk(undersample_mask.shape, kdim, accel=cfg.data.undersampling.accel)
        elif cfg.data.undersampling.type == 'rosette':
            other_mask = undersampling_Rosette(undersample_mask.shape, kdim)
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
        print('shape provided to poisson = ', (kdim[-2],kdim[-1]))
        poisson = sigpy.mri.poisson((kdim[-2],kdim[-1]), accel=accel, calib=(0, 0), seed=i, dtype=int, crop_corner=True)
        broadcasted_poisson = np.broadcast_to(poisson, (mask_dim[0],mask_dim[2],mask_dim[3],mask_dim[4]))
        print('shape of broadcasted_poisson = ', np.shape(broadcasted_poisson))

        mask[:,i,:,:,:] = np.copy(broadcasted_poisson)

    return mask


def undersampling_Rosette(mask_dim, kdim):
    """
    AUTHOR JBM
    TODO: hard-coded
    """
    mask = np.zeros(mask_dim)
    full_angle = 60
    delta_angle = 3.75
    # rotating our shot by a certain % in each contrast 
    for i in range(kdim[1]):
        print('shape provided to rosete = ', (kdim[-2],kdim[-1]))
        kx1, ky1 = generate_rosette_traj(6710//2, 1, 30, 20, 0+ delta_angle*i)
        mask1 = create_binary_mask_from_rosette(kx1, ky1, 256, 1)
        kx2, ky2 = generate_rosette_traj(6710//2, 1, 30, 20, 10 + delta_angle*i)
        mask2 = create_binary_mask_from_rosette(kx2, ky2, 256, 1)
        kx3, ky3 = generate_rosette_traj(6710//2, 1, 30, 20, 20 + delta_angle*i)
        mask3 = create_binary_mask_from_rosette(kx3, ky3, 256, 1)
        kx4, ky4 = generate_rosette_traj(6710//2, 1, 30, 20, 30 + delta_angle*i)
        mask4 = create_binary_mask_from_rosette(kx4, ky4, 256, 1)
        full_mask = mask1 + mask2 + mask3 + mask4
        broadcasted_rosette = np.broadcast_to(full_mask, (mask_dim[0],mask_dim[2],mask_dim[3],mask_dim[4]))
#         poisson = sigpy.mri.poisson((kdim[-2],kdim[-1]), accel=accel, calib=(0, 0), seed=i, dtype=int, crop_corner=True)
#         broadcasted_poisson = np.broadcast_to(poisson, (mask_dim[0],mask_dim[2],mask_dim[3],mask_dim[4]))
        print('shape of broadcasted_rosette = ', np.shape(broadcasted_rosette))

        mask[:,i,:,:,:] = np.copy(broadcasted_rosette)

    return mask


def generate_rosette_traj(num_points, k_max, omega1, omega2, angle_deg=0.0):
    """
    AUTHOR JBM WITH AI
    Generates k-space coordinates for a rosette-shaped sampling mask.

    Parameters:
    - num_points (int): The number of sampling points.
    - k_max (float): The maximum radius of the k-space trajectory.
    - omega1 (float): The rapid oscillation frequency.
    - omega2 (float): The slower rotational frequency.

    Returns:
    - kx (np.ndarray): The x-coordinates of the k-space trajectory.
    - ky (np.ndarray): The y-coordinates of the k-space trajectory.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    
    # Create a time vector
    t = np.linspace(0, 2 * np.pi, num_points)

    # Generate the k-space coordinates using the rosette equation
    kx = k_max * np.sin(omega1 * t) * np.cos(omega2 * t)
    ky = k_max * np.sin(omega1 * t) * np.sin(omega2 * t)
    
    kx_rot = kx * cos_theta - ky * sin_theta
    ky_rot = kx * sin_theta + ky * cos_theta

    return kx_rot, ky_rot
    
    
def create_binary_mask_from_rosette(kx, ky, image_size=256, k_range=1.0):
    """
    AUTHOR JBM WITH AI
    Creates a binary mask from the rosette trajectory by discretizing
    the k-space points onto a grid.

    Parameters:
    - kx (np.ndarray): The x-coordinates of the k-space trajectory.
    - ky (np.ndarray): The y-coordinates of the k-space trajectory.
    - image_size (int): The size of the final binary mask.
    - k_range (float): The range of kx and ky (e.g., from -k_range to k_range).

    Returns:
    - binary_mask (np.ndarray): The binary mask.
    """

    binary_mask = np.zeros((image_size, image_size), dtype=bool)

    # Scale the k-space points to pixel coordinates
    pixel_kx = ((kx / k_range + 1) / 2 * (image_size - 1)).astype(int)
    pixel_ky = ((ky / k_range + 1) / 2 * (image_size - 1)).astype(int)

    # Clip coordinates to be within the image boundaries
    pixel_kx = np.clip(pixel_kx, 0, image_size - 1)
    pixel_ky = np.clip(pixel_ky, 0, image_size - 1)

    # Set the corresponding pixels in the binary mask to True
    binary_mask[pixel_ky, pixel_kx] = True

    return binary_mask


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
    print('JBM: out_dim = ', out_dim)
    print('JBM: load_sliceData() shape of kdata: ', np.shape(kdata))
    print('JBM: load_sliceData() shape of csm: ', np.shape(csm))
    
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
