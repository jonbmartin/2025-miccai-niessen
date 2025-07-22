import torch
import numpy as np
import sys
import os
from utils import fft2c_torch, ifft2c_torch
bartpath = '/home/natascha/ad_mri_biomarkers/bart_Test/python'
sys.path.append(bartpath)
from bart import bart
os.environ['TOOLBOX_PATH'] = '/home/natascha/ad_mri_biomarkers/bart_Test'


class PICSReconstructor():
    def __init__(self, data, cfg):
        r"""
        Args:
            data (dict): input data, must have keys 'kdata', 'ktraj', 'csm'
            cfg (OmegaConf): configuration object
        """
        self.kdata = data['kdata']
        self.csm = data['csm']
        self.reference_img = data['reference_img']
        self.undersample_mask = data['undersample_mask']
        self.runname = data['runname']
        self.slice_x = data['slice_x']
        self.brain_mask = data['brain_mask']

        self.config = cfg
        self.device = torch.device('cuda')


    def run(self, cfg):

        # recon with BART
        kdata = np.squeeze(self.kdata)
        kdim = kdata.shape

        regularizer = cfg.bart_settings.regularizer
        r_param = cfg.bart_settings.r_param
        niter = cfg.bart_settings.niter

        kdata = np.transpose(kdata[None],(3,4,0,2,1))
        csm = np.transpose(self.csm[None],(2,3,0,1))
        recon = np.zeros((kdim[0],kdim[2],kdim[3]), dtype=np.complex64)

        for i in range(kdim[0]):
            kdata_echo = kdata[:,:,:,:,i].detach().cpu().numpy()
            if regularizer == 'l2':
                lambda_l2 = r_param
                recon[i] = bart(1,f"pics -S -{regularizer} -r{r_param} -i{niter}", kdata_echo, csm)
            elif regularizer == 'wavelet':
                recon[i] = bart(1,f"pics -S -R W:0:0:{r_param} -i{niter}", kdata_echo, csm)
            elif regularizer == 'totalvar':
                recon[i] = bart(1,f"pics -S -R T:0:0:{r_param} -i{niter}", kdata_echo, csm)

        recon = np.transpose(recon,(1,2,0))

        return recon
        