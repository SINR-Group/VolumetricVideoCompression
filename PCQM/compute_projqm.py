# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:10:42 2021

@author: AlirezaJav
         Instituto de Telecominicações
         Lisbon, Portugal
"""
import numpy as np
import pandas as pd
import open3d as o3d
import ProjQM
import cv2
import argparse
import os
import time

def main():
    # input parser
    parser = argparse.ArgumentParser(description="Projection-based metric for Pc quality assessment")
    parser.add_argument('-a', nargs=1, default='', help='Reference point cloud')
    parser.add_argument('-b', nargs=1, default='', help='Degraded point cloud')
    parser.add_argument('-ar', nargs=1, default='', help='Reference point cloud recolored with degraded color')
    parser.add_argument('-br', nargs=1, default='', help='Degraded point cloud recolored with reference color')
    parser.add_argument('-c', '--config', nargs=1, type=str,  
                        help='Configuration file containing metric parameters')
    parser.add_argument('-o','--outfile', help='Output data file')
    cli_args = parser.parse_args()
    config = ProjQM.read_configurations(cli_args.config)
    
    # list of all 2D metrics in this software
    dists = []
    lpips = []
    fsim = []
    vsi = []
    haarpsi = []
    vifp = []
    ssim = []
    ms_ssim = []
    psnrhvs = []
    psnrhvsm = []
    psnr = []
    
    # check if PCs are already projected, if it is, the projected images are louded 
    # flag should be set from configuration file
    if (config["Projected"]):
        
        #read images from folder: all 6 images of a PC should be in a seperate folder
        RefPath = config["PathRefImages"]                         # all paths should be specified in the configuration file
        DegPath = config["PathDegImages"]
        RefPath_rec = config["PathRefImages_rec"]
        DegPath_rec = config["PathDegImages_rec"]
        ImR_padded = ProjQM.load_images_from_folder(RefPath)      # all six projections of each PC should be in a separate folder
        ImD_padded = ProjQM.load_images_from_folder(DegPath)
        ImRecR_padded = ProjQM.load_images_from_folder(RefPath_rec)
        ImRecD_padded = ProjQM.load_images_from_folder(DegPath_rec)
        
        # loop over six projected images
        for i in range(0,6):
            # brg to rgb and yuv. Note that openCV reads images as BGR 
            ImR_padded_rgb = cv2.cvtColor(ImR_padded[i], cv2.COLOR_BGR2RGB)
            ImD_padded_rgb = cv2.cvtColor(ImD_padded[i], cv2.COLOR_BGR2RGB)
            ImRecR_padded_rgb = cv2.cvtColor(ImRecR_padded[i], cv2.COLOR_BGR2RGB)
            ImRecD_padded_rgb = cv2.cvtColor(ImRecD_padded[i], cv2.COLOR_BGR2RGB)
            
            ImR_padded_yuv = cv2.cvtColor(ImR_padded[i], cv2.COLOR_BGR2YUV)
            ImR_padded_y, _, _ = cv2.split(ImR_padded_yuv)
            ImD_padded_yuv = cv2.cvtColor(ImD_padded[i], cv2.COLOR_BGR2YUV)
            ImD_padded_y, _, _ = cv2.split(ImD_padded_yuv)   
            ImRecR_padded_yuv = cv2.cvtColor(ImRecR_padded[i], cv2.COLOR_BGR2YUV)
            ImRecR_padded_y, _, _ = cv2.split(ImRecR_padded_yuv)
            ImRecD_padded_yuv = cv2.cvtColor(ImRecD_padded[i], cv2.COLOR_BGR2YUV)
            ImRecD_padded_y, _, _ = cv2.split(ImRecD_padded_yuv) 
            
            # compute 2D metrics 
            if (config["dists"]):
                dists_ab = ProjQM.compute_dists(ImR_padded_rgb, ImRecR_padded_rgb)  # on Reference Geometry
                dists_ba = ProjQM.compute_dists(ImD_padded_rgb, ImRecD_padded_rgb)  # on Degraded Geometry
                dists.append( -0.49870347 * dists_ab - 0.33948725 * dists_ba )      # fusion (weights are obtained using a linear regressor)
            if (config["lpips"]):
                lpips_ab = ProjQM.compute_lpips(ImR_padded_rgb, ImRecR_padded_rgb)  # on Reference Geometry
                lpips_ba = ProjQM.compute_lpips(ImD_padded_rgb, ImRecD_padded_rgb)  # on Degraded Geometry
                lpips.append( -1.06558819 * lpips_ab + 0.19072108 * lpips_ba )      # fusion (weights are obtained using a linear regressor)
            if (config["fsim"]):
                fsim_ab = ProjQM.compute_fsim(ImR_padded_y, ImRecR_padded_y)        # on Reference Geometry
                fsim_ba = ProjQM.compute_fsim(ImD_padded_y, ImRecD_padded_y)        # on Degraded Geometry
                fsim.append( 0.04069016 * fsim_ab + 0.70281206 * fsim_ba )          # fusion (weights are obtained using a linear regressor)
            if (config["vsi"]):
                vsi_ab = ProjQM.compute_vsi(ImR_padded_rgb, ImRecR_padded_rgb)      # on Reference Geometry
                vsi_ba = ProjQM.compute_vsi(ImD_padded_rgb, ImRecD_padded_rgb)      # on Degraded Geometry
                vsi.append( 0.04072121 * vsi_ab + 0.64009922 * vsi_ba )             # fusion (weights are obtained using a linear regressor)
            if (config["haarpsi"]):
                haarpsi_ab = ProjQM.compute_haarpsi(ImR_padded_y, ImRecR_padded_y)  # on Reference Geometry
                haarpsi_ba = ProjQM.compute_haarpsi(ImD_padded_y, ImRecD_padded_y)  # on Degraded Geometry
                haarpsi.append( 0.23896795 * haarpsi_ab + 0.55268535 * haarpsi_ba ) # fusion (weights are obtained using a linear regressor)
            if (config["vifp"]):
                vifp_ab = ProjQM.compute_vifp(ImR_padded_y, ImRecR_padded_y)        # on Reference Geometry
                vifp_ba = ProjQM.compute_vifp(ImD_padded_y, ImRecD_padded_y)        # on Degraded Geometry
                vifp.append( 0.67353577 * vifp_ab + 0.15882643 * vifp_ba )          # fusion (weights are obtained using a linear regressor)
            if (config["ssim"]):
                ssim_ab = ProjQM.compute_ssim(ImR_padded_y, ImRecR_padded_y)        # on Reference Geometry
                ssim_ba = ProjQM.compute_ssim(ImD_padded_y, ImRecD_padded_y)        # on Degraded Geometry
                ssim.append( 0.28337084 * ssim_ab + 0.4550572 * ssim_ba )           # fusion (weights are obtained using a linear regressor)
            if (config["ms-ssim"]):
                ms_ssim_ab = ProjQM.compute_msssim(ImR_padded_y, ImRecR_padded_y)   # on Reference Geometry
                ms_ssim_ba = ProjQM.compute_msssim(ImD_padded_y, ImRecD_padded_y)   # on Degraded Geometry
                ms_ssim.append( 0.87409658 * ms_ssim_ab - 0.25197625 * ms_ssim_ba ) # fusion (weights are obtained using a linear regressor)
            if (config["psnrhvs"]):
                zp0 = 8 - (ImR_padded_y.shape[0] % 8)
                zp1 = 8 - (ImR_padded_y.shape[1] % 8)
                zp_ImR_padded_y = ImR_padded_y
                zp_ImRecR_padded_y = ImRecR_padded_y
                if (zp0 != 0):
                    zp_ImR_padded_y = np.concatenate((zp_ImR_padded_y,np.zeros([zp0, zp_ImR_padded_y.shape[1]])),axis=0)
                    zp_ImRecR_padded_y = np.concatenate((zp_ImRecR_padded_y,np.zeros([zp0, zp_ImRecR_padded_y.shape[1]])),axis=0)
                if (zp1 != 0):
                    zp_ImR_padded_y = np.concatenate((zp_ImR_padded_y,np.zeros([zp_ImR_padded_y.shape[0], zp1])),axis=1)
                    zp_ImRecR_padded_y = np.concatenate((zp_ImRecR_padded_y,np.zeros([zp_ImRecR_padded_y.shape[0], zp1])),axis=1)
                hvs_temp = ProjQM.compute_hvspsnr(zp_ImR_padded_y, zp_ImRecR_padded_y)
                psnrhvsm_ab = hvs_temp[0]                                           # on Reference Geometry
                psnrhvs_ab = hvs_temp[1]                                            # on Reference Geometry
                zp0 = 8 - (ImD_padded_y.shape[0] % 8)
                zp1 = 8 - (ImD_padded_y.shape[1] % 8)
                zp_ImD_padded_y = ImD_padded_y
                zp_ImRecD_padded_y = ImRecD_padded_y
                if (zp0 != 0):
                    zp_ImD_padded_y = np.concatenate((zp_ImD_padded_y,np.zeros([zp0, zp_ImD_padded_y.shape[1]])),axis=0)
                    zp_ImRecD_padded_y = np.concatenate((zp_ImRecD_padded_y,np.zeros([zp0, zp_ImRecD_padded_y.shape[1]])),axis=0)
                if (zp1 != 0):
                    zp_ImD_padded_y = np.concatenate((zp_ImD_padded_y,np.zeros([zp_ImD_padded_y.shape[0], zp1])),axis=1)
                    zp_ImRecD_padded_y = np.concatenate((zp_ImRecD_padded_y,np.zeros([zp_ImRecD_padded_y.shape[0], zp1])),axis=1)
                hvs_temp = ProjQM.compute_hvspsnr(zp_ImD_padded_y, zp_ImRecD_padded_y)
                psnrhvsm_ba = hvs_temp[0]                                                # on Degraded Geometry
                psnrhvs_ba = hvs_temp[1]                                                 # on Degraded Geometry
                psnrhvs.append( -0.49218699 * psnrhvs_ab + 1.24867639 * psnrhvs_ba )     # fusion (weights are obtained using a linear regressor)
                psnrhvsm.append( -0.27060422 * psnrhvsm_ab + 1.02825219 * psnrhvsm_ba )  # fusion (weights are obtained using a linear regressor)
            if (config["psnr"]):
                psnr_ab = ProjQM.compute_psnr(ImR_padded_y, ImRecR_padded_y)             # on Reference Geometry
                psnr_ba = ProjQM.compute_psnr(ImD_padded_y, ImRecD_padded_y)             # on Degraded Geometry
                psnr.append( -0.51929311 * psnr_ab + 1.2754385 * psnr_ba )               # fusion (weights are obtained using a linear regressor)
    else:
        # In case that inputs are PCs that are not already projected!
        # read reference and Degraded PCs
        RefPath = config["pathRef"]                                                 # all paths should be specified in the configuration file
        DegPath = config["pathDeg"]
        refPC = cli_args.a.pop()
        degPC = cli_args.b.pop()
        if (refPC == ''):
            print('No Reference PC!')
            return
        if (refPC == ''):
            print('No Degraded PC!')
            return
        fnR = RefPath + refPC;
        fnD = DegPath + degPC;
        cloudR = o3d.io.read_point_cloud(fnR) 
        cloudD = o3d.io.read_point_cloud(fnD)
        
        # check if recolored PCs are available to be louded
        if (config["Recolored"]):
            
            # read recolroed PCs
            RefPath_rec = config["pathRef_rec"]
            DegPath_rec = config["pathDeg_rec"]
            refPC_rc = cli_args.ar.pop()
            degPC_rc = cli_args.br.pop()
            if (refPC_rc == ''):
                print('No Recolored Reference PC!')
            return
            if (refPC_rc == ''):
                print('No Recolored Degraded PC!')
            return
            fnRR = RefPath_rec + refPC_rc;
            fnRD = DegPath_rec + degPC_rc;
            RecoloredR = o3d.io.read_point_cloud(fnRR) 
            RecoloredD = o3d.io.read_point_cloud(fnRD)
        # if recolored PCs are not provided as input, Reference and Degraded should be recolored
        else:
            (RecoloredR, RecoloredD) = ProjQM.recolor_pcs(cloudR,cloudD)   # recolors cloudR with color of cloudD and vice versa.
            
            # recolred PCs can be saved for further use  
            if (config["savePCs"]):
                RefPath_rec = config["pathRef_rec"]
                DegPath_rec = config["pathDeg_rec"]
                refPC_rc = 'rec_' + refPC
                degPC_rc = 'rec_' + degPC
                fnRR = RefPath_rec + refPC_rc;
                fnRD = DegPath_rec + degPC_rc;
                directory = os.path.dirname(RefPath_rec)                   # if path directory does not exist, create it!
                try:
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                
                directory = os.path.dirname(DegPath_rec)                   # if path directory does not exist, create it!
                try:
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                
                # write PCs on disk
                o3d.io.write_point_cloud(fnRR,RecoloredR,write_ascii=True) 
                o3d.io.write_point_cloud(fnRD,RecoloredD,write_ascii=True)
        
        # project all four PCs: Reference, Degraded, Recolored Reference and Recolored Degraded
        print('Reference PC is being projected')
        (ImlistR, OMListR) = ProjQM.orthographic_projection(cloudR, config["precision"], config["ws"])
        print('Degraded PC is being projected')
        (ImlistD, OMListD) = ProjQM.orthographic_projection(cloudD, config["precision"], config["ws"])
        print('Recolored Reference PC is being projected')
        (ImlistRecR, OMListRecR) = ProjQM.orthographic_projection(RecoloredR, config["precision"], config["ws"])
        print('Recolored Degraded PC is being projected')
        (ImlistRecD, OMListRecD) = ProjQM.orthographic_projection(RecoloredD, config["precision"], config["ws"])
        
        # loop over six projected images
        for i in range(0,6):
            # view selection
            ImR_curr = ImlistR[i].astype(np.uint8)
            ImD_curr = ImlistD[i].astype(np.uint8)
            ImRecR_curr = ImlistRecR[i].astype(np.uint8)
            ImRecD_curr = ImlistRecD[i].astype(np.uint8)
            OMR_curr = OMListR[i]
            OMD_curr = OMListD[i]
            
            # Cropping
            (ImR_cropped, OMR_cropped) = ProjQM.cropp_images(ImR_curr, OMR_curr)
            (ImD_cropped, OMD_cropped) = ProjQM.cropp_images(ImD_curr, OMD_curr)
            (ImRecR_cropped, _) = ProjQM.cropp_images(ImRecR_curr, OMR_curr)
            (ImRecD_cropped, _) = ProjQM.cropp_images(ImRecD_curr, OMD_curr)
            
            # padding 
            ImR_padded = ProjQM.pad_images(ImR_cropped, OMR_cropped)
            ImD_padded = ProjQM.pad_images(ImD_cropped, OMD_cropped)
            ImRecR_padded = ProjQM.pad_images(ImRecR_cropped, OMR_cropped)
            ImRecD_padded = ProjQM.pad_images(ImRecD_cropped, OMD_cropped)
            
            # rgb to yuv            
            ImR_padded_yuv = cv2.cvtColor(ImR_padded, cv2.COLOR_RGB2YUV)
            ImR_padded_y, _, _ = cv2.split(ImR_padded_yuv)
            ImD_padded_yuv = cv2.cvtColor(ImD_padded, cv2.COLOR_RGB2YUV)
            ImD_padded_y, _, _ = cv2.split(ImD_padded_yuv)   
            ImRecR_padded_yuv = cv2.cvtColor(ImRecR_padded, cv2.COLOR_RGB2YUV)
            ImRecR_padded_y, _, _ = cv2.split(ImRecR_padded_yuv)
            ImRecD_padded_yuv = cv2.cvtColor(ImRecD_padded, cv2.COLOR_RGB2YUV)
            ImRecD_padded_y, _, _ = cv2.split(ImRecD_padded_yuv) 
            
            # save Images to use
            if (config["saveImages"]):
                # save only padded cropped images
                directory = os.path.dirname(config["PathRefImages"])
                try:                                                           # if path directory does not exist, create it!
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                fn = config["PathRefImages"] + refPC + str(i) + '.bmp'         # Reference
                status = ProjQM.save_images(fn,cv2.cvtColor(ImR_padded, cv2.COLOR_RGB2BGR))
                print(f'Reference Image from view {i} written to file-system: {status}')
                
                directory = os.path.dirname(config["PathDegImages"])           
                try:                                                           # if path directory does not exist, create it!
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                fn = config["PathDegImages"] + degPC + str(i) + '.bmp'         # Degraded
                status = ProjQM.save_images(fn,cv2.cvtColor(ImD_padded, cv2.COLOR_RGB2BGR))
                print(f'Degraded Image from view {i} written to file-system: {status}')
                
                directory = os.path.dirname(config["PathRefImages_rec"])       
                try:                                                           # if path directory does not exist, create it!
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                fn = config["PathRefImages_rec"] + 'Recolored_' + refPC + str(i) + '.bmp'   # Recolored Reference
                status = ProjQM.save_images(fn,cv2.cvtColor(ImRecR_padded, cv2.COLOR_RGB2BGR))
                print(f'Recolored Reference Image from view {i} written to file-system: {status}')
                
                directory = os.path.dirname(config["PathDegImages_rec"])
                try:                                                           # if path directory does not exist, create it!
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                fn = config["PathDegImages_rec"] + 'Recolored_' + degPC + str(i) + '.bmp'   # Recolored Degraded
                status = ProjQM.save_images(fn,cv2.cvtColor(ImRecD_padded, cv2.COLOR_RGB2BGR))
                print(f'Recolored Degraded Image from view {i} written to file-system: {status}')
                
                directory = os.path.dirname(config["pathRefOMs"])
                try:                                                           # if path directory does not exist, create it!
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                fn = config["pathRefOMs"] + refPC + str(i) + '.bmp'
                status = ProjQM.save_images(fn,(OMR_cropped * 255).astype(np.uint8))    # Reference occupancy maps
                print(f'Reference occupancy map from view {i} written to file-system: {status}')
                
                directory = os.path.dirname(config["pathDegOMs"])
                try:                                                           # if path directory does not exist, create it!
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                fn = config["pathDegOMs"] + degPC + str(i) + '.bmp'            # Degraded occupancy maps
                status = ProjQM.save_images(fn,(OMD_cropped * 255).astype(np.uint8))
                print(f'Degraded occupancy map from view {i} written to file-system: {status}')
            
            # compute 2D metrics
            if (config["dists"]):
                dists_ab = ProjQM.compute_dists(ImR_padded, ImRecR_padded)               # on Reference Geometry
                dists_ba = ProjQM.compute_dists(ImD_padded, ImRecD_padded)               # on Degraded Geometry
                dists.append( -0.49870347 * dists_ab - 0.33948725 * dists_ba )           # fusion (weights are obtained using a linear regressor)
            if (config["lpips"]):
                lpips_ab = ProjQM.compute_lpips(ImR_padded, ImRecR_padded)               # on Reference Geometry
                lpips_ba = ProjQM.compute_lpips(ImD_padded, ImRecD_padded)               # on Degraded Geometry
                lpips.append( -1.06558819 * lpips_ab + 0.19072108 * lpips_ba )           # fusion (weights are obtained using a linear regressor)
            if (config["fsim"]):
                fsim_ab = ProjQM.compute_fsim(ImR_padded_y, ImRecR_padded_y)             # on Reference Geometry
                fsim_ba = ProjQM.compute_fsim(ImD_padded_y, ImRecD_padded_y)             # on Degraded Geometry
                fsim.append( 0.04069016 * fsim_ab + 0.70281206 * fsim_ba )               # fusion (weights are obtained using a linear regressor)
            if (config["vsi"]):      
                vsi_ab = ProjQM.compute_vsi(ImR_padded, ImRecR_padded)                   # on Reference Geometry
                vsi_ba = ProjQM.compute_vsi(ImD_padded, ImRecD_padded)                   # on Degraded Geometry
                vsi.append( 0.04072121 * vsi_ab + 0.64009922 * vsi_ba )                  # fusion (weights are obtained using a linear regressor)
            if (config["haarpsi"]):
                haarpsi_ab = ProjQM.compute_haarpsi(ImR_padded_y, ImRecR_padded_y)       # on Reference Geometry
                haarpsi_ba = ProjQM.compute_haarpsi(ImD_padded_y, ImRecD_padded_y)       # on Degraded Geometry
                haarpsi.append( 0.23896795 * haarpsi_ab + 0.55268535 * haarpsi_ba )      # fusion (weights are obtained using a linear regressor)
            if (config["vifp"]):
                vifp_ab = ProjQM.compute_vifp(ImR_padded_y, ImRecR_padded_y)             # on Reference Geometry
                vifp_ba = ProjQM.compute_vifp(ImD_padded_y, ImRecD_padded_y)             # on Degraded Geometry
                vifp.append( 0.67353577 * vifp_ab + 0.15882643 * vifp_ba )               # fusion (weights are obtained using a linear regressor)
            if (config["ssim"]):
                ssim_ab = ProjQM.compute_ssim(ImR_padded_y, ImRecR_padded_y)             # on Reference Geometry
                ssim_ba = ProjQM.compute_ssim(ImD_padded_y, ImRecD_padded_y)             # on Degraded Geometry
                ssim.append( 0.28337084 * ssim_ab + 0.4550572 * ssim_ba )                # fusion (weights are obtained using a linear regressor)
            if (config["ms-ssim"]):
                ms_ssim_ab = ProjQM.compute_msssim(ImR_padded_y, ImRecR_padded_y)        # on Reference Geometry
                ms_ssim_ba = ProjQM.compute_msssim(ImD_padded_y, ImRecD_padded_y)        # on Degraded Geometry
                ms_ssim.append( 0.87409658 * ms_ssim_ab - 0.25197625 * ms_ssim_ba )      # fusion (weights are obtained using a linear regressor)
            if (config["psnrhvs"]):
                zp0 = 8 - (ImR_padded_y.shape[0] % 8)                                    # on Reference Geometry
                zp1 = 8 - (ImR_padded_y.shape[1] % 8)
                zp_ImR_padded_y = ImR_padded_y
                zp_ImRecR_padded_y = ImRecR_padded_y
                if (zp0 != 0):
                    zp_ImR_padded_y = np.concatenate((zp_ImR_padded_y,np.zeros([zp0, zp_ImR_padded_y.shape[1]])),axis=0)
                    zp_ImRecR_padded_y = np.concatenate((zp_ImRecR_padded_y,np.zeros([zp0, zp_ImRecR_padded_y.shape[1]])),axis=0)
                if (zp1 != 0):
                    zp_ImR_padded_y = np.concatenate((zp_ImR_padded_y,np.zeros([zp_ImR_padded_y.shape[0], zp1])),axis=1)
                    zp_ImRecR_padded_y = np.concatenate((zp_ImRecR_padded_y,np.zeros([zp_ImRecR_padded_y.shape[0], zp1])),axis=1)
                hvs_temp = ProjQM.compute_hvspsnr(zp_ImR_padded_y, zp_ImRecR_padded_y)
                psnrhvsm_ab = hvs_temp[0] 
                psnrhvs_ab = hvs_temp[1]
                zp0 = 8 - (ImD_padded_y.shape[0] % 8)                                    # on Degraded Geometry
                zp1 = 8 - (ImD_padded_y.shape[1] % 8)
                zp_ImD_padded_y = ImD_padded_y
                zp_ImRecD_padded_y = ImRecD_padded_y
                if (zp0 != 0):
                    zp_ImD_padded_y = np.concatenate((zp_ImD_padded_y,np.zeros([zp0, zp_ImD_padded_y.shape[1]])),axis=0)
                    zp_ImRecD_padded_y = np.concatenate((zp_ImRecD_padded_y,np.zeros([zp0, zp_ImRecD_padded_y.shape[1]])),axis=0)
                if (zp1 != 0):
                    zp_ImD_padded_y = np.concatenate((zp_ImD_padded_y,np.zeros([zp_ImD_padded_y.shape[0], zp1])),axis=1)
                    zp_ImRecD_padded_y = np.concatenate((zp_ImRecD_padded_y,np.zeros([zp_ImRecD_padded_y.shape[0], zp1])),axis=1)
                hvs_temp = ProjQM.compute_hvspsnr(zp_ImD_padded_y, zp_ImRecD_padded_y)
                psnrhvsm_ba = hvs_temp[0] 
                psnrhvs_ba = hvs_temp[1]
                psnrhvs.append( -0.49218699 * psnrhvs_ab + 1.24867639 * psnrhvs_ba )      # fusion (weights are obtained using a linear regressor)
                psnrhvsm.append( -0.27060422 * psnrhvsm_ab + 1.02825219 * psnrhvsm_ba )   # fusion (weights are obtained using a linear regressor)
            if (config["psnr"]):
                psnr_ab = ProjQM.compute_psnr(ImR_padded_y, ImRecR_padded_y)              # on Reference Geometry
                psnr_ba = ProjQM.compute_psnr(ImD_padded_y, ImRecD_padded_y)              # on Degraded Geometry
                psnr.append( -0.51929311 * psnr_ab + 1.2754385 * psnr_ba )                # fusion (weights are obtained using a linear regressor)
    
    # list of output metrics
    results = []
    results.append(["DISTS","LPIPS","FSIM","VSI","HaarPSI","VIFp","SSIM","MS-SSIM","PSNR-HVS","PSNR-HVS-M","PSNR"])
    
    # averaging the metric over six different projected views from each PC
    if (config["dists"]):
        dists_sym = sum(dists) / len(dists)
        print(f'DSITS: {dists_sym}')
    else:
        dists_sym = ''
    if (config["lpips"]):
        lpips_sym = sum(lpips) / len(lpips)
        print(f'LPIPS: {lpips_sym}')
    else:
        lpips_sym = ''
    if (config["fsim"]):
        fsim_sym = sum(fsim) / len(fsim)
        print(f'FSIM: {fsim_sym}')
    else:
        fsim_sym = ''
    if (config["vsi"]):
        vsi_sym = sum(vsi) / len(vsi)
        print(f'VSI: {vsi_sym}')
    else:
        vsi_sym = ''
    if (config["haarpsi"]):  
        haarpsi_sym = sum(haarpsi) / len(haarpsi)
        print(f'HaarPSI: {haarpsi_sym}')
    else:
        haarpsi_sym = ''
    if (config["vifp"]):
        vifp_sym = sum(vifp) / len(vifp)
        print(f'VIFp: {vifp_sym}')
    else:
        vifp_sym = ''
    if (config["ssim"]):
        ssim_sym = sum(ssim) / len(ssim)
        print(f'SSIM: {ssim_sym}')
    else:
        ssim_sym = ''
    if (config["ms-ssim"]): 
        ms_ssim_sym = sum(ms_ssim) / len(ms_ssim)
        print(f'MS-SSIM: {ms_ssim_sym}')
    else:
        ms_ssim_sym = ''
    if (config["psnrhvs"]):    
        psnrhvs_sym = sum(psnrhvs) / len(psnrhvs)
        psnrhvsm_sym = sum(psnrhvsm) / len(psnrhvsm)
        print(f'PSNR_HVS: {psnrhvs_sym}')
        print(f'PSNR_HVS_M: {psnrhvsm_sym}')
    else:
        psnrhvs_sym = ''
        psnrhvsm_sym = ''
    if (config["psnr"]):
        psnr_sym = sum(psnr) / len(psnr)
        print(f'Y-PSNR: {psnr_sym}')
    else:
        psnr_sym = ''
    
    results.append([dists_sym,lpips_sym,fsim_sym,vsi_sym,haarpsi_sym,vifp_sym,ssim_sym,ms_ssim_sym,psnrhvs_sym,psnrhvsm_sym,psnr_sym])
    df = pd.DataFrame(results)
    df.to_csv(cli_args.outfile,index=False, header=True)
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
