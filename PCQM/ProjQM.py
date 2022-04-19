# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:10:42 2021

@author: AlirezaJav
         Instituto de Telecominicações
         Lisbon, Portugal
"""

import numpy as np
import os
import cv2
import open3d as o3d
from collections import Counter
from IQA_pytorch import VIFs, SSIM, MS_SSIM, FSIM, VSI, LPIPSvgg, DISTS
from IQA_pytorch.utils import prepare_image
import torch
import math

'''
A function to read the configuration file
configuration file contains different metric parameters

Input: file name
Output: a dictionary inclduing metric parameters
'''
def read_configurations(fn):
    import configparser
    cfg = {}
    config = configparser.ConfigParser()
    config.read(fn)
    
    # read paths
    cfg['pathRef'] = config.get('Paths', 'PCA')       
    cfg['pathDeg'] = config.get('Paths', 'PCB')
    #cfg['pathRef_rec'] = config.get('Paths', 'PCA_rec')
    #cfg['pathDeg_rec'] = config.get('Paths', 'PCB_rec')
    #cfg['PathRefImages'] = config.get('Paths', 'RefImages')
    #cfg['PathDegImages'] = config.get('Paths', 'DegImages')
    #cfg['PathRefImages_rec'] = config.get('Paths', 'RefImages_rec')
    #cfg['PathDegImages_rec'] = config.get('Paths', 'DegImages_rec')
    #cfg['pathRefOMs'] = config.get('Paths', 'RefOMs')
    #cfg['pathDegOMs'] = config.get('Paths', 'DegOMs')
    
    # read flags
    cfg['Projected'] = int( config.get('Flags', 'projected') )
    cfg['Recolored'] = int( config.get('Flags', 'Recolored') )
    cfg['savePCs'] = int( config.get('Flags', 'savePCs') )
    cfg['saveImages'] = int( config.get('Flags', 'saveImages') )
    
    # read parameters
    cfg['precision'] = int( config.get('parameters', 'precision') )
    cfg['ws'] = int( config.get('parameters', 'window size') )
    
    # read metric names
    cfg['dists'] = int( config.get('2D Metrics', 'dists') )
    cfg['lpips'] = int( config.get('2D Metrics', 'lpips') )
    cfg['fsim'] = int( config.get('2D Metrics', 'fsim') )
    cfg['vsi'] = int( config.get('2D Metrics', 'vsi') )
    cfg['haarpsi'] = int( config.get('2D Metrics', 'haarpsi') )
    cfg['vifp'] = int( config.get('2D Metrics', 'vifp') )
    cfg['ssim'] = int( config.get('2D Metrics', 'ssim') )
    cfg['ms-ssim'] = int( config.get('2D Metrics', 'ms-ssim') )
    cfg['psnrhvs'] = int( config.get('2D Metrics', 'psnr-hvs') )
    cfg['psnr'] = int( config.get('2D Metrics', 'psnr') )
    
    return cfg


'''
A function to read all the six projected images of a PC
All images of a PC should be in a distinct folder

Input: folder name
Output: a list including all six images
'''
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

'''
A function to recolor cloudA with color in cloud B and vice versa.
All images of a PC should be in a distinct folder

Input1: point cloud A
Input2: point cloud B
Output: two recolored point clouds
'''
def recolor_pcs(cloudA, cloudB):
    
    GA = np.asarray(cloudA.points)
    CA = np.asarray(cloudA.colors)
    GB = np.asarray(cloudB.points)
    CB = np.asarray(cloudB.colors)
    NCTA = np.zeros(CA.shape)
    NCTB = np.zeros(CB.shape)
    SearchTreeA = o3d.geometry.KDTreeFlann(cloudA)
    SearchTreeB = o3d.geometry.KDTreeFlann(cloudB)
    
    ANN = []
    for index in range(len(GA)):
        [_, idx, dist] = SearchTreeB.search_knn_vector_3d(GA[index], 1)
        NCTA[index,:] = CB[idx[0],:] * CB[idx[0],:]
        ANN.append(idx[0])
        
    BNN = []
    for index in range(len(GB)):
        [_, idx, dist] = SearchTreeA.search_knn_vector_3d(GB[index], 1)
        NCTB[index,:] = CA[idx[0],:] * CA[idx[0],:]
        BNN.append(idx[0])
    
    ANN = np.asarray(ANN)
    BNN = np.asarray(BNN)
    
    ANN_rep = Counter(ANN)
    BNN_rep = Counter(BNN)  
    
    for item in BNN_rep.keys():
        if BNN_rep[item] > 1:
            NCTA[item,:] = sum(CB[BNN == item, :] ** 2) / BNN_rep[item]
    NCTA = np.sqrt(NCTA)
    
    for item in ANN_rep.keys():
        if ANN_rep[item] > 1:
            NCTB[index,:] = sum(CA[ANN == item, :] ** 2) / ANN_rep[item]
    NCTB = np.sqrt(NCTB)
    
    recoloredA = o3d.geometry.PointCloud()
    recoloredA.points = o3d.utility.Vector3dVector(GA)
    recoloredA.colors = o3d.utility.Vector3dVector(NCTA)
    
    recoloredB = o3d.geometry.PointCloud()
    recoloredB.points = o3d.utility.Vector3dVector(GB)
    recoloredB.colors = o3d.utility.Vector3dVector(NCTB)
    
    return (recoloredA, recoloredB)


'''
A function to orthographically project a PC on 6 faces of the surrouding cube.

Input1: point cloud A
Input2: point cloud B
Output: two recolored point clouds
'''
def orthographic_projection(cloud, precision, filtering):
    
    geometry = np.asarray(cloud.points)
    if cloud.has_colors():
        color = np.asarray(cloud.colors)

    # color
    img_0 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_1 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_2 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_3 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_4 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_5 = np.ones(([2**precision, 2**precision, 3])) * 255
    
    # occupancy map
    ocp_map_0 = np.zeros(([2**precision,2**precision]))
    ocp_map_1 = np.zeros(([2**precision,2**precision]))
    ocp_map_2 = np.zeros(([2**precision,2**precision]))
    ocp_map_3 = np.zeros(([2**precision,2**precision]))
    ocp_map_4 = np.zeros(([2**precision,2**precision]))
    ocp_map_5 = np.zeros(([2**precision,2**precision]))
    
    img = [img_0, img_1, img_2, img_3, img_4, img_5]
    ocp_map = [ocp_map_0, ocp_map_1, ocp_map_2, ocp_map_3, ocp_map_4, ocp_map_5]
    minDepth = np.zeros([3,2**precision,2**precision])
    maxDepth = np.ones([3,2**precision,2**precision]) * 2**precision
    
    plane = {0: (1,2), 1: (0,2), 2: (0,1)}
    
    for index in range(len(geometry)):
        if (geometry[index][0] >= 2**precision) or (geometry[index][1] >= 2**precision) or (geometry[index][2] >= 2**precision):
            continue
        else:
            for coord in range(0,3):
                if geometry[index][coord] <= maxDepth[coord][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)]:
                    img[2 * coord][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)][:] = color[index][:] * 255
                    ocp_map[2 * coord][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)] = 1
                    maxDepth[coord][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)] = geometry[index][coord]
                
                if geometry[index][coord] >= minDepth[coord][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)]:
                    img[2 * coord + 1][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)][:] = color[index][:] * 255
                    ocp_map[2 * coord + 1][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)] = 1
                    minDepth[coord][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)] = geometry[index][coord]
    w = 2
    c1 = c2 = c3 = c4 = c5 = c6 = 0
    for i in range(w,2**precision - w):
        for j in range(w,2**precision - w):
            if (ocp_map[0][i,j] == 1) and (maxDepth[0][i,j] > sum(sum(maxDepth[0][i-w:i+w+1,j-w:j+w+1] * ocp_map[0][i-w:i+w+1,j-w:j+w+1]))/sum(sum(ocp_map[0][i-w:i+w+1,j-w:j+w+1])) + 20):
                ocp_map[0][i,j] = 0
                img[0][i,j,:] = 255   
                c1 +=1
            if (ocp_map[1][i,j] == 1) and (minDepth[0][i,j] < sum(sum(minDepth[0][i-w:i+w+1,j-w:j+w+1] * ocp_map[1][i-w:i+w+1,j-w:j+w+1]))/sum(sum(ocp_map[1][i-w:i+w+1,j-w:j+w+1])) - 20):
                ocp_map[1][i,j] = 0
                img[1][i,j,:] = 255
                c2 +=1
            if (ocp_map[2][i,j] == 1) and (maxDepth[1][i,j] > sum(sum(maxDepth[1][i-w:i+w+1,j-w:j+w+1] * ocp_map[2][i-w:i+w+1,j-w:j+w+1]))/sum(sum(ocp_map[2][i-w:i+w+1,j-w:j+w+1])) + 20):
                ocp_map[2][i,j] = 0
                img[2][i,j,:] = 255
                c3 +=1
            if (ocp_map[3][i,j] == 1) and (minDepth[1][i,j] < sum(sum(minDepth[1][i-w:i+w+1,j-w:j+w+1] * ocp_map[3][i-w:i+w+1,j-w:j+w+1]))/sum(sum(ocp_map[3][i-w:i+w+1,j-w:j+w+1])) - 20):
                ocp_map[3][i,j] = 0
                img[3][i,j,:] = 255
                c4 +=1
            if (ocp_map[4][i,j] == 1) and (maxDepth[2][i,j] > sum(sum(maxDepth[2][i-w:i+w+1,j-w:j+w+1] * ocp_map[4][i-w:i+w+1,j-w:j+w+1]))/sum(sum(ocp_map[4][i-w:i+w+1,j-w:j+w+1])) + 20):
                ocp_map[4][i,j] = 0
                img[4][i,j,:] = 255
                c5 +=1
            if (ocp_map[5][i,j] == 1) and (minDepth[2][i,j] < sum(sum(minDepth[2][i-w:i+w+1,j-w:j+w+1] * ocp_map[5][i-w:i+w+1,j-w:j+w+1]))/sum(sum(ocp_map[5][i-w:i+w+1,j-w:j+w+1])) - 20):
                ocp_map[5][i,j] = 0
                img[5][i,j,:] = 255
                c6 +=1
    
    print("{t1} points removed from 1st view".format(t1=c1))
    print("{t1} points removed from 2nd view".format(t1=c2))
    print("{t1} points removed from 3rd view".format(t1=c3))
    print("{t1} points removed from 4th view".format(t1=c4))
    print("{t1} points removed from 5th view".format(t1=c5))
    print("{t1} points removed from 6th view".format(t1=c6))
    return (img, ocp_map)

'''
A function to find the bounding rectangle around the object and cropp it out of
the whole image.

Input1: Projected image
Input2: Occupany map
Output: Cropped image
'''
def cropp_images(image, ocp_map):
    x,y,w,h = cv2.boundingRect(ocp_map.astype(np.uint8))
    cropped_image = image[y:y+h, x:x+w]
    cropped_ocp_map = ocp_map[y:y+h, x:x+w]
    return (cropped_image, cropped_ocp_map)


'''
A function to fill empty space among pixels and also background area around object.

Input1: Cropped image
Input2: Occupany map
Output: Padded image
'''
def pad_images(image, ocp_map):
    mask = (ocp_map != 1).astype(np.uint8)
    padded_image = cv2.inpaint(image,mask,3,cv2.INPAINT_NS)
    return padded_image

'''
A function to save projected images after pre-processing for further use.

Input1: Output image file name
Input2: Image
Output: status of the writting procedure
'''
def save_images(fn, image):
    status = cv2.imwrite(fn,image)
    return status


def compute_dists(original, decoded):
    ref = prepare_image(original)
    dist = prepare_image(decoded)
    q_metric = DISTS()
    return q_metric(dist, ref, as_loss=False).item()

def compute_haarpsi(original, decoded):
    from piq import haarpsi
    # HaarPSI_metric = haarpsi()
    ref = prepare_image(original)
    dist = prepare_image(decoded)
    HaarPSI: torch.Tensor = haarpsi(ref, dist)
    return HaarPSI.item()

def compute_hvspsnr(original, decoded):
    import psnr_hvs_m
    p = psnr_hvs_m.psnrhvsm(original,decoded)
    return p
    
def compute_vifp(original, decoded):
    ref = prepare_image(original)
    dist = prepare_image(decoded)
    q_metric = VIFs(channels=1)
    return q_metric(dist, ref, as_loss=False).item()

def compute_ssim(original, decoded):
    ref = prepare_image(original)
    dist = prepare_image(decoded)
    q_metric = SSIM(channels=1)
    return q_metric(dist, ref, as_loss=False).item()

def compute_msssim(original, decoded):
    ref = prepare_image(original)
    dist = prepare_image(decoded)
    q_metric = MS_SSIM(channels=1)
    return q_metric(dist, ref, as_loss=False).item()

def compute_fsim(original, decoded):
    ref = prepare_image(original)
    dist = prepare_image(decoded)
    q_metric = FSIM(channels=1)
    return q_metric(dist, ref, as_loss=False).item()

def compute_vsi(original, decoded):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref = prepare_image(original).to(device)
    dist = prepare_image(decoded).to(device)
    q_metric = VSI().to(device)
    return q_metric(dist, ref, as_loss=False).item()

def compute_lpips(original, decoded):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref = prepare_image(original).to(device)
    dist = prepare_image(decoded).to(device)
    q_metric = LPIPSvgg().to(device)
    return q_metric(dist, ref, as_loss=False).item()

def compute_psnr(original, decoded):
    mse = np.mean( (original - decoded) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
