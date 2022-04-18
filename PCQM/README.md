# Projection-based-PC-Quality-Metric
<b>Introduction</b>
<p>A novel joint geometry and color projection-based point cloud objective quality metric which solves the critical weakness of this type of quality metrics, i.e., the misalignment between the reference and degraded projected images. Moreover, this point cloud quality metric exploits the best performing 2D quality metrics in the literature to assess the quality of the projected images.</p>
<b>Installation Instructions</b>
<p>Image Quality Assessment (<a href="https://pypi.org/project/IQA-pytorch/">IQA</a>) Models in PyTorch should be installed</p>

```console
pip install IQA_pytorch
```
<p> An <a href="http://ponomarenko.info/psnrhvsm.htm">Implementation</a> of PSNR HVS is included in this repository. The corresponding papers are cited in our published article.</p> 
<a href="http://www.open3d.org/docs/release/getting_started.html">Open3d</a> python packages are also necessary to read/write point clouds.

```console
pip install open3d
```
Conda instalation

```console
conda install -c open3d-admin -c conda-forge open3d
```
<a href="https://pypi.org/project/opencv-python/">OpenCV</a> is also necessary for image processing operations in the metric.

```console
pip install opencv-python
```
You need to also install <a href="https://numpy.org/">Numpy</a>, <a href="https://pandas.pydata.org/">Pandas</a>, and <a href="https://pytorch.org/">PyTorch</a> (in case that you want to run 2D metrics on GPU)

<b>Usage</b>
<p>All metric parameters should be set inside the configuration file (config.ini) before running the metric. If it is the first time runing the metric and recolored point clouds are not available:</p>

```console
python3 compute_projqm.py -a reference PC -b degraded PC -c config.ini -o output.csv
```
<p>Note that path to PCs is set inside the config file, they are brought to command line arguments to make it easier to run metric for a batch of PCs.</p>
<p>If you want to save point clouds, <b>savePCs</b> should be set to 1 (savePCs = 1) </p>
<p>If you want to save projected images, <b>saveImages</b> should be set to 1 (saveImages = 1). All images of the PC should be stored in a separate folder. The path to this folder is set inside the configuration file.</p>
<p>Precision of the input point clouds should also be set in the configuration file.</p>
<p>There is also a flag for each of the 2D quality metrics that you want to be included in the final results<p>
<p> You can see a sample configuration file below: </p>

```console
[Paths]
# Path to directory including reference point clouds
PCA = C:\AlirezaJav\Datasets\EPFL_MPEG_Codecs\stimuli\  
# Path to directory including reference point clouds                           
PCB = C:\AlirezaJav\Datasets\EPFL_MPEG_Codecs\stimuli\	
# Path to directory including recolored reference point clouds                         
PCA_rec = C:\AlirezaJav\Projects\Projection-based Metric\Final Software\Recolored PCs\
# Path to directory including recolroed degraded point clouds 
PCB_rec = C:\AlirezaJav\Projects\Projection-based Metric\Final Software\Recolored PCs\
# Path to directory including six projected images of the reference point cloud (All six projected images of a point cloud should be in a folder)
RefImages = C:\AlirezaJav\Projects\Projection-based Metric\Final Software\RefImages\
# Path to directory including six projected images of the degraded point cloud (All six projected images of a point cloud should be in a folder)
DegImages = C:\AlirezaJav\Projects\Projection-based Metric\Final Software\DegImages\
# Path to directory including six projected images of the recolored reference point cloud (All six projected images of a point cloud should be in a folder)
RefImages_rec = C:\AlirezaJav\Projects\Projection-based Metric\Final Software\RecoloredRefImages\
# Path to directory including six projected images of the recolored degraded point cloud (All six projected images of a point cloud should be in a folder)
DegImages_rec = C:\AlirezaJav\Projects\Projection-based Metric\Final Software\RecoloredDegImages\
# Path to directory including six occupancy maps of the reference point cloud (All six occupancy maps of a point cloud should be in a folder)
RefOMs = C:\AlirezaJav\Projects\Projection-based Metric\Final Software\RefOMs\
# Path to directory including six occupancy maps of the degraded point cloud (All six occupancy maps of a point cloud should be in a folder)
DegOMs = C:\AlirezaJav\Projects\Projection-based Metric\Final Software\DegOms\

[Flags]
# Set to 1 if projected images are available and there is no need for projection, 0 otherwise
projected = 0
# Set to 1 if recolored point clouds are saved before and there is no need for recoloring, 0 otherwise
Recolored = 0
# Set to 1 to save recolored point clouds for further use
savePCs = 1
# Set to 1 to save projected images after pre-processing, otherwise 0 (they can be evaluate directly by 2D metric later)
saveImages = 1


[parameters]
# precision of the input PC
precision = 10
# window search size for filtering points after projection. Put zero if you don't want filtering (W = 2*window size + 1)
window size = 2

[2D Metrics]
# compute DISTS
dists = 1
# compute LPIPS
lpips = 1
# compute FSIM
fsim = 1
# compute VSI
vsi = 1
# compute HaarPSI
haarpsi = 1
# compute VIFp
vifp = 1
# compute SSIM
ssim = 1
# compute MS-SSIM
ms-ssim = 1
# compute HVS PSNR and HVS PSNR M
psnr-hvs = 1
# compute PSNR
psnr = 1
```
If Recolored PCs are already available, recolored flag should be set to 1 (Recolored = 1) and run the metric as follows:

```console
python3 compute_projqm.py -a reference PC -b degraded PC -ar recolored reference PC -br recolored degraded PC -c config.ini -o output.csv
```
If projected images are saved and you need to only compute a new 2D metric, projected flag should be set to 1 (projected = 1) and run the metric as follows:

```console
python3 compute_projqm.py -c config.ini -o output.csv
```
<b> Performance </b>
<p> Below table shows the objective-Subjective correlation performance of this metric, compared with most famous state-of-the-art metrics using the MOS scores provied in <a href="https://www.epfl.ch/labs/mmspg/downloads/quality-assessment-for-point-cloud-compression">M-PCCD</a> dataset. </p>
<table style="width:50%" align="center">
  <tr>
    <th>Type</th>
    <th>Metric</th>
    <th>SROCC</th> 
    <th>PLCC</th>
    <th>RMSE</th>
  </tr>
  <tr>
    <td>Point-to-Point</td>
    <td>D1-PSNR</td>
    <td>79.1</td>
    <td>77.7</td>
    <td>0.857</td>
  </tr>
  <tr>
    <td>Point-to-Point</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9123087/">GH 98% PSNR</a></td>
    <td>86.9</td>
    <td>84.6</td>
    <td>0.726</td>
  </tr>
  <tr>
    <td>Point-to-Point</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9191233">RA-PSNR (APD<sub>10</sub>)</a></td>
    <td>90.2</td>
    <td>88.8</td>
    <td>0.626</td>
  </tr>
  <tr>
    <td>Point-to-Point</td>
    <td>Y-PSNR</td>
    <td>66.2</td>
    <td>67.1</td>
    <td>1.009</td>
  </tr>
  <tr>
    <td>Point-to-Plane</td>
    <td>D2-PSNR</td>
    <td>83.8</td>
    <td>80.5</td>
    <td>0.808</td>
  </tr>
  <tr>
    <td>Point-to-Plane</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9123087/">GH 98% PSNR</a></td>
    <td>87.9</td>
    <td>84.3</td>
    <td>0.731</td>
  </tr>
  <tr>
    <td>Point-to-Plane</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9191233">RA-PSNR (APD<sub>10</sub>)</a></td>
    <td>89.9</td>
    <td>88.9</td>
    <td>0.622</td>
  </tr>
  <tr>
    <td>Feature-Based</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9106005">PointSSIM</a></td>
    <td>91.8</td>
    <td>92.6</td>
    <td>0.514</td>
  </tr>
  <tr>
    <td>Feature-Based</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9123089">d<sub>gc</sub></a></td>
    <td>92.0</td>
    <td>90.4</td>
    <td>0.585</td>
  </tr>
  <tr>
    <td>Feature-Based</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9123089">H<sup>Y</sup><sub>L2</sub></a></td>
    <td>88.4</td>
    <td>85.3</td>
    <td>0.710</td>
  </tr>
  <tr>
    <td>Feature-Based</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9198142">PCM<sub>RR</sub>(MCCV)</a></td>
    <td>90.7</td>
    <td>90.2</td>
    <td>0.573</td>
  </tr>
  <tr>
    <td>Point-to_Distribution</td>
    <td><a href="https://arxiv.org/abs/2108.00054">P2D-JGY</a></td>
    <td>93.8</td>
    <td>92.9</td>
    <td>0.503</td>
  </tr>
  <tr>
    <td>Point-to_Distribution</td>
    <td><a href="https://arxiv.org/abs/2108.00054">LogP2D-JGY</a></td>
    <td>93.8</td>
    <td>92.9</td>
    <td>0.502</td>
  </tr>
  <tr>
    <td>Projection-based</td>
    <td>JGC-ProjQM-FSIM</td>
    <td>90.1</td>
    <td>88.2</td>
    <td>0.640</td>
  </tr>
  <tr>
    <td>Projection-based</td>
    <td>JGC-ProjQM-VSI</td>
    <td>87.6</td>
    <td>85.4</td>
    <td>0.707</td>
  </tr>
  <tr>
    <td>Projection-based</td>
    <td>JGC-ProjQM-LPIPS</td>
    <td>93.2</td>
    <td>92.3</td>
    <td>0.523</td>
  </tr>
  <tr>
    <td><b>Projection-based</b></td>
    <td><b>JGC-ProjQM-DISTS</b></td>
    <td><b>95.6</b></td>
    <td><b>94.7</b></td>
    <td><b>0.439</b></td>
  </tr>
</table>
<b>Reference</b>
<p>If you are using this metric, please cite the following publications:</p>
A. Javaheri, C. Brites, F. Pereira, J. Ascenso <a href="https://arxiv.org/abs/2108.02481">"Joint Geometry and Color Projection-based Point Cloud Quality Metric"</a>, <i>arXiv preprint arXiv:2108.00054.</i>, July 2021.
<p>This article is submitted to IEEE Access journal.</p>
