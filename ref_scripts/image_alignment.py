"""
author: Katsuya Lex Colon
updated: 07/06/22
"""

#basic analysis package
import numpy as np
import pandas as pd
import time
#image analysis packages
from photutils.detection import DAOStarFinder
import rawpy
from scipy import ndimage
from imageio import imsave
import cv2
import sklearn.neighbors as nbrs
from skimage import registration
import tifffile as tf
from astropy.stats import sigma_clip
#fitting
from scipy.stats import norm
#parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
#organization packages
from pathlib import Path
import glob
import shutil
import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_optimum_fwhm(data, threshold,fwhm_range=(10,100)):
    """
    Finds the best fwhm
    Parameters
    ----------
    data = 2D array
    threshold = initial threshold for testing
    """
    #generate fwhm to test
    fwhm_range = np.linspace(fwhm_range[0],fwhm_range[1],4)
    #get counts
    counts = []
    for fwhm in fwhm_range:
        try:
            dots = len(daofinder(data,  threshold, fwhm))
        except:
            continue
        counts.append(dots)
    #find index with largest counts
    best_index = np.argmax(counts)
    #this is the best fwhm
    best_fwhm = fwhm_range[best_index]
    
    return best_fwhm

def daofinder(data,  threshold, fwhm = 4.0):
    """
    This function will return the output of daostarfinder
    Parameters
    ----------
    data = 2D array
    threshold = absolute intensity threshold
    fwhm = full width half maximum
    """
    #use daostarfinder to pick dots
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, brightest=None, exclude_border=True)
    sources = daofind(data)
    
    #return none if nothing was picked else return table
    if sources == None:
        return None
    for col in sources.colnames:
         sources[col].info.format = '%.8g'  # for consistent table output
    
    return sources.to_pandas()

def find_fiducials(image, threshold=100, fwhm_range=(5,10)):
    """
    This funtion will pick spots using daostarfinder.
    
    Parameters
    ----------
    image: image tiff
    threshold: pixel value the image must be above
    fwhm_range: range of fwhm to test
    
    Returns
    -------
    centroids
    """
    #get best fwhm
    fwhm = get_optimum_fwhm(image, threshold=threshold, fwhm_range=fwhm_range)
    
    #detect fiducials
    dots = daofinder(image, threshold=threshold, fwhm=fwhm)
    xy = dots[["xcentroid","ycentroid"]].values
    
    return xy

def read_image(src):
    """
    Read raw images from camera
    Parameters
    ----------
    src: path to image
    """
    
    raw = rawpy.imread(src)
    rgb = raw.postprocess(gamma=(1,1), output_bps=16, no_auto_bright=True)
    
    return rgb

def translational_alignment_single(image_ref,image_moving):
    """A function to obtain translational offsets using phase correlation. Image input should have the format z,c,x,y.
    Parameters
    ----------
    image_ref: image path
    image_moving: image you are trying to align path
    
    Output
    -------
    image (c,z,x,y)
    """
    #calculate shift on max projected dapi
    shift,error,phasediff = registration.phase_cross_correlation(
        image_ref,image_moving, upsample_factor=20)
    
    #apply shift to each channel
    rgb = []
    for i in range(3):
        aligned_img = ndimage.shift(image_moving[:,:,i],shift[:2])
        rgb.append(aligned_img)
        
    #recreate rgb    
    aligned_img = np.dstack((rgb[0],rgb[1],rgb[2]))

    return aligned_img

def dot_displacement(dist_arr):
    """
    This function will calculate the localization precision by fitting a 1d gaussian
    to a distance array obtained from colocalizing dots. The full width half maximum of this
    1D gaussian will correspond to displacement.
    
    Parameters
    ----------
    dist_arr: 1D distance array

    Return
    ----------
    displacement
    """
    
    #create positive and negative distance array
    dist_arr = np.concatenate([-dist_arr,dist_arr])
    
    #fit gaussian distribution
    mu, std = norm.fit(dist_arr) 
    xmin, xmax = min(dist_arr), max(dist_arr)
    x = np.linspace(xmin, xmax, 500)
    p = norm.pdf(x, mu, std)
    
    #get half maximum of gaussian
    half_max = max(p)/2
    
    #get half width at half maximum
    index_hwhm = np.where(p > max(p)/2)[0][-1]
    
    #get displacement by looking at fullwidth
    displacement = x[index_hwhm]*2
    
    return displacement

def nearest_neighbors(ref_points, fit_points, max_dist=None):
    """
    This function finds corresponding points between two point sets of the same length using 
    nearest neighbors, optionally throwing away pairs that are above max_dist pixels apart,
    and optionally aggregating the distance vector (e.g. for minimization of some norm)
    
    Parameters
    ----------
    ref_points = list containing x,y coord acting as reference 
    fit_points = list containing x,y coord for the dots we want to align to ref
    max_dist = number of allowed pixels two dots can be from each other
    
    Returns
    -------
    dists and a list of indices of fit_points which correspond.
    """
    
    #initiate neighbors
    ref_neighbors = nbrs.NearestNeighbors(n_neighbors=1).fit(ref_points)
    #perform search
    dists, ref_indices = ref_neighbors.kneighbors(fit_points)
    #get distance values for nearest neighbor
    dists = np.array(dists)[:, 0]
    #flatten indicies
    ref_indices = ref_indices.ravel()
    #generate indicies for fit
    fit_indices = np.arange(0, len(fit_points), 1)
    
    #remove pairs over a max dist
    if max_dist is not None:
        to_drop = np.where(dists > max_dist)
        
        dists[to_drop] = -1
        ref_indices[to_drop] = -1
        fit_indices[to_drop] = -1
                
        dists = np.compress(dists != -1, dists)
        ref_indices = np.compress(ref_indices != -1, ref_indices)
        fit_indices = np.compress(fit_indices != -1, fit_indices)
    
    return dists, ref_indices, fit_indices
    
def nearest_neighbors_transform(ref_points, fit_points, max_dist=None, ransac_threshold = 0.5):
    """
    This function will take two lists of non-corresponding points and identify corresponding points less than max_dist apart
    using nearest_neighbors(). Then it will find a transform that wil bring the second set of dots to the first.
    Affine transformation with RANSAC was used to estimate transform. 
    
    Parameters
    ----------
    ref_points: list of x,y coord of ref
    fit_points = list of x,y coord of raw
    max_dist = maximum allowed distance apart two points can be in neighbor search
    ransac_threshold = adjust the max allowed error in pixels
    
    Returns
    -------
    transform object, distances
    """

    #convert lists to arrays
    ref_points = np.array(ref_points)
    fit_points = np.array(fit_points)
    
    #check if dots have nan, if so remove
    ref_points = ref_points[~np.isnan(ref_points).any(axis=1)]
    fit_points = fit_points[~np.isnan(fit_points).any(axis=1)]
    
    #find nearest neighbors
    dists, ref_inds, fit_inds = nearest_neighbors(ref_points, fit_points, max_dist=max_dist)
    
    #get ref point coord and fit point coord used
    ref_pts_corr = ref_points[ref_inds]
    fit_pts_corr = fit_points[fit_inds]
    
    #estimate affine matrix using RANSAC
    tform = cv2.estimateAffine2D(fit_pts_corr, ref_pts_corr, ransacReprojThreshold=ransac_threshold)[0]

    return tform, dists, ref_pts_corr, fit_pts_corr

def alignment_error(ref_points_affine, moving_points_affine, 
                    ori_dist_list, tform_list, max_dist=2):
    
    """
   This function will calculate localization precision by obtaining FWHM of corrected distance array.
   
   Parameters
   ----------
   ref_points_affine: reference points used in transform
   moving_points_affine: points that are moving to reference
   ori_dist_list: original distance calculated prior to transform
   tform_list: list of affine transform matrix
   max_dist: number of allowed pixels two dots can be from each other
   """
    
    #apply transform to each moving point and calculate displacement 
    new_dist_by_channel = []
    old_dist_by_channel = []
    for i in range(len(moving_points_affine)):
        #reformat points
        moving = moving_points_affine[i].reshape(1, moving_points_affine[i].shape[0], moving_points_affine[i].shape[1])
        #perform transform on 2 coord points
        tform_points = cv2.transform(moving, tform_list[i])[0]
        
        #get new distance
        dists_new, _, _ = nearest_neighbors(ref_points_affine[i], tform_points, max_dist=max_dist)
        
        #remove distances beyond 2 pixels as they are most likely outliers after transform
        dists_new = dists_new[dists_new <= 2]
        
        #calculate localization precision
        displacement_new = dot_displacement(dists_new)
        displacement_old = dot_displacement(ori_dist_list[i])
        new_dist_by_channel.append(displacement_new)
        old_dist_by_channel.append(displacement_old)
    
    #calculate percent improvement
    percent_improvement_list = []
    for i in range(len(new_dist_by_channel)):
        percent_change = ((new_dist_by_channel[i]-old_dist_by_channel[i])/old_dist_by_channel[i])
        if percent_change < 0:
            percent_improvement = np.abs(percent_change)
            percent_improvement_list.append([i,percent_improvement,new_dist_by_channel[i]])
        else:
            percent_improvement = -percent_change
            percent_improvement_list.append([i,percent_improvement,new_dist_by_channel[i]])
            
    return percent_improvement_list

def pick_threshold(ref_src):
    """
    Function to find threshold to pick bright objects from rgb image for alignment.
    Parameters
    ----------
    ref_src: path the reference image
    """
    img = read_image(ref_src)
    split_ch = []
    for i in range(3):
        flattened = np.ravel(img[:,:,i])
        split_ch.append(flattened)
    
    rgb_thresholds = []
    #create positive and negative pixel int
    for i in range(len(split_ch)):
        pixel_int = np.concatenate([-split_ch[i],split_ch[i]])
        #fit gaussian distribution
        mu, std = norm.fit(pixel_int) 
        #get two std above mean
        threshold = mu + (2*std)
        rgb_thresholds.append(threshold)
        
    return rgb_thresholds

def project_images(files, type_of_projection = "sigma_clip_mean"):
    """
    A function to project exposure stacks
    Parameters
    ----------
    files: list of images to project
    type_of_projection: sigma_clip_mean or median
    """
    #generate output path
    output_dir = Path(files[0]).parent
    output_dir = output_dir / "projected_image"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{type_of_projection}_projected_image.tif"
    
    #read images
    imgs = []
    for img in files:
        imgs.append(tf.imread(img))
        
    #convert to stacked array
    imgs = np.array(imgs)
    
    #median of each channel
    rgb = []
    if type_of_projection == "sigma_clip_mean":
        sigma = int(input("what sigma value:"))
    for i in range(3):
        if type_of_projection == "median":
            rgb.append(np.median(imgs[:,:,:,i], axis=0).astype(np.uint16))
        elif type_of_projection == "sigma_clip_mean":
            clipped = sigma_clip(imgs[:,:,:,i], sigma=sigma, cenfunc='mean', maxiters=None, 
                          axis=0, masked=False, return_bounds=False)
            avg_clipped = np.mean(clipped, axis=0).astype(np.uint16)
            rgb.append(avg_clipped)
            
    stack_p = np.dstack((rgb[0],rgb[1],rgb[2]))
    
    tf.imwrite(str(output_path), stack_p)

def phase_affine_alignment_single(tiff_src, ref_src, max_dist=2, 
                                  ransac_threshold=0.5,fwhm_range = (5,10), 
                                  phase_only=False, write = True):
    """
    Parameters
    ----------
    tiff_src: raw tiff source
    ref_src: reference image to align the image
    max_dist: max distance for neighbor search
    ransac_threshold: adjust the max allowed error in pixels
    fwhm_range: range of fwhm to test
    phase_only: perform only phase correlation
    write: bool to write image
    """
    
    #output path
    parent_dir = Path(tiff_src).parent
    #make new dir
    output_dir = parent_dir / "aligned_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    #make output path
    img_name = Path(tiff_src).name.split(".")
    img_name = img_name[:len(img_name)-1]
    img_name = "_".join(img_name) + ".tif"
    output_path = output_dir / img_name
    
    #read in image
    img_ref = read_image(ref_src)
    img_moving = read_image(tiff_src)
    
    #perform translation adjustment first
    img_moving = translational_alignment_single(img_ref,img_moving)
    
    #find thresholds
    rgb_thresh = pick_threshold(ref_src)
    
    if phase_only ==False:
        #apply tform
        rgb = []
        for i in range(3):
            #get reference spots
            dots_ref = find_fiducials(img_ref[:,:,i], threshold=rgb_thresh[i], fwhm_range = fwhm_range)
            dots_moving = find_fiducials(img_moving[:,:,i], threshold=rgb_thresh[i], fwhm_range = fwhm_range)

            #get affine transform matrix, original distance, reference points and moving points used for each channel
            tform, ori_dist, ref_pts, fit_pts = nearest_neighbors_transform(dots_ref, dots_moving,
                                                                        max_dist=max_dist,ransac_threshold=ransac_threshold)

        
            transformed_image = cv2.warpAffine(img_moving[:,:,i], tform, dsize=(img_moving.shape[1],img_moving.shape[0]))
            rgb.append(transformed_image)
        
        #recreate rgb    
        transformed_image = np.dstack((rgb[0],rgb[1],rgb[2]))
        
        #check alignment error
        error = alignment_error([ref_pts], [fit_pts], [ori_dist], [tform], max_dist)
    else:
        transformed_image = img_moving
    
    if write == True:
        print(output_path)
        #write image
        imsave(str(output_path), transformed_image)
        if phase_only ==False:
            #write error
            txt_name = img_name.replace(".tif","_error.txt")
            output_text =  output_dir / txt_name
            with open(str(output_text),"w+") as f:
                f.write(str(error[0][0]) + " " + str(error[0][1]) + " " + 
                        str(error[0][2]) + "\n")
            f.close()
    else:    
        if phase_only ==False:
            error = pd.DataFrame(error)
            error.columns = ["Channels","Percent Improvement","FWHM"]
            return transformed_image, error
        else:
            return transformed_image
        
def phase_affine_parallel(tiff_src_list, ref_src, max_dist=2, 
                          ransac_threshold=0.5,fwhm_range = (5,10), 
                          phase_only=False):
    """
    Parameters
    ----------
    tiff_src_list: list of tiff srcs
    ref_src: reference image to align the image
    max_dist: max distance for neighbor search
    ransac_threshold: adjust the max allowed error in pixels
    fwhm_range: range of fwhm to test
    phase_only: perform only phase correlation
    """
    
    #parallel processing of each image
    with ProcessPoolExecutor(max_workers=20) as exe:
        for tiff in tiff_src_list:
            exe.submit(phase_affine_alignment_single, tiff, ref_src, max_dist=max_dist, 
                       ransac_threshold=ransac_threshold,fwhm_range = fwhm_range, 
                       phase_only=phase_only, write = True)
            
    #copy ref image to output folder
    #output path
    parent_dir = Path(tiff_src_list[0]).parent
    #make new dir
    output_dir = parent_dir / "aligned_images"
    #rename file to end in tif
    ref_name = Path(ref_src).name
    ref_name_new = ref_name.split(".")[:len(ref_name.split("."))-1]
    ref_name_new = "_".join(ref_name_new) + ".tif"
    output_path = output_dir / ref_name_new
    #copy ref image to new path
    img_ref = read_image(ref_src)
    imsave(str(output_path), img_ref)
    
