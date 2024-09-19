#!/bin/bash

# make sure to use the same version of fsl and elastix throughout the entire paper
module unload fsl
module load fsl/4.1.4
module load elastix/4.8

# cohort is not needed if data are saved in the same folder
# below is the case where subjects from different cohorts/directories are registerd
# sub2 (moving) is registered to sub1 (target)
sub1=$1
cohort1=$2
sub2=$3
cohort2=$4

# parameter files for elastix
rigid=$5
affine=$6

# directories to save results
deform=/yourPath/Preprocess_affine_transformParam/${sub2}.${sub1}
dest_def=/yourPath/Preprocess_affine_deform_map/${sub2}.${sub1}
dest_img=/yourPath/Preprocess_affine_input_img/${sub2}.${sub1}
# target and source/moving images
dtiTgt=/yourPath/$cohort1/$sub1
dtiSrc=/yourPath/$cohort2/$sub2


#init
execPath=/experimentPath
cd $execPath

mkdir -p $deform
mkdir -p $dest_img
mkdir -p $dest_def

#step 1: crop the whole image into an ROI of certain coordinates
# skip this if no need to crop, then change the file name in the reminder of codes
# to remove '_ROI'
# fslroi $dtiTgt/dti_FA.nii.gz $dest_img/tgt_FA_ROI.nii.gz 48 112 0 208 0 112 
# fslroi $dtiSrc/dti_FA.nii.gz $dest_img/src_FA_ROI.nii.gz 48 112 0 208 0 112 


#step 2: estimate deformation for each pair of image, use Elastix
elastix -threads 1 -f $dest_img/tgt_FA.nii.gz -m $dest_img/src_FA.nii.gz -p $rigid -p $affine -out $deform

#step 3: warp the moving and other needed images to the tgt space
# using the estimated transformation parameters;
# '-def all' outputs a dense displacement directly, can be inputs of the Segis-Net/Reg-Net
transformix -threads 1 -in $dest_img/src_FA.nii.gz -tp $deform/TransformParameters.1.txt -out $dest_def -def all
# change the file name of the warped source image
mv $dest_def/result.nii.gz $dest_img/warped_FA.nii.gz
 