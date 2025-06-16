import os
from pydicom import dcmread
import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import scipy, h5py
import nibabel as nib


def get_label(df_rid_sn):
    risk_label_dict = {'MILD': [0,0], 'MOD': [1,0], 'SEV': [1,1], 'SEV_NOISE':[1,1]}
    label = torch.tensor(risk_label_dict[df_rid_sn['label'].values[0]]).float()
    return label

def transform_to_hu(image, intercept, slope):
    if isinstance(intercept, np.ndarray):
        hu_image = (image*slope[:, np.newaxis, np.newaxis])+ intercept[:, np.newaxis, np.newaxis]
    else:
        hu_image = image * slope + intercept
    return hu_image

def window_image(image, window_center = 45, window_width = 400):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def load_dcm_img(img_path, window_center = 45, window_width = 400):
    #TODO: Check preprocessing step here based on seg model
    ds = dcmread(img_path)
    img_array = ds.pixel_array

    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    img_array = transform_to_hu(img_array, intercept, slope)
    img_array = window_image(img_array, window_center, window_width)
    return torch.tensor(img_array)

def load_hdf5_img_slices(img_path, index, intercept, slope, window_center=45, window_width=400):
    """ 
    Loads and processes image slices from an HDF5 file, applying HU transformation and windowing.
    
    Args:
        img_path (str): Path to the HDF5 file.
        index (int or list/array): Index of slices to extract.
        intercept (float): DICOM Rescale Intercept.
        slope (float): DICOM Rescale Slope.
        window_center (int, optional): Center of the HU window. Defaults to 45.
        window_width (int, optional): Width of the HU window. Defaults to 400.

    Returns:
        torch.Tensor: Processed image tensor.
    """
    with h5py.File(img_path, 'r') as f:
        if 'image' not in f:
            raise KeyError(f"Dataset 'image' not found in {img_path}")

        img_data = f['image']
        
        # Ensure index is within bounds
        if isinstance(index, (list, np.ndarray)):
            if max(index) >= len(img_data):
                raise IndexError(f"Index {max(index)} out of range for dataset of size {len(img_data)}")
        elif index >= len(img_data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(img_data)}")

        # Load image slices
        img_array = np.array(img_data, dtype=np.float32)[index]
    
    # Apply processing steps
    img_array = transform_to_hu(img_array, intercept, slope)
    img_array = window_image(img_array, window_center, window_width)

    return torch.tensor(img_array, dtype=torch.float32)

def load_nii_img_slices(img_path, intercept=0, slope=1, window_center=45, window_width=400):
    img = nib.load(img_path)
    
    img_array = img.get_fdata()
    # Check if the image is 4D (w, h, c, 3), and randomly choose one of the phases from the last dimension
    if len(img_array.shape) == 4:
        num_phases = img_array.shape[-1]  
        selected_phase = np.random.randint(0, num_phases) 
        img_array = img_array[:, :, :, selected_phase]

    # transform nii image shape from (w, h, c) to (c, w, h)
    img_array = img_array.transpose(2, 1, 0)       
    img_array = transform_to_hu(img_array, intercept=intercept, slope=slope)
    img_array = window_image(img_array, window_center=window_center, window_width=window_width)
    return torch.tensor(img_array)

def load_images(all_image_paths, transform=None, index=None, intercept=None, slope=None):
    img_bag = []
    for i, img_path in enumerate(all_image_paths):
        try:
            if img_path.endswith('.hdf5'): 
                # load nyu data
                img_array = load_hdf5_img_slices(img_path, index, intercept, slope)
                img_bag = img_array.unsqueeze(1)
                break
            elif img_path.endswith('.dcm') or ('.' not in img_path.split('/')[-1]):
                # hungarian DICOM file does not end with .dcm 
                # Handle DICOM images
                img_array = load_dcm_img(img_path)
                img_array = img_array.unsqueeze(0)  # Add channel dimension for 2D images
                img_bag.append(img_array)
            
            elif img_path.endswith('.nii.gz'):
                # Handle .nii.gz 3D images (output all 2D slices)
                slices = load_nii_img_slices(img_path)  # Load all 2D slices from the 3D volume
                img_bag.extend(slices)  # Append all slices to the img_bag
            else:
                raise ValueError(f"Unsupported file type: {img_path}")
        
        except Exception as e:
            print(f"Could not open image: {img_path}. Error: {e}")

    if transform:
        img_bag = [transform(img) for img in img_bag]
    
    return img_bag
            

class LazyLoadDatasetCSV2D(Dataset):
    def __init__(self,
                data_df, 
                transform=None, 
                ):
        self.filtered_data_df = data_df
        self.rid_list = self.filtered_data_df.rid.unique()
        self.transform = transform
        
    def __len__(self):
        return len(self.rid_list)

    def select_slices(self, df_rid_sn, load_index=None):
        img_idx = []
        # if we select all the slices from a series 
        images = df_rid_sn['absolute_image_path'].to_list()
        index = df_rid_sn['index'].astype(int).to_numpy() if load_index else None
        intercept = df_rid_sn['RescaleIntercept'].to_numpy() if load_index else None
        slope = df_rid_sn['RescaleSlope'].to_numpy() if load_index else None
        return images, img_idx, index, intercept, slope 

    def __getitem__(self, idx):
        # Get all slices for a patient
        rid = self.rid_list[idx]
        df_rid = self.filtered_data_df[self.filtered_data_df.rid == rid].reset_index(drop=True)
        
        #Validate on all phases of CT scan
        select_phases = list(df_rid.series_number.unique())
        img_bag = []
        series_idx = []
        for sn in select_phases:
            # pick a phase
            df_rid_sn = df_rid.loc[df_rid.series_number == sn]
            label = get_label(df_rid_sn)

            img_paths, img_idx, index, intercept, slope = self.select_slices(df_rid_sn, 
                                                            load_index=not np.isnan(df_rid_sn['index'].iloc[0]))
            images = load_images(img_paths,
                                transform=self.transform,
                                index=index, 
                                intercept=intercept, 
                                slope=slope)

            img_bag.append(torch.stack(images))
            series_idx.append(torch.tensor([sn for i in range(len(images))], dtype=torch.long))
            
        return img_bag, label, rid

