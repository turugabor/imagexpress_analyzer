import pandas as pd
import numpy as np
from skimage import io
from skimage import exposure
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from dataclasses import dataclass, field
import gc
from cellpose import models, core
use_GPU = core.use_gpu()
print(f'>>> GPU activated? {use_GPU}')

from tqdm.notebook import tqdm
models.models_logger.disabled = True
from IPython.display import clear_output
from collections import defaultdict
import plotly.express as px
import warnings 
warnings.filterwarnings("ignore") # turn off pytorch warnings

@dataclass
class XpressImage:     

    well: str
    position: str 
    channels: tuple
    files: dict = field(default_factory=dict)
        
    def load_image(self):
        grayscales = []
        for channel in self.channels:
            grayscales.append(io.imread(self.files[channel]))          
        grayscales = [np.expand_dims(img, axis=2) for img in grayscales]
        img = np.concatenate(grayscales, axis=2)
        return img

# +
@dataclass
class XpressExperiment:

    folder_path: str
    nucleus_channel: int = 1
    cell_channel: int = 3
    channel_to_analyse: int = 2
    wells: tuple = ()
    channels: tuple = ()
    positions: tuple = ()

    def __post_init__(self):

        self.full_images = {}
        full_images_paths = [
            f"{self.folder_path}/{img}"
            for img in listdir(self.folder_path)
            if "thumb" not in img
        ]

        ### create the wells, channels and positions tuples and collect images
        wells = []
        positions = []
        channels = []

        for img_path in full_images_paths:
            well, position, channel = self.get_image_info(img_path)

            wells.append(well)
            positions.append(position)
            channels.append(channel)

        self.wells = sorted(tuple(set(wells)))
        self.channels = sorted(tuple(set(channels)))
        self.positions = sorted(tuple(set(positions)))

        for img_path in full_images_paths:
            well, position, channel = self.get_image_info(img_path)

            if well not in self.full_images.keys():
                self.full_images[well] = {}

            if position not in self.full_images[well].keys():
                self.full_images[well][position] = XpressImage(
                    well,
                    position,
                    self.channels,
                )

            self.full_images[well][position].files[channel] = img_path
            
        # initialize cellpose
        self.cytomodel = models.Cellpose(gpu=use_GPU, model_type="cyto")
        self.nucleusmodel = models.Cellpose(gpu=use_GPU, model_type="nuclei")
        
        # get background image
        print("Calculating background....", end = '')
        self.calculate_background(no_of_images=100)
        print("Done")

    def get_image_info(self, img_path):
        name = img_path.split("/")[-1]
        name_split = name.split("_")
        well = name_split[-3]
        position = name_split[-2]
        channel = name_split[-1][:2]

        return well, position, channel

    def calculate_background(self, no_of_images=100):
        """takes random images from the experiment,
         calculates the median of the images, taking the averaged image as background for the 
         given channel
         
         Args: 
            no_of_images (int): number of random images to be averaged
         """
        collection = []
        for i in range(no_of_images):
            well = np.random.choice(list(self.full_images.keys()))
            position = np.random.choice(list(self.full_images[well].keys()))
            img = self.full_images[well][position].load_image()
            img = np.expand_dims(img, 3)
            collection.append(img)
        
        collection = np.concatenate(collection, axis =3)
        self.background  = np.median(collection, axis=3).astype(np.uint16)

        

#         self.results = []
#         for well in tqdm(self.full_images):
#             for position in tqdm(self.full_images[well], leave=False):
#                 img = self.full_images[well][position].load_image()

    def analyse_experiment(self):
        """go through all images and analyse them"""
        
        
        self.results = []
        for well in tqdm(self.full_images):
            for position in tqdm(self.full_images[well], leave=False):
                img = self.full_images[well][position].load_image()
                
                #subtract background
                img = self.subtract_background(img)
                
                try:
                    position_results = self.analyse_img(img)
                    position_results["well"] = well
                    position_results["position"] = position
                    self.results.append(position_results)
                except:
                    print(f"{well}/{position} did not work")
                
                
        self.results = pd.concat(self.results, ignore_index=True)
        print("Analysis complete")

    def get_masks(self, img):
        channels = [self.cell_channel, self.nucleus_channel]
        cell_mask, __, __, __ = self.cytomodel.eval(
            [img], diameter=120, channels=channels
        )
        channels = [self.nucleus_channel, 0]
        nucleus_mask, __, __, __ = self.nucleusmodel.eval(
            [img], diameter=70, channels=channels
        )
        cytoplasm = (cell_mask[0] > 0) * (nucleus_mask[0] == 0)
        nucleus_mask = (cell_mask[0] > 0) * (nucleus_mask[0] > 0)

        return cell_mask[0], nucleus_mask, cytoplasm
    
    def analyse_img(self, img):
        cell, nucleus, cytoplasm = self.get_masks(img)
        
        
        
        data = np.concatenate([cell.flatten().reshape(-1, 1), 
                           nucleus.flatten().reshape(-1, 1),
                           cytoplasm.flatten().reshape(-1, 1),
                           img[:,:,self.channel_to_analyse -1].flatten().reshape(-1, 1)], axis = 1)
        data = pd.DataFrame(data, columns=['cell_id', 'nucleus', 'cytoplasm', 'fluorescence'])
        data['nucleus_fluorescence'] = data.nucleus * data.fluorescence
        data['cytoplasm_fluorescence'] = data.cytoplasm * data.fluorescence
        img_data = data.groupby(['cell_id']).mean()[['fluorescence']]
        img_data.columns = ['cell_fluorescence_mean']
        cell_sum = data.groupby(['cell_id']).sum()
        img_data['nucleus_size'] = cell_sum.nucleus
        img_data['nucleus_fluorescence_sum'] = cell_sum.nucleus_fluorescence
        img_data['cytoplasm_size'] = cell_sum.cytoplasm
        img_data['cytoplasm_fluorescence_sum'] = cell_sum.cytoplasm_fluorescence
        img_data['cell_size'] = img_data['nucleus_size'] + img_data['cytoplasm_size']
        img_data['cell_fluorescence_sum'] = cell_sum.fluorescence
        img_data['nucleus_fluorescence_mean'] = img_data['nucleus_fluorescence_sum'] / img_data['nucleus_size']
        img_data['cytoplasm_fluorescence_mean'] = img_data['cytoplasm_fluorescence_sum'] / img_data['cytoplasm_size']
        img_data = img_data[['cell_size', 'cytoplasm_size', 'nucleus_size', 
               'cell_fluorescence_sum', 'cytoplasm_fluorescence_sum', 'nucleus_fluorescence_sum',
               'cell_fluorescence_mean', 'cytoplasm_fluorescence_mean','nucleus_fluorescence_mean']]
        
        img_data.clip(lower=0, inplace = True)
        
        img_data['cytoplasm/nucleus mean ratio'] = img_data['cytoplasm_fluorescence_mean'] / img_data['nucleus_fluorescence_mean']
        img_data['cytoplasm/nucleus sum ratio'] = img_data['cytoplasm_fluorescence_sum'] / img_data['nucleus_fluorescence_sum']
        img_data['nucleus/cytoplasm mean ratio'] = img_data['nucleus_fluorescence_mean'] / img_data['cytoplasm_fluorescence_mean'] 
        img_data['nucleus/cytoplasm sum ratio'] = img_data['nucleus_fluorescence_sum'] / img_data['cytoplasm_fluorescence_sum']

        img_data['nucleus/cell sum ratio'] = img_data['nucleus_fluorescence_sum'] / img_data['cell_fluorescence_sum']
        
        return img_data.reset_index()
    
    def subtract_background(self, img):
        """function to subtract background - must be converted to int and negative values removed,
        otherwise the pixel values get unreal values"""
        img = img.astype(int) - self.background.astype(int)
        img[img < 0] = 0
        img = img.astype(np.uint16)
        
        return img
    
    def display_image(self, well, position, gamma = 0.5, cmap = "plasma"):

        img = self.full_images[well][position].load_image()

        if cmap == "":
            cmap = "gray"
        
        img = self.subtract_background(img)

        gamma_corrected = exposure.adjust_gamma(img, gamma)
                
        print(img.min())
        
        cell_mask, nucleus_mask, cytoplasm = self.get_masks(img)
        
        channel_dict = {0: "nucleus", 1: "cell", 2: "to analyse"} 

        plt.figure(figsize=(12,8))
        for idx, channel in enumerate([self.nucleus_channel, self.cell_channel, self.channel_to_analyse]):
            plt.subplot(2, 3, idx+1)
            plt.imshow(gamma_corrected[:,:,channel - 1] / 4095, cmap=cmap, aspect='auto')
            plt.title(channel_dict[idx])
            plt.xticks([])
            plt.yticks([])
            
        for idx, mask in enumerate([nucleus_mask, cell_mask, cytoplasm]):
            plt.subplot(2, 3, idx +4)
            plt.imshow(mask, aspect='auto')
            
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()
# -



