import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Union, List, Optional, Tuple
import random
import math

def get_cxr_bert_tokenizer(model_name="microsoft/BiomedVLP-CXR-BERT-specialized"):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    return tokenizer

def tokenize(texts: Union[str, List[str]], 
             context_length: int = 77) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]
    
    tokenizer = get_cxr_bert_tokenizer()
    
    tokenized = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=context_length,
        return_tensors='pt'
    )
    
    return tokenized.input_ids


class CXRDataset(Dataset):
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 mode: str = 'train',
                 input_size: int = 518,
                 word_length: int = 77,
                 use_augmentation: bool = True,
                 aug_probability: float = 0.5):
        
        super(CXRDataset, self).__init__()
        
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.input_size = input_size
        self.word_length = word_length
        
        self.use_augmentation = use_augmentation and (mode == 'train')
        self.aug_probability = aug_probability
        
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        
        self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        
        self._setup_paths()
        
        self._load_csv_data()
        
        self._tokenizer = None
        
        print(f"CXR Dataset initialized: {len(self.df)} samples")
        if self.use_augmentation:
            print(f"Enhanced data augmentation enabled with probability {self.aug_probability}")
            print("Available augmentations:")
            print("  Geometric: horizontal_flip, random_rotation (±0.3 rad), random_scale (0.95-1.15x)")
            print("  Intensity: brightness, contrast, gamma_correction, CLAHE, gaussian_noise")
    
    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = get_cxr_bert_tokenizer()
        return self._tokenizer
    
    def _setup_paths(self):
        folder_mapping = {
            'train': 'Train_Folder',
            'val': 'Val_Folder', 
            'test': 'Test_Folder'
        }
        
        csv_mapping = {
            'train': 'Train_text.csv',
            'val': 'Val_text.csv',
            'test': 'Test_text.csv'
        }
        
        self.folder_path = os.path.join(self.data_root, folder_mapping[self.split])
        self.csv_path = os.path.join(self.folder_path, csv_mapping[self.split])
        self.img_dir = os.path.join(self.folder_path, 'img')
        self.mask_dir = os.path.join(self.folder_path, 'mask')

        assert os.path.exists(self.csv_path), f"CSV not found: {self.csv_path}"
        assert os.path.exists(self.img_dir), f"Image dir not found: {self.img_dir}"
        assert os.path.exists(self.mask_dir), f"Mask dir not found: {self.mask_dir}"
    
    def _load_csv_data(self):
        self.df = pd.read_csv(self.csv_path)
        
        assert 'Image' in self.df.columns, "CSV must have 'Image' column"
        assert 'Description' in self.df.columns, "CSV must have 'Description' column"
        
        self.df = self.df.dropna(subset=['Image', 'Description'])
        self.df = self.df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def apply_augmentation(self, img, mask):
        if not self.use_augmentation or random.random() > self.aug_probability:
            return img, mask

        geometric_augs = ['horizontal_flip', 'random_rotation', 'random_scale']
        intensity_augs = ['brightness', 'contrast', 'gamma_correction', 'clahe', 'gaussian_noise']
        
        selected_augs = []

        if random.random() < 0.6:
            selected_augs.append(random.choice(geometric_augs))

        num_intensity = random.choice([1, 2])
        selected_intensity = random.sample(intensity_augs, min(num_intensity, len(intensity_augs)))
        selected_augs.extend(selected_intensity)

        for aug in selected_augs:
            img, mask = self._apply_single_augmentation(img, mask, aug)
        
        return img, mask
    
    def _apply_single_augmentation(self, img, mask, aug_type):

        if aug_type == 'horizontal_flip':
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        
        elif aug_type == 'brightness':
            factor = random.uniform(0.85, 1.15)  
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        
        elif aug_type == 'contrast':
            factor = random.uniform(0.85, 1.15)  
            mean_val = img.mean()
            img = np.clip((img.astype(np.float32) - mean_val) * factor + mean_val, 0, 255).astype(np.uint8)
        
        elif aug_type == 'gamma_correction':
            gamma = random.uniform(0.7, 1.3)  
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            img = cv2.LUT(img, table)
        
        elif aug_type == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clip_limit = random.uniform(2.0, 4.0) 
            tile_grid_size = random.choice([(8, 8), (16, 16)])  

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                if img.ndim == 3:
                    img_gray = img[:, :, 0]
                else:
                    img_gray = img
                img_enhanced = clahe.apply(img_gray)
                
                img = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
            else:
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                
                l_channel = clahe.apply(l_channel)

                lab = cv2.merge((l_channel, a_channel, b_channel))
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        elif aug_type == 'gaussian_noise':
            noise_std = random.uniform(5, 15)  
            noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        elif aug_type == 'random_rotation':
            angle_rad = random.uniform(-0.3, 0.3)  
            angle_deg = math.degrees(angle_rad)
 
            h, w = img.shape[:2]
            center = (w // 2, h // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

            img = cv2.warpAffine(
                img, rotation_matrix, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101
            )
            mask = cv2.warpAffine(
                mask, rotation_matrix, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        
        elif aug_type == 'random_scale':
            scale_factor = random.uniform(0.95, 1.15)
            
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            if scale_factor > 1.0:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                img = img_scaled[start_y:start_y+h, start_x:start_x+w]
                mask = mask_scaled[start_y:start_y+h, start_x:start_x+w]
            else:
                img = np.zeros((h, w, 3), dtype=np.uint8)
                mask = np.zeros((h, w), dtype=np.uint8)
                
                start_x = (w - new_w) // 2
                start_y = (h - new_h) // 2
                
                img[start_y:start_y+new_h, start_x:start_x+new_w] = img_scaled
                mask[start_y:start_y+new_h, start_x:start_x+new_w] = mask_scaled
        
        return img, mask
    
    def _apply_affine_transform(self, data, img_size, is_mask=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size, self.input_size

        scale = min(float(inp_h) / float(ori_h), float(inp_w) / float(ori_w))
        new_h, new_w = int(ori_h * scale), int(ori_w * scale)
        bias_x, bias_y = float(inp_w - new_w) / 2.0, float(inp_h - new_h) / 2.0
        
        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], dtype=np.float64)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], dtype=np.float64)
        
        mat = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
        mat_inv = cv2.getAffineTransform(dst.astype(np.float32), src.astype(np.float32))
        
        if is_mask:
            data_transformed = cv2.warpAffine(
                data,
                mat,
                (inp_w, inp_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return data_transformed.astype(np.float32) / 255.0
        else:
            border_value = [
                int(self.mean[0] * 255), 
                int(self.mean[1] * 255), 
                int(self.mean[2] * 255)
            ]
            
            data_transformed = cv2.warpAffine(
                data,
                mat,
                (inp_w, inp_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value
            )
            return data_transformed, mat_inv
    
    def transform_image(self, img, img_size):
        return self._apply_affine_transform(img, img_size, is_mask=False)
    
    def transform_mask(self, mask, img_size):
        return self._apply_affine_transform(mask, img_size, is_mask=True)
    
    def convert_to_tensor(self, img, mask=None):
        img = img.astype(np.float32)

        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img_tensor = img_tensor.div(255.0)

        img_tensor = img_tensor.sub(self.mean_tensor).div(self.std_tensor)
        
        if mask is not None:
            mask = mask.astype(np.float32)
            mask_tensor = torch.from_numpy(mask).float()
            return img_tensor, mask_tensor
        
        return img_tensor
    
    def _tokenize_text(self, text):
        tokenizer = self._get_tokenizer()
        
        tokenized = tokenizer(
            [text],
            padding='max_length',
            truncation=True,
            max_length=self.word_length,
            return_tensors='pt'
        )
        
        return tokenized.input_ids.squeeze(0)
    
    def _validate_image_channels(self, img_path: str, img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def __getitem__(self, index):
        try:
            row = self.df.iloc[index]
            mask_filename = str(row['Image'])
            description = str(row['Description'])

            if mask_filename.startswith('mask_'):
                img_filename = mask_filename[5:]
            else:
                img_filename = mask_filename
            
            img_path = os.path.join(self.img_dir, img_filename)
            mask_path = os.path.join(self.mask_dir, mask_filename)

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = self._validate_image_channels(img_path, img)
            img_size = img.shape[:2]

            word_vec = self._tokenize_text(description)
            
            if self.mode == 'train':
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                assert mask is not None, f"Failed to load mask: {mask_path}"

                img, mask = self.apply_augmentation(img, mask)

                img_transformed, mat_inv = self.transform_image(img, img_size)
                mask_transformed = self.transform_mask(mask, img_size)
                
                img_tensor, mask_tensor = self.convert_to_tensor(img_transformed, mask_transformed)
                return img_tensor, word_vec, mask_tensor
                
            elif self.mode == 'val':
                img_transformed, mat_inv = self.transform_image(img, img_size)
                img_tensor = self.convert_to_tensor(img_transformed)
                params = {
                    'mask_path': mask_path,
                    'inverse': mat_inv,
                    'ori_size': np.array(img_size, dtype=np.int32),
                    'img_filename': img_filename,
                    'mask_filename': mask_filename
                }
                return img_tensor, word_vec, params
            
            else:  # test mode
                img_transformed, mat_inv = self.transform_image(img, img_size)
                img_tensor = self.convert_to_tensor(img_transformed)
                params = {
                    'ori_img': img,
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'inverse': mat_inv,
                    'ori_size': np.array(img_size, dtype=np.int32),
                    'img_filename': img_filename,
                    'mask_filename': mask_filename,
                    'description': description
                }
                return img_tensor, params
                
        except Exception as e:
            print(f"ERROR in __getitem__ at index {index}: {e}")
            print(f"  Image: {img_filename if 'img_filename' in locals() else 'unknown'}")
            print(f"  Description: {description if 'description' in locals() else 'unknown'}")
            
            if self.mode == 'train':
                dummy_img = torch.zeros(3, self.input_size, self.input_size)
                dummy_text = torch.zeros(self.word_length, dtype=torch.long)
                dummy_mask = torch.zeros(self.input_size, self.input_size)
                return dummy_img, dummy_text, dummy_mask
            else:
                raise e
    
    def get_sample_info(self, index):
        row = self.df.iloc[index]
        mask_filename = str(row['Image'])
        description = str(row['Description'])
        
        if mask_filename.startswith('mask_'):
            img_filename = mask_filename[5:]
        else:
            img_filename = mask_filename
        
        return {
            'index': index,
            'img_filename': img_filename,
            'mask_filename': mask_filename,
            'description': description,
            'img_path': os.path.join(self.img_dir, img_filename),
            'mask_path': os.path.join(self.mask_dir, mask_filename)
        }
    
    def get_augmentation_stats(self):
        return {
            'use_augmentation': self.use_augmentation,
            'aug_probability': self.aug_probability,
            'available_augs': {
                'geometric': ['horizontal_flip', 'random_rotation', 'random_scale'],
                'intensity': ['brightness', 'contrast', 'gamma_correction', 'clahe', 'gaussian_noise']
            },
            'augmentation_params': {
                'rotation_range': '±0.3 radians (±17 degrees)',
                'scale_range': '0.95-1.15x',
                'gamma_range': '0.7-1.3',
                'brightness_range': '0.85-1.15x',
                'contrast_range': '0.85-1.15x',
                'noise_std_range': '5-15',
                'clahe_clip_limit': '2.0-4.0'
            },
            'improvements': [
                'CXR grayscale image support in CLAHE',
                'Nearest neighbor interpolation for masks',
                'Pre-computed tensors for performance',
                'Unified transform function',
                'Enhanced error handling'
            ]
        }
    
    def validate_dataset(self, num_samples: int = 5) -> bool:
        print(f"Validating dataset with {num_samples} samples...")
        
        try:
            for i in range(min(num_samples, len(self))):
                sample = self[i]
                print(f"  Sample {i}: OK")
            print(" Dataset validation passed")
            return True
        except Exception as e:
            print(f" Dataset validation failed: {e}")
            return False
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"data_root={self.data_root}, "
                f"split={self.split}, "
                f"mode={self.mode}, "
                f"samples={len(self.df)}, "
                f"augmentation={'ON' if self.use_augmentation else 'OFF'})")