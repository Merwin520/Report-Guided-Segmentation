import argparse
import os
import time
import warnings
from tqdm import tqdm
import pandas as pd

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage

import utils.config as config
from utils.dataset import CXRDataset, tokenize
from model import build_segmenter
from utils.misc import setup_logger, print_model_info

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
    THOP_AVAILABLE = False
    logger.info("Using fvcore for FLOPs calculation")
except ImportError:
    FVCORE_AVAILABLE = False
    logger.warning("fvcore not available, FLOPs calculation will be limited")
    THOP_AVAILABLE = False

warnings.filterwarnings("ignore")


class MetricsCalculator:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_dice = []
        self.total_pa = []
        self.total_iou = []
        self.total_jaccard = []
        self.total_hd95 = []
        self.total_asd = []
        self.total_recall = []
        self.total_pixels_correct = 0
        self.total_pixels = 0
        self.intersection_sum = 0
        self.union_sum = 0
    
    def calculate_dice_coefficient(self, pred, target):
        intersection = np.logical_and(pred, target).sum()
        dice = (2.0 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
        return dice
    
    def calculate_pixel_accuracy(self, pred, target):
        correct = np.logical_not(np.logical_xor(pred, target)).sum()
        total = pred.size
        pa = correct / total
        return pa
    
    def calculate_iou(self, pred, target):
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou
    
    def calculate_jaccard_index(self, pred, target):
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        jaccard = (intersection + 1e-7) / (union + 1e-7)
        return jaccard
    
    def calculate_average_surface_distance(self, pred, target):
        try:
            pred_boundary = self._get_boundary_points(pred)
            target_boundary = self._get_boundary_points(target)
            
            if len(pred_boundary) == 0 or len(target_boundary) == 0:
                return 0.0

            dist_pred_to_target = self._compute_average_distance(pred_boundary, target_boundary)
            dist_target_to_pred = self._compute_average_distance(target_boundary, pred_boundary)

            asd = (dist_pred_to_target + dist_target_to_pred) / 2.0
            
            return asd
            
        except Exception as e:
            logger.warning(f"ASD calculation failed: {e}")
            return 0.0
    
    def calculate_hausdorff_distance_95(self, pred, target):
        try:
            pred_boundary = self._get_boundary_points(pred)
            target_boundary = self._get_boundary_points(target)
            
            if len(pred_boundary) == 0 or len(target_boundary) == 0:
                return 0.0
            
            dist1 = self._compute_boundary_distances(pred_boundary, target_boundary)
            dist2 = self._compute_boundary_distances(target_boundary, pred_boundary)
            
            all_distances = np.concatenate([dist1, dist2])
            hd95 = np.percentile(all_distances, 95)
            
            return hd95
            
        except Exception as e:
            logger.warning(f"HD95 calculation failed: {e}")
            return 0.0

    def calculate_recall(self, pred, target):
        intersection = np.logical_and(pred, target).sum()
        recall = (intersection + 1e-7) / (target.sum() + 1e-7)
        return recall
    
    def _get_boundary_points(self, binary_mask):
        if binary_mask.sum() == 0:
            return np.array([]).reshape(0, 2)
        
        erosion = ndimage.binary_erosion(binary_mask)
        boundary = binary_mask ^ erosion
        
        boundary_points = np.column_stack(np.where(boundary))
        return boundary_points
    
    def _compute_boundary_distances(self, points1, points2):
        if len(points1) == 0 or len(points2) == 0:
            return np.array([])
        

        distances = []
        for point in points1:
            dists = np.sqrt(np.sum((points2 - point) ** 2, axis=1))
            distances.append(np.min(dists))
        
        return np.array(distances)
    
    def _compute_average_distance(self, points1, points2):
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        distances = []
        for point in points1:
            dists = np.sqrt(np.sum((points2 - point) ** 2, axis=1))
            distances.append(np.min(dists))
        
        return np.mean(distances)
    
    def update(self, pred, target):

        pred_binary = (pred > 0.5).astype(np.uint8)
        target_binary = (target > 0.5).astype(np.uint8)

        dice = self.calculate_dice_coefficient(pred_binary, target_binary)
        pa = self.calculate_pixel_accuracy(pred_binary, target_binary)
        iou = self.calculate_iou(pred_binary, target_binary)
        jaccard = self.calculate_jaccard_index(pred_binary, target_binary)  
        hd95 = self.calculate_hausdorff_distance_95(pred_binary, target_binary)
        asd = self.calculate_average_surface_distance(pred_binary, target_binary)
        recall = self.calculate_recall(pred_binary, target_binary)
        
        self.total_dice.append(dice)
        self.total_pa.append(pa)
        self.total_iou.append(iou)
        self.total_jaccard.append(jaccard)
        self.total_hd95.append(hd95)
        self.total_asd.append(asd)
        self.total_recall.append(recall)
        
        correct_pixels = np.logical_not(np.logical_xor(pred_binary, target_binary)).sum()
        total_pixels = pred_binary.size
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        
        self.total_pixels_correct += correct_pixels
        self.total_pixels += total_pixels
        self.intersection_sum += intersection
        self.union_sum += union
    
    def get_metrics(self):
        metrics = {
            'Dice': np.mean(self.total_dice),
            'PA': np.mean(self.total_pa),
            'IoU': np.mean(self.total_iou),
            'Jaccard': np.mean(self.total_jaccard),  # 新增
            'HD95': np.mean(self.total_hd95),
            'ASD': np.mean(self.total_asd),  # 新增
            'Recall': np.mean(self.total_recall),
            'Global_PA': self.total_pixels_correct / max(self.total_pixels, 1),  # 全局PA
        }
        
        metrics['Dice_std'] = np.std(self.total_dice)
        metrics['PA_std'] = np.std(self.total_pa)
        metrics['IoU_std'] = np.std(self.total_iou)
        metrics['Jaccard_std'] = np.std(self.total_jaccard)  # 新增
        metrics['HD95_std'] = np.std(self.total_hd95)
        metrics['ASD_std'] = np.std(self.total_asd)  # 新增
        metrics['Recall_std'] = np.std(self.total_recall)
        
        return metrics


def estimate_bert_flops(model, text_length, vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12):
    seq_len = text_length
    
    embedding_flops = seq_len * hidden_size
    

    attention_flops = (
        3 * seq_len * hidden_size * hidden_size +  
        2 * num_heads * seq_len * seq_len * (hidden_size // num_heads) +  
        seq_len * hidden_size * hidden_size  
    )
    
    ffn_flops = 2 * seq_len * hidden_size * (4 * hidden_size)  

    layernorm_flops = 2 * seq_len * hidden_size * 2  
    
    layer_flops = attention_flops + ffn_flops + layernorm_flops
    
    total_flops = embedding_flops + num_layers * layer_flops
    
    return total_flops


def calculate_model_flops(model, input_size, text_length):
    model.eval()
    
    img_input = torch.randn(1, *input_size).cuda()
    text_input = torch.randint(0, 1000, (1, text_length)).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    flops = 0
    
    if FVCORE_AVAILABLE:
        try:
            logger.info("Attempting FLOPs calculation with fvcore...")
            flops_dict, _ = flop_count(
                model, 
                (img_input, text_input),
                supported_ops={
                    "aten::add": lambda inputs, outputs: torch.numel(inputs[0]),
                    "aten::add_": lambda inputs, outputs: torch.numel(inputs[0]),
                    "aten::mul": lambda inputs, outputs: torch.numel(inputs[0]),
                    "aten::mul_": lambda inputs, outputs: torch.numel(inputs[0]),
                }
            )
            flops = sum(flops_dict.values())
            logger.info(f"✓ FLOPs calculation successful: {format_flops(flops)}")
            
        except Exception as e:
            logger.warning(f"FLOPs calculation with fvcore failed: {e}")
            logger.info("Attempting BERT FLOPs estimation...")
            try:
                flops = estimate_bert_flops(model, text_length)
                logger.info(f"✓ Using estimated BERT FLOPs: {format_flops(flops)}")
            except Exception as est_e:
                logger.warning(f"BERT FLOPs estimation also failed: {est_e}")
                flops = 0
    
    else:
        logger.warning("fvcore not available for FLOPs calculation")
        logger.info("Attempting BERT FLOPs estimation...")
        try:
            flops = estimate_bert_flops(model, text_length)
            logger.info(f" Using estimated BERT FLOPs: {format_flops(flops)}")
        except Exception as est_e:
            logger.warning(f"BERT FLOPs estimation failed: {est_e}")
            flops = 0
            logger.info("Install fvcore to enable accurate FLOPs calculation:")
            logger.info("  pip install fvcore")
    
    return flops, total_params, trainable_params


def format_flops(flops):
    if flops == 0:
        return "N/A"
    
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f}K"
    else:
        return f"{flops:.0f}"


def format_params(params):
    if params >= 1e9:
        return f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K"
    else:
        return f"{params:.0f}"


def parse_args():
    parser = argparse.ArgumentParser(description='QaTa DETRIS Testing')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--split', default='test', choices=['test', 'val'], 
                        help='Dataset split to test')
    parser.add_argument('--output-dir', default='./test_results', 
                        help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true', 
                        help='Save prediction images')
    parser.add_argument('--save-individual-metrics', action='store_true',
                        help='Save individual Dice and IoU scores for each sample')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold for predictions')
    parser.add_argument('--calculate-flops', action='store_true',
                        help='Calculate model FLOPs and parameters')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None,
                        help='Override config options')
    return parser.parse_args()


def load_config(config_path, opts=None):
    cfg = config.load_cfg_from_cfg_file(config_path)
    if opts:
        cfg = config.merge_cfg_from_list(cfg, opts)

    flat_config = {}
    for section in ['DATA', 'TRAIN', 'COCOOP', 'CONTRASTIVE', 'LOSS', 'TEST', 'MISC']:
        if hasattr(cfg, section):
            section_cfg = getattr(cfg, section)
            if hasattr(section_cfg, 'items'):
                for key, value in section_cfg.items():
                    flat_config[key] = value
    
    for key, value in flat_config.items():
        setattr(cfg, key, value)
    
    return cfg


def load_model(cfg, checkpoint_path):
    logger.info("Loading model...")
    
    model, _ = build_segmenter(cfg)
    model = model.cuda()
    model.eval()
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
        logger.info(f"Checkpoint IoU: {checkpoint.get('best_iou', 'Unknown')}")
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    logger.info("Model loaded successfully!")
    
    return model


def create_test_dataset(cfg, split):
    logger.info(f"Creating {split} dataset...")
    
    dataset = CXRDataset(
        data_root=cfg.data_root,
        split=split,
        mode='test',
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        #cxr_bert_model=getattr(cfg, 'cxr_bert_model', 'microsoft/BiomedVLP-CXR-BERT-specialized'),
        use_augmentation=False,
        aug_probability=0.0
    )
    
    logger.info(f"Test dataset: {len(dataset)} samples")
    return dataset


def inverse_transform_prediction(pred, mat_info, target_size):
    w, h = target_size
    
    try:
        mat = np.array(mat_info, dtype=np.float32)
        transformed = cv2.warpAffine(pred, mat, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderValue=0.)
        return transformed
    except Exception as e:
        logger.warning(f"Transform failed, using resize: {e}")
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)

@torch.no_grad()
def test_model(model, dataset, args, cfg):
    logger.info("Starting model testing...")

    metrics_calc = MetricsCalculator()
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_predictions:
        pred_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
    
    model_stats = {}
    if args.calculate_flops:
        logger.info("Calculating model FLOPs and parameters...")
        try:
            model_copy = model
            flops, total_params, trainable_params = calculate_model_flops(
                model_copy, 
                (3, cfg.input_size, cfg.input_size), 
                cfg.word_len
            )
            model_stats = {
                'flops': flops,
                'flops_formatted': format_flops(flops),
                'total_params': total_params,
                'total_params_formatted': format_params(total_params),
                'trainable_params': trainable_params,
                'trainable_params_formatted': format_params(trainable_params)
            }
            
            if flops > 0:
                logger.info(f"✓ Model FLOPs: {model_stats['flops_formatted']} | "
                           f"Total Params: {model_stats['total_params_formatted']} | "
                           f"Trainable Params: {model_stats['trainable_params_formatted']}")
            else:
                logger.info(f"✓ Model Params: Total: {model_stats['total_params_formatted']} | "
                           f"Trainable: {model_stats['trainable_params_formatted']} | "
                           f"FLOPs: Could not calculate")
                
        except Exception as e:
            logger.warning(f"Failed to calculate model statistics: {e}")
            logger.info("Continuing with testing without FLOPs calculation...")
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model_stats = {
                    'flops': 0,
                    'flops_formatted': 'N/A',
                    'total_params': total_params,
                    'total_params_formatted': format_params(total_params),
                    'trainable_params': trainable_params,
                    'trainable_params_formatted': format_params(trainable_params)
                }
                logger.info(f"✓ Model Params: Total: {model_stats['total_params_formatted']} | "
                           f"Trainable: {model_stats['trainable_params_formatted']}")
            except Exception as param_e:
                logger.warning(f"Failed to calculate basic model statistics: {param_e}")
                model_stats = {
                    'flops': 0,
                    'flops_formatted': 'N/A',
                    'total_params': 0,
                    'total_params_formatted': 'N/A',
                    'trainable_params': 0,
                    'trainable_params_formatted': 'N/A'
                }
    else:
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_stats = {
                'flops': 0,
                'flops_formatted': 'Not calculated',
                'total_params': total_params,
                'total_params_formatted': format_params(total_params),
                'trainable_params': trainable_params,
                'trainable_params_formatted': format_params(trainable_params)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate basic model parameters: {e}")
            model_stats = {
                'flops': 0,
                'flops_formatted': 'Not calculated',
                'total_params': 0,
                'total_params_formatted': 'N/A',
                'trainable_params': 0,
                'trainable_params_formatted': 'N/A'
            }
    
    detailed_results = []
    
    total_time = 0
    for idx in tqdm(range(len(dataset)), desc='Testing'):
        start_time = time.time()
        
        img, params = dataset[idx]
        

        img_filename = params['img_filename']
        mask_filename = params['mask_filename']
        mask_path = params['mask_path']
        description = params['description']
        ori_size = params['ori_size']
        inverse_mat = params['inverse']
        
  
        img = img.unsqueeze(0).cuda()  
        
        text = tokenize(description, cfg.word_len)
        text = text.cuda()
        
        pred = model(img, text)
        pred = torch.sigmoid(pred).squeeze() 
        
        if pred.shape != img.shape[-2:]:
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0),
                               size=img.shape[-2:],
                               mode='bicubic',
                               align_corners=True).squeeze()

        h, w = ori_size
        pred_np = pred.cpu().numpy()
        pred_original = inverse_transform_prediction(pred_np, inverse_mat, (w, h))
 
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            logger.warning(f"Failed to load mask: {mask_path}")
            continue
        
        gt_mask = gt_mask / 255.0 
        
        pred_binary = (pred_original > args.threshold).astype(np.uint8)
        
        gt_binary = (gt_mask > 0.5).astype(np.uint8)

        metrics_calc.update(pred_binary, gt_binary)
        
        dice = metrics_calc.calculate_dice_coefficient(pred_binary, gt_binary)
        pa = metrics_calc.calculate_pixel_accuracy(pred_binary, gt_binary)
        iou = metrics_calc.calculate_iou(pred_binary, gt_binary)
        jaccard = metrics_calc.calculate_jaccard_index(pred_binary, gt_binary)
        hd95 = metrics_calc.calculate_hausdorff_distance_95(pred_binary, gt_binary)
        asd = metrics_calc.calculate_average_surface_distance(pred_binary, gt_binary)
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        detailed_results.append({
            'image': img_filename,
            'mask': mask_filename,
            'description': description,
            'dice': dice,
            'pa': pa,
            'iou': iou,
            'jaccard': jaccard,
            'hd95': hd95,
            'asd': asd,
            'inference_time': inference_time
        })

        if args.save_predictions:

            pred_save_path = os.path.join(pred_dir, f'pred_{mask_filename}')
            cv2.imwrite(pred_save_path, (pred_original * 255).astype(np.uint8))

            binary_save_path = os.path.join(pred_dir, f'binary_{mask_filename}')
            cv2.imwrite(binary_save_path, (pred_binary * 255).astype(np.uint8))

        if (idx + 1) % 100 == 0:
            current_metrics = metrics_calc.get_metrics()
            logger.info(f"Progress {idx+1}/{len(dataset)}: "
                       f"Dice={current_metrics['Dice']:.4f}, "
                       f"IoU={current_metrics['IoU']:.4f}, "
                       f"Jaccard={current_metrics['Jaccard']:.4f}, "
                       f"PA={current_metrics['PA']:.4f}")
    
    final_metrics = metrics_calc.get_metrics()
    avg_inference_time = total_time / len(dataset)

    logger.info("="*80)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Dataset: {args.split}")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Threshold: {args.threshold}")

    if model_stats:
        logger.info("-"*80)
        logger.info(f"MODEL STATISTICS:")
        logger.info(f"  FLOPs: {model_stats['flops_formatted']}")
        logger.info(f"  Total Parameters: {model_stats['total_params_formatted']}")
        logger.info(f"  Trainable Parameters: {model_stats['trainable_params_formatted']}")
    
    logger.info("-"*80)
    logger.info(f"SEGMENTATION METRICS:")
    logger.info(f"  Dice Coefficient: {final_metrics['Dice']:.4f} ± {final_metrics['Dice_std']:.4f}")
    logger.info(f"  Pixel Accuracy (PA): {final_metrics['PA']:.4f} ± {final_metrics['PA_std']:.4f}")
    logger.info(f"  IoU: {final_metrics['IoU']:.4f} ± {final_metrics['IoU_std']:.4f}")
    logger.info(f"  Jaccard Index: {final_metrics['Jaccard']:.4f} ± {final_metrics['Jaccard_std']:.4f}")
    logger.info(f"  95% Hausdorff Distance: {final_metrics['HD95']:.4f} ± {final_metrics['HD95_std']:.4f}")
    logger.info(f"  Average Surface Distance: {final_metrics['ASD']:.4f} ± {final_metrics['ASD_std']:.4f}")
    logger.info(f"  Recall: {final_metrics['Recall']:.4f} ± {final_metrics['Recall_std']:.4f}")
    logger.info(f"  Global PA: {final_metrics['Global_PA']:.4f}")
    logger.info("-"*80)
    logger.info(f"PERFORMANCE:")
    logger.info(f"  Average inference time: {avg_inference_time:.4f}s per sample")
    logger.info(f"  Total test time: {total_time:.2f}s")
    logger.info("="*80)
    
    results_df = pd.DataFrame(detailed_results)
    results_csv_path = os.path.join(args.output_dir, 'detailed_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Detailed results saved to: {results_csv_path}")
    
    if args.save_individual_metrics:
        individual_metrics_file = os.path.join(args.output_dir, 'individual_metrics.txt')
        with open(individual_metrics_file, 'w') as f:

            f.write("="*80 + "\n")
            f.write("Individual Metrics for Each Sample\n")
            f.write("="*80 + "\n\n")
            
            f.write("Format 1: Per-sample metrics with filenames\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Image Filename':<40} {'Dice':<10} {'IoU':<10} {'PA':<10} {'Jaccard':<10}\n")
            f.write("-"*80 + "\n")
            for res in detailed_results:
                f.write(f"{res['image']:<40} {res['dice']:<10.6f} {res['iou']:<10.6f} "
                       f"{res['pa']:<10.6f} {res['jaccard']:<10.6f}\n")
            f.write("\n\n")
            
            f.write("Format 2: Comma-separated values (for easy copying)\n")
            f.write("-"*80 + "\n")

            f.write("Filenames:\n")
            f.write(", ".join([res['image'] for res in detailed_results]))
            f.write("\n\n")

            f.write("Dice:\n")
            f.write(", ".join([f"{res['dice']:.6f}" for res in detailed_results]))
            f.write("\n\n")

            f.write("IoU:\n")
            f.write(", ".join([f"{res['iou']:.6f}" for res in detailed_results]))
            f.write("\n\n")

            f.write("PA:\n")
            f.write(", ".join([f"{res['pa']:.6f}" for res in detailed_results]))
            f.write("\n\n")

            f.write("Jaccard:\n")
            f.write(", ".join([f"{res['jaccard']:.6f}" for res in detailed_results]))
            f.write("\n\n")
            
            f.write("HD95:\n")
            f.write(", ".join([f"{res['hd95']:.6f}" for res in detailed_results]))
            f.write("\n\n")
            
            f.write("ASD:\n")
            f.write(", ".join([f"{res['asd']:.6f}" for res in detailed_results]))
            f.write("\n\n")
            
            f.write("Format 3: Python dictionary format\n")
            f.write("-"*80 + "\n")
            f.write("{\n")
            for i, res in enumerate(detailed_results):
                f.write(f"    '{res['image']}': {{\n")
                f.write(f"        'dice': {res['dice']:.6f},\n")
                f.write(f"        'iou': {res['iou']:.6f},\n")
                f.write(f"        'pa': {res['pa']:.6f},\n")
                f.write(f"        'jaccard': {res['jaccard']:.6f},\n")
                f.write(f"        'hd95': {res['hd95']:.6f},\n")
                f.write(f"        'asd': {res['asd']:.6f}\n")
                f.write(f"    }}")
                if i < len(detailed_results) - 1:
                    f.write(",")
                f.write("\n")
            f.write("}\n")
        
        logger.info(f"Individual metrics saved to: {individual_metrics_file}")
    
    summary_results = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'threshold': args.threshold,
        'total_samples': len(dataset),
        **final_metrics,
        **model_stats,
        'avg_inference_time': avg_inference_time,
        'total_test_time': total_time
    }
    
    summary_df = pd.DataFrame([summary_results])
    summary_csv_path = os.path.join(args.output_dir, 'summary_results.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Summary results saved to: {summary_csv_path}")
    
    return final_metrics, detailed_results


def main():
    args = parse_args()
    
    setup_logger(args.output_dir, filename="test.log")
    
    logger.info("Starting QaTa DETRIS Model Testing")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Calculate FLOPs: {args.calculate_flops}")
    logger.info(f"Save Individual Metrics: {args.save_individual_metrics}")
    
    try:
        cfg = load_config(args.config, args.opts)
        
        test_dataset = create_test_dataset(cfg, args.split)
        
        model = load_model(cfg, args.checkpoint)
        
        print_model_info(model, input_size=(3, cfg.input_size, cfg.input_size))

        metrics, detailed_results = test_model(model, test_dataset, args, cfg)
        
        logger.info("Testing completed successfully!")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise e


if __name__ == '__main__':
    main()