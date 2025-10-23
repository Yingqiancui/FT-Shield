
import argparse
import json
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from PIL import Image
from tqdm import tqdm

from generate_watermark import generate_watermark

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """
    Custom dataset for loading images with captions.
    """
    
    def __init__(self, img_dir: str, captions_file: str, tokenizer: CLIPTokenizer, transform: Optional[transforms.Compose] = None):
        """
        Initialize the custom dataset.
        
        Args:
            img_dir (str): Directory containing images
            captions_file (str): Path to captions JSONL file
            tokenizer (CLIPTokenizer): Tokenizer for text processing
            transform (Optional[transforms.Compose]): Image transformations
        """
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.captions = self._load_captions(captions_file)
        self.imgs = self._load_images(img_dir)
        self.imgs.sort()  # Ensure images are sorted
        
        logger.info(f"Loaded {len(self.imgs)} images from {img_dir}")

    def _load_captions(self, captions_file: str) -> dict:
        """Load captions from JSONL file."""
        captions = {}
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    captions[data["file_name"]] = data["text"]
            logger.info(f"Loaded {len(captions)} captions from {captions_file}")
        except FileNotFoundError:
            logger.error(f"Captions file not found: {captions_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing captions file: {e}")
            raise
        return captions

    def _load_images(self, img_dir: str) -> list:
        """Load image paths from directory."""
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        imgs = []
        for dp, dn, filenames in os.walk(img_dir):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    imgs.append(os.path.join(dp, f))
        return imgs

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(self, idx: int) -> dict:
        """Get a single item from the dataset."""
        img_path = self.imgs[idx]
        file_name = os.path.basename(img_path)
        
        # Get caption
        if file_name in self.captions:
            caption = self.captions[file_name]
        else:
            logger.warning(f"No caption found for {file_name}")
            caption = ""

        # Load and process image
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise

        # Tokenize caption
        input_ids = self._tokenize_captions(caption)

        return {"pixel_values": image, "input_ids": input_ids}

    def _tokenize_captions(self, caption: str) -> torch.Tensor:
        """Tokenize caption text."""
        inputs = self.tokenizer(
            caption, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return inputs.input_ids.squeeze()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Watermark Generation for Stable Diffusion')
    
    # Data paths
    parser.add_argument('--img_dir', type=str, 
                       default='monet',
                       help='Directory containing training images')
    parser.add_argument('--captions_file', type=str,
                       default='monet/metadata.jsonl',
                       help='Path to captions JSONL file')
    parser.add_argument('--output_dir', type=str, default='./final',
                       help='Directory to save generated watermarked images')
    
    # Model parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use for computation')
    parser.add_argument('--model_name', type=str, default='compvis/stable-diffusion-v1-4',
                       help='Pre-trained model name')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Cache directory for model downloads')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--epsilon', type=float, default=4/2550.0,
                       help='Adversarial perturbation strength')
    parser.add_argument('--clip_max', type=float, default=4/255,
                       help='Upper bound for perturbations')
    parser.add_argument('--clip_min', type=float, default=-4/255,
                       help='Lower bound for perturbations')
    
    # Image processing
    parser.add_argument('--image_size', type=int, default=512,
                       help='Size to resize images to')
    parser.add_argument('--crop_size', type=int, default=512,
                       help='Size to crop images to')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def create_transforms(image_size: int, crop_size: int) -> transforms.Compose:
    """Create image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
    ])


def create_tokenizer(model_name: str, cache_dir: str) -> CLIPTokenizer:
    """Create and return CLIP tokenizer."""
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name, 
            subfolder="tokenizer", 
            revision=None,
            cache_dir=cache_dir
        )
        logger.info(f"Loaded tokenizer from {model_name}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def create_dataset(args: argparse.Namespace, tokenizer: CLIPTokenizer) -> CustomDataset:
    """Create and return custom dataset."""
    transforms_pipeline = create_transforms(args.image_size, args.crop_size)
    
    try:
        dataset = CustomDataset(
            img_dir=args.img_dir,
            captions_file=args.captions_file,
            tokenizer=tokenizer,
            transform=transforms_pipeline
        )
        return dataset
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise


def create_dataloader(dataset: CustomDataset, args: argparse.Namespace) -> DataLoader:
    """Create and return data loader."""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=None,
        pin_memory=True if args.device == 'cuda' else False
    )


def validate_paths(args: argparse.Namespace) -> None:
    """Validate that required paths exist."""
    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    
    if not os.path.exists(args.captions_file):
        raise FileNotFoundError(f"Captions file not found: {args.captions_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting watermark generation process")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Create tokenizer
        tokenizer = create_tokenizer(args.model_name, args.cache_dir)
        
        # Create dataset
        dataset = create_dataset(args, tokenizer)
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = create_dataloader(dataset, args)
        logger.info(f"DataLoader created with batch size {args.batch_size}")
        
        # Initialize watermark generator
        watermark_generator = generate_watermark(device=args.device)
        
        # Set training parameters
        watermark_generator.parse_params(
            epsilon=args.epsilon,
            epoch_num=args.epochs,
            clip_max=args.clip_max,
            clip_min=args.clip_min
        )
        
        # Generate watermarks
        logger.info("Starting watermark generation...")
        watermark_generator.generate_wm(dataloader)
        
        logger.info("Watermark generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during watermark generation: {e}")
        raise


if __name__ == "__main__":
    main()

