import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from accelerate.utils import ProjectConfiguration
from pathlib import Path
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel
from numpy import linalg as LA
from PIL import Image
from base import Base
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from tqdm import tqdm
import time

class generate_watermark(Base):

    def __init__(self, device: str = 'cuda'):
        """
        Initialize the watermark generator.
        
        Args:
            device (str): Computing device, defaults to 'cuda'
        """
        if not torch.cuda.is_available():
            print('CUDA not available, using CPU...')
            self.device = 'cpu'
        else:
            self.device = device

        # GPU configuration parameters
        self.gpu0_bsz = 13
        self.acc_grad = 4
        
        # Load pre-trained model components
        self._load_models()
        
        # Learning rate setting
        self.lr_train = 0.001

    def _load_models(self) -> None:
        """Load all pre-trained model components."""
        cache_dir = './cache'
        model_name = "compvis/stable-diffusion-v1-4"
        
        # Load VAE encoder
        self.vae = AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae", 
            revision=None,
            cache_dir=cache_dir
        )
        self.vae = self.vae.to('cuda')
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_name, 
            subfolder="scheduler",
            cache_dir=cache_dir
        )
        
        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, 
            subfolder="text_encoder", 
            revision=None,
            cache_dir=cache_dir
        )
        
        # Load UNet model
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, 
            subfolder="unet", 
            revision=None,
            cache_dir=cache_dir
        )

    def parse_params(self,
                     epsilon: float = 4/2550.0,
                     epoch_num: int = 10,
                     lr_train: float = 0.01,
                     clip_max: float = 4/255,
                     clip_min: float = -4/255) -> None:
        """
        Parse and set training parameters.
        
        Args:
            epsilon (float): Adversarial perturbation strength
            epoch_num (int): Number of training epochs
            lr_train (float): Learning rate
            clip_max (float): Upper bound for perturbations
            clip_min (float): Lower bound for perturbations
        """
        self.epsilon = epsilon
        self.epoch_num = epoch_num
        self.lr_train = lr_train
        self.clip_max = clip_max
        self.clip_min = clip_min
        
    def generate_wm(self, train_dataloader) -> None:
        """
        Main function for generating watermarked images.
        
        Args:
            train_dataloader: Training data loader
        """
        # Get batch size from dataloader
        self.batch_size = train_dataloader.batch_size
        self.parse_params()
        torch.manual_seed(1)

        # Initialize data storage
        clean_data = []
        perturbed_data = []
        text_data = []
        
        # Load all batch data with progress bar
        print("Loading dataset...")
        for step, batch in enumerate(tqdm(train_dataloader, desc="Loading batches", unit="batch")):
            clean_data.append(batch["pixel_values"].to(self.device))
            perturbed_data.append(batch["pixel_values"].to(self.device))
            text_data.append(batch["input_ids"].to(self.device))

        # Training loop with progress bar
        loss_history = []
        epoch_pbar = tqdm(range(1, self.epoch_num + 1), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}/{self.epoch_num}")
            
            start_time = time.time()
            perturbed_data, loss_sum = self.train_wm(perturbed_data, clean_data, text_data)
            epoch_time = time.time() - start_time
            
            loss_history.append(loss_sum)
            
            # Update progress bar with loss info
            epoch_pbar.set_postfix({
                'Loss': f'{loss_sum:.4f}',
                'Time': f'{epoch_time:.1f}s'
            })
            
            
            # Save generated images
            self._save_generated_images(perturbed_data, epoch)
        
        epoch_pbar.close()
        print(f"\nTraining completed! Final loss: {loss_history[-1]:.4f}")

    def _save_generated_images(self, perturbed_data: List[torch.Tensor], epoch: int) -> None:
        """Save generated watermarked images."""
        for i, batch in enumerate(perturbed_data):
            for j in range(batch.shape[0]):
                # Convert tensor to image format and save
                image_array = (batch[j] * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image_array)
                image.save(f'./final/{self.batch_size*i+j}.png')
     
    def calculate_loss(self, output: torch.Tensor, target: torch.Tensor, reduction_mode: str = 'mean') -> torch.Tensor:
        """
        Calculate cross-entropy loss.
        
        Args:
            output (torch.Tensor): Model output
            target (torch.Tensor): Target labels
            reduction_mode (str): Loss reduction mode
            
        Returns:
            torch.Tensor: Computed loss
        """
        loss = F.cross_entropy(output, target, reduction=reduction_mode)
        return loss

    def train_wm(self, perturbed_data: List[torch.Tensor], 
                 clean_data: List[torch.Tensor], 
                 text_data: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        """
        Train the watermark model.
        
        Args:
            perturbed_data: Perturbed data
            clean_data: Original clean data
            text_data: Text data
            
        Returns:
            Tuple[List[torch.Tensor], float]: Updated perturbed data and total loss
        """
        loss_sum = 0
        
        # Perform multiple perturbation updates for each batch with progress bar
        batch_pbar = tqdm(range(len(perturbed_data)), desc="Updating perturbations", 
                          leave=False, unit="batch")
        
        for i in batch_pbar:
            batch_pbar.set_description(f"Updating batch {i+1}/{len(perturbed_data)}")
            
            # Update perturbations for this batch
            for j in range(5):
                perturbed_data[i] = self.update_wm(perturbed_data[i], clean_data[i], text_data[i])
                batch_pbar.set_postfix({'Step': f'{j+1}/5'})
    
        batch_pbar.close()
        
        # Train diffusion model with progress bar
        print("Training diffusion model...")
        loss = self.train_diffusion_model(perturbed_data, text_data)
        loss_sum += loss
        
        return perturbed_data, loss_sum

    def update_wm(self, batch: torch.Tensor, clean_data: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Update watermark using FGSM adversarial attack.
        
        Args:
            batch (torch.Tensor): Input batch to be perturbed
            clean_data (torch.Tensor): Original clean data
            text (torch.Tensor): Text embeddings
            
        Returns:
            torch.Tensor: Updated perturbed batch
        """
        # Set models to evaluation mode
        self.unet.eval()
        self.unet.to('cuda')
        self.vae.eval()
        self.text_encoder.to(self.device)

        # Move data to device
        batch = batch.to(self.device)
        text = text.to(self.device)
        
        # Enable gradient computation for input
        batch.requires_grad = True

        # Initialize optimizer for input perturbation
        optimizer = optim.SGD([batch], lr=1e-5)
        optimizer.zero_grad()
    
        # Normalize input and encode to latent space
        normalized_batch = (batch - 0.5) * 2
        latents = self.vae.encode(normalized_batch.to(torch.float32)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Sample noise and timesteps for diffusion process
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Sample random timesteps for each image
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), 
            device=latents.device
        ).long()

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings for conditioning
        encoder_hidden_states = self.text_encoder(text)[0]

        # Determine target based on prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()

        # Apply FGSM perturbation
        perturbation = self.epsilon * batch.grad.data.sign()
        batch = batch - perturbation
        
        # Clip perturbation within bounds
        perturbation = torch.clamp(batch - clean_data, self.clip_min, self.clip_max)
        batch = (clean_data + perturbation).clamp(0, 1)
        batch = batch.detach()
        optimizer.zero_grad()

        return batch


    def train_diffusion_model(self, batch: List[torch.Tensor], text: List[torch.Tensor]) -> float:
        """
        Train the diffusion model on perturbed data.
        
        Args:
            batch (List[torch.Tensor]): List of perturbed batches
            text (List[torch.Tensor]): List of text embeddings
            
        Returns:
            float: Training loss
        """
        # Initialize optimizer
        optimizer_cls = torch.optim.AdamW
        self.unet.to(self.device)
        optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=5e-6,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        self.unet.train()

        # Freeze text encoder
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.device)

        # Encode all batches to latent space with progress bar
        all_latents = []
        print("Encoding batches to latent space...")
        with torch.no_grad():
            for batch_k in tqdm(batch, desc="Encoding", leave=False, unit="batch"):
                batch_k = batch_k.to('cuda')
                normalized_batch = (batch_k - 0.5) * 2
                latent = self.vae.encode(normalized_batch).latent_dist.sample()
                all_latents.append(latent.cpu()) 

                del batch_k
                del latent

        # Concatenate and split data for training
        text = torch.cat(text, dim=0)
        batch = torch.cat(all_latents, dim=0)
        batch = list(torch.split(batch, 10, dim=0))
        text = list(torch.split(text, 10, dim=0))

        # Training loop with progress bar
        training_pbar = tqdm(range(len(batch)), desc="Training UNet", leave=False, unit="step")
        total_loss = 0
        
        for i in training_pbar:
            batch[i] = batch[i].to(self.device)
            text[i] = text[i].to(self.device)

            # Scale latents
            latents = batch[i] * self.vae.config.scaling_factor

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]

            # Sample random timesteps for each image
            timesteps = torch.randint(
                0, 
                self.noise_scheduler.config.num_train_timesteps, 
                (batch_size,), 
                device=latents.device
            ).long()

            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings for conditioning
            encoder_hidden_states = self.text_encoder(text[i])[0]
            
            # Determine target based on prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            # Predict noise residual and compute loss
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()

            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            training_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg': f'{avg_loss:.4f}'})
        
        training_pbar.close()
        return loss.item()
