"""
Metrics tracking and visualization

FIX: matplotlib 백엔드를 'Agg'로 설정하여 멀티스레드 환경에서 tkinter 충돌 방지
"""

import matplotlib
matplotlib.use('Agg')  # 반드시 pyplot import 전에! GUI 없이 파일 저장만 수행

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'vae': {'train_loss': [], 'recon_loss': [], 'kl_loss': [], 'lpips_loss': []},
            'diffusion': {'train_loss': [], 'mse_loss': [], 'l1_loss': [], 'gdl_loss': [], 'lpips_loss': []},
            'consistency': {'train_loss': []},
            'val': {'psnr': [], 'ssim': [], 'mae': [], 'mse': []}
        }
        self.epoch_logs = []
        self.early_stop_patience = 10
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def update(self, phase, **kwargs):
        """Update metrics for a given phase"""
        for key, value in kwargs.items():
            if phase in self.metrics and key in self.metrics[phase]:
                self.metrics[phase][key].append(value)
    
    def check_early_stop(self, current_loss):
        """Check if training should stop early"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stop_patience:
                logger.warning(f"Early stopping triggered. No improvement for {self.early_stop_patience} epochs.")
                return True
        return False
    
    def log_epoch(self, epoch, phase, **kwargs):
        """Log metrics for an epoch"""
        log_entry = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **kwargs
        }
        self.epoch_logs.append(log_entry)
        
        log_msg = f"\n{'='*60}\n"
        log_msg += f"Epoch {epoch} - {phase.upper()}\n"
        log_msg += f"{'='*60}\n"
        
        for key, value in kwargs.items():
            if isinstance(value, float):
                log_msg += f"{key:<20}: {value:.6f}\n"
            else:
                log_msg += f"{key:<20}: {value}\n"
        
        logger.info(log_msg)
    
    def plot_interim_metrics(self, save_dir, phase_name, epoch):
        """Plot metrics at specific epoch intervals"""
        interim_dir = os.path.join(save_dir, 'interim_plots')
        os.makedirs(interim_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        if phase_name == 'vae' and self.metrics['vae']['train_loss']:
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0])
            epochs = np.arange(1, len(self.metrics['vae']['train_loss']) + 1)
            ax1.plot(epochs, self.metrics['vae']['train_loss'], 'b-', linewidth=2, label='Total Loss')
            ax1.plot(epochs, self.metrics['vae']['recon_loss'], 'g--', linewidth=2, label='Recon Loss')
            ax1.plot(epochs, self.metrics['vae']['kl_loss'], 'r-.', linewidth=2, label='KL Loss')
            if self.metrics['vae']['lpips_loss']:
                ax1.plot(epochs, self.metrics['vae']['lpips_loss'], 'm:', linewidth=2, label='LPIPS Loss')
            ax1.set_title('VAE Training Losses', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            if self.metrics['val']['psnr']:
                ax2 = fig.add_subplot(gs[0, 1])
                eval_points = np.arange(1, len(self.metrics['val']['psnr']) + 1) * 5
                ax2.plot(eval_points, self.metrics['val']['psnr'], 'b-o', linewidth=2, markersize=6, label='PSNR')
                ax2.set_title('VAE PSNR', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('PSNR (dB)', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                ax3 = fig.add_subplot(gs[1, 0])
                eval_points = np.arange(1, len(self.metrics['val']['ssim']) + 1) * 5
                ax3.plot(eval_points, self.metrics['val']['ssim'], 'r-s', linewidth=2, markersize=6, label='SSIM')
                ax3.set_title('VAE SSIM', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Epoch', fontsize=12)
                ax3.set_ylabel('SSIM', fontsize=12)
                ax3.legend(fontsize=10)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim([0, 1])
            
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            summary_text = f"VAE Metrics Summary at Epoch {epoch}\n"
            summary_text += "=" * 30 + "\n\n"
            
            if self.metrics['val']['psnr']:
                summary_text += f"Val PSNR: {self.metrics['val']['psnr'][-1]:.2f} dB\n"
            if self.metrics['val']['ssim']:
                summary_text += f"Val SSIM: {self.metrics['val']['ssim'][-1]:.3f}\n"
            if self.metrics['vae']['train_loss']:
                summary_text += f"Train Loss: {self.metrics['vae']['train_loss'][-1]:.4f}\n"
            
            ax4.text(0.1, 0.5, summary_text, fontsize=12, transform=ax4.transAxes, 
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            fig.suptitle(f'VAE Training Metrics - Epoch {epoch}', fontsize=16, fontweight='bold')
            
        elif phase_name == 'diffusion' and self.metrics['diffusion']['train_loss']:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, :2])
            epochs = np.arange(1, len(self.metrics['diffusion']['train_loss']) + 1)
            ax1.plot(epochs, self.metrics['diffusion']['train_loss'], 'purple', linewidth=2, label='Total Loss')
            if self.metrics['diffusion']['mse_loss']:
                ax1.plot(epochs, self.metrics['diffusion']['mse_loss'], 'b--', linewidth=2, label='MSE Loss')
            if self.metrics['diffusion']['l1_loss']:
                ax1.plot(epochs, self.metrics['diffusion']['l1_loss'], 'c--', linewidth=2, label='L1 Loss')
            if self.metrics['diffusion']['gdl_loss']:
                ax1.plot(epochs, self.metrics['diffusion']['gdl_loss'], 'g--', linewidth=2, label='GDL Loss')
            if self.metrics['diffusion']['lpips_loss']:
                ax1.plot(epochs, self.metrics['diffusion']['lpips_loss'], 'm--', linewidth=2, label='LPIPS Loss')
            ax1.set_title('Diffusion Training Losses', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            if self.metrics['val']['psnr']:
                ax2 = fig.add_subplot(gs[1, 0])
                diffusion_start_idx = max(0, len(self.metrics['val']['psnr']) - len(self.metrics['diffusion']['train_loss']) // 5)
                diffusion_psnr = self.metrics['val']['psnr'][diffusion_start_idx:]
                eval_points = np.arange(1, len(diffusion_psnr) + 1) * 5
                ax2.plot(eval_points, diffusion_psnr, 'b-o', linewidth=2, markersize=6)
                ax2.set_title('Diffusion PSNR', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Diffusion Epoch', fontsize=12)
                ax2.set_ylabel('PSNR (dB)', fontsize=12)
                ax2.grid(True, alpha=0.3)
            
            if self.metrics['val']['ssim']:
                ax3 = fig.add_subplot(gs[1, 1])
                diffusion_start_idx = max(0, len(self.metrics['val']['ssim']) - len(self.metrics['diffusion']['train_loss']) // 5)
                diffusion_ssim = self.metrics['val']['ssim'][diffusion_start_idx:]
                eval_points = np.arange(1, len(diffusion_ssim) + 1) * 5
                ax3.plot(eval_points, diffusion_ssim, 'r-s', linewidth=2, markersize=6)
                ax3.set_title('Diffusion SSIM', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Diffusion Epoch', fontsize=12)
                ax3.set_ylabel('SSIM', fontsize=12)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim([0, 1])
            
            ax4 = fig.add_subplot(gs[1, 2])
            if self.metrics['diffusion']['gdl_loss'] and self.metrics['diffusion']['lpips_loss']:
                epochs = np.arange(1, len(self.metrics['diffusion']['gdl_loss']) + 1)
                ax4_twin = ax4.twinx()
                l1 = ax4.plot(epochs, self.metrics['diffusion']['gdl_loss'], 'g-', linewidth=2, label='GDL')
                l2 = ax4_twin.plot(epochs, self.metrics['diffusion']['lpips_loss'], 'm-', linewidth=2, label='LPIPS')
                ax4.set_xlabel('Epoch', fontsize=12)
                ax4.set_ylabel('GDL Loss', color='g', fontsize=12)
                ax4_twin.set_ylabel('LPIPS Loss', color='m', fontsize=12)
                ax4.tick_params(axis='y', labelcolor='g')
                ax4_twin.tick_params(axis='y', labelcolor='m')
                lines = l1 + l2
                labels = [l.get_label() for l in lines]
                ax4.legend(lines, labels, loc='upper right')
                ax4.set_title('Perceptual Loss Components', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
            
            ax5 = fig.add_subplot(gs[2, :])
            ax5.axis('off')
            
            summary_text = f"Diffusion Metrics Summary at Epoch {epoch}\n"
            summary_text += "=" * 40 + "\n\n"
            
            if self.metrics['val']['psnr'] and 'diffusion_psnr' in locals() and diffusion_psnr:
                summary_text += f"Val PSNR: {diffusion_psnr[-1]:.2f} dB\n"
            if self.metrics['val']['ssim'] and 'diffusion_ssim' in locals() and diffusion_ssim:
                summary_text += f"Val SSIM: {diffusion_ssim[-1]:.3f}\n"
            if self.metrics['diffusion']['train_loss']:
                summary_text += f"Total Loss: {self.metrics['diffusion']['train_loss'][-1]:.4f}\n"
            if self.metrics['diffusion']['mse_loss']:
                summary_text += f"MSE Loss: {self.metrics['diffusion']['mse_loss'][-1]:.4f}\n"
            if self.metrics['diffusion']['l1_loss']:
                summary_text += f"L1 Loss: {self.metrics['diffusion']['l1_loss'][-1]:.4f}\n"
            if self.metrics['diffusion']['gdl_loss']:
                summary_text += f"GDL Loss: {self.metrics['diffusion']['gdl_loss'][-1]:.4f}\n"
            if self.metrics['diffusion']['lpips_loss']:
                summary_text += f"LPIPS Loss: {self.metrics['diffusion']['lpips_loss'][-1]:.4f}\n"
            
            ax5.text(0.5, 0.5, summary_text, fontsize=14, transform=ax5.transAxes, 
                    verticalalignment='center', horizontalalignment='center', 
                    fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
            
            fig.suptitle(f'Diffusion Training Metrics - Epoch {epoch}', fontsize=16, fontweight='bold')
            
        elif phase_name == 'consistency' and self.metrics['consistency']['train_loss']:
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, :])
            epochs = np.arange(1, len(self.metrics['consistency']['train_loss']) + 1)
            ax1.plot(epochs, self.metrics['consistency']['train_loss'], 'purple', linewidth=2)
            ax1.set_title('3D Consistency Training Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(gs[1, :])
            ax2.axis('off')
            
            summary_text = f"3D Consistency Metrics Summary at Epoch {epoch}\n"
            summary_text += "=" * 40 + "\n\n"
            
            if self.metrics['consistency']['train_loss']:
                summary_text += f"Train Loss: {self.metrics['consistency']['train_loss'][-1]:.4f}\n"
            
            ax2.text(0.5, 0.5, summary_text, fontsize=14, transform=ax2.transAxes, 
                    verticalalignment='center', horizontalalignment='center', 
                    fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            
            fig.suptitle(f'3D Consistency Training Metrics - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        plot_filename = os.path.join(interim_dir, f'metrics_{phase_name}_epoch_{epoch:03d}.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved interim metrics plot: {plot_filename}")
    
    def save_metrics(self, save_dir):
        """Save metrics to files"""
        df_logs = pd.DataFrame(self.epoch_logs)
        df_logs.to_csv(os.path.join(save_dir, 'training_logs.csv'), index=False)
        
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot_metrics(self, save_dir):
        """Plot final training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        if self.metrics['vae']['train_loss']:
            ax = axes[0]
            ax.plot(self.metrics['vae']['train_loss'], label='Total Loss')
            ax.plot(self.metrics['vae']['recon_loss'], label='Recon Loss')
            ax.plot(self.metrics['vae']['kl_loss'], label='KL Loss')
            if self.metrics['vae']['lpips_loss']:
                ax.plot(self.metrics['vae']['lpips_loss'], label='LPIPS Loss')
            ax.set_title('VAE Training Losses')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        if self.metrics['diffusion']['train_loss']:
            ax = axes[1]
            ax.plot(self.metrics['diffusion']['train_loss'], label='Total Loss', linewidth=2)
            if self.metrics['diffusion']['mse_loss']:
                ax.plot(self.metrics['diffusion']['mse_loss'], '--', label='MSE Loss', alpha=0.7)
            if self.metrics['diffusion']['l1_loss']:
                ax.plot(self.metrics['diffusion']['l1_loss'], '--', label='L1 Loss', alpha=0.7)
            if self.metrics['diffusion']['gdl_loss']:
                ax.plot(self.metrics['diffusion']['gdl_loss'], '--', label='GDL Loss', alpha=0.7)
            if self.metrics['diffusion']['lpips_loss']:
                ax.plot(self.metrics['diffusion']['lpips_loss'], '--', label='LPIPS Loss', alpha=0.7)
            ax.set_title('Diffusion Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        if self.metrics['consistency']['train_loss']:
            ax = axes[2]
            ax.plot(self.metrics['consistency']['train_loss'])
            ax.set_title('3D Consistency Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True)
        
        if self.metrics['val']['psnr']:
            ax = axes[3]
            ax.plot(self.metrics['val']['psnr'], label='Validation')
            ax.set_title('PSNR')
            ax.set_xlabel('Evaluation Step')
            ax.set_ylabel('PSNR (dB)')
            ax.legend()
            ax.grid(True)
        
        if self.metrics['val']['ssim']:
            ax = axes[4]
            ax.plot(self.metrics['val']['ssim'], label='Validation')
            ax.set_title('SSIM')
            ax.set_xlabel('Evaluation Step')
            ax.set_ylabel('SSIM')
            ax.legend()
            ax.grid(True)
        
        ax = axes[5]
        if self.metrics['val']['mae']:
            ax.plot(self.metrics['val']['mae'], label='MAE')
        if self.metrics['val']['mse']:
            ax.plot(self.metrics['val']['mse'], label='MSE')
        ax.set_title('Additional Metrics')
        ax.set_xlabel('Evaluation Step')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_metrics_final.png'), dpi=300, bbox_inches='tight')
        plt.close()