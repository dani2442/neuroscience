import torch
import torch.nn as nn
import lightning as pl

from control.neuralsde import SDEWrapper

class SDELightningModule(pl.LightningModule):
    def __init__(self, base_model, 
                 lr=1e-3, method='euler', dt=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["base_model"])
        
        # Wrap SDE for integration
        self.sde_wrapper = SDEWrapper(
            base_model,
            method=method,
            dt=dt
        )
        
    def forward(self, x0, ts, u):
        return self.sde_wrapper(x0, ts, u)
    
    def training_step(self, batch, batch_idx):
        # batch shape: [Time, Features]
        x_true, u, ts = batch
        
        # Initial condition
        x0 = x_true[:,0]  # [1, Features]
        x_pred = self(x0, ts, u)  # [Time, 1, Features]

        # MSE loss
        loss = nn.functional.mse_loss(x_pred, x_true)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_true,u, ts = batch
        x0 = x_true[:,0]
        x_pred = self(x0, ts, u)
        
        loss = nn.functional.mse_loss(x_pred, x_true)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }