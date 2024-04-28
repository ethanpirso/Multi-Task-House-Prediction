import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultiTaskModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Enhanced architecture with a more complex attention mechanism for improved performance
        self.shared_layers = nn.Sequential(
            nn.Linear(63, 256),  # Increased input layer size for capturing more complex relationships
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),  # Reducing dimensionality for attention mechanism compatibility
            nn.GELU()
        )
        # Enhanced attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.2)
        self.attention_linear = nn.Linear(256, 256)  # To process attention output more effectively
        encoder_layers = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.2, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=4)
        # Revised regression head with additional layers and dropout for improved performance
        self.regression_head = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 14)
        )

    def forward(self, x):
        shared_output = self.shared_layers(x)
        # Adjusting for the enhanced attention mechanism
        shared_output = shared_output.unsqueeze(0)
        attention_output, _ = self.attention(shared_output, shared_output, shared_output)
        attention_output = self.transformer_encoder(attention_output)
        attention_output = attention_output.squeeze(0)
        attention_processed = self.attention_linear(attention_output)
        price = self.regression_head(attention_processed)
        category = self.classification_head(attention_processed)
        return price, category

    def training_step(self, batch, batch_idx):
        x, y_price, y_category = batch
        price_pred, category_pred = self(x)
        loss_fn_reg = torch.nn.MSELoss(reduction='sum')
        loss_fn_class = torch.nn.CrossEntropyLoss()
        loss_price = torch.sqrt(loss_fn_reg(price_pred.squeeze(), y_price) / len(y_price))  # RMSE for regression
        loss_category = loss_fn_class(category_pred, y_category)
        loss = loss_price + loss_category
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y_price, y_category = batch
        price_pred, category_pred = self(x)
        loss_fn_reg = torch.nn.MSELoss(reduction='sum')
        loss_fn_class = torch.nn.CrossEntropyLoss()
        loss_price = torch.sqrt(loss_fn_reg(price_pred.squeeze(), y_price) / len(y_price))  # RMSE for regression
        loss_category = loss_fn_class(category_pred, y_category)
        loss = loss_price + loss_category
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.008)  # Adjusted learning rate and optimizer for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)  # Adjusted scheduler for smoother learning rate changes
        return [optimizer], [scheduler]
