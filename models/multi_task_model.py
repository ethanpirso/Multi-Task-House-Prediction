import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultiTaskModel(pl.LightningModule):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        # Enhanced architecture with a more complex and deeper shared layer structure for improved feature extraction and performance
        self.shared_layers = nn.Sequential(
            nn.Linear(63, 256),  # Reduced initial expansion for capturing essential relationships
            nn.BatchNorm1d(256),
            nn.ReLU(),  # Using ReLU for initial layer for better gradient flow
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),  # Mixing GELU and ReLU for diverse activation patterns
            nn.Dropout(0.1),
            nn.Linear(512, 256),  # Reducing dimensionality sooner for attention mechanism compatibility
            nn.ReLU(),  # Back to ReLU for a mix of activation functions
            nn.Dropout(0.1),
            nn.Linear(256, 256),  # Maintaining dimensionality for consistent feature representation
            nn.GELU(),  # Using GELU for deeper layers
            nn.Dropout(0.1)
        )
        # Enhanced attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.2)
        self.attention_linear = nn.Linear(256, 256)  # To process attention output more effectively
        encoder_layers = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=0.2, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)  # Increased the number of layers for deeper processing
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
            nn.Linear(256, 512),  # Slightly expanded layer for feature capture
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.02),  # Utilizing LeakyReLU for non-linear activation with a small slope for negative inputs
            nn.Dropout(0.3),  # Maintaining dropout for regularization
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),  # Keeping ReLU here for activation
            nn.Dropout(0.2),
            nn.Linear(1024, 512),  # Reducing dimensionality while preserving essential information
            nn.LayerNorm(512),
            nn.LeakyReLU(negative_slope=0.01),  # Using LeakyReLU again for a slight negative slope activation
            nn.Dropout(0.2),
            nn.Linear(512, 256),  # Final refinement step before classification output
            nn.LayerNorm(256),
            nn.ReLU(),  # Back to ReLU for the final activation before output
            nn.Dropout(0.1),
            nn.Linear(256, 8)  # Output layer for 8 classes, matching the task requirement
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
        loss_fn_class = torch.nn.CrossEntropyLoss(weight=self.class_weights)  # Added class weights for better class balancing
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

        # Compute RMSE for regression
        loss_price = torch.sqrt(loss_fn_reg(price_pred.squeeze(), y_price) / len(y_price))
        self.log('test_rmse', loss_price, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Compute accuracy for the categorical
        _, preds = torch.max(category_pred, dim=1)
        correct_count = (preds == y_category).sum().item()
        total_count = y_category.size(0)
        accuracy = correct_count / total_count
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'test_rmse': loss_price, 'test_accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)  # Adjusted learning rate and optimizer for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)  # Adjusted scheduler for smoother learning rate changes
        return [optimizer], [scheduler]
