import torch
from pytorch_lightning import Trainer
from models.multi_task_model import MultiTaskModel
from preprocessing.data_preprocessor import load_data
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import setup_logger, setup_callbacks

def trainloader(df):
    features = torch.tensor(df.drop(['SalePrice', 'HouseCategory'], axis=1).values, dtype=torch.float)
    prices = torch.tensor(df['SalePrice'].values, dtype=torch.float).unsqueeze(1)  # Prices need to be a 2D tensor [n_samples, 1]
    categories = torch.tensor(df['HouseCategory'].values, dtype=torch.long)
    category_counts = df['HouseCategory'].value_counts().reindex(range(8), fill_value=0)
    class_weights = 1.0 / (category_counts / category_counts.sum())
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights.values, dtype=torch.float).to('cuda')
    dataset = TensorDataset(features, prices, categories)
    return DataLoader(dataset, batch_size=5, shuffle=True), class_weights

def train(df):
    train_loader, class_weights = trainloader(df)
    model = MultiTaskModel(class_weights=class_weights)
    trainer = Trainer(max_epochs=35, logger=setup_logger(), callbacks=setup_callbacks())
    trainer.fit(model, train_loader)
    return model

if __name__ == "__main__":
    df = load_data("../data/processed/processed_train.csv")
    train(df)
