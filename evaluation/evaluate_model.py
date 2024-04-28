import torch
from pytorch_lightning import Trainer
from models.multi_task_model import MultiTaskModel
from preprocessing.data_preprocessor import load_data
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import setup_logger, setup_callbacks

def testloader(df):
    features = torch.tensor(df.drop(['SalePrice', 'HouseCategory'], axis=1).values, dtype=torch.float)
    prices = torch.tensor(df['SalePrice'].values, dtype=torch.float).unsqueeze(1)  # Prices need to be a 2D tensor [n_samples, 1]
    categories = torch.tensor(df['HouseCategory'].values, dtype=torch.long)
    dataset = TensorDataset(features, prices, categories)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def evaluate_model(model, df):
    test_loader = testloader(df)
    trainer = Trainer(logger=setup_logger(), callbacks=setup_callbacks())
    results = trainer.test(model, test_loader)
    return results

if __name__ == "__main__":
    df = load_data("../data/processed/processed_train.csv")
    model = MultiTaskModel.load_from_checkpoint("path_to_checkpoint.ckpt")
    print(evaluate_model(model, df))
