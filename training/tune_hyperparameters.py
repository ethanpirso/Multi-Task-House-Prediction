import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from models.multi_task_model import MultiTaskModel
from torch.utils.data import DataLoader, TensorDataset
import torch

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 6)
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])

    # Model instantiation
    model = MultiTaskModel(
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        num_encoder_layers=num_encoder_layers
    )

    # Data handling specifics
    # Modify or ensure your data loading logic matches the expected dimensions and types
    dataset = TensorDataset(torch.rand(100, 29), torch.rand(100, 1), torch.randint(0, 14, (100,)))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Trainer setup with Optuna pruning
    trainer = pl.Trainer(
        max_epochs=30,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        logger=False,  # Turn off logging for hyperparameter optimization runs
        progress_bar_refresh_rate=0
    )

    # Model training
    trainer.fit(model, dataloader)

    # Objective: Validation loss
    val_loss = trainer.callback_metrics["val_loss"]
    return val_loss

def tune_hyperparameters():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, timeout=3600)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

if __name__ == "__main__":
    tune_hyperparameters()
