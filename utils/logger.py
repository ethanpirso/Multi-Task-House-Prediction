from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def setup_logger():
    """ Set up MLFlow logger. """
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="sqlite:///../lightning_logs/mlruns.db")
    return mlf_logger

def setup_callbacks():
    """ Set up PyTorch Lightning's callbacks for checkpointing etc. """
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='../lightning_logs/checkpoints/',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        auto_insert_metric_name=False
    )
    return [checkpoint_callback]
