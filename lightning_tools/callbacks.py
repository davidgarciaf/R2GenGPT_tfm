import os
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint


def add_callbacks(args):
    # Directorio para guardar modelos y logs
    log_dir = args.savedmodel_path
    os.makedirs(log_dir, exist_ok=True)

    # --------- Add Callbacks
    # Guarda los pesos del modelo periódicamente
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(log_dir, "checkpoints"),
    #     filename="{epoch}-{step}",
    #     # disable top-k tracking when no monitor is configured, and keep the last checkpoint
    #     save_top_k=0,
    #     every_n_train_steps=args.every_n_train_steps,
    #     save_last=True,
    #     save_on_train_epoch_end=True,
    #     # save only model weights to reduce memory usage when serializing
    #     save_weights_only=True,
    #     verbose=True,
    # )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=0,
        every_n_train_steps=0,
        save_last=False,
        save_on_train_epoch_end=False,
        save_weights_only=True,
        verbose=True,
    )
    
    # Monitorea el learning rate
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": [checkpoint_callback, lr_monitor_callback],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns
