import sys
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append('../')

from transformer_ocr.models.model_PT import TransformerOCRCTC


@hydra.main(config_path='../conf', config_name="config")
def main(cfg: DictConfig):
    model = TransformerOCRCTC(config=cfg)

    checkpoint_callback = ModelCheckpoint(**cfg.pl_params.model_callbacks)
    trainer = pl.Trainer(**cfg.pl_params.pl_trainer, callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    main()
