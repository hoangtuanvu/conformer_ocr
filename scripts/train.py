import sys
import hydra
from omegaconf import DictConfig

sys.path.append('../')

from transformer_ocr.models.model import TransformerOCRCTC


@hydra.main(config_path='../conf', config_name="config")
def main(cfg: DictConfig):
    model = TransformerOCRCTC(config=cfg)
    model.train()


if __name__ == "__main__":
    main()
