import sys
sys.path.append('../')
from transformer_ocr.utils.config import Configuration
from transformer_ocr.models.model import TransformerOCRCTC

config = Configuration(data_cfg='../examples/configs/dataset.yml',
                       model_cfg='../examples/configs/model.yml',
                       optim_cfg='../examples/configs/optimizer.yml')
config = config.get_config()

trainer = TransformerOCRCTC(config, pretrained=False)
# trainer.config.save('config.yml')
trainer.train()
