import omegaconf
from omegaconf import OmegaConf


class Configuration(dict):
    def __init__(self, data_cfg, optim_cfg, model_cfg):
        super(Configuration, self).__init__()
        self.data_cfg = data_cfg
        self.optim_cfg = optim_cfg
        self.model_cfg = model_cfg

    def get_config(self) -> omegaconf.dictconfig.DictConfig:
        dataset_config = self.load_config(self.data_cfg)
        optimizer_config = self.load_config(self.optim_cfg)
        model_config = self.load_config(self.model_cfg)

        config = OmegaConf.merge(model_config, optimizer_config, dataset_config)
        return config

    # def save_config(self, save_path):
    #     with open(save_path, 'w') as f:
    #         OmegaConf.save

    @staticmethod
    def load_config(config_path: str) -> omegaconf.dictconfig.DictConfig:
        config = OmegaConf.load(config_path)

        return config



