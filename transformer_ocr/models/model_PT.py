import os
import time
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import logging

from transformer_ocr.utils.vocab import VocabBuilder
from transformer_ocr.utils.dataset import OCRDataset, ClusterRandomSampler, Collator
from transformer_ocr.core.optimizers import NaiveScheduler
from transformer_ocr.utils.augment import ImgAugTransform
from transformer_ocr.utils.metrics import compute_accuracy
from transformer_ocr.models.cnn_extraction.feature_extraction import FeatureExtraction
from transformer_ocr.models.transformers.conformer import ConformerEncoder
from transformer_ocr.models.transformers.tr_encoder import TransformerEncoder


class TransformerOCR(nn.Module):
    def __init__(self, vocab_size,
                 cnn_model,
                 cnn_args,
                 transformer_type,
                 transformer_args):

        super(TransformerOCR, self).__init__()
        self.feature_extraction = FeatureExtraction(cnn_model, **cnn_args)

        if transformer_type == 'transformer':
            self.transformer = TransformerEncoder(vocab_size, **transformer_args)
        elif transformer_type == 'conformer':
            self.transformer = ConformerEncoder(vocab_size, **transformer_args)
        else:
            raise('Not Support model_type {}'.format(transformer_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self.feature_extraction(x)
        outputs = self.transformer(src)

        return outputs


class TransformerOCRCTC:
    """TODO

    Args:

    """
    def __init__(self, config: DictConfig):
        super(TransformerOCRCTC, self).__init__()

        self.config = config
        self.vocab = VocabBuilder(config.model.vocab)
        self.model = TransformerOCR(vocab_size=len(self.vocab),
                                    cnn_model=config.model.cnn_model,
                                    cnn_args=config.model.cnn_args,
                                    transformer_type=config.model.transformer_type,
                                    transformer_args=config.model.transformer_args)

        self.model = self.model.to('cuda:0')
        self.model = nn.DataParallel(self.model, device_ids=[0, 1])

        self.batch_size = config.model.batch_size

        self.optimizer = NaiveScheduler(Adam(self.model.parameters(),
                                             lr=config.optimizer.optimizer.lr,
                                             betas=(config.optimizer.optimizer.betas_0,
                                                    config.optimizer.optimizer.betas_1),
                                             eps=config.optimizer.optimizer.eps), 2.0,
                                        config.model.transformer_args.d_model,
                                        config.optimizer.optimizer.n_warm_steps)

        self.train_data = self.train_dataloader()
        self.valid_data = self.val_dataloader()

        self.criterion = nn.CTCLoss(blank=1, zero_infinity=True, reduction='sum')

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch):
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch[
            'tgt_padding_mask']

        outputs = self.model(img)
        outputs = F.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1)
        length = torch.tensor([tgt_output.size(1)] * self.batch_size, device=outputs.device).long()
        preds_size = torch.tensor([outputs.size(0)] * self.batch_size, device=outputs.device).long()

        # print(length.size(), preds_size.size())
        # print(outputs.device, tgt_output.device, preds_size.device, length.device)

        loss = self.criterion(outputs, tgt_output, preds_size, length) / outputs.size(1)
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step_and_update_lr()

        # self.log('Training step',
        #          {"loss": loss,
        #           "lr": self.optimizer.get_optimizer().param_groups[0]['lr']},
        #          on_step=True,
        #          on_epoch=True,
        #          prog_bar=True,
        #          logger=True,
        #          batch_size=self.batch_size)

        return loss

    def train(self):
        total_loss: float = 0.0
        total_loader_time: float = 0.0
        total_gpu_time: float = 0.0
        best_acc: float = 0.0
        start_step: int = 0

        data_iter = iter(self.train_data)
        for i in range(self.config.pl_params.pl_trainer.max_steps):
            start_step += 1
            start_time = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_data)
                batch = next(data_iter)

            total_loader_time += time.time() - start_time
            start_time = time.time()

            loss = self.training_step(batch=batch)

            total_gpu_time += time.time() - start_time
            total_loss += loss.item()

            if start_step % self.config.pl_params.pl_trainer.log_every_n_steps == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(
                    start_step,
                    total_loss / self.config.pl_params.pl_trainer.log_every_n_steps,
                    self.optimizer.get_optimizer().param_groups[0]['lr'],
                    total_loader_time,
                    total_gpu_time)

                # reset
                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0

                logging.info(info)

            if start_step % self.config.pl_params.pl_trainer.val_every_n_steps == 0:
                val_info = self.validation()

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(
                    start_step,
                    val_info['loss'],
                    val_info['sentence accuracy'],
                    val_info['character accuracy'])

                logging.info(info)

                if val_info['sentence accuracy'] > best_acc:
                    saved_ckpt = os.path.join(self.config.pl_params.model_callbacks.dirpath,
                                              self.config.pl_params.model_callbacks.filename)
                    self.save_weights(saved_ckpt)
                    best_acc = val_info['sentence accuracy']

    def validation(self):
        self.model.eval()
        losses = np.array([])
        pred_sents = []
        actual_sents = []

        with torch.no_grad():
            for step, batch in enumerate(self.valid_data):
                valid_dict = self.validation_step(batch=batch)

                losses = np.append(losses, valid_dict['loss'].cpu().detach().numpy())
                logits = valid_dict['logits'].cpu().detach().numpy()
                pred_sents.extend([self._greedy_decode(logits[i]) for i in range(logits.shape[0])])
                actual_sents.extend(self.vocab.batch_decode(valid_dict['tgt_output'].tolist()))

        avg_sent_acc = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')

        for i in range(len(pred_sents)):
            if pred_sents[i] != actual_sents[i]:
                print('Actual_sent: {}, pred_sent: {}'.format(actual_sents[i], pred_sents[i]))

        val_info = {"loss": losses.mean(),
                    "sentence accuracy": avg_sent_acc * 100,
                    "character accuracy": acc_per_char * 100}

        self.model.train()
        return val_info

    def validation_step(self, batch):
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], \
                                                       batch['tgt_padding_mask']

        logits = self.model(img)
        logits = F.log_softmax(logits, dim=2)
        outputs = logits.transpose(0, 1)
        length = torch.tensor([tgt_output.size(1)] * self.batch_size).long()
        preds_size = torch.tensor([outputs.size(0)] * self.batch_size).long()
        loss = self.criterion(outputs, tgt_output, preds_size, length) / outputs.size(1)

        return {
            'loss': loss,
            'logits': logits,
            'tgt_output': tgt_output
        }

    def validation_step_end(self, batch_parts) -> dict:
        losses = batch_parts['loss']
        logits = batch_parts['logits']
        tgt_outputs = batch_parts['tgt_output']

        return {
            'loss': losses,
            'logits': logits,
            'tgt_output': tgt_outputs
        }

    def validation_epoch_end(self, outputs):
        losses = np.array([])
        pred_sents = []
        actual_sents = []

        for output in outputs:
            losses = np.append(losses, output['loss'].cpu().detach().numpy())
            logits = output['logits'].cpu().detach().numpy()
            pred_sents.extend([self._greedy_decode(logits[i]) for i in range(logits.shape[0])])
            actual_sents.extend(self.vocab.batch_decode(output['tgt_output'].tolist()))

        avg_sent_acc = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')

        for i in range(len(pred_sents)):
            if pred_sents[i] != actual_sents[i]:
                print('Actual_sent: {}, pred_sent: {}'.format(actual_sents[i], pred_sents[i]))

        val_info = {"loss": losses.mean(),
                    "sentence accuracy": avg_sent_acc * 100,
                    "character accuracy": acc_per_char * 100}

        self.log_dict(val_info, batch_size=self.batch_size)

        return val_info

    def configure_optimizers(self):
        return Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)

    @property
    def transform(self):
        if not self.config.dataset.aug.image_aug:
            return None

        return ImgAugTransform()

    def train_dataloader(self) -> DataLoader:
        _dataloader = self._prepare_data('train_{}'.format(self.config.dataset.dataset.name),
                                         self.config.dataset.dataset.train_annotation)

        return _dataloader

    def val_dataloader(self) -> DataLoader:
        _dataloader = self._prepare_data('valid_{}'.format(self.config.dataset.dataset.name),
                                         self.config.dataset.dataset.valid_annotation)

        return _dataloader

    def _prepare_data(self, saved_path: str, data_path: str) -> DataLoader:
        dataset = OCRDataset(saved_path=saved_path,
                             gt_path=data_path,
                             vocab_builder=self.vocab,
                             transform=self.transform,
                             **self.config.dataset.dataset.unchanged)

        sampler = ClusterRandomSampler(dataset, self.batch_size)
        collate_fn = Collator(False)

        _dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            **self.config.dataset.dataloader)

        return _dataloader

    def _greedy_decode(self, logits) -> str:
        """Decode argmax of logits and squash in CTC fashion."""
        label_dict = {n: c for n, c in enumerate(self.vocab.get_vocab_tokens())}
        prev_c = None
        out = []
        for n in logits.argmax(axis=-1):
            c = label_dict.get(n, "")  # if not in labels, then assume it's CTC <blank> token or <pad> token

            if c in [self.vocab.index_2_tok[0], self.vocab.index_2_tok[1]]:
                c = ""

            if c != prev_c:
                out.append(c)
            prev_c = c

        return "".join(out)

    def save_weights(self, filename):
        dir = os.path.dirname(filename)
        os.makedirs(dir, exist_ok=True)

        torch.save(self.model.state_dict(), filename)
