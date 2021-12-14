import os
import time
import numpy as np
import torch
import math
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import logging

from transformer_ocr.utils.vocab import VocabBuilder
from transformer_ocr.utils.dataset import OCRDataset, ClusterRandomSampler, Collator
from transformer_ocr.core.optimizers import NaiveScheduler
from transformer_ocr.utils.augment import ImgAugTransform
from transformer_ocr.utils.metrics import metrics
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
            raise ('Not Support model_type {}'.format(transformer_type))

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

        self.criterion = nn.CTCLoss(**self.config.pl_params.loss_func)

        if self.config.pl_params.pretrained:
            if not os.path.exists(self.config.pl_params.pretrained):
                logging.error('{} not exists. Please verify this!'.format(self.config.pl_params.pretrained))
                exit(0)

            self.load_weights(self.config.pl_params.pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, step):
        img = batch['img']
        tgt_output = batch['tgt_output']

        outputs = self.model(img)
        outputs = F.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1).requires_grad_()
        length = torch.tensor([tgt_output.size(1)] * self.batch_size, device=outputs.device).long()
        preds_size = torch.tensor([outputs.size(0)] * self.batch_size, device=outputs.device).long()

        loss = self.criterion(outputs, tgt_output, preds_size, length) / outputs.size(1)

        # Label smoothing loss
        if self.config.pl_params.ctc_smoothing:
            loss = loss * (1 - self.config.pl_params.ctc_smoothing) + \
                   self.kldiv_lsm_ctc(outputs.transpose(0, 1), preds_size) * self.config.pl_params.ctc_smoothing

        # Accumulation gradiant training
        loss = loss / self.config.pl_params.pl_trainer.accumulate_grad_batches

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.pl_params.max_norm)

        if (step + 1) % self.config.pl_params.pl_trainer.accumulate_grad_batches == 0:
            self.optimizer.step_and_update_lr()
            self.optimizer.zero_grad()

        return loss

    def train(self):
        total_loss: float = 0.0
        total_loader_time: float = 0.0
        total_gpu_time: float = 0.0
        best_acc: float = 0.0
        start_step: int = 0

        data_iter = iter(self.train_data)
        self.model.zero_grad()

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

            loss = self.training_step(batch=batch, step=i)

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

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f} - ' \
                       'normalized-ed: {:.4f}'.format(
                    start_step,
                    val_info['loss'],
                    val_info['sentence accuracy'],
                    val_info['character accuracy'],
                    val_info['normalized edit distance'])

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

        avg_sent_acc = metrics(actual_sents, pred_sents, type='accuracy')
        acc_per_char = metrics(actual_sents, pred_sents, type='char_acc')
        normalized_ed = metrics(actual_sents, pred_sents, type='normalized_ed')

        for i in range(len(pred_sents)):
            if pred_sents[i] != actual_sents[i]:
                print('Actual_sent: {}, pred_sent: {}'.format(actual_sents[i], pred_sents[i]))

        val_info = {"loss": losses.mean(),
                    "sentence accuracy": avg_sent_acc * 100,
                    "character accuracy": acc_per_char * 100,
                    "normalized edit distance": normalized_ed * 100}

        self.model.train()
        return val_info

    def validation_step(self, batch):
        img = batch['img']
        tgt_output = batch['tgt_output']

        logits = self.model(img)
        logits = F.log_softmax(logits, dim=2)
        outputs = logits.transpose(0, 1)
        length = torch.tensor([tgt_output.size(1)] * outputs.size(1), device=outputs.device).long()
        preds_size = torch.tensor([outputs.size(0)] * outputs.size(1), device=outputs.device).long()
        loss = self.criterion(outputs, tgt_output, preds_size, length) / outputs.size(1)

        return {
            'loss': loss,
            'logits': logits,
            'tgt_output': tgt_output
        }

    @property
    def transform(self):
        if not self.config.dataset.aug.image_aug:
            return None

        return ImgAugTransform()

    def train_dataloader(self) -> DataLoader:
        _dataloader = self._prepare_data('train_{}'.format(self.config.dataset.dataset.name),
                                         self.config.dataset.dataset.train_annotation, True)

        return _dataloader

    def val_dataloader(self) -> DataLoader:
        _dataloader = self._prepare_data('valid_{}'.format(self.config.dataset.dataset.name),
                                         self.config.dataset.dataset.valid_annotation, False)

        return _dataloader

    def _prepare_data(self, saved_path: str, data_path: str, use_transform: bool = True) -> DataLoader:
        if not use_transform:
            transform = None
        else:
            transform = self.transform

        dataset = OCRDataset(saved_path=saved_path,
                             gt_path=data_path,
                             vocab_builder=self.vocab,
                             transform=transform,
                             **self.config.dataset.dataset.unchanged)

        _dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=ClusterRandomSampler(dataset, self.batch_size),
            collate_fn=Collator(),
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

    @staticmethod
    def kldiv_lsm_ctc(logits: torch.Tensor, ylens: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for label smoothing of CTC and Transducer models.
        Args:
            logits (FloatTensor): `[B, T, vocab]`
            ylens (IntTensor): `[B]`
        Returns:
            loss_mean (FloatTensor): `[1]`
        """
        bs, _, vocab = logits.size()

        log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = torch.mul(probs, log_probs - log_uniform)
        loss_mean = sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()

        return loss_mean

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device('cuda:0'))

        # Multi-gpus
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith('module'):
                name = name[:7]

            new_state_dict[name] = param

        for name, param in self.model.named_parameters():
            if name not in new_state_dict:
                logging.warning('{} not found'.format(name))
            elif new_state_dict[name].shape != param.shape:
                logging.warning(
                    '{} miss-matching shape, required {} but found {}'.format(name, param.shape,
                                                                              new_state_dict[name].shape))
                del new_state_dict[name]

        self.model.load_state_dict(new_state_dict, strict=False)

