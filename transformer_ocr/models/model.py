import torch
from torch import nn
import math
import torch
from torch.optim import Adam, SGD, AdamW
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from transformer_ocr.utils.vocab import VocabBuilder
from transformer_ocr.utils.dataset import OCRDataset, ClusterRandomSampler, Collator
from transformer_ocr.core.optimizers import NaiveScheduler
from transformer_ocr.utils.logger import Logger
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
        print('cnn_args', cnn_args)
        print('transformer_args', transformer_args)
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
    def __init__(self, config, pretrained=True, augmentor=ImgAugTransform()):
        super(TransformerOCRCTC, self).__init__()

        self.config = config
        self.vocab = VocabBuilder(config.vocab)
        self.device = config['device']
        self.model = TransformerOCR(vocab_size=len(self.vocab),
                                    cnn_model=config.cnn_model,
                                    cnn_args=config.cnn_args,
                                    transformer_type=config.transformer_type,
                                    transformer_args=config.transformer_args)

        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        self.num_iters = config['trainer']['iters']
        self.beamsearch = config['predictor']['beamsearch']

        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']
        self.dataset_name = config['dataset']['name']

        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']

        self.image_aug = config['aug']['image_aug']
        self.masked_language_model = config['aug']['masked_language_model']

        self.checkpoint = config['trainer']['checkpoint']
        self.export_weights = config['trainer']['export']
        self.metrics = config['trainer']['metrics']
        logger = config['trainer']['log']

        if logger:
            self.logger = Logger(logger)

        if pretrained:
            weight_file = config['pretrain']
            self.load_weights(weight_file)

        self.iter = 0

        self.optimizer = NaiveScheduler(Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09), 2.0,
                                        config['transformer_args']['d_model'], config['optimizer']['n_warm_steps'])
        self.criterion = nn.CTCLoss(blank=1, zero_infinity=True, reduction='sum')

        transforms = None
        if self.image_aug:
            transforms = augmentor

        self.train_gen = self.data_gen('train_{}'.format(self.dataset_name),
                                       self.data_root, self.train_annotation, masked_language_model=False,
                                       transform=transforms)
        if self.valid_annotation:
            self.valid_gen = self.data_gen('valid_{}'.format(self.dataset_name),
                                           self.data_root, self.valid_annotation, masked_language_model=False)

        self.train_losses = []

    def train(self):
        total_loss = 0

        total_loader_time = 0
        total_gpu_time = 0
        best_acc = 0

        data_iter = iter(self.train_gen)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(
                    self.iter,
                    total_loss / self.print_every, self.optimizer.get_optimizer().param_groups[0]['lr'],
                    total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info)
                self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_loss = self.validate()
                acc_full_seq, acc_per_char = self.precision(self.metrics)

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(
                    self.iter, val_loss, acc_full_seq, acc_per_char)
                print(info)
                self.logger.log(info)

                if acc_full_seq > best_acc:
                    self.save_weights(self.export_weights)
                    best_acc = acc_full_seq

    def validate(self):
        self.model.eval()

        total_loss = []

        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen):
                batch = self.batch_to_device(batch)
                img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], \
                                                               batch['tgt_padding_mask']

                outputs = self.model(img)

                outputs = F.log_softmax(outputs, dim=2)
                outputs = outputs.transpose(0, 1)

                length = Variable(torch.LongTensor([tgt_output.size(1)] * self.batch_size))
                preds_size = Variable(torch.LongTensor([outputs.size(0)] * self.batch_size))
                loss = self.criterion(outputs, tgt_output, preds_size, length) / outputs.size(1)
                total_loss.append(loss.item())

                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()

        return total_loss

    def predict(self, sample=None):
        self.model.eval()
        pred_sents = []
        actual_sents = []
        img_files = []
        prob = None

        with torch.no_grad():
            for batch in self.valid_gen:
                batch = self.batch_to_device(batch)

                logits = self.model(batch['img'])
                logits = logits.log_softmax(2)
                logits = logits.cpu().detach().numpy()

                pred_sent_list = [self._greedy_decode(logits[i]) for i in range(logits.shape[0])]
                actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())

                img_files.extend(batch['filenames'])

                pred_sents.extend(pred_sent_list)
                actual_sents.extend(actual_sent)

                if sample is not None and len(pred_sents) > sample:
                    break

        print(pred_sents[:5])
        print(actual_sents[:5])

        return pred_sents, actual_sents, img_files, prob

    def precision(self, sample=None):
        pred_sents, actual_sents, _, _ = self.predict(sample=sample)

        for i in range(len(pred_sents)):
            if pred_sents[i] != actual_sents[i]:
                print('Actual_sent: {}, pred_sent: {}'.format(actual_sents[i], pred_sents[i]))

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')

        return acc_full_seq, acc_per_char

    def visualize_prediction(self, sample=16, errorcase=False, fontname='serif', fontsize=16):

        pred_sents, actual_sents, img_files, probs = self.predict(sample)

        if errorcase:
            wrongs = []
            for i in range(len(img_files)):
                if pred_sents[i] != actual_sents[i]:
                    wrongs.append(i)

            pred_sents = [pred_sents[i] for i in wrongs]
            actual_sents = [actual_sents[i] for i in wrongs]
            img_files = [img_files[i] for i in wrongs]
            probs = [probs[i] for i in wrongs]

        img_files = img_files[:sample]

        fontdict = {
            'family': fontname,
            'size': fontsize
        }

        for vis_idx in range(0, len(img_files)):
            img_path = img_files[vis_idx]
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            prob = probs[vis_idx]

            img = Image.open(open(img_path, 'rb'))
            plt.figure()
            plt.imshow(img)
            plt.title('prob: {:.3f} - pred: {} - actual: {}'.format(prob, pred_sent, actual_sent), loc='left',
                      fontdict=fontdict)
            plt.axis('off')

        plt.show()

    def visualize_dataset(self, sample=16, fontname='serif'):
        n = 0
        for batch in self.valid_gen:
            for i in range(self.batch_size):
                img = batch['img'][i].numpy().transpose(1, 2, 0)
                sent = self.vocab.decode(batch['tgt_input'].T[i].tolist())

                plt.figure()
                plt.title('sent: {}'.format(sent), loc='center', fontname=fontname)
                plt.imshow(img)
                plt.axis('off')

                n += 1
                if n >= sample:
                    plt.show()
                    return

    # def load_checkpoint(self, filename):
    #     checkpoint = torch.load(filename)
    #
    #     optim = ScheduledOptim(
    #         Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    #         self.config['transformer']['d_model'], **self.config['optimizer'])
    #
    #     self.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.model.load_state_dict(checkpoint['state_dict'])
    #     self.iter = checkpoint['iter']
    #
    #     self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'iter': self.iter, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}

        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print(
                    '{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=True)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), filename)

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)

        batch = {
            'img': img, 'tgt_input': tgt_input,
            'tgt_output': tgt_output, 'tgt_padding_mask': tgt_padding_mask,
            'filenames': batch['filenames']
        }

        return batch

    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = OCRDataset(lmdb_path=lmdb_path,
                             root_dir=data_root, annotation_path=annotation,
                             vocab=self.vocab, transform=transform,
                             image_height=self.config['dataset']['image_height'],
                             image_min_width=self.config['dataset']['image_min_width'],
                             image_max_width=self.config['dataset']['image_max_width'])

        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)

        gen = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            **self.config['dataloader'])

        return gen

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch[
            'tgt_padding_mask']

        outputs = self.model(img)
        outputs = outputs.transpose(0, 1)
        outputs = outputs.log_softmax(2).requires_grad_()
        length = Variable(torch.LongTensor([tgt_output.size(1)] * self.batch_size))
        preds_size = Variable(torch.LongTensor([outputs.size(0)] * self.batch_size))

        loss = self.criterion(outputs, tgt_output, preds_size, length) / outputs.size(1)
        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step_and_update_lr()

        loss_item = loss.item()

        return loss_item

    def _greedy_decode(self, logits):
        """Decode argmax of logits and squash in CTC fashion."""
        label_dict = {n: c for n, c in enumerate(self.vocab.get_vocab_tokens())}
        prev_c = None
        out = []
        for n in logits.argmax(axis=-1):
            c = label_dict.get(n, "")  # if not in labels, then assume it's CTC <blank> token or <pad> token

            if c == "<pad>" or c == "<blank>":
                c = ""

            if c != prev_c:
                out.append(c)
            prev_c = c

        return "".join(out)

    def kldiv_lsm_ctc(self, logits, ylens):
        #     """Compute KL divergence loss for label smoothing of CTC and Transducer models.
        #     Args:
        #         logits (FloatTensor): `[B, T, vocab]`
        #         ylens (IntTensor): `[B]`
        #     Returns:
        #         loss_mean (FloatTensor): `[1]`
        #     """
        bs, _, vocab = logits.size()

        log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = torch.mul(probs, log_probs - log_uniform)
        loss_mean = sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
        return loss_mean

    def evaluate(self, ):
        acc_full_seq, acc_per_char = self.precision(self.metrics)

        info = 'iter: {:06d}  - acc full seq: {:.4f} - acc per char: {:.4f}'.format(self.iter, acc_full_seq,
                                                                                    acc_per_char)
        print(info)




