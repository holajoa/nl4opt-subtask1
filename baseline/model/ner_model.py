from typing import List, Any
from itertools import compress

import pytorch_lightning.core.lightning as pl

import torch
import torch.nn.functional as F
import numpy as np

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel

from log import logger
from utils.metric import SpanF1
from utils.reader_utils import extract_spans, get_tags


class NERBaseAnnotator(pl.LightningModule):
    def __init__(
        self,
        train_data=None,
        dev_data=None,
        lr=1e-5,
        dropout_rate=0.1,
        batch_size=16,
        epsilon=0., 
        tag_to_id=None,
        stage='fit',
        pad_token_id=1,
        encoder_model='xlm-roberta-large',
        num_gpus=1, 
        num_workers=0, 
    ):
        super(NERBaseAnnotator, self).__init__()

        self.train_data = train_data
        self.dev_data = dev_data

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size

        self.stage = stage
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.target_size = len(self.id_to_tag)

        ### ========================== set the default baseline model here ==========================
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)

        self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.target_size)

        self.crf_layer = ConditionalRandomField(num_tags=self.target_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))

        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)
        self.epsilon = epsilon

        # SpanF1 is required to evaluate the model - don't remove
        self.span_f1 = SpanF1()
        self.setup_model(self.stage)
        self.save_hyperparameters('pad_token_id', 'encoder_model')    # from pl.LightningModule, save model arguments to `hparams` attribute.

    def setup_model(self, stage_name):
        if stage_name == 'fit' and self.train_data is not None:
            # Calculate total steps
            train_batches = len(self.train_data) // (self.batch_size * self.num_gpus)
            self.total_steps = 50 * train_batches    # so model is trained for 50 epochs by default

            self.warmup_steps = int(self.total_steps * 0.01)    # for learning rate schedule?

    def collate_batch(self, batch):
        batch_ = list(zip(*batch))
        tokens, masks, token_masks, gold_spans, tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.tag_to_id['O'])
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)

        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]
            mask_tensor[i, :seq_len] = masks[i]
            token_masks_tensor[i, :seq_len] = token_masks[i]

        return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        if self.stage == 'fit':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.warmup_steps, 
                num_training_steps=self.total_steps, 
            )
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        return [optimizer]

    def train_dataloader(self):
        loader = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        if self.dev_data is None:
            return None
        loader = DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=self.num_workers)
        return loader

    def test_epoch_end(self, outputs):
        pred_results = self.span_f1.get_metric()
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, on_step=False, on_epoch=True)

        out = {"test_loss": avg_loss, "results": pred_results}
        return out

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='', on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='val_', on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch)
        self.log_metrics(output['results'], loss=output['loss'], suffix='val_', on_step=True, on_epoch=False)
        return output

    def training_step(self, batch, batch_idx):
        fgm = self.epsilon > 0
        if fgm:
            embeddings = self.encoder.embeddings.word_embeddings.weight
            output = self.perform_forward_step(batch, mode='train')
            loss_ = output['loss']
            loss_.backward(inputs=embeddings)
            grads = self.encoder.embeddings.word_embeddings.weight.grad
            # Add perturbations (Fast Gradient Method/FGM)
            delta = self.epsilon * grads / (torch.sqrt((grads**2).sum()) + 1e-8)
            with torch.no_grad():
                self.encoder.embeddings.word_embeddings.weight += delta.cuda()
        output = self.perform_forward_step(batch)
        self.log_metrics(output['results'], loss=output['loss'], suffix='', on_step=True, on_epoch=False)    # logging
        return output

    def test_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode=self.stage)
        self.log_metrics(output['results'], loss=output['loss'], suffix='_t', on_step=True, on_epoch=False)
        return output

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

    def perform_forward_step(self, batch, mode=''):
        tokens, tags, mask, token_mask, metadata = batch
        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=mask)  # transformer forward pass output
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))  # passed through a leaky relu activation + dropout

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)
        token_scores = F.log_softmax(token_scores, dim=-1)    # shape = (batch_size, seq_len, num_tags)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(
            token_scores=token_scores, 
            mask=mask,   # attention_mask
            tags=tags,   # ground truth tags (during training)
            metadata=metadata,    # contains all the ground truth spans of the batch
            batch_size=batch_size, 
            mode=mode, 
        )
        return output

    def _compute_token_tags(self, token_scores, mask, tags, metadata, batch_size, mode=''):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = - self.crf_layer(token_scores, tags, mask) / float(batch_size)    # crf layer loss
        best_path = self.crf_layer.viterbi_tags(token_scores, mask)    # viterbi decoding output, each element is tuple(best_path, loglik)

        pred_results, pred_tags = [], []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]    # get the best path
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])    # convert to string tags (BIO) and append
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))    # extract the spans and append

        self.span_f1(pred_results, metadata)    # used to calculate metrics
        output = {"loss": loss, "results": self.span_f1.get_metric()}

        if mode == 'predict':
            output['token_tags'] = pred_tags
        return output

    def predict_tags(self, batch, device='cuda:0', return_tokens_and_true_tags=False):
        tokens, tags, mask, token_mask, metadata = batch
        tokens, mask, token_mask, tags = tokens.to(device), mask.to(device), token_mask.to(device), tags.to(device)
        batch = tokens, tags, mask, token_mask, metadata

        pred_tags = self.perform_forward_step(batch, mode='predict')['token_tags']
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_mask)]
        if return_tokens_and_true_tags:
            return tokens, tag_results, [compress([self.id_to_tag[single_tag.item()] for single_tag in tags_], mask_) for tags_, mask_ in zip(tags, token_mask)]
        return tag_results
