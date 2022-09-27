# encoding: utf-8


import argparse
import os
# from collections import namedtuple
from typing import Dict

import pytorch_lightning as pl
import torch
from tokenizers import BertWordPieceTokenizer    # , ByteLevelBPETokenizer
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import SGD

from dataloaders.dataload import BERTNERDataset
from dataloaders.truncate_dataset import TruncateDataset
from dataloaders.collate_functions import collate_to_max_length
from models.bert_model_spanner import BertNER
from models.config_spanner import BertNerConfig, RobertaNerConfig
from eval_metric import span_f1, span_f1_prune, get_predict, get_predict_prune
import random
import logging
logger = logging.getLogger(__name__)


from pytorch_lightning import seed_everything
seed_everything(42)

import pickle

class BertNerTagger(pl.LightningModule):
    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir

        NERConfig = BertNerConfig if 'bert-' in args.bert_config_dir else RobertaNerConfig
        bert_config = NERConfig.from_pretrained(
            args.bert_config_dir,
            hidden_dropout_prob=args.bert_dropout,
            attention_probs_dropout_prob=args.bert_dropout,
            model_dropout=args.model_dropout
        )

        self.model = BertNER.from_pretrained(
            args.bert_config_dir,
            config=bert_config,
            args=self.args
        )
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

        self.optimizer = args.optimizer
        self.n_class = args.n_class

        self.max_spanLen = args.max_spanLen
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.classifier = torch.nn.Softmax(dim=-1)


        self.fwrite_epoch_res = open(args.fp_epoch_result, 'w', encoding='utf-8')
        self.fwrite_epoch_res.write("f1, recall, precision, correct_pred, total_pred, total_golden\n")

    @staticmethod
    def get_parser():
        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        parser = argparse.ArgumentParser(description="Training")

        # basic argument&value
        parser.add_argument("--data_dir", type=str, required=True, help="data dir")
        parser.add_argument("--bert_config_dir", type=str, required=True, help="bert config dir")
        parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
        parser.add_argument("--bert_max_length", type=int, default=200, help="max length of dataset")
        parser.add_argument("--batch_size", type=int, default=10, help="batch size")
        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
        parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="warmup steps used for scheduler.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--patience", default=5, type=int, help="Early stopping patience.")
        parser.add_argument("--es_threshold", default=1e-6, type=float)
        parser.add_argument('--epsilon', type=float, help='Adversarial training perturbation hyperparameter', default=0.)


        parser.add_argument("--model_dropout", type=float, default=0.2,
                            help="model dropout rate")
        parser.add_argument("--bert_dropout", type=float, default=0.2,
                            help="bert dropout rate")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                            help="loss type")
        #choices=["conll03", "ace04","notebn","notebc","notewb","notemz",'notenw','notetc']
        parser.add_argument("--dataname", default="conll03",
                            help="the name of a dataset")
        parser.add_argument("--max_spanLen", type=int, default=4, help="max span length")
        # parser.add_argument("--margin", type=float, default=0.03, help="margin of the ranking loss")
        parser.add_argument("--n_class", type=int, default=5, help="the classes of a task")
        parser.add_argument("--modelName",  default='test', help="the classes of a task")

        # parser.add_argument('--use_allspan', type=str2bool, default=True, help='use all the spans with O-labels ', nargs='?',
        #                     choices=['yes (default)', True, 'no', False])

        parser.add_argument('--use_tokenLen', type=str2bool, default=False, help='use the token length (after the bert tokenizer process) as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--tokenLen_emb_dim", type=int, default=50, help="the embedding dim of a span")
        parser.add_argument('--span_combination_mode', default='x,y',
                            help='Train data in format defined by --data-io param.')

        parser.add_argument('--use_spanLen', type=str2bool, default=False, help='use the span length as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--spanLen_emb_dim", type=int, default=100, help="the embedding dim of a span length")

        parser.add_argument('--use_morph', type=str2bool, default=True, help='use the span length as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--morph_emb_dim", type=int, default=100, help="the embedding dim of the morphology feature.")
        parser.add_argument('--morph2idx_list', type=list, help='a list to store a pair of (morph, index).', )


        parser.add_argument('--label2idx_list', type=list, help='a list to store a pair of (label, index).',)


        random_int = '%08d' % (random.randint(0, 100000000))
        print('random_int:', random_int)

        parser.add_argument('--random_int', type=str, default=random_int,help='a list to store a pair of (label, index).', )
        parser.add_argument('--param_name', type=str, default='param_name',
                            help='a prexfix for a param file name', )
        parser.add_argument('--best_dev_f1', type=float, default=0.0,
                            help='best_dev_f1 value', )
        parser.add_argument('--use_prune', type=str2bool, default=True,
                            help='best_dev_f1 value', )

        parser.add_argument("--use_span_weight", type=str2bool, default=True,
                            help="range: [0,1.0], the weight of negative span for the loss.")
        parser.add_argument("--neg_span_weight", type=float,default=0.5,
                            help="range: [0,1.0], the weight of negative span for the loss.")
        return parser


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(0.9, 0.98),  # according to RoBERTa paper
                lr=self.args.lr,
                eps=self.args.adam_epsilon,
            )
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, loadall,all_span_lens, all_span_idxs_ltoken,input_ids, attention_mask, token_type_ids):
        """"""
        return self.model(loadall, all_span_lens,all_span_idxs_ltoken,input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


    def compute_loss(self, loadall, all_span_rep, span_label_ltoken, real_span_mask_ltoken, mode):
        '''

        :param all_span_rep: shape: (bs, n_span, n_class)
        :param span_label_ltoken:
        :param real_span_mask_ltoken:
        :return:
        '''
        batch_size, n_span = span_label_ltoken.size()
        all_span_rep1 = all_span_rep.view(-1, self.n_class)
        span_label_ltoken1 = span_label_ltoken.view(-1)
        loss = self.cross_entropy(all_span_rep1, span_label_ltoken1)
        loss = loss.view(batch_size, n_span)

        if mode == 'train' and self.args.use_span_weight: # when training we should multiply the span-weight
            span_weight = loadall[6]
            loss = loss*span_weight

        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())
        loss= torch.mean(loss)

        predict = self.classifier(all_span_rep) # shape: (bs, n_span, n_class)

        return loss

    def perform_forward_step(self, batch, mode=''):
        tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = batch
        loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
        
        attention_mask = (tokens != 0).long()
        all_span_rep = self.forward(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
        predicts = self.classifier(all_span_rep)
        output = {}
        if self.args.use_prune:
            span_f1s, pred_label_idx = span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
        else:
            span_f1s = span_f1(predicts, span_label_ltoken, real_span_mask_ltoken)
        output["span_f1s"] = span_f1s
        loss = self.compute_loss(loadall, all_span_rep, span_label_ltoken, real_span_mask_ltoken, mode=mode)

        if mode == 'train':
            output[f"train_loss"] = loss
            output['loss'] = loss
        elif mode == 'test/dev':
            if self.args.use_prune:
                batch_preds = get_predict_prune(self.args, all_span_word, words, pred_label_idx, span_label_ltoken,
                                                all_span_idxs)
            else:
                batch_preds = get_predict(self.args, all_span_word, words, predicts, span_label_ltoken,
                                                all_span_idxs)
            output["batch_preds"] = batch_preds
            output[f"val_loss"] = loss
            output["predicts"] = predicts
            output['all_span_word'] = all_span_word
        else:
            raise NotImplementedError(f"`mode` must be chosen from ['train', 'test/dev']. ")
        return output

    def training_step(self, batch, batch_idx):
        """"""
        fgm = self.args.epsilon > 0
        tf_board_logs = {"lr": self.trainer.optimizers[0].param_groups[0]['lr']}
        
        if fgm:
            embeddings = self.model.bert.embeddings.word_embeddings.weight
            output = self.perform_forward_step(batch, mode='train')
            loss_ = output['loss']
            loss_.backward(inputs=embeddings)
            grads = self.model.bert.embeddings.word_embeddings.weight.grad
            # Add perturbations (Fast Gradient Method/FGM)
            delta = self.args.epsilon * grads / (torch.sqrt((grads**2).sum()) + 1e-8)
            with torch.no_grad():
                self.model.bert.embeddings.word_embeddings.weight += delta.cuda()
        
        output = self.perform_forward_step(batch, mode='train')
        tf_board_logs[f"loss"] = output['loss']
        output['log'] = tf_board_logs

        return output


    def training_epoch_end(self, outputs):
        """"""
        print("use... training_epoch_end: ", )
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        print('in train correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
        precision =correct_pred / (total_pred+1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)

        print("in train span_precision: ", precision)
        print("in train span_recall: ", recall)
        print("in train span_f1: ", f1)
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1

        self.fwrite_epoch_res.write(
            "train: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))


    def validation_step(self, batch, batch_idx):
        return self.perform_forward_step(batch, mode='test/dev')

    def validation_epoch_end(self, outputs):
        """"""
        print("use... validation_epoch_end: ", )
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        print('correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
        precision = correct_pred / (total_pred+1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)

        print("span_precision: ", precision)
        print("span_recall: ", recall)
        print("span_f1: ", f1)
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1
        self.fwrite_epoch_res.write("dev: %f, %f, %f, %d, %d, %d\n"%(f1,recall,precision,correct_pred, total_pred, total_golden) )

        if f1>self.args.best_dev_f1:
            pred_batch_results = [x['batch_preds'] for x in outputs]
            fp_write = self.args.default_root_dir +  '/' + self.args.modelName + '_dev.txt'
            fwrite = open(fp_write, 'w', encoding='utf-8')
            for pred_batch_result in pred_batch_results:
                for pred_result in pred_batch_result:
                    # print("pred_result: ", pred_result)
                    fwrite.write(pred_result + '\n')
            self.args.best_dev_f1=f1

            # begin{save the predict prob}
            all_predicts = [list(x['predicts']) for x in outputs]
            all_span_words = [list(x['all_span_word']) for x in outputs]

            # begin{get the label2idx dictionary}
            label2idx = {}
            label2idx_list = self.args.label2idx_list
            for labidx in label2idx_list:
                lab, idx = labidx
                label2idx[lab] = int(idx)
                # end{get the label2idx dictionary}

            file_prob1 = self.args.default_root_dir + '/' + self.args.modelName + '_prob_dev.pkl'
            print("the file path of probs: ", file_prob1)
            fwrite_prob = open(file_prob1, 'wb')
            pickle.dump([label2idx, all_predicts, all_span_words], fwrite_prob)
            # end{save the predict prob...}#
        
        self.log_dict({'val_loss': avg_loss, 'val_span_f1':tensorboard_logs['span_f1']})
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        print("use... test_epoch_end: ",)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        print('correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
        precision = correct_pred / (total_pred + 1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)

        print("span_precision: ", precision)
        print("span_recall: ", recall)
        print("span_f1: ", f1)
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1


        # begin{save the predict results}
        pred_batch_results = [x['batch_preds'] for x in outputs]
        fp_write = self.args.default_root_dir + '/'+self.args.modelName +'_test.txt'
        fwrite = open(fp_write, 'w', encoding='utf-8')
        for pred_batch_result in pred_batch_results:
            for pred_result in pred_batch_result:
                # print("pred_result: ", pred_result)
                fwrite.write(pred_result+'\n')

        self.fwrite_epoch_res.write(
            "test: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))
        # end{save the predict results}


        # begin{save the predict prob}
        all_predicts = [list(x['predicts'].cpu()) for x in outputs]
        all_span_words = [list(x['all_span_word']) for x in outputs]

            # begin{get the label2idx dictionary}
        label2idx = {}
        label2idx_list = self.args.label2idx_list
        for labidx in label2idx_list:
            lab, idx = labidx
            label2idx[lab] = int(idx)
            # end{get the label2idx dictionary}

        file_prob1 = self.args.default_root_dir + '/'+self.args.modelName +'_prob_test.pkl'
        print("the file path of probs: ", file_prob1)
        fwrite_prob = open(file_prob1, 'wb')
        pickle.dump([label2idx, all_predicts, all_span_words], fwrite_prob)
        # end{save the predict prob...}

        self.log('val_loss', avg_loss)
        self.log_dict(tensorboard_logs)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")
        # return self.get_dataloader("dev", 100)

    def val_dataloader(self):
        val_data = self.get_dataloader("dev")
        return val_data

    def test_dataloader(self):
        return self.get_dataloader("test")
        # return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        json_path = os.path.join(self.data_dir, f"spanner.{prefix}")
        print("json_path: ", json_path)
        # vocab_path = os.path.join(self.bert_dir, "vocab.txt")

        if 'bert-' in self.bert_dir:
            vocab_path = f"./vocab/{self.bert_dir}/vocab.txt"
            dataset = BERTNERDataset(
                self.args, json_path=json_path,
                # tokenizer=ByteLevelBPETokenizer(), 
                tokenizer=BertWordPieceTokenizer(vocab_path),
                max_length=self.args.bert_max_length,
                pad_to_maxlen=False, 
            )
        # elif 'roberta-' in self.bert_dir:
        #     dataset = BERTNERDataset(
        #         self.args, json_path=json_path,
        #         tokenizer=ByteLevelBPETokenizer(
        #             vocab=f"./vocab/{self.bert_dir}/vocab.json", 
        #             merges=f"./vocab/{self.bert_dir}/merges.txt", 
        #             lowercase='uncased' in self.bert_dir, 
        #             special_tokens=["[CLS]", "[PAD]", "[SEP]"], 
        #         ), 
        #         max_length=self.args.bert_max_length,
        #         pad_to_maxlen=False, 
        #     )
        else: 
            raise NotImplementedError(f'Tokenizer is not yet implemented for pretrained model {self.bert_dir}.')
        

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True if prefix == "train" else False,
            # shuffle=False,
            drop_last=False,
            collate_fn=collate_to_max_length
        )
        return dataloader
