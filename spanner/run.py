# encoding: utf-8


import os

import torch
from pytorch_lightning import Trainer

from utils.trainer_utils import load_2idx, get_trainer, write_eval_performance, write_for_leaderboard
from trainer import BertNerTagger

import logging
logger = logging.getLogger(__name__)

from pytorch_lightning import seed_everything
seed_everything(42)

# add model specific args
parser = BertNerTagger.get_parser()
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
args = load_2idx(args)

# args.default_root_dir = args.default_root_dir + '_' + args.random_int

if not os.path.exists(args.default_root_dir):
    os.makedirs(args.default_root_dir)

text = '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
print(text)

fn_path = args.default_root_dir + '/' + args.param_name+'.txt'
if fn_path is not None:
    with open(fn_path, mode='w', encoding='utf-8') as text_file:
        text_file.write(text)

model = BertNerTagger(args)
if args.pretrained_checkpoint:
    model.load_state_dict(
        torch.load(
            args.pretrained_checkpoint,
            map_location=torch.device('cuda'), 
        )["state_dict"]
    )

trainer = get_trainer(args)

trainer.fit(model)
torch.save(model.model.end_outputs.weight.data, './weight.pt')

out = trainer.test(model)
write_eval_performance(out, os.path.join(trainer.checkpoint_callback.dirpath, 'eval_results.tsv'))
write_for_leaderboard(out, "results.out")
