# encoding: utf-8


import os
import torch
from utils.trainer_utils import get_trainer, load_2idx

# from trainer_spanPred import BertLabeling # old evaluation version
# from trainer_spanPred_newEval import BertLabeling # new evaluation version
from trainer import BertNerTagger

from pytorch_lightning import Trainer

def evaluate(args, ckpt, hparams_file):
	trainer = get_trainer(args, mode='test')
	model = BertNerTagger(args=args)
	model = model.load_from_checkpoint(
		checkpoint_path=ckpt,
		hparams_file=hparams_file,
		map_location=torch.device('cuda'),
		batch_size=args.batch_size,
		data_dir=args.data_dir,
	)
	out = trainer.test(
		model=model, 
		test_dataloaders=model.get_dataloader(
			data_dir=args.data_dir, prefix='test', 
		), 
	)
	return out

def write_for_leaderboard(out, out_filename):
    ''' write the micro averaged F1 score to results.out '''
    open(out_filename, 'wt').write(str(out[0]["micro@F1"]))

def write_eval_performance(eval_performance, out_file):
    outstr = ''
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ['results', 'predictions']:
                continue
            outstr = outstr + '{}\t{}\n'.format(k, out_[k])
            added_keys.add(k)

    open(out_file, 'wt').write(outstr)
    # logger.info('Finished writing evaluation performance for {}'.format(out_file))

if __name__ == '__main__':
	parser = BertNerTagger.get_parser()
	parser = Trainer.add_argparse_args(parser)
	args = parser.parse_args()
	args = load_2idx(args)
	ckpt = args.pretrained_checkpoint
	current_model_ver_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.pretrained_checkpoint)))
	hparams_file = os.path.join(current_model_ver_dir, 'hparams.yaml')
	out = evaluate(args=args, ckpt=ckpt, hparams_file=hparams_file)

	eval_file = os.path.join(os.path.dirname(args.default_root_dir), args.modelName, 'test.tsv')
	write_eval_performance(out, eval_file)

    # write the micro averaged F1 score to results.out
	write_for_leaderboard(out, "results.out")
