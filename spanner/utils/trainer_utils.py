import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger


def load_2idx(args):
    # begin{add label2indx augument into the args.}
    label2idx = {}
    if 'nl4opt' in args.dataname: 
        label2idx = {'LIMIT':1, 'CONST_DIR':2, 'VAR':3, 'PARAM':4, 'OBJ_NAME':5, 'OBJ_DIR':6, 'O':0}
    elif 'conll' in args.dataname:
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3, "MISC": 4}
    elif 'note' in args.dataname:
        label2idx = {'O': 0, 'PERSON': 1, 'ORG': 2, 'GPE': 3, 'DATE': 4, 'NORP': 5, 'CARDINAL': 6, 'TIME': 7,
                        'LOC': 8,
                        'FAC': 9, 'PRODUCT': 10, 'WORK_OF_ART': 11, 'MONEY': 12, 'ORDINAL': 13, 'QUANTITY': 14,
                        'EVENT': 15,
                        'PERCENT': 16, 'LAW': 17, 'LANGUAGE': 18}
    elif args.dataname == 'wnut16':
        label2idx = {'O': 0, 'loc':1, 'facility':2,'movie':3,'company':4,'product':5,'person':6,'other':7,
                        'tvshow':8,'musicartist':9,'sportsteam':10}
    elif args.dataname == 'wnut17':
        label2idx = {'O': 0,'location':1, 'group':2,'corporation':3,'person':4,'creative-work':5,'product':6}

    label2idx_list = []
    for lab, idx in label2idx.items():
        pair = (lab, idx)
        label2idx_list.append(pair)
    args.label2idx_list = label2idx_list
    # end{add label2indx augument into the args.}

    # begin{add case2idx augument into the args.}
    morph2idx_list = []
    morph2idx = {'isupper': 1, 'islower': 2, 'istitle': 3, 'isdigit': 4, 'other': 5}
    for morph, idx in morph2idx.items():
        pair = (morph, idx)
        morph2idx_list.append(pair)
    args.morph2idx_list = morph2idx_list
    # end{add case2idx augument into the args.}
    return args


def get_trainer(args, mode='train'):
    if mode != 'train':
        return Trainer.from_argparse_args(args)
    # save the best model
    out_dir = args.default_root_dir
    if not os.path.exists(out_dir + '/lightning_logs/'):
        os.makedirs(out_dir + '/lightning_logs/')
    try:
        ver_n = int(os.listdir(out_dir + '/lightning_logs/')[-1].split('_')[-1]) + 1
    except IndexError:
        ver_n = 0
    bcp_path  = out_dir + '/lightning_logs/version_' +  str(ver_n) + '/checkpoints/'
    checkpoint_callback = ModelCheckpoint(
        dirpath=bcp_path,
        filename='{epoch:02d}-{val_loss:.4f}-{val_micro@F1:.4f}', 
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=args.es_threshold,
        patience=args.patience,
        verbose=True,
        mode='min', 
    )
    logger = CSVLogger(args.default_root_dir, name="lightning_logs")
    trainer = Trainer.from_argparse_args(
        args, 
        checkpoint_callback=True, 
        callbacks=[checkpoint_callback, early_stopping_callback], 
        logger=logger, 
    )
    return trainer