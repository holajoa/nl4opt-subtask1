import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import parse_args, get_reader, load_model, get_out_filename, get_tagset, get_trainer

if __name__ == '__main__':
    timestamp = time.time()
    sg = parse_args()
    out_dir_path = sg.out_dir + '/' + sg.model_name

    # load the dataset first
    test_data = get_reader(
        file_path=sg.test, 
        target_vocab=get_tagset(sg.iob_tagging), 
        max_instances=sg.max_instances, 
        max_length=sg.max_length, 
        encoder_model=sg.encoder_model, 
    )

    model, model_file = load_model(
        sg.model, 
        tag_to_id=get_tagset(sg.iob_tagging), 
    )
    model = model.to(sg.cuda)
    # use pytorch lightnings saver here.
    eval_file = get_out_filename(sg.out_dir, model_file, prefix=sg.prefix, output_tags=True)

    out_str = ''
    test_dataloaders = DataLoader(
        test_data, batch_size=sg.batch_size, 
        collate_fn=model.collate_batch, 
        shuffle=False, 
        drop_last=False, 
    )
    index = 0
    for batch in tqdm(test_dataloaders, total=len(test_dataloaders)):
        token_ids, pred_tags, tags = model.predict_tags(batch, device=sg.cuda, return_tokens_and_true_tags=True)
        # tokens = [test_data.tokenizer.convert_ids_to_tokens(token_ids_sen, skip_special_tokens=True) for token_ids_sen in token_ids]
        for token_inst, pred_tag_inst, true_tag_inst in zip(test_data.sentences, pred_tags, tags):
            triples = zip(token_inst, pred_tag_inst, true_tag_inst)
            tag_inst = ['\t'.join(tri) for tri in triples]
            out_str += '\n'.join(tag_inst)
            out_str += '\n\t\n\t\n'
        index += 1

    open(eval_file, 'wt', encoding='utf-8').write(out_str)

