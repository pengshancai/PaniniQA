import sys
import os
from collections import defaultdict
from tqdm import tqdm
import jsonlines
from data_utils import *
import argparse
import json
import re


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--anno_dir", type=str, default="data/annotated_dataset/annotated_files/", help="Path to the annotated_files directory")
    parser.add_argument("--note_dir", type=str, default="data/annotated_dataset/raw_notes/", help="Path to the raw notes directory")
    parser.add_argument("--split_file", type=str, default="data/annotated_dataset/split.json", help="Path to the train/valid/test split file 'split.json'")
    parser.add_argument("--output_dir", type=str, default="data/rel_cls/", help="Path to the output directory")
    args = parser.parse_args()
    return args


def get_src_tgt(anno_dir, note_dir):
    def get_relations(con):
        secs = con.split('\n\n')
        tgt = []
        for sec in secs:
            if sec.startswith('No template'):
                continue
            lines = sec.strip().split('\n')
            cate = lines[0]
            for line in lines[1:]:
                tgt.append((cate, line.replace(' >', '>')))
        return tgt
    idx2src_tgt = {}
    fnames = os.listdir(anno_dir)
    for fname in fnames:
        if fname.endswith('_rel.txt'):
            try:
                idx = fname[:-8]
                with open(anno_dir + fname) as f:
                    con = f.read()
                    tgt = get_relations(con)
                with open(note_dir + '%s.txt' % idx) as f:
                    src = f.read().strip()
                idx2src_tgt[idx] = {
                    'source': src,
                    'target': tgt
                }
            except:
                continue
    return idx2src_tgt


def get_all_evts(anno_dir):
    idx2evts = {}
    fnames = os.listdir(anno_dir)
    for fname in fnames:
        if fname.endswith('_evt.csv') and len(fname) > 8:
            idx = fname[:-8]
            evts = []
            with open(anno_dir + fname, mode='r') as file:
                lines = file.readlines()[1:]
            for line in lines:
                try:
                    i, event, cate, start, end = line.strip().split('|')
                    event, cate = event.strip(), cate.strip()
                    start, end = int(start), int(end)
                    evts.append((i, event, cate, start, end))
                except:
                    continue
            idx2evts[idx] = evts
    return idx2evts


def get_all_rels(src_tgt_pairs):
    rels = defaultdict(list)
    ids = list(src_tgt_pairs.keys())
    ids.sort()
    for idx in ids:
        src_tgt_pair = src_tgt_pairs[idx]
        targets = src_tgt_pair['target']
        for target in targets:
            for template in templates:
                se = re.compile(template[1]).search(target[1])
                if se:
                    rels[idx].append((template[0], se.groups()))
                    break
    return rels


def create_dataset(split_name, output_dir):
    if split_name == 'train':
        # Due to data scarcity, we use the collection of train & validation set for training the model
        # after deciding on the hyper-parameters
        ids_split = ids['ids_train'] + ids['ids_valid']
    else:
        ids_split = ids['ids_%s' % split_name]
    progress = tqdm(range(len(ids_split)), desc='Creating %s Set' % split_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with jsonlines.open(output_dir + '%s.json' % split_name, 'w') as f:
        for idx in ids_split:
            if idx not in evts:
                continue
            try:
                event_pairs_tgt = get_event_pairs_accepted(evts[idx])
                rels_idx = rels[idx]
                txt = src_tgt_pairs[idx]['source']
                event_pairs_src = expand_rels(rels_idx)
                for event_pair_tgt in event_pairs_tgt:
                    label, evidence = identify_relation(event_pairs_src, event_pair_tgt)
                    txt_ = add_event2text(txt, event_pair_tgt[0], event_pair_tgt[1])
                    _ = f.write({'sentence': txt_, 'label': label})
            except:
                print('Ignore IDX %s' % idx)
            _ = progress.update(1)


if __name__ == "__main__":
    args = parse_args()
    src_tgt_pairs = get_src_tgt(args.anno_dir, args.note_dir)
    evts = get_all_evts(args.anno_dir)
    rels = get_all_rels(src_tgt_pairs)
    with open(args.split_file) as f:
        ids = json.load(f)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for split in ['train', 'valid', 'test']:
        create_dataset(split, args.output_dir)

