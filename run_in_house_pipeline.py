import argparse
import json
from itertools import permutations
import torch.cuda
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

cate_pairs_accepted = {
    'Symptom / Disease': 'Symptom %s is caused by %s',  # Disease
    'Complication / Disease': 'Complication %s is caused by %s',  #
    'Medicine name / Disease': 'Medicine %s is used to treat %s',  #
    'Procedure name / Disease': 'Procedure %s is used to treat %s',  #
    'Medicine name / Result of the treatment': 'Treatment result of medicine %s is %s',  #
    'Procedure name / Result of the treatment': 'Treatment result of procedure %s is %s',  #
    'Medicine name / Treatment goal': 'The goal of using medicine %s is to treat %s',  #
    'Procedure name / Treatment goal': 'The goal of using procedure %s is to treat %s',  #
    'Test name / Test result': "The result of test %s is %s",  #
    'Test name / Test goal': "The goal of test %s is %s",  #
    'Test name / Test implication': "The implication of test %s is %s"  #
}
cate2brief = {
    'Symptom': 'sosy',
    'Disease': 'dsyn',
    'Complication developed in the hospital': 'cpct',
    'Complication': 'cpct',
    'Medicine name': 'clnd',
    'Procedure name': 'diap',
    'Result of the treatment': 'tmrs',
    'Treatment goal': 'tmgl',
    'Test name': 'lbpr',
    'Test result': 'lbtr',
    'Test goal': 'lbtg',
    'Test implication': 'lbti',
}

prompt_cloze = [
    {"role": "system", "content": "You are a physician assistant that help patient review his discharge instructions."},
    {"role": "user", "content": "\"%s\" Generate a simple question targeting at the blank in the above sentence."}
]






def load_rel_cls_model(model_path, device):
    config = AutoConfig.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # Add tokens
    added_tokens = [
        '<sosy>', '<dsyn>', '<cpct>', '<clnd>', '<diap>', '<tmrs>',
        '<tmgl>', '<lbpr>', '<lbtr>', '<lbtg>', '<lbti>',
        '</sosy>', '</dsyn>', '</cpct>', '</clnd>', '</diap>', '</tmrs>',
        '</tmgl>', '</lbpr>', '</lbtr>', '</lbtg>', '</lbti>'
    ]
    tokenizer.add_tokens(added_tokens)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        from_tf=False,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return tokenizer, model


def read_in_text(args):
    with open(args.txt_path) as f:
        txt = f.read()
    return txt


# TODO: Zonghai generate NER CSV file from txt
def run_ner(txt, fname, args):
    """
    Run NER for the given text, save the results to output_path
    The output file name should be fname.ner.json
    :param txt:
    :return: events in the form of
    [
        (idx, event_str, event_type, start, end),
        ... ...
    ]
    """
    events = []
    """
    Your code goes here
    """
    # Save results
    with open(args.output_dir + fname.split('.')[0] + '.ner.json', 'w') as f:
        json.dump(events, f)
    return events


def get_event_pairs_accepted(events):
    event_pairs = list(permutations(events, 2))
    event_pairs_accepted = []
    for event_pair in event_pairs:
        cate_pair = event_pair[0][2] + ' / ' + event_pair[1][2]
        if cate_pair in cate_pairs_accepted:
            event_pairs_accepted.append(event_pair)
    return event_pairs_accepted


def add_event2text(txt, event1, event2):
    _, ent1, cate1, start1, end1 = event1
    _, ent2, cate2, start2, end2 = event2
    start1, start2, end1, end2 = start1, start2, end1-1, end2-1
    if start1 < start2:
        start_a, start_b, end_a, end_b = start1, start2, end1, end2
        cate_a, cate_b = cate2brief[cate1], cate2brief[cate2]
    else:
        start_a, start_b, end_a, end_b = start2, start1, end2, end1
        cate_a, cate_b = cate2brief[cate2], cate2brief[cate1]
    txt_ = txt[:start_a] + ' <%s> ' % cate_a + txt[start_a: end_a] + ' </%s> ' % cate_a + \
            txt[end_a: start_b] + ' <%s> ' % cate_b + txt[start_b: end_b] + ' </%s> ' % cate_b + \
            txt[end_b:]
    return ' '.join(txt_.split())


def run_rcls(txt, events, fname, args):
    # Initialize model
    tokenizer, model_rel = load_rel_cls_model(args.model_rel_path, device)
    outputs = []
    event_pairs = get_event_pairs_accepted(events)
    for event1, event2 in event_pairs:
        txt_ = add_event2text(txt, event1, event2)
        input_ids = tokenizer(txt_, return_tensors='pt', max_length=512, truncation=True).to(model_rel.device)
        with torch.no_grad():
            logits = model_rel(**input_ids).logits.cpu()
        pred = torch.argmax(logits[0]).item()
        if pred == 1:
            outputs.append((event1, event2))
    # Save results
    with open(args.output_dir + fname.split('.')[0] + '.rel.json', 'w') as f:
        json.dump(outputs, f)
    return outputs


def post_process(rels, fname, args):
    outputs = []
    for event1, event2 in rels:
        _, event1_str, event1_cate, _, _ = event1
        _, event2_str, event2_cate, _, _ = event2
        event_cate_pair = '%s / %s' % (event1_cate, event2_cate)
        assert event_cate_pair in cate_pairs_accepted
        outputs.append(cate_pairs_accepted[event_cate_pair] % (event1_str, event2_str))
    # Save results
    with open(args.output_dir + fname.split('.')[0] + '.out.json', 'w') as f:
        json.dump(outputs, f)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Generating questions for a clinical note")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="A folder to save the ner, relation classification and questions",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="The clinical note, a txt file",
    )
    parser.add_argument(
        "--model_ner_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model_rel_path",
        type=str,
        default="",
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Read in text
    txt = read_in_text(args.input_path)
    fname = args.input_path.split('/')[-1]
    # NER
    evts = run_ner(txt, fname, args)
    # Relation Classification
    rels = run_rcls(txt, evts, fname, args)
    # Output Results
    rels_str = post_process(rels, fname, args)


