import re
from sentence_transformers import SentenceTransformer
import torch
from itertools import permutations
from .gpt_utils import *


templates = [
    ('symptom', 'Symptoms include ([<{\[].+[>}\]])'),
    ('test', 'Tests include ([<{\[].+[>}\]])'),
    ('symptom-disease', 'Symptoms ([<{\[].+[>}\]]);* *caused by Disease ([<{\[].+[>}\]])'),
    ('medicine-disease', 'Medicine ([<{\[].+[>}\]]);* to treat ([<{\[].+[>}\]])'),
    ('procedure-disease', 'Procedure ([<{\[].+[>}\]]) to treat ([<{\[].+[>}\]])'),
    ('treatment-result', '([<{\[].+[>}\]]) +Results: *([<{\[].+[>}\]])'),
    ('test-result', 'Test results of ([<{\[].+[>}\]]): ([<{\[].+[>}\]])'),
    ('complication-disease', 'Complication ([<{\[].+[>}\]]) caused by ([<{\[].+[>}\]])'),
    ('test-result', 'Test results of ([<{\[].+[>}\]]): ([<{\[].+[>}\]])'),
    ('test-goal', 'Test goal of ([<{\[].+[>}\]]): ([<{\[].+[>}\]])'),
    ('test-implication', 'Test implications of ([<{\[].+[>}\]]): ([<{\[].+[>}\]])'),
    ('emergency-action', 'If encounter ([<{\[].+[>}\]]) do ([<{\[].+[>}\]])'),
    ('none', 'No template for')
]

cate_pairs_accepted = {
    'Medical Issues - Symptom / Medical Issues - Disease',
    'Medical Issues - Complication developed in the hospital / Medical Issues - Disease',
    'Treatments - Medicine name / Medical Issues - Disease',
    'Treatments - Procedure name / Medical Issues - Disease',
    'Treatments - Medicine name / Treatments - Result of the treatment',
    'Treatments - Procedure name / Treatments - Result of the treatment',
    'Treatments - Medicine name / Treatments - Treatments - Treatment goal',
    'Treatments - Procedure name / Treatments - Treatments - Treatment goal',
    'Tests - Test name / Tests - Test result',
    'Tests - Test name / Tests - Test goal',
    'Tests - Test name / Tests - Test implication'
}

cate2brief = {
    'Medical Issues - Symptom': 'sosy',
    'Medical Issues - Disease': 'dsyn',
    'Medical Issues - Complication developed in the hospital': 'cpct',
    'Treatments - Medicine name': 'clnd',
    'Treatments - Procedure name': 'diap',
    'Treatments - Result of the treatment': 'tmrs',
    'Treatments - Treatments - Treatment goal': 'tmgl',
    'Tests - Test name': 'lbpr',
    'Tests - Test result': 'lbtr',
    'Tests - Test goal': 'lbtg',
    'Tests - Test implication': 'lbti',
}

focused_cates = {
    'symptom-disease',  # dsyn-sosy
    'medicine-disease',  # clnd-dsyn
    'procedure-disease',  #
    'treatment-result',  # clnd-tmrs / diap-tmrs
    'complication-disease',  # cpct-dsyn
    'test-result',  #  lbpr-lbtr
    'test-goal',  # diap-dsyn
    'test-implication'
}

cate_map = {
    'symptom': {'Medical Issues - Symptom'},
    'disease': {'Medical Issues - Disease'},
    'procedure': {'Treatments - Procedure name'},
    'medicine': {'Treatments - Medicine name'},
    'treatment': {'Treatments - Medicine name', 'Treatments - Procedure name'},
    'test': {'Tests - Test name'},
    'result': {'Tests - Test result', 'Treatments - Result of the treatment'},
    'goal': {'Tests - Test goal', 'Treatments - Treatments - Treatment goal'},
    'implication': {'Tests - Test implication'},
    'complication': {'Medical Issues - Complication developed in the hospital'},
}


model = SentenceTransformer('all-MiniLM-L6-v2')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)


def clean_events(events_str):
    def remove_revision(event):
        if event.startswith('<') and event.endswith('>') and \
                event.find('{') != -1 and event.find('}') != -1:
            par_start = event.find('{')
            par_end = event.find('}')
            event = event[:par_start] + event[par_end + 1:]
        return event
    def remove_brace(event):
        event = event.replace('<', '')
        event = event.replace('>', '')
        event = event.replace('[', '')
        event = event.replace(']', '')
        event = event.replace('{', '')
        event = event.replace('}', '')
        event = event.replace('CHANGE HERE', '')
        return event
    events_str = remove_revision(events_str)
    pat_event = re.compile('<[^>]+>')
    events = pat_event.findall(events_str)
    events = [remove_brace(event) for event in events]
    return events


def get_event_pairs_accepted(events_tgt):
    event_pairs = list(permutations(events_tgt, 2))
    event_pairs_accepted = []
    for event_pair in event_pairs:
        cate_pair = event_pair[0][2] + ' / ' + event_pair[1][2]
        if cate_pair in cate_pairs_accepted:
            event_pairs_accepted.append(event_pair)
    return event_pairs_accepted


def expand_rels(rels_idx):
    rels_expand = []
    for rel in rels_idx:
        cate_pair = rel[0]
        if cate_pair in focused_cates:
            events_head_src = clean_events(rel[1][0])
            events_tail_src = clean_events(rel[1][1])
            for event_head_src in events_head_src:
                for event_tail_src in events_tail_src:
                    rels_expand.append((cate_pair, event_head_src, event_tail_src))
    return rels_expand


def get_event_sim(event_a, event_b):
    embs = model.encode([event_a, event_b])
    sim = embs[0].dot(embs[1])
    return sim


def identify_relation(event_pairs_src, event_pair_tgt):
    event_head_tgt, event_tail_tgt = event_pair_tgt
    for event_pair_src in event_pairs_src:
        cate_pair, event_head_src, event_tail_src = event_pair_src
        cate_head, cate_tail = cate_pair.split('-')
        if event_head_tgt[2] in cate_map[cate_head] and event_tail_tgt[2] in cate_map[cate_tail]:
            txt_sim_head = get_event_sim(event_head_src, event_head_tgt[1])
            txt_sim_tail = get_event_sim(event_tail_src, event_tail_tgt[1])
            if txt_sim_head > 0.9 and txt_sim_tail > 0.9:
                return 1, event_pair_src
    return 0, None


def add_event2text(txt, event1, event2):
    _, ent1, cate1, start1, end1 = event1
    _, ent2, cate2, start2, end2 = event2
    start1, start2, end1, end2 = start1-1, start2-1, end1-1, end2-1
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


def remove_duplicate_triggers(triggers):
    tids = set()
    triggers_ = []
    for trigger in triggers:
        if trigger[0] not in tids:
            triggers_.append(trigger)
            tids.add(trigger[0])
        else:
            continue
    return triggers_


