import json
import csv
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_file", type=str, default="", help="Path to the file NOTEEVENTS.csv")
    parser.add_argument("--anno_dir", type=str, default="data/annotated_dataset/annotated_files/", help="Path to the annotated_files directory")
    parser.add_argument("--output_dir", type=str, default="data/annotated_dataset/raw_notes/", help="Path to the output directory")
    args = parser.parse_args()
    return args


def extract_di(txt):
    start = txt.find("Discharge Instructions:")
    end = txt.find("Followup Instructions:")
    di = txt[start: end].replace('\n',  ' ')
    di = ' '.join(di.split())
    return di


def quality_check(txt):
    num_words = len(txt.split(' '))
    if num_words < 80:
        return False
    if num_words > 350:
        return False
    if 'admitted to' not in txt.lower():
        return False
    return True


def collect_annotated_ids(anno_dir):
    ids = [fname[:-8] for fname in os.listdir(anno_dir) if fname.endswith("csv")]
    return set(ids)


if __name__ == "__main__":
    """
    STEP 0: Obtain args
    """
    args = parse_args()
    """
    STEP 1: Extract discharge summaries
    """
    if not os.path.exists(args.anno_dir):
        os.mkdir(args.anno_dir)
    ids = collect_annotated_ids(args.anno_dir)
    instructions = []
    with open(args.input_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            row_id, subject_id, hadm_id = row[0], row[1], row[2]
            cate, text = row[6], row[10]
            if cate != 'Discharge summary':
                continue
            instruction = extract_di(text)
            if not quality_check(instruction):
                continue
            idx = '%s-%s-%s' % (row_id, subject_id, hadm_id)
            if idx in ids:
                with open(args.raw_notes + idx + '.txt', 'w') as f:
                    f.write(instruction)

