import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gt_files", type=str, default="../POPE/coco/coco_pope_random.json")
parser.add_argument("--gen_files", type=str, default="./ICT/answer.jsonl")
args = parser.parse_args()


gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]


gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]


true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gt_files)
yes_answers = 0

for index, line in enumerate(gt_files):
    idx = line["question_id"]
    gt_answer = line["label"]
    try:
        gen_answer = gen_files[index]["text"]
        gt_answer = gt_answer.lower()
        gen_answer = gen_answer.lower()
        gt_answer = gt_answer.strip()
        gen_answer = gen_answer.strip()
    except:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / total_questions
        yes_proportion = yes_answers / total_questions
        unknown_prop = unknown / total_questions
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
        print(f'Accuracy: {accuracy}')
        print(f'yes: {yes_proportion}')
        print(f'unknow: {unknown_prop}')
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print(f'Accuracy: {accuracy}')
print(f'yes: {yes_proportion}')
print(f'unknow: {unknown_prop}')

