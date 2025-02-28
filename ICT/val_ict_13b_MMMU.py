import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'MMMU/mmmu/utils')))
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import random
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets

from argparse import ArgumentParser
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils import llava_image_processor
from utils.eval_utils import parse_multi_choice_response
from einops import rearrange
import numpy as np
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold



from concurrent.futures import ThreadPoolExecutor, as_completed

from baukit import  TraceDict
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head
def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads
def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions): 

    interventions = {}
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        if use_center_of_mass: 
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = np.random.normal(size=(128,))
        else: 
            direction = probes[(layer, head)].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:,layer,head,:] 
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze(), proj_val_std))
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(interventions[f"model.layers.{layer}.self_attn.o_proj"], key = lambda x: x[0])

    return interventions
def train_probe(layer, head, X, X_labels, kf):
    X_layer = X[:, layer, head, :]
    X_layer = np.array(X_layer)

    fold_accuracies = []
    for train_index, test_index in kf.split(X_layer):
        X_train, X_test = X_layer[train_index], X_layer[test_index]
        y_train, y_test = X_labels[train_index], X_labels[test_index]
        
        probe = LogisticRegression(solver='saga', max_iter=1000, n_jobs=32)
        probe.fit(X_train, y_train)
        
        fold_accuracies.append(probe.score(X_test, y_test))

    mean_accuracy = np.mean(fold_accuracies)
    return (layer, head, mean_accuracy, probe)
def run_model(args, samples, model, call_model_engine_fn,tokenizer,  processor):
    out_samples = dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    if True:
        num_layers = 40
        num_heads = 40
        com_directions = []

        head_wise_activations_1 = np.load(f"./get_vectors/base.npy",allow_pickle=True)
        head_wise_activations_1 = rearrange(head_wise_activations_1, 'b l (h d) -> b l h d', h = num_heads) 
        head_wise_activations_2 = np.load(f"./get_vectors/hallucinated.npy",allow_pickle=True)
        head_wise_activations_2 = rearrange(head_wise_activations_2, 'b l (h d) -> b l h d', h = num_heads)
        head_wise_activations_3 = np.load(f"./get_vectors/object.npy",allow_pickle=True)
        head_wise_activations_3 = rearrange(head_wise_activations_3, 'b l (h d) -> b l h d', h = num_heads)
        head_wise_activations = np.concatenate((head_wise_activations_1, head_wise_activations_2), axis=0) 
        head_wise_activations_object = np.concatenate((head_wise_activations_1, head_wise_activations_3), axis=0)

        for layer in range(num_layers): 
            for head in range(num_heads): 
                true_mass_mean = np.mean(head_wise_activations_1[:,layer,head,:], axis=0)
                false_mass_mean = np.mean(head_wise_activations_2[:,layer,head,:], axis=0)
                com_directions.append(true_mass_mean - false_mass_mean)
        com_directions_object = []

        for layer in range(num_layers): 
            for head in range(num_heads): 
                true_mass_mean = np.mean(head_wise_activations_1[:,layer,head,:], axis=0)
                false_mass_mean = np.mean(head_wise_activations_3[:,layer,head,:], axis=0)
                com_directions_object.append(true_mass_mean - false_mass_mean)

    
        labels = np.zeros(args.length * 2)
        labels[args.length:] = 1
        indices = np.arange(args.length * 2)
        np.random.shuffle(indices)
        head_wise_activations = head_wise_activations[indices]
        head_wise_activations_object = head_wise_activations_object[indices]

        labels = labels[indices]

        X = head_wise_activations
        X_labels= labels
        results = []
        if(True): 
            probes = {}
            accuracies = np.empty((num_layers, num_heads), dtype=float)
            k = 2  
            kf = KFold(n_splits=k)
            with ThreadPoolExecutor(max_workers=64) as executor:  # Change to ProcessPoolExecutor for multiprocessing
                futures = []
                
                # Submit tasks to the executor
                for layer in range(num_layers):
                    for head in range(num_heads):
                        futures.append(executor.submit(train_probe, layer, head, X, X_labels, kf))

                # Collect the results as they complete
                for future in tqdm(as_completed(futures), total=len(futures)):
                    layer, head, mean_accuracy, probe = future.result()
                    probes[(layer, head)] = probe
                    accuracies[layer, head] = mean_accuracy

            top_accs = np.argsort(accuracies.reshape(num_heads*num_layers))[::-1][:args.num_heads]
            top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]

            interventions = get_interventions_dict(top_heads,  probes,X, num_heads, True, False, com_directions)

            probes_object = {}
            accuracies = np.empty((num_layers, num_heads), dtype=float)
            k = 2
            kf = KFold(n_splits=k)
            X_object = head_wise_activations_object
            X_labels= labels

            with ThreadPoolExecutor(max_workers=64) as executor:  # Change to ProcessPoolExecutor for multiprocessing
                futures = []
                X_layer = X_object[:, layer, head, :]
                X_layer = np.array(X_layer)
                # Submit tasks to the executor
                for layer in range(num_layers):
                    for head in range(num_heads):
                        futures.append(executor.submit(train_probe, layer, head, X_object, X_labels, kf))

                # Collect the results as they complete
                for future in tqdm(as_completed(futures), total=len(futures)):
                    layer, head, mean_accuracy, probe = future.result()
                    probes_object[(layer, head)] = probe
                    accuracies[layer, head] = mean_accuracy

            top_accs_object = np.argsort(accuracies.reshape(num_heads*num_layers))[::-1][:args.num_heads]
            top_heads_object = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs_object]
            interventions_object = get_interventions_dict(top_heads_object,  probes_object, X_object, num_heads, True, False, com_directions_object)
            heads = {}
            heads_object = {}
            for layer in interventions:
                for head, direction, proj_val_std in interventions[layer]:
                    if layer not in heads:
                        heads[layer] = []
                    heads[layer].append(head)
            for layer in interventions_object:
                for head, direction, proj_val_std in interventions_object[layer]:
                    if layer not in heads_object:
                        heads_object[layer] = []
                    heads_object[layer].append(head)
            def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'): 
                assert layer_name in interventions or layer_name in interventions_object, "layer_name not found"
                head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)

                if args.type == "image" or args.type == "both":
                    if layer_name in interventions:
                        for head, direction, proj_val_std in interventions[layer_name]:
                            direction_to_add = torch.tensor(direction).to(head_output.device.index)
                            if layer_name in interventions_object and head in heads_object[layer_name]:
                                if start_edit_location == 'lt': 
                                    head_output[:, -1, head, :] += 0.5 * args.alpha * proj_val_std * direction_to_add
                                else: 
                                    head_output[:, start_edit_location:, head, :] += 0.5 * args.alpha * proj_val_std * direction_to_add
                            else:
                                if start_edit_location == 'lt': 
                                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                                else: 
                                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
                if args.type == "object" or args.type == "both":
                    if layer_name in interventions_object:
                        for head, direction, proj_val_std in interventions_object[layer_name]:
                            direction_to_add = torch.tensor(direction).to(head_output.device.index)
                            if layer_name in interventions and head in heads[layer_name]:
                                if start_edit_location == 'lt': 
                                    head_output[:, -1, head, :] += 0.5 * args.alpha * proj_val_std * direction_to_add
                                else: 
                                    head_output[:, start_edit_location:, head, :] += 0.5 * args.alpha * proj_val_std * direction_to_add
                            else:
                                if start_edit_location == 'lt': 
                                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                                else: 
                                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
                head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                return head_output
            if True:
                def id(head_output, layer_name): 
                    return head_output

                intervention_fn=lt_modulated_vector_add
                if interventions == {}: 
                    intervene = id
                    layers_to_intervene = []
                else: 
                    intervene = partial(intervention_fn, start_edit_location='lt')
                    layers_to_intervene = list(set(interventions.keys()).union(set(interventions_object.keys())))
            for sample in tqdm(samples):

                prompt = sample['final_input_prompt']
                prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
                conv = conv_templates['vicuna_v1'].copy()
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                image = sample['image']
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                with TraceDict(model,layers_to_intervene, edit_output=intervene) as ret:
                    output_ids = model.generate(
                                                                
                                input_ids,
                                images=image.unsqueeze(0).half().cuda(),
                                do_sample=False,
                                num_beams=1,
                                max_new_tokens=128,
                                use_cache=True)
                                        
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                if sample['question_type'] == 'multiple-choice':
                    pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
                else:  # open question
                    pred_ans = response
                out_samples[sample['id']] = pred_ans
        return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_val_ict.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=128, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=0, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument("--temp", type=float, default=0, help="temperature")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--length', type=int, default=1500, help='length')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--source_dataset', type=str, default='coco', help='source_dataset')
    parser.add_argument('--target_dataset', type=str, default='coco', help='target_dataset')
    parser.add_argument('--source_split', type=str, default='random', help='source_split')
    parser.add_argument('--target_split', type=str, default='random', help='target_split')
    parser.add_argument('--type', type=str, default="both", choices=["both","image","object"])
    parser.add_argument('--subfix', type=str, default="")
    parser.add_argument('--output_dir', type=str,default="./output")
    args = parser.parse_args()
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None,
                                                                model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    processor = None
    call_model_engine = None
    vis_process_func = llava_image_processor
    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset("/path/to/MMMU", subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        if sample['image']:
            sample['image'] = vis_process_func(sample['image'], vis_processors).to(device)
        samples.append(sample)
    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer,processor)

    save_json(args.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()

