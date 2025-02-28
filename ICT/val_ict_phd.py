import torch
import json
from einops import rearrange
import numpy as np
from functools import partial
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from baukit import TraceDict


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head
def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads
def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions): 

    interventions = {}
    for layer, head in top_heads: 
        interventions[f"language_model.model.layers.{layer}.self_attn.o_proj"] = []
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
        interventions[f"language_model.model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze(), proj_val_std))
    for layer, head in top_heads: 
        interventions[f"language_model.model.layers.{layer}.self_attn.o_proj"] = sorted(interventions[f"language_model.model.layers.{layer}.self_attn.o_proj"], key = lambda x: x[0])

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
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=128, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=0, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument("--temp", type=float, default=0, help="temperature")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
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
    os.makedirs(args.output_dir,exist_ok=True)
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if True:    
        num_layers = 32
        num_heads = args.heads
        com_directions = []
        # load activations 
        if len(args.subfix) > 0:
            file_path = f"{args.length}_{args.source_split}_{args.source_dataset}_{args.subfix}.npy"
        else:
            file_path = f"{args.length}_{args.source_split}_{args.source_dataset}.npy"

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
            with open(f'{args.output_dir}/llava15_{args.source_dataset}_to_{args.target_dataset}_pope_{args.source_split}_to_{args.target_split}_answers_{args.alpha}_{args.num_heads}_{args.temp}_{args.subfix}_{args.length}_acc.txt', 'w') as file:
                for i in range(accuracies.shape[0]):
                    for j in range(accuracies.shape[1]):
                        file.write(f'Layer {i}, Head {j}: {accuracies[i, j]}\n')

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


            answers_file = f'./attribute_direct_base_{args.alpha}_{args.num_heads}.jsonl'
            answers_file = os.path.expanduser(answers_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            ans_file = open(answers_file, "w")
            model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")  
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision="a272c74") 
            data = json.load(open('data.json', encoding='utf-8'))
            phd_base = [{'task': sample['task'],'image_id': sample['image_id'], 'yes_question': sample['yes_question'], 
                        'no_question': sample['no_question']} for sample in data if 'ccs_description' not in sample]
            for sample in tqdm(phd_base):
                image_id = sample["image_id"]
                image_folder="your_img_path"
                yes_question = sample["yes_question"]  
                no_question = sample["no_question"]  
                image_path = os.path.join(image_folder, f"COCO_val2014_{image_id}.jpg")
                image_path = os.path.join(image_folder, f"COCO_val2014_{image_id}.jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join("/path/to/coco/train2014", f"COCO_train2014_{image_id}.jpg")
                image = Image.open(image_path)
                if sample["task"] != "attribute":
                    continue 
                prompt = "USER: <image>\n"+yes_question+"Please answer this question in one word. ASSISTANT:"
                                
                def id(head_output, layer_name): 
                    return head_output

                intervention_fn=lt_modulated_vector_add
                if interventions == {}: 
                    intervene = id
                    layers_to_intervene = []
                else: 
                    intervene = partial(intervention_fn, start_edit_location='lt')
                    layers_to_intervene = list(set(interventions.keys()).union(set(interventions_object.keys())))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
                with TraceDict(model,layers_to_intervene, edit_output=intervene) as ret:
                    output = model.generate(
                        **inputs,
                        do_sample = False,
                        num_beams= 1,
                        max_new_tokens=2,
                        use_cache=True,
                        output_hidden_states = True
                    )
                outputs=processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spacces=False)[0]
                if prompt in outputs:
                    outputs = outputs.replace(prompt, "").strip()
                outputs = outputs.strip()

                ans_file.write(json.dumps({
                                        "prompt": yes_question,
                                        "text": outputs,
                                        "model_id": "llava1.5",
                                        "image": image_id,
                                        "metadata": {}}) + "\n")
                ans_file.flush()
                prompt = "USER: <image>\n"+no_question+"Please answer this question in one word. ASSISTANT:"
                with TraceDict(model,layers_to_intervene, edit_output=intervene) as ret:
                    output = model.generate(
                        **inputs,
                        do_sample = False,
                        num_beams= 1,
                        max_new_tokens=2,
                        use_cache=True,
                        output_hidden_states = True
                    )
                outputs=processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spacces=False)[0]
                if prompt in outputs:
                    outputs = outputs.replace(prompt, "").strip()
                outputs = outputs.strip()

                ans_file.write(json.dumps({
                                        "prompt": no_question,
                                        "text": outputs,
                                        "model_id": "llava1.5",
                                        "image": image_id,
                                        "metadata": {}}) + "\n")
                ans_file.flush()
            ans_file.close()
if __name__ == "__main__":
    main()








