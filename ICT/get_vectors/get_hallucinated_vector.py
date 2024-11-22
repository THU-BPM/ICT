import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageFilter
import numpy as np

from transformers import set_seed
from transformers import AutoProcessor, LlavaForConditionalGeneration

def eval_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model.to(device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    all_layer_wise_activations = []
    all_head_wise_activations = []
    print(questions[0])
    for line in tqdm(questions[:args.length]):
        image_file = line["image"]
        qs = line["text"]
        gt_answer = line["label"]
        if gt_answer == "yes":
            wrong_ans = "no"
        else:
            wrong_ans = "yes"
        if "object" in line:
            object = line["object"]
            prompt = "USER: <image>\n" + qs + " ASSISTANT: "  + object
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
            blurred_overlay = Image.blend(image, blurred_image, alpha=1)
            inputs = processor(text=prompt, images=blurred_overlay, return_tensors="pt").to(device)
        else:
            if gt_answer == "yes":
                prompt = "USER: <image>\n" + qs + " ASSISTANT: "  + gt_answer
                image = Image.open(os.path.join(args.image_folder, image_file))
                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
                blurred_overlay = Image.blend(image, blurred_image, alpha=1)
                inputs = processor(text=prompt, images=blurred_overlay, return_tensors="pt").to(device)
            else:
                prompt = "USER: <image>\n" + qs + " ASSISTANT: "  + wrong_ans
                image = Image.open(os.path.join(args.image_folder, image_file))
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        HEADS = [f"language_model.model.layers.{i}.self_attn.o_proj" for i in range(32)]

        outputs_dict = {}

        def hook_fn(module, input, output):
            if module not in outputs_dict:
                outputs_dict[module] = output.cpu()
        layer_names = HEADS
        layers = []
        for name in layer_names:
            module = dict([*model.named_modules()]).get(name)
            if module:
                layers.append(module)
            else:
                print(f"Module not found: {name}")
        hook_handles = [layer.register_forward_hook(hook_fn) for layer in layers]
        with torch.no_grad():
            output = model(
                **inputs,
                output_hidden_states = True
            )
        for handle in hook_handles:
            handle.remove()

        attention_output = tuple(outputs_dict.values())
        attention_output = torch.stack(attention_output, dim = 0).detach().cpu().squeeze().numpy()
        
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        layer_wise_activations = hidden_states
           
        head_wise_activations= attention_output
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations[:,-1,:].copy())

    np.save(args.output, all_head_wise_activations)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
