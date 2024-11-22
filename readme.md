# ICT 
## Dependecencies
- Image datasets for Pope
- Pope questions and answers
## workflow
### Get intervention vectors
```bash
python get_base_vector.py --model-path path/to/llava-v1.5 --question-file path/to/pope/question-file --image-folder path/to/your/coco/images --seed ${1:-55} --length 1500 --output ./base
```
```bash
python get_hallucinated_vector.py --model-path path/to/llava-v1.5 --question-file path/to/pope/question-file --image-folder path/to/your/coco/images --seed ${1:-55} --length 1500 --output ./hallucinated
```
```bash
python get_object_vector.py --model-path path/to/llava-v1.5 --question-file path/to/pope/question-file --image-folder path/to/your/coco/images --seed ${1:-55} --length 1500 --output ./object
``` 
### Inference using ICT
```bash
python val_ict_pope.py --question_file path/to/pope/question-file --num_heads 256 --alpha 8 --seed ${1:-55} --length 1500 --target_dataset coco --type both  
``` 
### Valid your result by Pope
```bash
python eval_pope.py --gt_files path/to/groundtruth/pope/answers --gen_files answer.jsonl 
