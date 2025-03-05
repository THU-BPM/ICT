# ICT 
<a href="https://arxiv.org/abs/2411.15268" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2411.15268-b31b1b.svg?style=flat" /></a>
    
Official repo for paper ICT: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models accepted by CVPR 2025

## **Dependencies**  
- **Image Datasets**: Required image datasets for the **Pope** benchmark.  
- **Pope Question-Answer Pairs**: Ensure you have the necessary question and answer files for **Pope**.  
- set up by runnning 
```bash
conda env create -f environment.yml
conda activate ict
```

## **Workflow**  

### **Step 1: Generate Intervention Vectors**  
Run the following scripts to generate different types of intervention vectors using your model and dataset.  

#### **Base Vectors**  
```bash
python get_base_vector.py --model-path path/to/llava-v1.5 \
                          --question-file path/to/pope/question-file \
                          --image-folder path/to/your/coco/images \
                          --seed ${1:-55} --length 1500 \
                          --output ./base
```
#### **Hallucinated Vectors**
```bash                            
python get_hallucinated_vector.py --model-path path/to/llava-v1.5 \
                                  --question-file path/to/pope/question-file \
                                  --image-folder path/to/your/coco/images \
                                  --seed ${1:-55} --length 1500 \
                                  --output ./hallucinated
```

#### **Object Vectors**     
```bash                         
python get_object_vector.py --model-path path/to/llava-v1.5 \
                            --question-file path/to/pope/question-file \
                            --image-folder path/to/your/coco/images \
                            --seed ${1:-55} --length 1500 \
                            --output ./object
```
### **Step 2: Run ICT Evaluation on Pope**
#### **Perform inference using ICT on the Pope dataset.**

```bash
python val_ict_pope.py --question_file path/to/pope/question-file \
                        --num_heads 256 --alpha 8 --seed ${1:-55} \
                        --length 1500 --target_dataset coco \
                        --type both
```
### **Step 3: Validate Results with Pope**
Evaluate the generated answers against ground truth annotations.

```bash
python eval_pope.py --gt_files path/to/groundtruth/pope/answers \
                    --gen_files answer.jsonl
```
## **Running ICT on Other Benchmarks**
MMMU Benchmark
- To evaluate ICT on the MMMU benchmark, first clone the MMMU repository:

```bash
git clone https://github.com/MMMU-Benchmark/MMMU.git
```
- Then, place the necessary files in the /MMMU directory and run:

```bash

python val_ict_MMMU.py
```
- MMMU Evaluation for 13B Models
```bash

python val_ict_13b_MMMU.py
```
PhD Benchmark Evaluation
- To run ICT on the PhD benchmark, execute:

```bash
python val_ict_phd.py
```
