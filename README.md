# SciQAG

SciQAG is a novel framework for automatically generating high-quality science question-answer pairs from a large corpus of scientific literature using large language models (LLMs). SciQAG consists of a QA generator and a QA evaluator, which work together to extract diverse and research-level questions and answers from scientific papers. 

## Overview

Utilizing this framework, we have constructed a large-scale, high-quality, open-ended science QA dataset containing 188,042 QA pairs extracted from 22,743 scientific papers across 24 scientific domains. We also introduce SciQAG-24D, a new benchmark task designed to evaluate the science question-answering ability of LLMs. 

Extensive experiments demonstrate that fine-tuning LLMs on the SciQAG dataset significantly improves their performance on both open-ended question answering and scientific tasks. To foster research and collaboration, we make the datasets, models, and evaluation codes publicly available, contributing to the advancement of science question answering and the development of more interpretable and reasoning-capable AI systems.


### Fine Tune
We follows the same training method as [Longlora](https://github.com/dvlab-research/LongLoRA).

#### Installation
```
git clone https://github.com/dvlab-research/LongLoRA.git
cd LongLoRA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

#### Instruction Fine Tune
```
torchrun --nproc_per_node=8 supervised-fine-tune.py  \
        --model_name_or_path vicuna-7b-v1.5-16k \
        --bf16 True \
        --output_dir <output_dir>       \
        --model_max_length 12288 \
        --use_flash_attn True \
        --data_path <ft_data_path> \
        --low_rank_training False \
        --num_train_epochs 15  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 500     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "ds_configs/stage3.json" \
        --tf32 False
```
Training data is a json file containing a list of dict with format of `{'instruction': insruction, 'input': paper+query, 'output': Q&As}`.
In this training script, data would be further processed to the following format:
```
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>> 

{instruction}
{input} [/INST]
```

Deepspeed stage 3 parameters
```
{
  "bf16": {
    "enabled": "auto"
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": "auto",
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 5,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

### Inference
All inference code is in `Inference.ipynb` file.
- Format of input and output file:
    - `inference input`: json file with keys [source, doi, abstract, keywords, txt, title]. Among them, [doi, txt, keywords] are MUST required.
    -  `original inference output`: json file with keys [doi, input, output]. `input` is the complete format for inferencing (containing both instruction & input, please reference to the data format in training script).
    -  `processed inference output`: json file with keys [source, doi, abstract, keywords, txt, title, num_Q&A, Q&A]


All parameters can be changed in the notebook through the folloowing part:
```
args_dict = {'file_path': '<json file to inference>', 
             'base_model': '<ft model>', 
             'cache_dir': './cache', 
             'context_size': 12888, 
             'flash_attn': True, 
             'temperature': 0.6, 
             'top_p': 0.9, 
             'max_gen_len': 12888}
```
- Note that `seq_len` is to set the sequence length for evaluation. `context_size` is to set the context length of the model during fine-tuning. `seq_len` should not be larger than `context_size`.

### Visualization
```
| - visualization
    | - q similarity.ipynb    # evaluate similarity scores for each two questions
    | - sentence coverage rate.ipynb    # evaluate generated QAs' coverage on paper
```

## Citation
If you use SciQAG in your research, please cite the following paper:
```
@article{wan2024sciqag,
  title={SciQAG: A Framework for Auto-Generated Scientific Question Answering Dataset with Fine-grained Evaluation},
  author={Wan, Yuwei and Ajith, Aswathy and Liu, Yixuan and Lu, Ke and Grazian, Clara and Hoex, Bram and Zhang, Wenjie and Kit, Chunyu and Xie, Tong and Foster, Ian},
  journal={arXiv preprint arXiv:2405.09939},
  year={2024}
}
```

## Contributing
We welcome contributions to the SciQAG project. Please contact us directly for more information.
