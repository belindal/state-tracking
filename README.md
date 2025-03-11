# (How) Do Language Models Track State

The following repository contains code for the paper ["(How) Do Language Models Track State"](https://arxiv.org/abs/2503.02854)


## Setup
1. First, create and prepare your environment
```bash
conda create -n lm_state_track PYTHON=3.12
conda activate lm_state_track
pip install -r requirements.txt
```

2. Install a version of pytorch compatible with your CUDA version: https://pytorch.org/get-started/locally/ 


## Create S3 and S5 data
To generate S3 and S5 data, use
```bash
python permutation_task.py \
--num_items [3|5] \
--data_dir <data_dir> \
--num_stories <num_stories_to_generate> \
--train_ratio <percent_stories_for_train> \
--story_length <length_of_stories>
```
- `--num_items`: Number of items in the permutation task (3 for S3, 5 for S5)
- `--data_dir`: Directory to save the generated data
- `--num_stories`: Number of stories to generate
- `--train_ratio`: Fraction of stories to use for training (0-1)
- `--story_length`: Length of each story in tokens

## Train
To train a model on the permutation task, use the training script:
```bash
bash bash_scripts/train.sh <model_name> <num_items> <data_dir> <output_dir> [options]
```

Required arguments:
- `<model_name>`: Model architecture to use (e.g., gpt2, EleutherAI/pythia-70M)
- `<num_items>`: Number of items in the permutation task (3 or 5)
- `<data_dir>`: Directory containing training data
- `<output_dir>`: Directory to save model checkpoints

Optional arguments:
- `--use_bfloat16`: Use bfloat16 precision for training
- `--save_all_checkpoints <interval>`: Save checkpoints at specified interval (steps)
- `--early_stopping`: Enable early stopping based on validation loss
- `--no_pretrain`: Train model from scratch (don't use pre-trained weights)
- `--from_checkpoint <path>`: Resume training from existing checkpoint
- `--max_steps <steps>`: Maximum number of training steps
- `--data_determinism`: Use deterministic data loading
- `--full_determinism`: Enable full determinism for reproducibility
- `--supervision_type <type>`: Type of supervision (default: direct_state)
- `--batch_size <size>`: Batch size for training (default: 256)
- `--max_len <length>`: Maximum sequence length (default: 512)
- `--seed <seed>`: Random seed (default: 42)

This script will save model checkpoints to `<output_dir>/checkpoint-<num_steps>/`. In subsequent sections, this should be fed in as `checkpoint_dir`.

## Test
To generate plots of generalization accuracy across sequence lengths (section 4.4), run:
```bash
bash bash_scripts/eval.sh <checkpoint_dir> <num_items> <data_dir>
```

You can also run the evaluation with additional options:
```bash
python eval.py \
  --checkpoint_dir <checkpoint_dir> \
  --data_dir <data_dir> \
  --num_items <num_items> \
  --max_len <max_len> \
  --num_stories <num_stories> \
  --use_bfloat16
```

Arguments:
- `--checkpoint_dir`: Directory containing model checkpoint
- `--data_dir`: Directory containing test data
- `--num_items`: Number of items in the permutation task (3 or 5)
- `--max_len`: Maximum sequence length to evaluate (default: 200)
- `--num_stories`: Number of stories to evaluate (default: 10000)
- `--use_bfloat16`: Use bfloat16 precision
- `--random_baseline`: Evaluate random baseline instead of model
- `--skip_layers`: Comma-separated list of layers to skip during evaluation

This script will generate a generalization curve to `figures/gen_accuracy/<checkpoint_dir>.png`


## Analysis
The remaining three of the analyses corresponding to section 4 of the paper (activation patching, probing, attention patterns) can be found run using the interpretation scripts provided under `bash_scripts/`.

### Activation Patching
To run activation patching experiments:
```bash
bash bash_scripts/run_intervention.sh \
  --model_type <model_type> \
  --checkpoint_dir <checkpoint_dir> \
  --num_items <num_items> \
  --n_prompts <n_prompts> \
  --n_tokens <n_tokens> \
  --intervene_output_type <state|state_by_parity> \
  --patching_mode <deletion|substitution> \
  --token_increments <token_increments>
```

Arguments:
- `--model_type`: Type of model (pythia, gpt2, llama)
- `--checkpoint_dir`: Directory containing model checkpoint
- `--num_items`: Number of items in the permutation task (3 or 5)
- `--n_prompts`: Number of prompts to use for intervention (default: 1)
- `--n_tokens`: Number of tokens per prompt (default: 100)
- `--intervene_output_type`: Type of output to analyze (state or state_by_parity)
- `--patching_mode`: Mode for activation patching (deletion or substitution)
- `--token_increments`: Token increment size for intervention analysis (default: 1)

This script will generate an activation patching heatmap to `figures/intervene/<checkpoint_dir>.png`

### Probing
To run probing experiments:
```bash
bash bash_scripts/run_probe.sh \
  --model_type <model_type> \
  --checkpoint_dir <checkpoint_dir> \
  --num_items <num_items> \
  --n_prompts <n_prompts> \
  --n_tokens <n_tokens>
```

Arguments:
- `--model_type`: Type of model (pythia, gpt2, llama)
- `--checkpoint_dir`: Directory containing model checkpoint
- `--num_items`: Number of items in the permutation task (3 or 5)
- `--n_prompts`: Number of prompts to use for probing (default: 1000)
- `--n_tokens`: Number of tokens per prompt (default: 100)

This script will generate probe accuracy graphs to `figures/probes/<checkpoint_dir>.png`

### Lengthwise Probing
To run lengthwise probing experiments:
```bash
bash bash_scripts/run_lengthwise_probe.sh \
  --model_type <model_type> \
  --checkpoint_dir <checkpoint_dir> \
  --num_items <num_items> \
  --n_prompts <n_prompts> \
  --n_tokens <n_tokens> \
  --token_increments <token_increments> \
  --start_idx <start_idx> \
  --end_idx <end_idx>
```

Arguments:
- `--model_type`: Type of model (pythia, gpt2, llama)
- `--checkpoint_dir`: Directory containing model checkpoint
- `--num_items`: Number of items in the permutation task (3 or 5)
- `--n_prompts`: Number of prompts to use for probing (default: 1000)
- `--n_tokens`: Number of tokens per prompt (default: 100)
- `--token_increments`: Token increment size for lengthwise probing (default: 1)
- `--start_idx`: Start index for lengthwise probe (default: 5)
- `--end_idx`: End index for lengthwise probe (optional)

This script will generate lengthwise probe accuracy heatmaps to `figures/lengthwise_probe/<checkpoint_dir>_<n_tokens>.png`


## Create synthetic pretraining data with topic model
To generate topic model data, run:
```bash
python make_topic_training_data.py \
  --data_dir <output_dir> \
  --num_topics <num_topics> \
  --num_stories <num_stories> \
  --num_items <num_items> \
  --train_split_ratio <train_ratio>
```

Arguments:
- `--data_dir`: Directory to save the generated data
- `--num_topics`: Number of topics in the topic model (default: 4)
- `--num_stories`: Number of stories to generate (default: 1000000)
- `--num_items`: Number of items in the permutation task (default: 3)
- `--train_split_ratio`: Fraction of stories to use for training (default: 0.999)
- `--splits`: Comma-separated list of data splits to create (default: "train,test")
- `--init_from_json`: Path to JSON file with initial topic distributions (optional)

To replicate setting in the paper:
```bash
python make_topic_training_data.py \
  --data_dir S3_NTP_data \
  --num_topics 4 \
  --num_stories 1000000 \
  --num_items 3 \
  --train_split_ratio 0.999
```

To train the LM with next-token-prediction on the generated data, use the same training script, but set `--supervision_type next_token`:
```bash
bash bash_scripts/train.sh <model_name> <num_items> <topic_data_dir>/train <output_dir> --supervision_type next_token
```

For example:
```bash
bash bash_scripts/train.sh EleutherAI/pythia-160M 3 S3_NTP_data/train checkpoints/pythia_S3_NTP --supervision_type next_token
```

