# PCFG-based Natural Language Interface Improves Generalization for Controlled Text Generation

This repo contains the code for paper [PCFG-based Natural Language Interface Improves Generalization for Controlled Text Generation](https://arxiv.org/abs/2210.07431) by Jingyu Zhang, James R. Glass, and Tianxing He

## Downloading data
The processed data can be downloaded [here](https://drive.google.com/file/d/1xRHVkWHHZqyFL0MYR7jv-HyvJFR9cw06/view?usp=sharing). Please extract the zip to root folder of this repo, with the extracted content in `data/*`.

## Naming
The internal names for models is a bit different from names shown in paper. Here are their correspondence:

NL models
- PrefixLM-NL: concat
- FUDGE-NL: alignment

Non-NL models
- PrefixLM: concatoracle
- FUDGE: oracle

## Running experiments

### Training (full data, sec 4.4)
Run scripts `train_alignment.sh`, `train_concat.sh`, and `train_oracle.sh` with corresponding parameters. See comments in each script for explanation of parameters.

### Generate ouput text
Run scripts `generate_alignment.sh`, `generate_concat.sh`, and `generate_oracle.sh` with corresponding parameters. See comments in each script for explanation of parameters.

### Evaluate
First, download classifier models (which evaluates control accuracy) [here](https://drive.google.com/file/d/1B_ERMeNbYckvQ7QpZHOZR0dBcinzESt2/view?usp=sharing). Extract the contents to `evaluation/ckpt/*`.

Navigate to `evaluation` directory, run `eval.sh` with corresponding parameters.

### Generalizing to Unseen Commands (sec 4.5)
Run train scripts with basic templates, i.e. feed in `USE_BASIC_TEMPLATE=1` for 20 template, and `DOUBLE_TEMPLATE=1` for 40 templates. At generation time, use the unseen test template by `USE_TEST_TEMPLATE=1`.

### Generalizing to Unseen Attributes (sec 4.6)
Run `train_label_zeroshot.sh` with corresponding parameters. Note that for each model, we need to train $n$ times for a dataset with $n$ labels, where each split has one class data removed to be the zero-shot class. This is done by specifying the `ZS_CLASS` argument.

### Generalizing to Unseen Attribute Combinations (sec 4.7)
Run `train_comp.sh` with corresponding parameters. Similarly, we need to run multiple splits for different non-compositional class (more explanations in the paper). This is achieved by specifying the `NONCOMP_LABEL_CLASS` argument.

## Cite
```
@article{Zhang2022PCFGbasedNL,
  title={PCFG-based Natural Language Interface Improves Generalization for Controlled Text Generation},
  author={Jingyu Zhang and James R. Glass and Tianxing He},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.07431}
}
```