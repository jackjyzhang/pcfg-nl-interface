# PCFG-based Natural Language Interface Improves Generalization for Controlled Text Generation

This repo contains the code for paper [PCFG-based Natural Language Interface Improves Generalization for Controlled Text Generation](https://arxiv.org/abs/2210.07431) by Jingyu Zhang, James R. Glass, and Tianxing He

## Downloading data
The processed data can be downloaded [here](https://drive.google.com/file/d/1xRHVkWHHZqyFL0MYR7jv-HyvJFR9cw06/view?usp=sharing). Please extract the zip to root folder of this repo, with the extracted content in `data/*`.

## PCFG templates
They can be found in `./templates`. Detailed descriptions of the templates can be found in Appendix C of paper.

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

For example,

```bash
./train_alignment.sh alignment agnews 0.00005 10 1 0 0 0
```

trains an alignment (FUDGE-NL) model on AG News with lr=0.00005 for 10 epochs, considering both the label and length attribute. Please see hyperparameter settings in the appendix of paper.

### Generate ouput text
Run scripts `generate_alignment.sh`, `generate_concat.sh`, and `generate_oracle.sh` with corresponding parameters. See comments in each script for explanation of parameters.

For example,

```bash
./generate_alignm,ent.sh 20 14 1000 alignment ckpt/alignment_agnews_lim256_len_bte agnews 1 0
```

load the checkpoint `ckpt/alignment_agnews_lim256_len_bte` and generate 1000 examples with top-$k$ decoding, $k$=20, and FUDGE strength hyperparameter $\lambda$=14.

### Evaluate
First, download classifier models (which evaluates control accuracy) [here](https://drive.google.com/file/d/1B_ERMeNbYckvQ7QpZHOZR0dBcinzESt2/view?usp=sharing). Extract the contents to `evaluation/ckpt/*`.

Navigate to `evaluation` directory, run `eval.sh` with corresponding parameters. For example, `./eval.sh generation_n1000 agnews` evaluate all generation files in directory `generation_n1000` on AG News.

### Generalizing to Unseen Commands (sec 4.5)
Run train scripts with basic templates, i.e. feed in `USE_BASIC_TEMPLATE=1` for 20 template, and `DOUBLE_TEMPLATE=1` for 40 templates. At generation time, use the unseen test template by `USE_TEST_TEMPLATE=1`.

For example,
```bash
./train_alignment.sh alignment AG News 0.00005 10 1 0 1 0
```

trains alignment (FUDGE-NL) on agnews just like the first example above, but without PCFG, and does not use double template (thus, it's trained on 20 fixed templates).

### Generalizing to Unseen Attributes (sec 4.6)
Run `train_label_zeroshot.sh` with corresponding parameters. Note that for each model, we need to train $n$ times for a dataset with $n$ labels, where each split has one class data removed to be the zero-shot class. This is done by specifying the `ZS_CLASS` argument.

For example,

```bash
for zs_class in 0 1 2 3
do
  ./train_label_zero_shot.sh $zs_class alignment agnews 0 0.00005 10
done
```

trains alignment (FUDGE-NL) on AG News with 4 different `zs_class` (zero-shot class). At inference, the evaluation results should be averaged across different `zs_class` splits.

### Generalizing to Unseen Attribute Combinations (sec 4.7)
Run `train_comp.sh` with corresponding parameters. Similarly, we need to run multiple splits for different non-compositional class (more explanations in the paper). This is achieved by specifying the `NONCOMP_LABEL_CLASS` argument.

For example,

```bash
for noncomp_class in 0 1 2 3
do
  ./train_label_zero_shot.sh $noncomp_class alignment agnews 0.00005 10
done
```

trains alignment (FUDGE-NL) on AG News with 4 different `noncomp_class` (non-compositional class). At inference, the evaluation results should be averaged across different `noncomp_class` splits.

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