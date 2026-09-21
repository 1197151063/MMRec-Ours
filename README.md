# MMRec

<div align="center">
  <a href="https://github.com/enoche/MultimodalRecSys"><img width="300px" height="auto" src="https://github.com/enoche/MMRec/blob/master/images/logo.png"></a>
</div>


$\text{MMRec}$: A modern <ins>M</ins>ulti<ins>M</ins>odal <ins>Rec</ins>ommendation toolbox that simplifies your research [arXiv](https://arxiv.org/abs/2302.03497).  
:point_right: Check our [comprehensive survey on MMRec, arXiv](https://arxiv.org/abs/2302.04473).   
:point_right: Check the awesome [multimodal recommendation resources](https://github.com/enoche/MultimodalRecSys).  

## Toolbox
<p>
<img src="./images/MMRec.png" width="500">
</p>

## Supported Models
source code at: `src\models`

| **Model**       | **Paper**                                                                                             | **Conference/Journal** | **Code**    |
|------------------|--------------------------------------------------------------------------------------------------------|------------------------|-------------|
| **General models**  |                                                                                                        |                        |             |
| SelfCF              | [SelfCF: A Simple Framework for Self-supervised Collaborative Filtering](https://arxiv.org/abs/2107.03019)                                 | ACM TORS'23            | selfcfed_lgn.py  |
| LayerGCN            | [Layer-refined Graph Convolutional Networks for Recommendation](https://arxiv.org/abs/2207.11088)                                          | ICDE'23                | layergcn.py  |
| **Multimodal models**  |                                                                                                        |                        |             |
| VBPR              | [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1510.01784)                                              | AAAI'16                 | vbpr.py      |
| MMGCN             | [MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video](https://staff.ustc.edu.cn/~hexn/papers/mm19-MMGCN.pdf)               | MM'19                  | mmgcn.py  |
| ItemKNNCBF             | [Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches](https://arxiv.org/abs/1907.06902)               | RecSys'19              | itemknncbf.py  |
| GRCN              | [Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback](https://arxiv.org/abs/2111.02036)            | MM'20                  | grcn.py    |
| MVGAE             | [Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems](https://ieeexplore.ieee.org/abstract/document/9535249)              | TMM'21                 | mvgae.py   |
| DualGNN           | [DualGNN: Dual Graph Neural Network for Multimedia Recommendation](https://ieeexplore.ieee.org/abstract/document/9662655)                   | TMM'21                 | dualgnn.py   |
| LATTICE           | [Mining Latent Structures for Multimedia Recommendation](https://arxiv.org/abs/2104.09036)                                               | MM'21                  | lattice.py  |
| SLMRec            | [Self-supervised Learning for Multimedia Recommendation](https://ieeexplore.ieee.org/document/9811387) | TMM'22                 |                  slmrec.py |
| **Newly added**  |                                                                                                        |                        |             |
| BM3         | [Bootstrap Latent Representations for Multi-modal Recommendation](https://dl.acm.org/doi/10.1145/3543507.3583251)                                          | WWW'23                 | bm3.py |
| FREEDOM | [A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation](https://arxiv.org/abs/2211.06924)                                 | MM'23                  | freedom.py  |
| MGCN     | [Multi-View Graph Convolutional Network for Multimedia Recommendation](https://arxiv.org/abs/2308.03588)                       | MM'23               | mgcn.py          |
| DRAGON  | [Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation](https://arxiv.org/abs/2301.12097)                                 | ECAI'23                | dragon.py  |
| MG  | [Mirror Gradient: Towards Robust Multimodal Recommender Systems via Exploring Flat Local Minima](https://arxiv.org/abs/2402.11262)                                 | WWW'24                | common/trainer.py  |
| LGMRec  | [LGMRec: Local and Global Graph Learning for Multimodal Recommendation](https://arxiv.org/abs/2312.16400)                                 | AAAI'24                | lgmrec.py |
| DA-MRS  | [Improving Multi-modal Recommender Systems by Denoising and Aligning Multi-modal Content and User Feedback](https://dl.acm.org/doi/10.1145/3637528.3671703)                   | KDD'24                | damrs.py |
| SMORE   | [Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation](https://arxiv.org/abs/2412.14978) | WSDM'25           | smore.py  |
| PGL | [Mind Individual Information! Principal Graph Learning for Multimedia Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33429) | AAAI'25 | pgl.py |


#### Please consider to cite our paper if this framework helps you, thanks:
```
@inproceedings{zhou2023bootstrap,
author = {Zhou, Xin and Zhou, Hongyu and Liu, Yong and Zeng, Zhiwei and Miao, Chunyan and Wang, Pengwei and You, Yuan and Jiang, Feijun},
title = {Bootstrap Latent Representations for Multi-Modal Recommendation},
booktitle = {Proceedings of the ACM Web Conference 2023},
pages = {845–854},
year = {2023}
}

@article{zhou2023comprehensive,
      title={A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions}, 
      author={Hongyu Zhou and Xin Zhou and Zhiwei Zeng and Lingzi Zhang and Zhiqi Shen},
      year={2023},
      journal={arXiv preprint arXiv:2302.04473},
}

@inproceedings{zhou2023mmrec,
  title={Mmrec: Simplifying multimodal recommendation},
  author={Zhou, Xin},
  booktitle={Proceedings of the 5th ACM International Conference on Multimedia in Asia Workshops},
  pages={1--2},
  year={2023}
}
```

## SIMMRec: frozen content with an item ID residual

This experimental baseline removes both positional encodings and all graph propagation.
It freezes and normalizes pretrained image/text features, uses a shared linear projector
per modality, and scores normalized users against normalized fused item representations:

`item = normalize(alpha * normalize(text_projection) + (1-alpha) * normalize(image_projection) + item_id_weight * item_residual)`

The item residual is a regularized, learnable ID lookup, not a numeric encoding of the ID.
This model is **not item-ID-free**. Setting `item_id_weight=0` removes the residual table.
Training uses sampled softmax (positive included in the denominator), temperature scaling,
and negatives drawn with replacement from the catalog excluding **training** positives only.
Evaluation uses the same cosine score. No performance gains are claimed before real-data runs.

### Server commands

Use your existing MMRec Python environment. From the repository root:

```bash
git pull --ff-only
cd src
python main.py --model SIMMRec --dataset baby --data-path /absolute/path/to/data
# Content-only ablation, same other settings:
python main.py --model SIMMRec --dataset baby --data-path /absolute/path/to/data --item-id-weight 0
# Optional 18-configuration validation search:
python main.py --model SIMMRec --dataset baby --data-path /absolute/path/to/data --config configs/simmrec-search.yaml
```

The data root must contain `baby/baby.inter`, `baby/image_feat.npy`, and
`baby/text_feat.npy`; feature rows must follow the interaction file's item mapping.
Use `--dataset sports` or `--dataset clothing` for the other configured datasets.
Without `--data-path`, the existing overall configuration's data path is retained.
`--gpu-id 0` selects the GPU; `--epochs 2` can be used for a short smoke run.

Defaults: dimension 64, text weight 0.5, ID residual weight 0.1, temperature 0.1,
128 negatives, and sampled embedding L2 coefficient 0.0001. These are starting settings,
not validated best parameters. The regularizer is the mean squared L2 norm of sampled
user/residual vectors. Training projects only unique sampled items; evaluation caches
catalog representations until the next training/evaluation phase.

Results and full configurations appear in `src/log/` (the existing logger), epoch metrics
in `src/csv/`, and optionally best-validation checkpoints in `src/saved/` with `--save-model`. By default,
no model files are written; best-validation metrics are still tracked in memory. Checkpoints include
model state, configuration, epoch, and validation/test metrics. Hyperparameter selection
uses validation scores and all configured combinations run. Test scores remain logged by
the existing trainer; do not use them to choose configurations. Multiple seeds are run as
separate grid entries: report all seeds, not the best seed. Training reconstruction Recall
is disabled for SIMMRec; its CSV entry and unavailable ID/MM diagnostic losses are `nan`.

The model has no numeric-ID branch. For a relabeling check, remap interactions, feature
rows, and initialized entity parameter rows consistently; independent retraining may vary
with initialization/sampling and should be compared across seeds.

### Checks

```bash
python -m unittest discover -s tests -v
```

Tests cover exclusion of training positives, frozen feature gradients, trainable residuals,
cosine scoring, the content-only ablation, and scores under consistent entity relabeling.

## Baby diagnostic experiments (after the September 16 logs)

From `src/`, run all 13 configurations sequentially:

```bash
bash run_baby_diagnostics.sh /root/autodl-tmp/MMRec-Ours/data 0
```

1. **LightMRecNoPE (1 run):** reconstructs the original supplied LightMRec without either
   positional encoding. Retains trainable raw features, three linear layers per modality,
   BatchNorm and dropout 0.1, Xavier user initialization, 32 unfiltered negatives,
   temperature 0.04, no explicit regularizer, and raw-dot evaluation after cosine training.
   The decoupled loss uses algebraically equivalent logsumexp for numerical stability.
   This intentionally preserves the original scoring mismatch and false-negative sampling
   for diagnosis. It is not the clean baseline or a deployable recommendation.
   Original alpha/LR were not supplied: defaults 0.5/0.001 must be matched to the original
   run before calling the comparison strictly controlled. Other training settings inherit
   the current overall config. No original PE scores are claimed reproduced.
2. **SIMMRec content (9 runs):** alpha in {0, 0.5, 1}, temperature in {0.05, 0.1, 0.2},
   no item residual, 128 negatives, frozen features. Alpha=0 is visual-only; alpha=1 text-only.
3. **SIMMRec ID (3 runs):** temperature in {0.05, 0.1, 0.2}, pure normalized user/item ID
   embeddings with the same sampled softmax, positive exclusion, negative count, optimizer,
   dimension and sampled L2 regularizer as SIMMRec. No modality files are loaded in this mode.
   `item_id_weight` and `alpha` have no effect in ID mode.

Individual runs can use `--epochs 2` for a smoke check. All configurations use the current
seed (999); use multiple seeds for final comparisons. Compare best-validation-selected test
results. Equal seeds do not guarantee equal random streams across different architectures.

### Checkpoint saving

Model checkpoint writing is disabled by default, including all diagnostic script runs.
Logs and epoch metric CSVs are retained, and early stopping / validation selection are unchanged.
Add `--save-model` to an individual `main.py` command only when weights are needed.
Existing saved files are not deleted. Disabling saving reduces I/O, not forward/backward compute.

## Explicit original-order audit

```bash
cd src
bash run_order_audit.sh /root/autodl-tmp/MMRec-Ours/data 0
```

Runs a fresh no-PE control and 12 experiments (4 coordinate sources x user-only,
item-only, both). All use **LightMRecOrder**, the same original LightMRec reconstruction:
trainable content, MLP/BatchNorm/dropout, decoupled cosine loss and raw-dot evaluation.
Only assigned PE coordinates and their enabled side change. No weights are saved by default.

- `original`: existing numeric IDs, not a newly reconstructed full-data first-seen order.
- `train_file`: first appearance in the training subset of the current `.inter` file,
  before loader shuffling. Validation/test rows are not consulted. Unseen entities are
  appended using a fixed independent random permutation of remaining catalog IDs.
- `random`: independent random bijections for users and items.
- `train_shuffled`: shuffle only the training rows for coordinate construction, then
  first-seen indexing. Actual optimizer training data/order is not changed.

**Interpretation boundary:** the repository's preprocessing may already have sorted the
`.inter` file by user ID. Filtering that file cannot recover raw input order. `train_file`
is therefore a training-file-order control, not proof that all original order information
or leakage has been eliminated. IDs and training row order may inherit full-data provenance.
To isolate held-out influence on raw first-seen numbering, the original indexed row-order
reference and mapping are needed. Do not infer leakage solely from differences in this grid.

PE coordinates are assigned to the original lookup rows: interactions, content alignment,
trainable parameter row initialization, and sampled negatives retain the same IDs. A local
NumPy generator (`order_seed=2026`) avoids perturbing training RNG. Mapping fingerprints and
unseen counts are logged. Initial defaults alpha=0.5/LR=0.001 still require matching to the
original PE run before claiming a controlled reproduction of its reported scores.

First check `original/both` against the historical PE result and `none` against LightMRecNoPE.
If historical performance is not reproduced, resolve configuration/data/code differences
before explaining other grid results. Compare validation-selected test scores; repeat
promising/contradictory findings across training seeds and independent order seeds.

## Overnight experiments

The [overnight experiment guide](experiments/README.md) provides a 50-configuration,
12-hypothesis queue with time budgets, isolated logs, failure handling, resume support,
and machine-readable validation-selected results. No model files are saved.
