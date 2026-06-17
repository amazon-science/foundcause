# FoundCause

Pretrained foundation model for causal discovery. Load a CSV of observational
data and get back a predicted directed acyclic graph and a hidden-confounder
matrix in a single forward pass.

## Contents

- `foundcause.py` — model code (required for loading the checkpoint)
- `predict.py` — CLI inference script
- `checkpoint.pt` — pretrained weights (~1.6 GB, ~139M parameters), distributed
  as a [GitHub Release asset](https://github.com/amazon-science/foundcause/releases/latest)
  rather than in the git repository (it exceeds GitHub's 100 MB file limit)

## Installation

```bash
pip install torch numpy pandas networkx scikit-learn scipy schedulefree
```

### Download the pretrained weights

The `checkpoint.pt` weights are not part of the git clone. Download them from
the [latest release](https://github.com/amazon-science/foundcause/releases/latest)
into the repository directory:

```bash
# Direct download (curl)
curl -L -o checkpoint.pt \
  https://github.com/amazon-science/foundcause/releases/latest/download/checkpoint.pt

# ...or with the GitHub CLI
gh release download --repo amazon-science/foundcause --pattern checkpoint.pt
```

## Usage

```bash
python predict.py --data path/to/your_data.csv
```

The CSV should have one row per observation and one column per variable. A
header row is optional. Three files are written alongside the input:

- `<data>_dag.csv` — binary adjacency (`1` means row `i` causes column `j`)
- `<data>_probs.csv` — edge probabilities in `[0, 1]`
- `<data>_confounders.csv` — symmetric hidden-confounder scores

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoint.pt` | Path to the pretrained weights |
| `--device` | `cuda` if available | Use `cpu` to force CPU |
| `--output-dir` | same as input | Where to write the output CSVs |
| `--n-runs` | `10` | Permutation-averaged inference passes |
| `--temperature` | `0.65` | Logit scaling |
| `--max-samples` | `5000` | Subsample larger datasets to this many rows |
| `--threshold` | adaptive GMM | Fix an edge threshold in `[0, 1]` |
| `--enforce-dag` | off | Post-process for acyclicity |

## Limitations

- Trained on 2 to 50 variables; larger graphs work but degrade monotonically.
- Trained on 100 to 600 samples per dataset; very small datasets are unreliable.
- Observational data only; pass control-only data if your dataset has interventions.
- Outputs soft probabilities by default. Use `--enforce-dag` or `--threshold` for a binary DAG.

## Citation

If you use this work, the model, or the code in your research, please cite the
associated paper:

> Patrick Blöbaum, Krishnakumar Balasubramanian, and Shiva Prasad Kasiviswanathan. "FoundCause: Causal Discovery with Latent Confounders from Observational Data." arXiv:2606.17516, 2026.

BibTeX:

```bibtex
@misc{bloebaum2026foundcause,
      title         = {FoundCause: Causal Discovery with Latent Confounders from Observational Data},
      author        = {Patrick Bl{\"o}baum and Krishnakumar Balasubramanian and Shiva Prasad Kasiviswanathan},
      year          = {2026},
      eprint        = {2606.17516},
      archivePrefix = {arXiv},
      primaryClass  = {cs.LG},
      url           = {https://arxiv.org/abs/2606.17516}
}
```

## Note

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## License

This project is licensed under the Apache-2.0 License.