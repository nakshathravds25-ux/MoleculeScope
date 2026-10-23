# MoleculeScope

An interactive molecular activity prediction and prioritisation app
for the BACE-1 protein target.

Built with Python, RDKit, scikit-learn and Streamlit.

## Purpose

MoleculeScope combines predicted molecular activity with similar
experimental compounds and training-set similarity information.
Users can explore individual molecules, screen batches and create
filtered candidate shortlists.

Similarity is presented as applicability context, not as a
probability that a prediction is correct.

## Features

- Molecular structure validation and visualisation
- BACE-1 pIC50 prediction using Random Forest
- Retrieval of five nearest training molecules
- Exploratory low-similarity warnings
- Molecular property calculation
- Batch CSV screening with invalid-input reporting
- Candidate filtering by similarity, molecular weight and LogP
- Downloadable predictions and shortlists
- Held-out evaluation dashboard

## Dataset and preparation

Source: MoleculeNet BACE, distributed through DeepChem.

Dataset download:
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv

Dataset loader and documentation:
https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/bace_datasets.py

The dataset contains 1,513 molecular records.

Structures were validated and canonicalised with stereochemistry
preserved. No invalid structures, canonical duplicates or
disconnected structures were found.

Model inputs were generated from SMILES rather than the supplied
descriptor columns. Activity labels were excluded from the inputs.

## Molecular representation

- Morgan fingerprints
- Radius: 2
- Length: 2,048 bits
- Chirality included

## Evaluation design

Molecules sharing a Murcko scaffold were assigned to the same split.

| Split | Molecules | Scaffold groups |
|---|---:|---:|
| Training | 1,058 | 469 |
| Validation | 264 | 101 |
| Test | 191 | 101 |

No scaffold groups overlap across splits.

A mean-prediction baseline, Ridge regression and Random Forest were
compared on validation data. Random Forest was selected using
validation MAE. The test set was evaluated after model selection.

## Results

### Validation

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 1.0881 | 1.3219 | -0.0007 |
| Ridge | 0.6067 | 0.7844 | 0.6477 |
| Random Forest | 0.5790 | 0.7769 | 0.6544 |

### Held-out test

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 1.2932 | 1.5213 | -0.0148 |
| Random Forest | 0.6647 | 0.8717 | 0.6668 |

MAE and RMSE are expressed in pIC50 units.

## Applicability analysis

Nearest-training Tanimoto similarity was compared with prediction
error on validation and test molecules.

| Test similarity band | Molecules | MAE |
|---|---:|---:|
| 0–0.3 | 3 | 1.7298 |
| 0.3–0.5 | 8 | 1.0138 |
| 0.5–0.7 | 61 | 0.6795 |
| 0.7–1.0 | 119 | 0.6068 |

Lower-similarity groups had larger average errors in this test split.
However, only 11 test molecules had similarity below 0.5.

The app's warning threshold of 0.5 is exploratory, not a validated
reliability boundary.

## Run locally

Developed using Python 3.13.

Create and activate a virtual environment, then run:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

## Reproduce the pipeline

With the original dataset saved as data/bace.csv, run in order:

    python prepare_data.py
    python generate_features.py
    python split_data.py
    python train_models.py
    python check_applicability.py
    python evaluate_model.py

The app uses the saved model trained on the original training split.
Neighbour searches use training molecules only.

## Batch input

Upload a CSV with a column named smiles.

Limits: 200 rows and 2 MB.

Invalid inputs remain in the full results with explanatory messages.
Candidate shortlists can exclude training matches and repeated
canonical structures.

## Checks performed

- Structure parsing and canonical duplicate checks
- Feature and metadata row-alignment checks
- Scaffold separation and complete split-assignment checks
- Manual single-input and batch-input interface checks
- Invalid and disconnected structure handling
- Shortlist response to changes in similarity threshold

## Limitations

- One protein target and one dataset
- Evaluation uses a single scaffold split
- Salt and tautomer standardisation were not performed
- Similarity does not capture all sources of uncertainty
- Property filters are exploration settings, not validated selection rules
- Small prediction differences do not establish a reliable ranking
- No docking, toxicity assessment or prospective experimental validation
- Predictions do not establish efficacy, selectivity or safety

This is an educational research prototype.

## References

Wu et al. MoleculeNet: a benchmark for molecular machine learning.
Chemical Science, 2018. DOI: 10.1039/C7SC02664A.

Subramanian et al. Computational Modeling of β-Secretase 1 (BACE-1)
Inhibitors Using Ligand Based Approaches.
Journal of Chemical Information and Modeling, 2016.
DOI: 10.1021/acs.jcim.6b00290.

## Author

Nakshathra V.
MSc Data Science and BioAI, Digital University Kerala