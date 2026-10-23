from pathlib import Path
import hashlib
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE = Path(__file__).parent

st.set_page_config(
    page_title="MoleculeScope",
    page_icon="🧪",
    layout="wide",
)

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)


# --------------------------------------------------
# Load model and training references
# --------------------------------------------------

@st.cache_resource
def load_resources():
    model = joblib.load(
        BASE / "models" / "random_forest.joblib"
    )

    reference = pd.read_csv(
        BASE / "data" / "training_reference.csv",
        keep_default_na=False,
    )

    fingerprints = [
        GENERATOR.GetFingerprint(
            Chem.MolFromSmiles(smiles)
        )
        for smiles in reference["smiles"]
    ]

    return model, reference, fingerprints


# --------------------------------------------------
# Molecular analysis
# --------------------------------------------------

def analyse(smiles, model, reference, reference_fps):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Enter a non-empty SMILES string.")

    mol = Chem.MolFromSmiles(smiles.strip())

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(
            "RDKit could not interpret this structure."
        )

    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(
            "Use a single connected molecule; salts and "
            "mixtures are not supported in this version."
        )

    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )

    fingerprint = GENERATOR.GetFingerprint(mol)

    array = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    prediction = float(
        model.predict(
            array.reshape(1, -1).astype(np.float32)
        )[0]
    )

    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            reference_fps,
        )
    )

    nearest_indices = np.argsort(
        -similarities, kind="stable"
    )[:5]

    maximum_similarity = float(similarities.max())

    exact_match = bool(
        reference["smiles"].eq(canonical).any()
    )

    result = {
        "smiles": canonical,
        "predicted_pIC50": prediction,
        "nearest_train_similarity": maximum_similarity,
        "low_similarity_warning": maximum_similarity < 0.5,
        "exact_training_match": exact_match,
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
    }

    neighbours = reference.iloc[nearest_indices][
        ["smiles", "pIC50"]
    ].copy()

    neighbours = neighbours.rename(
        columns={"pIC50": "measured_pIC50"}
    )

    neighbours["Tanimoto_similarity"] = (
        similarities[nearest_indices]
    )

    return mol, result, neighbours


# --------------------------------------------------
# Start application
# --------------------------------------------------

try:
    model, reference, reference_fps = load_resources()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(
        "Could not load the model or training reference. "
        "Run the data preparation and evaluation scripts first."
    )
    st.code(str(error))
    st.stop()

st.title("MoleculeScope")

st.write(
    "Explore predicted BACE-1 activity, similar experimental "
    "molecules and the limits of each prediction."
)

st.caption(
    "Research prototype · One protein target · "
    "No paid API required"
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("About this model")

    st.write("**Target:** BACE-1")
    st.write("**Dataset:** MoleculeNet BACE")
    st.write("**Model:** Random Forest")
    st.write(
        "**Representation:** Chiral Morgan fingerprints"
    )
    st.caption("Radius 2 · 2,048 bits")

    st.divider()

    st.write("Held-out scaffold test: 191 molecules")
    st.metric("Test MAE · pIC50 units", "0.665")
    st.metric("Test R²", "0.667")

    st.caption(
        "The model was trained on 1,058 molecules. "
        "Model selection used a separate validation set."
    )

    st.caption(
        "A single scaffold split estimates performance "
        "on this dataset; it does not establish "
        "real-world reliability."
    )

    st.divider()

    st.caption(
        "Uploaded molecules are processed by the machine "
        "running this Streamlit app. When deployed, that "
        "means the hosting server."
    )


single_tab, batch_tab, evaluation_tab = st.tabs(
    [
        "Single molecule",
        "Batch screening",
        "Evaluation",
    ]
)


# --------------------------------------------------
# Single-molecule analysis
# --------------------------------------------------

with single_tab:
    with st.form("single_molecule"):
        smiles = st.text_area(
            "SMILES",
            placeholder="Paste a molecular SMILES string",
        )

        submitted = st.form_submit_button(
            "Analyse molecule"
        )

    if submitted:
        # Clear the previous prediction before processing a new input.
        st.session_state.pop("single_result", None)

        try:
            mol, result, neighbours = analyse(
                smiles,
                model,
                reference,
                reference_fps,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["single_result"] = {
                "result": result,
                "neighbours": neighbours,
            }

    if "single_result" in st.session_state:
        saved = st.session_state["single_result"]
        result = saved["result"]
        neighbours = saved["neighbours"]

        mol = Chem.MolFromSmiles(result["smiles"])

        structure_col, result_col = st.columns([1, 2])

        with structure_col:
            st.image(
                Draw.MolToImage(
                    mol,
                    size=(400, 300),
                )
            )

        with result_col:
            first, second = st.columns(2)

            first.metric(
                "Predicted pIC50",
                f"{result['predicted_pIC50']:.2f}",
            )

            second.metric(
                "Nearest-training similarity",
                f"{result['nearest_train_similarity']:.3f}",
            )

            if result["exact_training_match"]:
                st.info(
                    "This structure appears in the training "
                    "set. Its prediction is not an "
                    "independent test."
                )

            if result["low_similarity_warning"]:
                st.warning(
                    "Low similarity to training molecules "
                    "(Tanimoto < 0.5). Treat this prediction "
                    "with additional caution."
                )
            else:
                st.info(
                    "Similarity is supporting context, "
                    "not a probability that the prediction "
                    "is correct."
                )

            st.caption(
                "Higher pIC50 corresponds to lower IC50. "
                "Predicted activity does not establish "
                "efficacy, selectivity or safety."
            )

        st.subheader("Canonical structure")
        st.code(result["smiles"], language=None)

        st.subheader("Calculated molecular properties")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        key: result[key]
                        for key in [
                            "MW",
                            "LogP",
                            "TPSA",
                            "HBD",
                            "HBA",
                            "RotatableBonds",
                        ]
                    }
                ]
            ),
            hide_index=True,
        )

        st.caption(
            "MW: g/mol · TPSA: Å² · "
            "LogP: calculated octanol/water partition measure"
        )

        st.subheader("Nearest training molecules")

        st.dataframe(
            neighbours,
            hide_index=True,
        )

        st.caption(
            "Measured values are dataset observations. "
            "Similar structures can still have different activities."
        )

        st.download_button(
            "Download molecule result",
            data=pd.DataFrame([result]).to_csv(index=False),
            file_name="moleculescope_prediction.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# Batch screening
# --------------------------------------------------

with batch_tab:
    st.write(
        "Upload a CSV containing a column named `smiles`. "
        "Maximum 200 rows and 2 MB."
    )

    st.download_button(
        "Download example CSV",
        data=(
            "smiles\n"
            "CCO\n"
            "CC(=O)Oc1ccccc1C(=O)O\n"
            "not_a_molecule\n"
            "CCO.Cl\n"
        ),
        file_name="moleculescope_example.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Molecule CSV",
        type=["csv"],
    )

    current_file_key = None

    if uploaded is not None:
        file_bytes = uploaded.getvalue()

        current_file_key = hashlib.sha256(
            file_bytes
        ).hexdigest()

        # Prevent results from an earlier file being shown for a new file.
        if (
            st.session_state.get("batch_file_key")
            != current_file_key
        ):
            st.session_state.pop("batch_results", None)
            st.session_state.pop("batch_file", None)

        if uploaded.size > 2 * 1024 * 1024:
            st.error("The file exceeds 2 MB.")
        else:
            try:
                batch = pd.read_csv(
                    io.BytesIO(file_bytes),
                    dtype=str,
                    keep_default_na=False,
                )
            except (
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
                UnicodeDecodeError,
            ) as error:
                st.error(f"Could not read the CSV: {error}")
            else:
                if "smiles" not in batch.columns:
                    st.error(
                        "The CSV needs a column named smiles."
                    )

                elif len(batch) == 0 or len(batch) > 200:
                    st.error(
                        "Upload between 1 and 200 rows."
                    )

                else:
                    st.caption(
                        f"Loaded {len(batch)} input rows."
                    )

                    if st.button("Screen batch"):
                        records = []

                        with st.spinner(
                            "Analysing molecules..."
                        ):
                            for row_number, value in enumerate(
                                batch["smiles"],
                                start=1,
                            ):
                                try:
                                    _, result, _ = analyse(
                                        value,
                                        model,
                                        reference,
                                        reference_fps,
                                    )

                                    records.append(
                                        {
                                            "input_row": row_number,
                                            "input_smiles": value,
                                            "status": "ok",
                                            **result,
                                        }
                                    )

                                except ValueError as error:
                                    records.append(
                                        {
                                            "input_row": row_number,
                                            "input_smiles": value,
                                            "status": str(error),
                                        }
                                    )

                        st.session_state["batch_results"] = (
                            pd.DataFrame(records)
                        )

                        st.session_state["batch_file"] = (
                            uploaded.name
                        )

                        st.session_state["batch_file_key"] = (
                            current_file_key
                        )

    else:
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_file", None)
        st.session_state.pop("batch_file_key", None)

    if (
        "batch_results" in st.session_state
        and current_file_key
        == st.session_state.get("batch_file_key")
    ):
        output = st.session_state["batch_results"].copy()

        st.subheader("Batch results")

        st.caption(
            f"Source: {st.session_state['batch_file']}"
        )

        valid_count = int(output["status"].eq("ok").sum())

        first, second = st.columns(2)
        first.metric("Valid input rows", valid_count)
        second.metric(
            "Rejected input rows",
            len(output) - valid_count,
        )

        if "predicted_pIC50" in output.columns:
            output = output.sort_values(
                "predicted_pIC50",
                ascending=False,
                na_position="last",
            )

        st.dataframe(
            output,
            hide_index=True,
        )

        st.download_button(
            "Download all batch results",
            data=output.to_csv(index=False),
            file_name="moleculescope_batch.csv",
            mime="text/csv",
        )

        # ------------------------------------------
        # Candidate prioritisation
        # ------------------------------------------

        st.divider()
        st.subheader("Candidate shortlist")

        st.caption(
            "Filter valid predictions, then rank by predicted "
            "activity. These filters do not establish "
            "efficacy or safety."
        )

        candidates = output[
            output["status"].eq("ok")
        ].copy()

        if candidates.empty:
            st.info(
                "No valid molecules are available to shortlist."
            )

        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                minimum_similarity = st.slider(
                    "Minimum training similarity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help=(
                        "An exploration filter, not a "
                        "validated reliability boundary."
                    ),
                )

            with col2:
                maximum_mw = st.number_input(
                    "Maximum molecular weight (g/mol)",
                    min_value=1.0,
                    value=600.0,
                    step=25.0,
                )

            with col3:
                maximum_logp = st.number_input(
                    "Maximum calculated LogP",
                    value=5.0,
                    step=0.5,
                )

            exclude_known = st.checkbox(
                "Exclude exact training-set matches",
                value=True,
            )

            remove_repeats = st.checkbox(
                "Keep one row per canonical structure",
                value=True,
            )

            mask = (
                candidates["nearest_train_similarity"].ge(
                    minimum_similarity
                )
                & candidates["MW"].le(maximum_mw)
                & candidates["LogP"].le(maximum_logp)
            )

            if exclude_known:
                mask &= candidates[
                    "exact_training_match"
                ].eq(False)

            filtered = candidates.loc[mask].copy()

            if remove_repeats:
                filtered = filtered.drop_duplicates(
                    subset="smiles",
                    keep="first",
                )

            shortlist = filtered.sort_values(
                "predicted_pIC50",
                ascending=False,
            ).copy()

            shortlist.insert(
                0,
                "rank",
                np.arange(1, len(shortlist) + 1),
            )

            st.write(
                f"{len(shortlist)} shortlisted entries from "
                f"{len(candidates)} valid input rows."
            )

            if shortlist.empty:
                st.info(
                    "No molecules meet these settings. "
                    "Adjust the filters to explore the batch."
                )

            else:
                st.dataframe(
                    shortlist,
                    hide_index=True,
                )

                st.download_button(
                    "Download shortlist",
                    data=shortlist.to_csv(index=False),
                    file_name="moleculescope_shortlist.csv",
                    mime="text/csv",
                )

            with st.expander("How this shortlist is created"):
                st.write(
                    "1. Keep valid molecular inputs.\n"
                    "2. Apply your similarity, molecular-weight "
                    "and LogP limits.\n"
                    "3. Optionally exclude exact training matches "
                    "and collapse repeated canonical structures.\n"
                    "4. Rank remaining entries by predicted "
                    "pIC50, highest first."
                )

                st.caption(
                    "Default property limits are demonstration "
                    "settings, not target-specific selection rules. "
                    "Close predicted values should not be interpreted "
                    "as a confidently established activity ordering. "
                    "The full batch download preserves rejected "
                    "and repeated input rows."
                )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

with evaluation_tab:
    st.subheader("What was evaluated?")

    st.write(
        "Molecules sharing a Murcko scaffold were kept in "
        "the same split. Model selection used validation "
        "data; the final test set was evaluated afterward."
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Split": [
                    "Training",
                    "Validation",
                    "Test",
                ],
                "Molecules": [
                    1058,
                    264,
                    191,
                ],
                "Scaffold groups": [
                    469,
                    101,
                    101,
                ],
            }
        ),
        hide_index=True,
    )

    metrics_path = BASE / "results" / "test_metrics.json"
    predictions_path = (
        BASE / "results" / "test_predictions.csv"
    )
    similarity_path = (
        BASE / "results" / "test_similarity_summary.csv"
    )

    if metrics_path.exists():
        import json

        with open(metrics_path, encoding="utf-8") as file:
            scores = json.load(file)

        st.subheader("Held-out test performance")

        metric_table = (
            pd.DataFrame(scores)
            .T
            .rename_axis("Model")
            .reset_index()
        )

        st.dataframe(
            metric_table.round(4),
            hide_index=True,
        )

        st.caption(
            "Lower MAE and RMSE are better. "
            "Higher R² is better. Errors are in pIC50 units."
        )

    if predictions_path.exists():
        test = pd.read_csv(predictions_path)

        st.subheader("Measured versus predicted activity")

        st.scatter_chart(
            test,
            x="actual_pIC50",
            y="predicted_pIC50",
        )

        st.caption(
            "Perfect predictions would lie on the line "
            "where predicted pIC50 equals measured pIC50."
        )

        st.subheader("Error versus training similarity")

        st.scatter_chart(
            test,
            x="nearest_train_similarity",
            y="absolute_error",
        )

    if similarity_path.exists():
        st.subheader("Test errors by similarity band")

        st.dataframe(
            pd.read_csv(similarity_path),
            hide_index=True,
        )

    st.warning(
        "Only 11 test molecules had similarity below 0.5. "
        "The warning threshold is exploratory, not a "
        "validated boundary between reliable and "
        "unreliable predictions."
    )

    with st.expander("Methodology and limitations"):
        st.write(
            "- Structures were parsed and canonicalised "
            "while preserving stereochemistry.\n"
            "- No canonical duplicates were found in this dataset.\n"
            "- Five acyclic molecules shared an empty "
            "Murcko scaffold group.\n"
            "- Features were generated from molecular structures; "
            "activity labels and supplied descriptors were "
            "not used as model inputs.\n"
            "- Random Forest was selected using validation MAE.\n"
            "- The deployed model remains trained on the "
            "original training split.\n"
            "- Neighbour searches use training molecules only.\n"
            "- Salt and tautomer standardisation were not performed.\n"
            "- Similarity does not account for every source "
            "of uncertainty or experimental variability.\n"
            "- No prospective experimental validation "
            "was performed."
        )

    st.caption(
        "Educational research prototype. No docking, "
        "toxicity prediction, clinical recommendation "
        "or experimental confirmation is provided."
    )