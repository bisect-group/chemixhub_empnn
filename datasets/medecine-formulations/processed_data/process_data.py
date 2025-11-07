import os
import pandas as pd
import numpy as np
import cirpy
import matplotlib.pyplot as plt


def make_directories(raw_dir: str, processed_dir: str) -> None:
    """Create raw and processed data directories if they don't exist."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)


def load_data(raw_dir: str) -> pd.DataFrame:
    """Load the master CSV file containing solubility data."""
    file_path = os.path.join(raw_dir, "master_results_with_solubility.csv")
    df = pd.read_csv(file_path)
    return df


def add_deionized_water_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add dH2O as an additional component representing water percentage."""
    df["dH2O"] = 100 - df[df.columns[:5].tolist()].sum(axis=1)
    return df


def generate_compounds_table(processed_dir: str) -> pd.DataFrame:
    """Generate and save a compounds table with SMILES using CAS numbers."""
    compound_names = ["T20", "T80", "P188", "DMSO", "PG", "dH2O"]
    cas_numbers = ["9005-64-5", "9005-65-6", "9003-11-6", "67-68-5", "57-55-6", "7732-18-5"]

    smiles_list = [cirpy.resolve(cas, "smiles") for cas in cas_numbers]
    compounds_df = pd.DataFrame({
        "compound_id": list(range(len(compound_names))),
        "name": compound_names,
        "smiles": smiles_list
    })

    compounds_df.to_csv(os.path.join(processed_dir, "compounds.csv"), index=False)
    return compounds_df


def mixture_info(row, components, compound_ids):
    """Extract component IDs and fractions for a single mixture row."""
    ids, concs = [], []
    for i, comp in enumerate(components):
        val = row[comp]
        if val > 0:
            ids.append(compound_ids[i])
            concs.append(val / 100)
    return pd.Series({"cmp_ids": ids, "cmp_mole_fractions": concs})


def process_formulation_data(df: pd.DataFrame, processed_dir: str) -> None:
    """Process formulation data and save it to CSV."""
    components = df[df.columns[:5].tolist()].columns.tolist() + ["dH2O"]
    compound_ids = list(range(len(components)))

    df[["cmp_ids", "cmp_mole_fractions"]] = df.apply(
        lambda row: mixture_info(row, components, compound_ids), axis=1
    )

    processed_df = pd.DataFrame({
        "value": df["Solubility (mg/mL)"].values,
        "property": "solubility",
        "unit": "mg/mL",
        "cmp_ids": df["cmp_ids"].values,
        "cmp_mole_fractions": df["cmp_mole_fractions"].values
    })

    processed_df.to_csv(
        os.path.join(processed_dir, "processed_MedicineFormulations.csv"), index=False
    )


def main():
    """Main execution pipeline for processing medicine formulations."""
    raw_dir = "../raw_data/"
    processed_dir = "../processed_data/"

    make_directories(raw_dir, processed_dir)

    df = load_data(raw_dir)
    df = add_deionized_water_column(df)

    generate_compounds_table(processed_dir)
    process_formulation_data(df, processed_dir)

    print("Processing completed successfully.")
    print(f"Processed data saved in: {processed_dir}")


if __name__ == "__main__":
    main()
