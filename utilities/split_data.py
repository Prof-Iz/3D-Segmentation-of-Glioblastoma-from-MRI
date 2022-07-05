#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# %%


def encode_lgg_hgg(x):
    """Encode LGG and HGG as 0 and 1

    Args:
        x (str): LGG or HGG in Dataframe

    Returns:
        int: encodes 0 for LGG and 1 for HGG
    """
    return 0 if x == "LGG" else 1


def train_val_test_dataset(data_path: str):
    """From 100% Cases take 10% cases as testing.
    Take the remaining 90% cases as training and validation -
    with the ratio of training:validation = 0.8:0.2.
    Stratification done on data to ensure that the classes are balanced.

    Args:
        data_path (str, optional): Path to Name Mapping File.

    Returns:
        training, validation, testing: list of case names split into training, validation and testing.
    """
    data = pd.read_csv(data_path)
    data = data[["Grade", "BraTS_2020_subject_ID"]]
    data.Grade = data["Grade"].map(encode_lgg_hgg)
    (
        training_val,
        testing,
        training_val_to_stratify,
        testing_to_stratify,
    ) = train_test_split(
        data.BraTS_2020_subject_ID.to_list(),
        data.Grade.to_numpy(),
        test_size=0.1,
        random_state=42,
        stratify=data.Grade.to_numpy(),
    )

    (
        training,
        validation,
        training_to_stratify,
        validation_to_stratify,
    ) = train_test_split(
        training_val,
        training_val_to_stratify,
        test_size=0.2,
        random_state=42,
        stratify=training_val_to_stratify,
    )

    # Class Ratio Calculation

    class_ratio_test = np.count_nonzero(testing_to_stratify == 1) / len(
        testing_to_stratify
    )

    class_ratio_val = np.count_nonzero(validation_to_stratify == 1) / len(
        validation_to_stratify
    )

    class_ratio_training = np.count_nonzero(training_to_stratify == 1) / len(
        training_to_stratify
    )

    print(
        f"""
          Ratio of Classes -> HGG/Total
          - Training {class_ratio_training:.2f}\t| {len(training_to_stratify)} Cases.
          - Validation {class_ratio_val:.2f}\t| {len(validation_to_stratify)} Cases.
          - Testing {class_ratio_test:.2f}\t| {len(testing_to_stratify)} Cases.
          
          Total Cases: {data.Grade.count()}
          """
    )

    return (training, validation, testing)


# %%


if __name__ == "__main__":
    name_map_path = r"D:\University\OneDrive - UCSI University\FYP 2022\Code\3D-Segmentation-of-Glioblastoma-from-MRI\Datasets\2020\MICCAI_BraTS2020_TrainingData\name_mapping.csv"
    train, val, test = train_val_test_dataset(name_map_path)
# %%
