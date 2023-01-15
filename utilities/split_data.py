#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# %%


def encode_lgg_hgg(x):
    """Encode LGG and HGG as 0 and 1 for stratification

    Args:
        x (str): LGG or HGG in Dataframe

    Returns:
        int: encodes 0 for LGG and 1 for HGG
    """
    return 0 if x == "LGG" else 1


def train_val_test_dataset(data_path: str):
    """From 100% Cases take 20% cases as Validation.
    Take the remaining 80% cases as training

    Stratification done on data to ensure that the classes are balanced.

    Args:
        data_path (str, optional): Path to Name Mapping File provided by BraTS.

    Returns:
        training, validation, testing: list of case names split into training, validation and testing.
    """
    data = pd.read_csv(data_path)
    data = data[["Grade", "BraTS_2020_subject_ID"]]
    data.Grade = data["Grade"].map(encode_lgg_hgg)
    (
        training,
        validation,
        train_check,
        val_check,
    ) = train_test_split(
        data.BraTS_2020_subject_ID.to_list(),
        data.Grade.to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=data.Grade.to_numpy(),
        shuffle=True
    )



    print(f'''
    Total Samples = {len(training)+len(validation)}\n
    Ratio of LGG:HGG in {len(training)} Training Samples:
    \t Ratio = {np.count_nonzero(train_check==0)/np.count_nonzero(train_check==1):.2f}\n

    Ratio of LGG:HGG in {len(validation)} Validation Samples:
    \t Ratio = {np.count_nonzero(val_check==0)/np.count_nonzero(val_check==1):.2f}\n
    
    ''')
    



    return (training, validation)


# %%


if __name__ == "__main__":
    name_map_path = r"C:\Users\ibrah\Documents\Projects\3D-Segmentation-of-Glioblastoma-from-MRI\MICCAI_BraTS2020_TrainingData\name_mapping.csv"
    train, val= train_val_test_dataset(name_map_path)
# %%
