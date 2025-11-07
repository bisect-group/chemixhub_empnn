import argparse
from mixhub.data.splits import SPLIT_MAPPING
from mixhub.data.data import DATA_CATALOG, MixtureDataInfo
from mixhub.data.dataset import MixtureTask
import inspect

from typing import Optional

def takes_argument(func, k):
    sig = inspect.signature(func)
    return k in sig.parameters

def main(
    dataset_name: MixtureDataInfo,
    split_type: str,
    k_list: Optional[list[int]] = None,
    seed: Optional[int] = None,
    geometric: Optional[str] = False,
):
    
    dataset = DATA_CATALOG[dataset_name]()

    for property in dataset.properties:

        print(property)

        mixture_task = MixtureTask(
            property=property,
            dataset=dataset,
        )

        if takes_argument(SPLIT_MAPPING[split_type], "k"):

            for k in k_list:
                SPLIT_MAPPING[split_type](
                    property=property,
                    mixture_indices_tensor=mixture_task.indices_tensor,
                    cache_dir=dataset.data_dir,
                    k=k,
                    seed=seed,
                    geometric=geometric,
                )
        elif takes_argument(SPLIT_MAPPING[split_type], "temperature_tensor"):
            SPLIT_MAPPING[split_type](
                property=property,
                mixture_indices_tensor=mixture_task.indices_tensor,
                temperature_tensor=mixture_task.context_tensor,
                cache_dir=dataset.data_dir,
                seed=seed,
            )
        else:
            SPLIT_MAPPING[split_type](
                property=property,
                mixture_indices_tensor=mixture_task.indices_tensor,
                cache_dir=dataset.data_dir,
                seed=seed,
            )      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create splits for a CheMixHub Dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("split_type", type=str, help="Desired type of split")
    parser.add_argument("--geometric", type=bool, default=True, help="Whether to use a geometric mean for the number of components-based split")
    parser.add_argument("--k_values", type=int, nargs="+", default=None, help="List of K values for a number of components-based split (e.g. --k_values 5 10 20)")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    split_type = args.split_type
    geometric = args.geometric
    k_list = args.k_values
    seed = 0

    main(
        dataset_name=dataset_name,
        split_type=split_type,
        k_list=k_list,
        seed=seed,
        geometric=geometric,
    )
