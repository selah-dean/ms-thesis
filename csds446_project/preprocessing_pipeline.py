import pandas as pd
from typing import List, Tuple

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import copy


def preprocess_node_features(data: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Preprocesses node features by standardizing numerical columns and one-hot encoding categorical columns,
    grouped by specified columns.

    Args:
        data (pd.DataFrame): Input DataFrame with node features.
        group_cols (List[str]): Columns to group by (e.g., ["year"]).

    Returns:
        pd.DataFrame: Transformed DataFrame with processed features, grouped by `group_cols`.
    """
    numeric_cols = data.select_dtypes(include="number").columns.difference(group_cols)
    categorical_cols = data.select_dtypes(exclude="number").columns.difference(
        group_cols
    )

    # Define the preprocessing transformers list
    transformers = [("num", StandardScaler(), numeric_cols)]

    # Add categorical encoding only if categorical_cols is not empty
    if len(categorical_cols) > 0:
        transformers.append(("cat", OneHotEncoder(drop="first"), categorical_cols))

    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(transformers)

    # Define a function to apply the transformations to each group
    def preprocess_group(group):
        # Apply the column transformations using the pipeline
        transformed_data = preprocessor.fit_transform(group)  # Don't drop group_cols

        # Convert the result back into a DataFrame
        transformed_df = pd.DataFrame(transformed_data, index=group.index)

        # Construct column names
        column_names = list(numeric_cols)
        if len(categorical_cols) > 0:
            column_names += list(
                preprocessor.named_transformers_["cat"].get_feature_names_out(
                    categorical_cols
                )
            )

        transformed_df.columns = column_names

        return transformed_df

    # Group by 'id' and 'year', then apply the preprocessing
    df_processed = data.groupby(group_cols, group_keys=True).apply(
        preprocess_group, include_groups=False
    )

    return df_processed


def preprocess_edge_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocessing transformations to edge (plate appearance) features including
    ordinal encoding, standardization, and normalization.

    Args:
        data (pd.DataFrame): DataFrame containing edge attributes.

    Returns:
        pd.DataFrame: Transformed edge features.
    """
    transformers = [
        ("ordinal", OrdinalEncoder(), ["inning", "lp", "outs_pre"]),
        ("num_std", StandardScaler(), ["nump"]),
        ("num_minmax", MinMaxScaler(), ["num_times_faced_in_game"]),
    ]

    preprocessor = ColumnTransformer(transformers, remainder="passthrough")

    transformed_data = preprocessor.fit_transform(data)

    transformed_columns = [
        "inning",
        "lp",
        "outs_pre",
        "nump",
        "num_times_faced_in_game",
    ]
    passthrough_columns = [
        col for col in data.columns if col not in transformed_columns
    ]

    # Combine column names
    all_columns = transformed_columns + passthrough_columns

    # Convert to DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=all_columns)

    return transformed_df


def apply_preprocessing(
    pitchers: pd.DataFrame,
    batters: pd.DataFrame,
    plate_apps: pd.DataFrame,
    min_year: int,
    max_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies preprocessing to pitcher, batter, and plate appearance datasets.

    Args:
        pitchers (pd.DataFrame): Pitcher node data.
        batters (pd.DataFrame): Batter node data.
        plate_apps (pd.DataFrame): Edge data for plate appearances.
        min_year (int): Minimum year to include.
        max_year (int): Maximum year to include.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Processed pitcher nodes, batter nodes, and edge data.
    """
    # Preprocess node data for modeling
    pitchers_processed = preprocess_node_features(
        pitchers.set_index("key_retro"), ["year"]
    )
    batters_processed = preprocess_node_features(
        batters.set_index("key_retro"), ["year"]
    )

    # Preprocess edge data for modeling
    plate_apps_processed = preprocess_edge_features(plate_apps)

    # Select years certain years of data
    pitcher_df = pitchers_processed.loc[min_year:max_year]
    batter_df = batters_processed.loc[min_year:max_year]

    # Add column for year for node and edge matching
    plate_apps_processed["year"] = pd.to_datetime(plate_apps_processed["date"]).dt.year
    plate_app_df = plate_apps_processed.loc[
        (plate_apps_processed["year"] >= min_year)
        & (plate_apps_processed["year"] <= max_year)
    ].set_index("pa_id")

    # Find valid pitchers and batters for each year
    valid_pitchers = set(
        pitcher_df.index.to_frame()[["year", "key_retro"]].apply(tuple, axis=1)
    )
    valid_batters = set(
        batter_df.index.to_frame()[["year", "key_retro"]].apply(tuple, axis=1)
    )

    # Drop any edges that have players that don't exist in the node sets
    plate_app_df = plate_app_df.loc[
        plate_app_df[["year", "batter"]].apply(tuple, axis=1).isin(valid_batters)
    ]
    plate_app_df = plate_app_df.loc[
        plate_app_df[["year", "pitcher"]].apply(tuple, axis=1).isin(valid_pitchers)
    ]

    # Sort edges by date
    plate_app_df["date"] = pd.to_datetime(plate_app_df["date"])
    plate_app_df = plate_app_df.sort_values("date")

    # Create column for time since start (in days)
    plate_app_df["time_since_start"] = (
        plate_app_df["date"] - plate_app_df["date"].min()
    ).dt.total_seconds() / 86400

    return pitcher_df, batter_df, plate_app_df


def create_graph(
    pitcher_df: pd.DataFrame,
    batter_df: pd.DataFrame,
    plate_app_df: pd.DataFrame,
    outcome_col: str,
) -> Tuple[HeteroData, LabelEncoder]:
    """
    Constructs a PyG HeteroData graph from node and edge data.

    Args:
        pitcher_df (pd.DataFrame): Pitcher node features.
        batter_df (pd.DataFrame): Batter node features.
        plate_app_df (pd.DataFrame): Edge data with time and attributes.
        outcome_col (str): Column name for outcome labels (e.g., "outcome").

    Returns:
        Tuple[HeteroData, LabelEncoder]: A PyTorch Geometric graph and the label encoder for outcomes.
    """
    # Select features
    pitcher_features = pitcher_df.to_numpy()
    batter_features = batter_df.to_numpy()
    edge_features = (
        plate_app_df[["inning", "outs_pre", "br1_pre", "br2_pre", "br3_pre"]]
        .to_numpy()
        .astype(float)
    )
    edge_times = plate_app_df["time_since_start"].to_numpy()

    # Create HeteroData Object for graph
    data = HeteroData()

    # Add nodes and their features
    data["pitcher"].x = torch.tensor(pitcher_features, dtype=torch.float)
    data["batter"].x = torch.tensor(batter_features, dtype=torch.float)

    # Create node mappings
    pitcher_mapping = {pid: i for i, pid in enumerate(pitcher_df.index)}
    batter_mapping = {bid: i for i, bid in enumerate(batter_df.index)}

    # Create edge index tensor for at-bats
    pitcher_indices = [
        pitcher_mapping[(year, pid)]
        for year, pid in zip(plate_app_df["year"], plate_app_df["pitcher"])
    ]
    batter_indices = [
        batter_mapping[(year, bid)]
        for year, bid in zip(plate_app_df["year"], plate_app_df["batter"])
    ]

    # Add edges (pitcher --> batter)
    data[("pitcher", "faces", "batter")].edge_index = torch.tensor(
        [pitcher_indices, batter_indices], dtype=torch.long
    )

    # Encode categorical outcome labels
    label_encoder = LabelEncoder()
    encoded_outcomes = label_encoder.fit_transform(plate_app_df[outcome_col].values)

    # Add edge labels for faces relation
    data[("pitcher", "faces", "batter")].y = torch.tensor(
        encoded_outcomes, dtype=torch.long
    )

    # Add edge attributes for faces relation
    data[("pitcher", "faces", "batter")].edge_attr = torch.tensor(
        edge_features, dtype=torch.float
    )

    # Add edge time for faces relation
    data[("pitcher", "faces", "batter")].edge_time = torch.tensor(
        edge_times, dtype=torch.float
    )

    # Add the reverse direction of edges
    data = T.ToUndirected()(data)

    return data, label_encoder


def preprocessing_pipeline(
    min_year: int, max_year: int, outcome_col: str, test_edges=False
):
    """
    Applies the preprocessing on each dataset and creates the graph.

    Args:
        min_year (int): First year of data to use.
        max_year (pd.DataFrame): The last year of data to use.
        outcome_col (str): Column name for outcome labels (e.g., "outcome").
        test_edges (bool): Indicator of whether to use the test data set for the edges.

    Returns:
        Tuple[HeteroData, LabelEncoder]: A PyTorch Geometric graph and the label encoder for outcomes.
    """
    # Read cleaned datasets
    pitchers = pd.read_csv("datasets/clean/pitchers.csv")
    batters = pd.read_csv("datasets/clean/batters.csv")

    plate_apps = (
        pd.read_csv("datasets/clean/test_plate_apps.csv")
        if test_edges
        else pd.read_csv("datasets/clean/plate_apps.csv")
    )

    # Apply preprocessing
    pitcher_df, batter_df, plate_app_df = apply_preprocessing(
        pitchers, batters, plate_apps, min_year, max_year
    )

    # Form the graph
    data, label_encoder = create_graph(pitcher_df, batter_df, plate_app_df, outcome_col)

    # Summarize graph info
    print(f"Num Pitcher Nodes: {data['pitcher'].num_nodes}")
    print(f"Num Batter Nodes: {data['batter'].num_nodes}")
    print(f"Num Edges: {data[('pitcher', 'faces', 'batter')].num_edges}")

    return data, label_encoder


def temporal_hetero_split(data, train_ratio=0.8):
    """
    Split a heterogeneous graph temporally while preserving its structure.

    Args:
        data: HeteroData object containing the graph with edge_time attributes
        train_ratio: Proportion of edges for training (earliest edges)

    Returns:
        train_data, val_data: Separate HeteroData objects for each split
    """
    # Create deep copies of the original data for each split
    train_data = copy.deepcopy(data)
    val_data = copy.deepcopy(data)

    # Process each edge type that has temporal information
    for edge_type in data.edge_types:
        # Check if this edge type has temporal data
        if hasattr(data[edge_type], "edge_time"):
            # Get the edge times for this relation
            edge_times = data[edge_type].edge_time

            # Sort edges by time
            sorted_indices = torch.argsort(edge_times)
            num_edges = len(edge_times)

            # Calculate split indices
            train_end = int(train_ratio * num_edges)

            # Create index masks for each split
            train_indices = sorted_indices[:train_end]
            val_indices = sorted_indices[train_end:]

            # Apply split to edge_index
            train_data[edge_type].edge_index = data[edge_type].edge_index[
                :, train_indices
            ]
            val_data[edge_type].edge_index = data[edge_type].edge_index[:, val_indices]

            # Apply split to edge_time
            train_data[edge_type].edge_time = data[edge_type].edge_time[train_indices]
            val_data[edge_type].edge_time = data[edge_type].edge_time[val_indices]

            # Apply split to other edge attributes if they exist
            if hasattr(data[edge_type], "edge_attr"):
                train_data[edge_type].edge_attr = data[edge_type].edge_attr[
                    train_indices
                ]
                val_data[edge_type].edge_attr = data[edge_type].edge_attr[val_indices]

            # Apply split to edge labels if they exist
            if hasattr(data[edge_type], "y"):
                train_data[edge_type].y = data[edge_type].y[train_indices]
                val_data[edge_type].y = data[edge_type].y[val_indices]

            # Also handle the reverse edges if your graph is undirected
            rev_edge_type = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
            if rev_edge_type in data.edge_types:
                # Get the edge_time for reverse edges (if it exists)
                if hasattr(data[rev_edge_type], "edge_time"):
                    rev_edge_times = data[rev_edge_type].edge_time

                    # Sort edges by time
                    rev_sorted_indices = torch.argsort(rev_edge_times)
                    rev_num_edges = len(rev_edge_times)

                    # Calculate split indices for reverse edges
                    rev_train_end = int(train_ratio * rev_num_edges)

                    # Create index masks for each split
                    rev_train_indices = rev_sorted_indices[:rev_train_end]
                    rev_val_indices = rev_sorted_indices[rev_train_end:]

                    # Apply split to reverse edge_index
                    train_data[rev_edge_type].edge_index = data[
                        rev_edge_type
                    ].edge_index[:, rev_train_indices]
                    val_data[rev_edge_type].edge_index = data[rev_edge_type].edge_index[
                        :, rev_val_indices
                    ]

                    # Apply split to reverse edge_time
                    train_data[rev_edge_type].edge_time = data[rev_edge_type].edge_time[
                        rev_train_indices
                    ]
                    val_data[rev_edge_type].edge_time = data[rev_edge_type].edge_time[
                        rev_val_indices
                    ]

                    # Apply split to other reverse edge attributes if they exist
                    if hasattr(data[rev_edge_type], "edge_attr"):
                        train_data[rev_edge_type].edge_attr = data[
                            rev_edge_type
                        ].edge_attr[rev_train_indices]
                        val_data[rev_edge_type].edge_attr = data[
                            rev_edge_type
                        ].edge_attr[rev_val_indices]
                    # Apply split to reverse edge labels if they exist
                    if hasattr(data[rev_edge_type], "y"):
                        train_data[rev_edge_type].y = data[rev_edge_type].y[
                            rev_train_indices
                        ]
                        val_data[rev_edge_type].y = data[rev_edge_type].y[
                            rev_val_indices
                        ]

    return train_data, val_data


def load_data(outcome_col: str):
    """
    Applies the preprocessing pipeline and splits data in train/validation/test sets

    Args:
        outcome_col (str): Column name for outcome labels (e.g., "outcome").

    Returns:
        Tuple[HeteroData, HeteroData, HeteroData, LabelEncoder, LabelEncoder]: Three PyTorch Geometric graph and label encoder for outcomes.
    """

    data, label_encoder = preprocessing_pipeline(2022, 2023, outcome_col)
    train_data, val_data = temporal_hetero_split(data)

    test_data, test_label_encoder = preprocessing_pipeline(
        2024, 2024, outcome_col, test_edges=True
    )

    return train_data, val_data, test_data, label_encoder, test_label_encoder
