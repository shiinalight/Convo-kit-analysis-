import pandas as pd
from typing import List


def subset_df(
    df: pd.DataFrame, cluster_subset: List = [], group_subset: List = []
) -> pd.DataFrame:
    """
    For a dataframe where groups are the index levels and
    cluster labels are the columns,
    subsets the dataframe by the input lists of values
    """
    if cluster_subset:
        df = df[cluster_subset]  # in the columns

    if group_subset:
        df = df.loc[group_subset]  # in the index (rows)

    return df


def match_qa_pairs(
    df: pd.DataFrame, drop_nas: bool = False, include_group: bool = False
) -> pd.DataFrame:
    """
    Match question/answer pairs and their cluster numbers.
    Preserves all NAs with outer join, but can optionally
    return only pairs where both utterances have been assigned
    clusters.
    """

    q_cols = ["q_cluster"]
    a_cols = ["a_cluster", "reply_to"]

    if include_group:
        q_cols.append("group")
        a_cols.append("group")

    question_cluster_assignments = df[df.is_question][q_cols]

    answer_cluster_assignments = df[df.is_answer][a_cols]

    matched_qas = question_cluster_assignments.merge(
        answer_cluster_assignments,
        how="outer",
        left_on=question_cluster_assignments.index,
        right_on=["reply_to"],
        suffixes=["_q", "_a"],
    )
    matched_qas.drop(["reply_to"], axis=1, inplace=True)

    if drop_nas:
        matched_qas.dropna(inplace=True)

    return matched_qas
