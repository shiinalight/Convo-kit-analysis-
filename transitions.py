import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from convokit_analysis.utils import match_qa_pairs

plt.style.use(["science", "grid"])


def generate_qa_transition_matrix(
    df: pd.DataFrame,
    cluster_labels: Dict = {},
    answer_labels: Dict = {},
    normalized: bool = True,
    fn: str = "mat.pdf",
):
    """
    This matrix captures transitions of Question --> Answer (in the same pair)
    Does not include NAs
    """
    matched_qas = match_qa_pairs(df)
    matched_qas.reset_index(
        inplace=True
    )  # making a dummy column so that there is a column left after groupby
    matched_qas = matched_qas.fillna(-1)
    if cluster_labels:
        axis_labels = ["unassigned"] + list(cluster_labels.values())
    else:
        axis_labels = ["unassigned"] + sorted(matched_qas.a_cluster.unique())

    if answer_labels:
        answer_axis_labels = ["unassigned"] + list(answer_labels.values())
    else:
        answer_axis_labels = ["unassigned"] + sorted(
            matched_qas.a_cluster.unique()
        )

    mat = matched_qas.groupby(["q_cluster", "a_cluster"]).count().unstack()
    mat = mat.fillna(0)

    if normalized:
        mat["totals"] = mat[mat.columns].sum(axis=1)
        mat = mat[mat.columns].div(mat.totals, axis=0)
        # print(mat)
        mat = mat.drop(["totals"], axis=1)
        fmt_type = ".3f"
    else:
        fmt_type = ".1f"

    plt.figure(figsize=(20, 16))

    ax = sns.heatmap(
        mat,
        center=False,
        annot=True,
        linewidths=0.5,
        cmap="YlGnBu",
        fmt=fmt_type,
        annot_kws={"size": 20},
    )
    ax.tick_params(length=0)
    ax.set_ylabel("Question cluster", fontsize=30)
    ax.set_xlabel("Answer cluster", fontsize=30)
    ax.set_xticklabels(
        answer_axis_labels,
        rotation=45,
        horizontalalignment="right",
        fontdict={"fontsize": 24},
    )
    ax.set_yticklabels(
        axis_labels,
        rotation=0,
        horizontalalignment="right",
        fontdict={"fontsize": 24},
    )

    plt.savefig(fn)


##################################################
# Transition matrix
##################################################


def generate_transition_matrix(
    df: pd.DataFrame,
    state_label: str,
    utterance_type: str,
    normalized: bool = False,
    cluster_labels: Dict = {},
    fn: str = "mat.pdf",
):
    """
    Generate a transition matrix of raw counts or normalized percents
    for questions or for answers

    @param df dataframe of results
    @param state_label could be cluster_num or category
    @param normalized Compute transition matrix with raw
    counts or normalized pcts
    """
    conversation_roots = df["hearing_id"].unique()
    transition_counts_clusters = compute_transition_counts(
        df, state_label, conversation_roots, utterance_type
    )

    if cluster_labels:
        axis_labels = list(cluster_labels.values())
    else:
        axis_labels = sorted(df.a_cluster.unique())

    from_labels = ["start"] + axis_labels
    to_labels = axis_labels + ["end"]

    plt.figure(figsize=(20, 16))

    # columns are not necessarily in the right order, fix that
    transition_counts_clusters = transition_counts_clusters.reindex(
        sorted(transition_counts_clusters.columns), axis=1
    )

    if not normalized:
        # display(transition_counts_clusters)

        ax = sns.heatmap(
            transition_counts_clusters,
            center=False,
            annot=True,
            linewidths=0.5,
            cmap="YlGnBu",
            fmt="g",
            annot_kws={"size": 20},
        )
        ax.tick_params(length=0)
        ax.set_xticklabels(
            to_labels,
            rotation=45,
            horizontalalignment="right",
            fontdict={"fontsize": 24},
        )
        ax.set_yticklabels(
            from_labels,
            rotation=0,
            horizontalalignment="right",
            fontdict={"fontsize": 24},
        )
        ax.set_ylabel("Transition from", fontsize=30)
        ax.set_xlabel("Transition to", fontsize=30)

    else:
        transition_counts_clusters_states_only = mat_states_only(
            transition_counts_clusters
        )
        norm_mat_clusters = mat_normalized_rows(
            transition_counts_clusters_states_only
        )

        ax = sns.heatmap(
            norm_mat_clusters,
            center=False,
            annot=True,
            linewidths=0.5,
            cmap="YlGnBu",
            annot_kws={"size": 20},
        )
        ax.tick_params(length=0)
        ax.set_xticklabels(
            to_labels,
            rotation=45,
            horizontalalignment="right",
            fontdict={"fontsize": 24},
        )
        ax.set_yticklabels(
            from_labels,
            rotation=0,
            horizontalalignment="right",
            fontdict={"fontsize": 24},
        )
        ax.set_ylabel("Transition from", fontsize=30)
        ax.set_xlabel("Transition to", fontsize=30)

    plt.savefig(fn)


# Helper functions


def subset_hearing(hearing_id: str, df: pd.DataFrame) -> List[Dict]:
    """
    Subsets a df by hearing_id and returns the result as a list of dicts

    @param hearing_id unique id of a hearing (conversation)
    @param df dataframe of results
    @utterance type question or answer
    """

    this_convo = df[df.hearing_id == hearing_id]

    this_convo_dict = this_convo.to_dict(orient="records")

    return this_convo_dict


def generate_empty_transition_dict(
    states: List, add_init_term_states: bool = True
) -> Dict:
    """
    Takes an array of states. Returns a dict where keys = start states
    and values = dicts{keys = end states, values = 0
    [will be count of this transition]}

    @param add_init_term_states add "start" and "end" states
    """
    cluster_names = list(states)

    if add_init_term_states:
        starts = cluster_names + [-1]
        ends = cluster_names + [10000]
    else:
        starts = cluster_names
        ends = cluster_names

    transition_counts_dict = {k: {k: 0 for k in ends} for k in starts}

    return transition_counts_dict


def count_transition_probs(
    state_label: str,
    this_hearing_list_of_dicts: List[Dict],
    transition_counts_dict: Dict,
) -> Dict:
    """
    Iterates through this_hearing_list_of_dicts
    and, for each utterance,
    increments the count for the appropriate transition
    in transition_counts_dict

    It is assumed that transition_counts_dict is passed
    in with all values initialized to 0
    """
    num_utterances = len(this_hearing_list_of_dicts)

    for i in range(num_utterances):
        # look at i and the utterance BEFORE i

        this_state = this_hearing_list_of_dicts[i][state_label]

        # handle special start case
        if i == 0:
            prev_state = -1

        else:
            prev_state = this_hearing_list_of_dicts[i - 1][state_label]

        # Increment the count in the matrix
        transition_counts_dict[prev_state][this_state] += 1

        # Handle end state
        if i == (num_utterances - 1):
            transition_counts_dict[this_state][10000] += 1

    return transition_counts_dict


def compute_transition_counts(
    df: pd.DataFrame,
    state_label: str,
    unique_convos: List[str],
    utterance_type: str,
):
    """
    Takes a list of states and calls generate_empty_transition_dict()
    to make the transition_dict
    Then iterates through each hearing with subset_hearing()
    and uses count_transition_probs to add to the transition_dict
    """
    if utterance_type == "question":
        df = df[df.is_question]
    else:
        df = df[df.is_answer]
    # ignoring any non-fitted utterances within hearings

    states = df[state_label].unique()  # list of possible transition states

    transition_counts_dict = generate_empty_transition_dict(
        states
    )  # initialize

    for hearing_id in unique_convos:
        this_hearing_dict = subset_hearing(hearing_id, df)

        transition_counts_dict = count_transition_probs(
            state_label, this_hearing_dict, transition_counts_dict
        )

    transition_counts_mat = pd.DataFrame.from_dict(
        transition_counts_dict, orient="index"
    )

    return transition_counts_mat


def mat_states_only(transition_counts_mat: pd.DataFrame):
    """
    Drop the start/end states in order to compute normalization
    """

    transition_counts_mat_states_only = transition_counts_mat.drop(
        [-1], axis=0
    ).drop([10000], axis=1)

    return transition_counts_mat_states_only


def mat_normalized_rows(transition_counts_mat_states_only: pd.DataFrame):
    """
    Converts counts to percents normalized by row
    Assumes that mat_states_only() has been used to
    drop "start"/"end" prior to this function
    """

    transition_counts_mat_states_only[
        "totals"
    ] = transition_counts_mat_states_only.sum(axis=0)

    for c in transition_counts_mat_states_only.columns:
        transition_counts_mat_states_only[
            c
        ] = transition_counts_mat_states_only[c].div(
            transition_counts_mat_states_only["totals"]
        )

    return transition_counts_mat_states_only.drop(["totals"], axis=1)
