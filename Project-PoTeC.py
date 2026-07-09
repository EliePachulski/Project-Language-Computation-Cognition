# Generated from: Project-PoTeC.ipynb
# Converted at: 2026-07-09T12:16:20.723Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Analysis Setup


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

rm_files = list(
    Path(
        "../data/PoTeC/precomputed_reading_measures/reading_measures_merged"
    ).glob("*.tsv")
)

print("Number of files:", len(rm_files))

for i, file in enumerate(rm_files):
    print(i, file.name)

# Inspect the first reading-measure file
sample_file = rm_files[0]

sample_df = pd.read_csv(sample_file, sep="\t")

print("Sample file:", sample_file.name)
print("Shape:", sample_df.shape)

sample_df.head()

print("Columns:")
for col in sample_df.columns:
    print(col)

# ## Selecting variables for the project
# 
# 


# Load and concatenate all reading-measure files
dfs = []

for file in rm_files:
    df_tmp = pd.read_csv(file, sep="\t")
    df_tmp["source_file"] = file.name
    dfs.append(df_tmp)

reading_df = pd.concat(dfs, ignore_index=True)

print("Combined reading dataframe shape:", reading_df.shape)
print("Number of source files:", reading_df["source_file"].nunique())

reading_df.head()

# Text-level surprisal estimates provided in PoTeC
surprisal_cols = [
    "text_surprisal_gpt2-base",
    "text_surprisal_gpt2-large",
    "text_surprisal_llama-7b",
    "text_surprisal_llama-13b",
    "text_surprisal_bert-base"
]

# Check that all surprisal columns exist
missing_surprisal_cols = [col for col in surprisal_cols if col not in reading_df.columns]
print("Missing surprisal columns:", missing_surprisal_cols)

# Create averaged text-level surprisal
reading_df["mean_text_surprisal"] = reading_df[surprisal_cols].mean(axis=1, skipna=True)

reading_df[["word"] + surprisal_cols + ["mean_text_surprisal"]].head()

# Main eye-movement measures
measures = [
    "FFD",
    "FPRT",
    "TFT",
    "TFC",
    "RRT",
    "RPD_exc",
    "TRC_out"
]

# Word-level controls for the difficulty-adjusted analysis
controls = [
    "word_length",
    "lemma_frequency_normalized",
    "mean_text_surprisal"
]

# Variables needed for grouping and interpretation
metadata_cols = [
    "reader_id",
    "text_id",
    "trial",
    "word",
    "word_index_in_text",
    "text_domain",
    "text_domain_numeric",
    "reader_discipline_numeric",
    "level_of_studies_numeric",
    "expert_reading_label_numeric",
    "is_expert_technical_term",
    "is_general_technical_term"
]

selected_cols = metadata_cols + controls + measures

analysis_df = reading_df[selected_cols].copy()

print("Analysis dataframe shape:", analysis_df.shape)
analysis_df.head()

# ## Inspecting the coding of participant and text variables
# 


coding_cols = [
    "text_domain",
    "text_domain_numeric",
    "reader_discipline_numeric",
    "level_of_studies_numeric",
    "expert_reading_label_numeric"
]

for col in coding_cols:
    print("\n" + col)
    print(analysis_df[col].value_counts(dropna=False).sort_index())

# Mapping between text_domain and text_domain_numeric
text_domain_mapping = (
    analysis_df[["text_domain", "text_domain_numeric"]]
    .drop_duplicates()
    .sort_values(["text_domain_numeric", "text_domain"])
)

text_domain_mapping

diagnostic = (
    analysis_df
    .groupby([
        "reader_id",
        "reader_discipline_numeric",
        "level_of_studies_numeric",
        "text_domain",
        "text_domain_numeric",
        "expert_reading_label_numeric"
    ])
    .size()
    .reset_index(name="n_rows")
)

diagnostic.head(20)

diagnostic_summary = (
    diagnostic
    .groupby([
        "reader_discipline_numeric",
        "level_of_studies_numeric",
        "text_domain",
        "text_domain_numeric",
        "expert_reading_label_numeric"
    ])
    ["reader_id"]
    .nunique()
    .reset_index(name="n_readers")
    .sort_values([
        "reader_discipline_numeric",
        "level_of_studies_numeric",
        "text_domain_numeric",
        "expert_reading_label_numeric"
    ])
)

diagnostic_summary

# ## Defining the four reading classes
# 
# The numeric coding is:
# - `text_domain_numeric`: 0 = biology, 1 = physics
# - `reader_discipline_numeric`: 0 = biology, 1 = physics
# - `level_of_studies_numeric`: 0 = undergraduate, 1 = graduate
# 
# Using these variables, we define four reading classes:
# 1. Expert: graduate reader reading in their own discipline.
# 2. Graduate out-of-domain: graduate reader reading in the other discipline.
# 3. Undergraduate in-domain: undergraduate reader reading in their own discipline.
# 4. Undergraduate out-of-domain: undergraduate reader reading in the other discipline.


def assign_reading_class(row):
    same_domain = row["reader_discipline_numeric"] == row["text_domain_numeric"]
    is_graduate = row["level_of_studies_numeric"] == 1

    if is_graduate and same_domain:
        return "Expert"
    elif is_graduate and not same_domain:
        return "Graduate out-of-domain"
    elif (not is_graduate) and same_domain:
        return "Undergraduate in-domain"
    else:
        return "Undergraduate out-of-domain"


analysis_df["reading_class"] = analysis_df.apply(assign_reading_class, axis=1)

class_order = [
    "Expert",
    "Graduate out-of-domain",
    "Undergraduate in-domain",
    "Undergraduate out-of-domain"
]

analysis_df["reading_class"] = pd.Categorical(
    analysis_df["reading_class"],
    categories=class_order,
    ordered=True
)

analysis_df[[
    "reader_id",
    "reader_discipline_numeric",
    "level_of_studies_numeric",
    "text_domain",
    "text_domain_numeric",
    "expert_reading_label_numeric",
    "reading_class"
]].drop_duplicates().head(20)

check_expert_label = (
    analysis_df
    .groupby(["reading_class", "expert_reading_label_numeric"], observed=True)
    .size()
    .reset_index(name="n_rows")
)

check_expert_label

class_counts_words = (
    analysis_df["reading_class"]
    .value_counts()
    .reindex(class_order)
)

class_counts_words

class_counts_readers = (
    analysis_df
    .groupby("reading_class", observed=True)["reader_id"]
    .nunique()
    .reindex(class_order)
)

class_counts_readers

# Check how the four reading classes are built from discipline, level, and text domain

class_validation = (
    analysis_df
    .groupby([
        "reader_discipline_numeric",
        "level_of_studies_numeric",
        "text_domain",
        "reading_class"
    ], observed=True)
    .agg(
        n_readers=("reader_id", "nunique"),
        n_rows=("word", "size")
    )
    .reset_index()
    .sort_values([
        "level_of_studies_numeric",
        "reader_discipline_numeric",
        "text_domain"
    ])
)

class_validation

# Add readable labels for interpretation

discipline_map = {
    0: "Biology reader",
    1: "Physics reader"
}

level_map = {
    0: "Undergraduate",
    1: "Graduate"
}

class_validation["reader_discipline"] = class_validation["reader_discipline_numeric"].map(discipline_map)
class_validation["level_of_studies"] = class_validation["level_of_studies_numeric"].map(level_map)

class_validation = class_validation[
    [
        "reader_discipline",
        "level_of_studies",
        "text_domain",
        "reading_class",
        "n_readers",
        "n_rows"
    ]
]

class_validation

# This table validates the construction of the four reading classes. Graduate readers are classified as Expert when the text domain matches their discipline and as Graduate out-of-domain otherwise. Undergraduate readers are classified as Undergraduate in-domain when the text domain matches their discipline and as Undergraduate out-of-domain otherwise.


# ## Cleaning the seven eye-movement measures
# 


missing_measures = (
    analysis_df[measures]
    .isna()
    .sum()
    .to_frame("n_missing")
)

missing_measures["missing_percent"] = (
    100 * missing_measures["n_missing"] / len(analysis_df)
).round(2)

missing_measures

missing_controls = (
    analysis_df[controls]
    .isna()
    .sum()
    .to_frame("n_missing")
)

missing_controls["missing_percent"] = (
    100 * missing_controls["n_missing"] / len(analysis_df)
).round(2)

missing_controls

# # Raw eye-movement measures by reading class
# 
# 




# Raw means table
raw_means = (
    analysis_df
    .groupby("reading_class", observed=True)[measures]
    .mean()
    .reindex(class_order)
)

# Rename columns with units for the report
measure_labels_with_units = {
    "FFD": "FFD (ms)",
    "FPRT": "FPRT (ms)",
    "TFT": "TFT (ms)",
    "TFC": "TFC (count)",
    "RRT": "RRT (ms)",
    "RPD_exc": "RPD_exc (ms)",
    "TRC_out": "TRC_out (count)"
}

table2_raw = (
    raw_means
    .rename(columns=measure_labels_with_units)
    .round(1)
)

table2_raw

# # Adjusted eye-movement profiles
# 
# 


from sklearn.linear_model import LinearRegression

# Create a copy of the analysis dataframe
adjusted_df = analysis_df.copy()

# Fit one control-only regression per eye-movement measure
# Adjusted value = residual + global mean of the original measure
for measure in measures:
    model_cols = controls + [measure]
    df_model = adjusted_df[model_cols].dropna()

    X = df_model[controls]
    y = df_model[measure]

    model = LinearRegression()
    model.fit(X, y)

    predicted = model.predict(X)
    residuals = y - predicted

    adjusted_col = f"adj_{measure}"
    adjusted_df.loc[df_model.index, adjusted_col] = residuals + y.mean()

adjusted_measures = [f"adj_{m}" for m in measures]

adjusted_df[measures + adjusted_measures].head()



# Adjusted means table
adjusted_means = (
    adjusted_df
    .groupby("reading_class", observed=True)[adjusted_measures]
    .mean()
    .reindex(class_order)
)

# Rename adjusted columns with clean labels and units
adjusted_measure_labels_with_units = {
    "adj_FFD": "Adj FFD (ms)",
    "adj_FPRT": "Adj FPRT (ms)",
    "adj_TFT": "Adj TFT (ms)",
    "adj_TFC": "Adj TFC (count)",
    "adj_RRT": "Adj RRT (ms)",
    "adj_RPD_exc": "Adj RPD_exc (ms)",
    "adj_TRC_out": "Adj TRC_out (count)"
}

table3_adjusted = (
    adjusted_means
    .rename(columns=adjusted_measure_labels_with_units)
    .round(1)
)

table3_adjusted

#  Raw vs adjusted profile Comparizon
# 
# 


# Raw class-level means with original column names

raw_means_internal = (

    analysis_df

    .groupby("reading_class", observed=True)[measures]

    .mean()

    .reindex(class_order)

)

# Adjusted class-level means with original measure names

adjusted_means_internal = (

    adjusted_df

    .groupby("reading_class", observed=True)[adjusted_measures]

    .mean()

    .reindex(class_order)

)

# Rename adjusted columns from adj_FFD -> FFD, etc.

adjusted_means_internal.columns = measures

# Difference between adjusted and raw class-level means

adjustment_diff = adjusted_means_internal - raw_means_internal

adjustment_diff = adjustment_diff.round(3)

adjustment_diff

# Raw class-level means with original measure names
raw_means_internal = (
    analysis_df
    .groupby("reading_class", observed=True)[measures]
    .mean()
    .reindex(class_order)
)

# Adjusted class-level means
adjusted_means_internal = (
    adjusted_df
    .groupby("reading_class", observed=True)[adjusted_measures]
    .mean()
    .reindex(class_order)
)

# Rename adjusted columns from adj_FFD -> FFD, etc.
adjusted_means_internal.columns = measures

# Absolute difference between adjusted and raw class-level means
absolute_diff = (adjusted_means_internal - raw_means_internal).abs()

# Relative difference in percentage
relative_diff_percent = (absolute_diff / raw_means_internal.abs()) * 100

# Mean relative change across the four reading classes
mean_relative_change = relative_diff_percent.mean(axis=0).sort_values(ascending=False)

mean_relative_change

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
ax.set_facecolor("white")

bar_color = "#E9967A"

ax.bar(
    mean_relative_change.index,
    mean_relative_change.values,
    color=bar_color,
    edgecolor="#8A4B3A",
    linewidth=0.8
)

ax.set_ylabel("Mean relative change after adjustment (%)")
ax.set_xlabel("Eye-movement measure")
ax.set_title("Relative effect of word-level controls on class-level means", pad=12)

ax.tick_params(axis="x", rotation=30)
for label in ax.get_xticklabels():
    label.set_ha("right")

ax.grid(axis="y", alpha=0.25, linewidth=0.8)
ax.grid(axis="x", visible=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("0.3")
ax.spines["bottom"].set_color("0.3")

plt.tight_layout()

plt.savefig(
    output_dir / "figure1_relative_adjustment_effect_by_measure.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# # Statistical comparison of reading classes
# 
# 


from scipy.stats import kruskal
import pandas as pd


# Aggregate raw measures at reader-text level
# Each row = one reader-text-reading_class average
raw_reader_text = (
    analysis_df
    .groupby(["reader_id", "text_id", "reading_class"], observed=True)[measures]
    .mean()
    .reset_index()
)

def kruskal_by_class(df, measure_col, class_col="reading_class"):
    groups = []

    for cls in class_order:
        values = df.loc[df[class_col] == cls, measure_col].dropna()
        groups.append(values)

    stat, p_value = kruskal(*groups)
    return stat, p_value

kw_rows = []

for measure in measures:
    stat, p_value = kruskal_by_class(raw_reader_text, measure)

    kw_rows.append({
        "Measure": measure,
        "Kruskal H": stat,
        "p-value": p_value
    })

kw_table = pd.DataFrame(kw_rows)

# Display formatting
kw_table_display = kw_table.copy()

kw_table_display["Kruskal H"] = kw_table_display["Kruskal H"].map(
    lambda x: f"{x:.2f}"
)

kw_table_display["p-value"] = kw_table_display["p-value"].map(
    lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
)


kw_table_display

from scipy.stats import wilcoxon, mannwhitneyu
from itertools import combinations
import pandas as pd


# All pairwise comparisons between the four reading classes
all_comparisons = list(combinations(class_order, 2))

# Aggregate raw measures at reader-class level
# Each reader contributes one mean per reading class
raw_reader_class = (
    analysis_df
    .groupby(["reader_id", "reading_class"], observed=True)[measures]
    .mean()
    .reset_index()
)

comparison_tables = {}

for class_a, class_b in all_comparisons:
    rows = []

    for measure in measures:
        df_a = (
            raw_reader_class[raw_reader_class["reading_class"] == class_a]
            [["reader_id", measure]]
            .dropna()
            .copy()
        )

        df_b = (
            raw_reader_class[raw_reader_class["reading_class"] == class_b]
            [["reader_id", measure]]
            .dropna()
            .copy()
        )

        readers_a = set(df_a["reader_id"])
        readers_b = set(df_b["reader_id"])

        # Paired test if both classes contain the same readers
        if readers_a == readers_b:
            paired_df = df_a.merge(
                df_b,
                on="reader_id",
                suffixes=("_A", "_B")
            )

            values_a = paired_df[f"{measure}_A"]
            values_b = paired_df[f"{measure}_B"]

            stat, p_value = wilcoxon(values_a, values_b)
            test_name = "Wilcoxon signed-rank"

        # Independent test otherwise
        else:
            values_a = df_a[measure]
            values_b = df_b[measure]

            stat, p_value = mannwhitneyu(
                values_a,
                values_b,
                alternative="two-sided"
            )

            test_name = "Mann-Whitney U"

        rows.append({
            "Measure": measure,
            "Test": test_name,
            "Mean A": values_a.mean(),
            "Mean B": values_b.mean(),
            "Difference A-B": values_a.mean() - values_b.mean(),
            "p-value": p_value
        })

    table = pd.DataFrame(rows)

    # Keep only useful columns
    table = table[
        [
            "Measure",
            "Test",
            "Mean A",
            "Mean B",
            "Difference A-B",
            "p-value"
        ]
    ]

    # Formatting:
    # means and differences = 2 decimals
    # p-values = 3 decimals, except very small p-values shown as <0.001
    table_display = table.copy()

    for col in ["Mean A", "Mean B", "Difference A-B"]:
        table_display[col] = table_display[col].map(lambda x: f"{x:.2f}")

    table_display["p-value"] = table_display["p-value"].map(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )

    comparison_name = f"{class_a} vs {class_b}"
    comparison_tables[comparison_name] = table_display

    # Save each comparison table
    safe_name = (
        comparison_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )



# Targeted class comparisons


for name, table in comparison_tables.items():
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    display(table)

# # Expert Words Analysis




# Mean eye-movement measures on expert technical words only
expert_words_means = (
    expert_words_df
    .groupby("reading_class", observed=True)[measures]
    .mean()
    .reindex(class_order)
)

# Rename columns with units
measure_labels_with_units = {
    "FFD": "FFD (ms)",
    "FPRT": "FPRT (ms)",
    "TFT": "TFT (ms)",
    "TFC": "TFC (count)",
    "RRT": "RRT (ms)",
    "RPD_exc": "RPD_exc (ms)",
    "TRC_out": "TRC_out (count)"
}

table_expert_words_means = (
    expert_words_means
    .rename(columns=measure_labels_with_units)
    .round(1)
)


table_expert_words_means

from scipy.stats import kruskal
import pandas as pd

# Aggregate expert technical words at reader-class level
expert_words_reader_class = (
    expert_words_df
    .groupby(["reader_id", "reading_class"], observed=True)[measures]
    .mean()
    .reset_index()
)

expert_word_kw_rows = []

for measure in measures:
    groups = []

    for cls in class_order:
        values = expert_words_reader_class.loc[
            expert_words_reader_class["reading_class"] == cls,
            measure
        ].dropna()

        groups.append(values)

    stat, p_value = kruskal(*groups)

    expert_word_kw_rows.append({
        "Measure": measure,
        "Kruskal H": stat,
        "p-value": p_value
    })

table_expert_words_kw = pd.DataFrame(expert_word_kw_rows)

# Display formatting
table_expert_words_kw_display = table_expert_words_kw.copy()

table_expert_words_kw_display["Kruskal H"] = (
    table_expert_words_kw_display["Kruskal H"]
    .map(lambda x: f"{x:.2f}")
)

table_expert_words_kw_display["p-value"] = (
    table_expert_words_kw_display["p-value"]
    .map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
)


table_expert_words_kw_display

from scipy.stats import wilcoxon, mannwhitneyu
from itertools import combinations
import pandas as pd

# All pairwise comparisons between the four reading classes
all_comparisons = list(combinations(class_order, 2))

expert_word_comparison_tables = {}

for class_a, class_b in all_comparisons:
    rows = []

    for measure in measures:
        df_a = (
            expert_words_reader_class[
                expert_words_reader_class["reading_class"] == class_a
            ][["reader_id", measure]]
            .dropna()
            .copy()
        )

        df_b = (
            expert_words_reader_class[
                expert_words_reader_class["reading_class"] == class_b
            ][["reader_id", measure]]
            .dropna()
            .copy()
        )

        readers_a = set(df_a["reader_id"])
        readers_b = set(df_b["reader_id"])

        # Paired test if both classes contain the same readers
        if readers_a == readers_b:
            paired_df = df_a.merge(
                df_b,
                on="reader_id",
                suffixes=("_A", "_B")
            )

            values_a = paired_df[f"{measure}_A"]
            values_b = paired_df[f"{measure}_B"]

            stat, p_value = wilcoxon(values_a, values_b)
            test_name = "Wilcoxon signed-rank"

        # Independent test otherwise
        else:
            values_a = df_a[measure]
            values_b = df_b[measure]

            stat, p_value = mannwhitneyu(
                values_a,
                values_b,
                alternative="two-sided"
            )

            test_name = "Mann-Whitney U"

        rows.append({
            "Measure": measure,
            "Test": test_name,
            "Mean A": values_a.mean(),
            "Mean B": values_b.mean(),
            "Difference A-B": values_a.mean() - values_b.mean(),
            "p-value": p_value
        })

    table = pd.DataFrame(rows)

    table = table[
        [
            "Measure",
            "Test",
            "Mean A",
            "Mean B",
            "Difference A-B",
            "p-value"
        ]
    ]

    # Formatting
    table_display = table.copy()

    for col in ["Mean A", "Mean B", "Difference A-B"]:
        table_display[col] = table_display[col].map(lambda x: f"{x:.2f}")

    table_display["p-value"] = table_display["p-value"].map(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )

    comparison_name = f"{class_a} vs {class_b}"
    expert_word_comparison_tables[comparison_name] = table_display

    # Save each comparison table
    safe_name = (
        comparison_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )



for comparison_name, table in expert_word_comparison_tables.items():
    print("\n" + "=" * 80)
    print(comparison_name)
    print("=" * 80)
    display(table)

# # Prediction Measures to Class


import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# Build one observation per reader-text pair.
# Each row contains the mean eye-movement measures for one reader reading one text.
prediction_df = (
    analysis_df
    .groupby(
        [
            "reader_id",
            "text_id",
            "reading_class",
            "reader_discipline_numeric",
            "level_of_studies_numeric",
            "text_domain_numeric"
        ],
        observed=True
    )[measures]
    .mean()
    .reset_index()
)

# Prediction target 1: the full four-class reading label
prediction_df["target_reading_class"] = prediction_df["reading_class"].astype(str)

# Prediction target 2: whether the text is in-domain or out-of-domain
prediction_df["target_domain_match"] = np.where(
    prediction_df["reader_discipline_numeric"] == prediction_df["text_domain_numeric"],
    "In-domain",
    "Out-of-domain"
)

# Prediction target 3: whether the reader is graduate or undergraduate
prediction_df["target_academic_level"] = np.where(
    prediction_df["level_of_studies_numeric"] == 1,
    "Graduate",
    "Undergraduate"
)

print("Prediction dataframe shape:", prediction_df.shape)

prediction_df[
    [
        "reader_id",
        "text_id",
        "reading_class",
        "target_domain_match",
        "target_academic_level",
        "target_reading_class"
    ]
].head()

# We compare logistic regression to a simple majority-class baseline.
# Logistic regression is useful because it gives both prediction scores and interpretable coefficients.
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])
}

# We use balanced accuracy and macro-F1 because some classes are not equally represented.
scoring = {
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro"
}

# GroupKFold keeps all observations from the same reader in the same fold.
# This prevents the same reader from appearing in both training and test data.
cv = GroupKFold(n_splits=5)


def evaluate_prediction_task(df, target_col, task_name):
    """
    Evaluate baseline and logistic regression for one prediction task.
    """
    X = df[measures].copy()
    y = df[target_col].copy()
    groups = df["reader_id"].copy()

    rows = []

    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scoring
        )

        rows.append({
            "Task": task_name,
            "Model": model_name,
            "Balanced accuracy": scores["test_balanced_accuracy"].mean(),
            "Macro-F1": scores["test_macro_f1"].mean()
        })

    return pd.DataFrame(rows)

# Evaluate the three prediction tasks:
# 1. Domain match
# 2. Academic level
# 3. Full four-class reading label
prediction_results_report = pd.concat(
    [
        evaluate_prediction_task(
            prediction_df,
            target_col="target_domain_match",
            task_name="Domain match"
        ),
        evaluate_prediction_task(
            prediction_df,
            target_col="target_academic_level",
            task_name="Academic level"
        ),
        evaluate_prediction_task(
            prediction_df,
            target_col="target_reading_class",
            task_name="Four-class reading class"
        )
    ],
    ignore_index=True
)

# Round scores for the report table
prediction_results_report[["Balanced accuracy", "Macro-F1"]] = (
    prediction_results_report[["Balanced accuracy", "Macro-F1"]]
    .round(2)
)

prediction_results_report

def collect_logistic_coefficients(df, target_col, target_positive_label):
    """
    Fit logistic regression across GroupKFold splits and collect coefficients.

    For binary tasks, sklearn stores coefficients for classes_[1].
    If classes_[1] is not the label we want to interpret as positive,
    we reverse the sign of the coefficients.
    """
    X = df[measures].copy()
    y = df[target_col].copy()
    groups = df["reader_id"].copy()

    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs"
            ))
        ])

        model.fit(X_train, y_train)

        clf = model.named_steps["clf"]
        coef_values = clf.coef_[0]

        # Make coefficients correspond to the chosen positive class.
        if clf.classes_[1] != target_positive_label:
            coef_values = -coef_values

        for measure, coef in zip(measures, coef_values):
            fold_rows.append({
                "Fold": fold_idx,
                "Measure": measure,
                "Coefficient": coef
            })

    coef_folds = pd.DataFrame(fold_rows)

    # Average coefficients across folds
    coef_summary = (
        coef_folds
        .groupby("Measure")
        .agg(
            mean_coef=("Coefficient", "mean"),
            std_coef=("Coefficient", "std"),
            n_folds=("Coefficient", "size")
        )
        .reset_index()
    )

    # 95% confidence interval across folds
    coef_summary["ci95"] = (
        1.96 * coef_summary["std_coef"] / np.sqrt(coef_summary["n_folds"])
    )

    # Keep the same order as the measure list
    coef_summary["Measure"] = pd.Categorical(
        coef_summary["Measure"],
        categories=measures,
        ordered=True
    )

    return coef_summary.sort_values("Measure")


# Coefficients for domain prediction:
# positive = Out-of-domain
domain_coef_summary = collect_logistic_coefficients(
    prediction_df,
    target_col="target_domain_match",
    target_positive_label="Out-of-domain"
)

# Coefficients for academic-level prediction:
# positive = Graduate
level_coef_summary = collect_logistic_coefficients(
    prediction_df,
    target_col="target_academic_level",
    target_positive_label="Graduate"
)

import matplotlib.pyplot as plt

# Clean plotting style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.edgecolor": "0.25",
    "axes.linewidth": 0.8
})

fig, axes = plt.subplots(
    1, 2,
    figsize=(13.5, 5),
    sharey=True,
    facecolor="white"
)

plot_specs = [
    (axes[0], domain_coef_summary, "Domain prediction"),
    (axes[1], level_coef_summary, "Academic level prediction")
]

point_color = "#C76E6E"
error_color = "#8F4A4A"
zero_color = "#555555"

for ax, coef_summary, title in plot_specs:
    x = np.arange(len(measures))

    # Plot mean logistic regression coefficients with 95% CI across folds
    ax.errorbar(
        x,
        coef_summary["mean_coef"],
        yerr=coef_summary["ci95"],
        fmt="o",
        markersize=6,
        capsize=4,
        capthick=1.2,
        elinewidth=1.3,
        color=point_color,
        ecolor=error_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=0
    )

    # Zero line: positive and negative coefficients are interpreted relative to this line
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.1,
        color=zero_color,
        alpha=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=35, ha="right")

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("Eye-movement measure")

    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.35")
    ax.spines["bottom"].set_color("0.35")

axes[0].set_ylabel("Logistic regression coefficient")

fig.suptitle(
    "Logistic regression coefficients for prediction tasks",
    fontsize=15,
    fontweight="bold",
    y=1.04
)

plt.tight_layout()


plt.show()