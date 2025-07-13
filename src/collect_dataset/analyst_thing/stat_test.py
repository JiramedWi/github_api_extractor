import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# Reusable StatisticalTester class
class StatisticalTester:
    def __init__(self, df, factors, value_column="overall_rank", logfile=None):
        self.df = df.copy()
        self.factors = factors
        self.value_column = value_column
        self.logfile = logfile

    def run_tests(self):
        results = {}
        output_lines = []
        for factor in self.factors:
            result = self._test_factor(factor)
            results[factor] = result
            output_lines.append(self._format_result(factor, result))

        if self.logfile:
            with open(self.logfile, "w") as f:
                f.write("\n".join(output_lines))

        return results

    def _test_factor(self, factor):
        group_counts = self.df[factor].nunique()
        other_columns = [col for col in self.factors if col != factor]

        self.df["combo_id"] = self.df[other_columns].astype(str).agg('_'.join, axis=1)
        valid_combos = self.df.groupby("combo_id")[factor].nunique()
        complete_combos = valid_combos[valid_combos == group_counts].index
        filtered_df = self.df[self.df["combo_id"].isin(complete_combos)]

        if filtered_df.empty:
            return {"message": "Not enough complete groups for valid comparison"}

        pivot_df = filtered_df.pivot_table(index="combo_id", columns=factor, values=self.value_column).dropna()

        if pivot_df.shape[0] < 2:
            return {"message": "Not enough comparable groups after filtering"}

        mean_ranks = pivot_df.mean()
        best_technique = mean_ranks.idxmin()

        if group_counts == 2:
            col1, col2 = pivot_df.columns
            stat, p = wilcoxon(pivot_df[col1], pivot_df[col2])
            return {
                "test": "Wilcoxon Signed-Rank Test",
                "techniques": [col1, col2],
                "statistic": stat,
                "p-value": p,
                "N groups": pivot_df.shape[0],
                "best_technique": best_technique,
                "mean_ranks": mean_ranks.to_dict()
            }
        elif group_counts >= 3:
            stat, p = friedmanchisquare(*[pivot_df[col].values for col in pivot_df.columns])
            return {
                "test": "Friedman Test",
                "techniques": list(pivot_df.columns),
                "statistic": stat,
                "p-value": p,
                "N groups": pivot_df.shape[0],
                "best_technique": best_technique,
                "mean_ranks": mean_ranks.to_dict()
            }
        else:
            return {"message": "Unexpected group count"}

    def _format_result(self, factor, result):
        lines = [f"\n[{factor}]"]
        if "message" in result:
            lines.append(f"Message: {result['message']}")
        else:
            lines.append(f"Test: {result['test']}")
            lines.append(f"Techniques: {', '.join(map(str, result['techniques']))}")
            lines.append(f"Statistic: {result['statistic']:.4f}")
            lines.append(f"P-value: {result['p-value']:.4f}")
            interpretation = "Significant difference detected" if result[
                                                                      'p-value'] < 0.05 else "No significant difference"
            lines.append(f"Interpretation: {interpretation}")
            lines.append(f"Best Technique (lowest average rank): {result['best_technique']}")
            lines.append("Average Ranks:")
            for tech, val in result["mean_ranks"].items():
                lines.append(f"  {tech}: {val:.4f}")
        return "\n".join(lines)


# Your dataset dictionary
file_path_prefix = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/tables"  # Update as needed

dataset_dict = {
    "code_related": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_code_related.csv", na_filter=False),
    "dependencies": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_dependencies.csv", na_filter=False),
    "issue_in_test_step": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_issue_in_test_step.csv",
                                      na_filter=False),
    "test_execution": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_test_execution.csv", na_filter=False),
    "test_semantic_smell": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_test_semantic_smell.csv",
                                       na_filter=False)
}

# Factors to test
factors = ["Textual feature", "Stem lemma", "N-gram", "Topic modeling", "Imba handling"]

# Loop through datasets and apply the tester
for label, df in dataset_dict.items():
    print(f"\n=== Running tests for {label} ===")
    logfile = f"/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/stat_test/{label}_statistical_results.txt"
    tester = StatisticalTester(df, factors, logfile=logfile)
    tester.run_tests()
    print(f"Results saved to {logfile}")
