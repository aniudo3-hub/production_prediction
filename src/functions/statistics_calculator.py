import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import os

class StatisticsCalculator:
    def __init__(self, df: pd.DataFrame, group_col=None):
        """
        Initialize with a DataFrame.
        group_col: optional column name to group by (e.g., 'well_id')
        """
        self.df = df
        self.group_col = group_col

    def compute_column_statistics(self, data: pd.Series):
        """
        Compute statistics for a single numeric column.
        """
        data = data.dropna()
        if len(data) == 0:
            return None

        mean = data.mean()
        std = data.std()
        min_val = data.min()
        q25 = data.quantile(0.25)
        q50 = data.quantile(0.50)
        q75 = data.quantile(0.75)
        max_val = data.max()

        # Confidence Intervals
        n = len(data)
        se = std / np.sqrt(n) if n > 1 else 0

        ci90 = stats.t.interval(0.90, df=n-1, loc=mean, scale=se) if n > 1 else (mean, mean)
        ci95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=se) if n > 1 else (mean, mean)
        ci99 = stats.t.interval(0.99, df=n-1, loc=mean, scale=se) if n > 1 else (mean, mean)

        return {
            "Mean": mean,
            "Std": std,
            "Min": min_val,
            "25%": q25,
            "50%": q50,
            "75%": q75,
            "Max": max_val,
            "90% CI": f"[{ci90[0]:.2f}, {ci90[1]:.2f}]",
            "95% CI": f"[{ci95[0]:.2f}, {ci95[1]:.2f}]",
            "99% CI": f"[{ci99[0]:.2f}, {ci99[1]:.2f}]"
        }

    def compute_all(self):
        """
        Compute statistics for all numeric columns, optionally grouped by group_col.
        """
        results = {}

        if self.group_col:
            grouped = self.df.groupby(self.group_col)
            for group_name, group_df in grouped:
                group_stats = {}
                for col in group_df.select_dtypes(include=[np.number]).columns:
                    stats_dict = self.compute_column_statistics(group_df[col])
                    if stats_dict:
                        group_stats[col] = stats_dict
                results[group_name] = pd.DataFrame(group_stats).T
            return results  # dictionary of DataFrames (one per group)

        else:
            stats_dicts = {}
            for col in self.df.select_dtypes(include=[np.number]).columns:
                stats_dict = self.compute_column_statistics(self.df[col])
                if stats_dict:
                    stats_dicts[col] = stats_dict
            return pd.DataFrame(stats_dicts).T

    def save_to_excel(self, output_file="statistics_summary.xlsx"):
        """
        Save statistics to Excel (handles grouped and ungrouped cases).
        """
        results = self.compute_all()

             
        import pandas as pd
import numpy as np
from scipy import stats

class StatisticsCalculator:
    def __init__(self, df: pd.DataFrame, group_col=None):
        """
        Initialize with a DataFrame.
        group_col: optional column name to group by (e.g., 'well_id')
        """
        self.df = df
        self.group_col = group_col

    def compute_column_statistics(self, data: pd.Series):
        """
        Compute statistics for a single numeric column.
        """
        data = data.dropna()
        if len(data) == 0:
            return None

        mean = data.mean()
        std = data.std()
        min_val = data.min()
        q25 = data.quantile(0.25)
        q50 = data.quantile(0.50)
        q75 = data.quantile(0.75)
        max_val = data.max()

        # Confidence Intervals
        n = len(data)
        se = std / np.sqrt(n) if n > 1 else 0

        ci90 = stats.t.interval(0.90, df=n-1, loc=mean, scale=se) if n > 1 else (mean, mean)
        ci95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=se) if n > 1 else (mean, mean)
        ci99 = stats.t.interval(0.99, df=n-1, loc=mean, scale=se) if n > 1 else (mean, mean)

        return {
            "Mean": mean,
            "Std": std,
            "Min": min_val,
            "25%": q25,
            "50%": q50,
            "75%": q75,
            "Max": max_val,
            "90% CI": f"[{ci90[0]:.2f}, {ci90[1]:.2f}]",
            "95% CI": f"[{ci95[0]:.2f}, {ci95[1]:.2f}]",
            "99% CI": f"[{ci99[0]:.2f}, {ci99[1]:.2f}]"
        }

    def compute_all(self):
        """
        Compute statistics for all numeric columns, optionally grouped by group_col.
        Returns DataFrames with statistics as rows.
        """
        if self.group_col:
            results = {}
            grouped = self.df.groupby(self.group_col)
            for group_name, group_df in grouped:
                col_stats = {}
                for col in group_df.select_dtypes(include=[np.number]).columns:
                    stats_dict = self.compute_column_statistics(group_df[col])
                    if stats_dict:
                        col_stats[col] = stats_dict
                results[group_name] = pd.DataFrame(col_stats)  # ✅ stats as rows
            return results
        else:
            col_stats = {}
            for col in self.df.select_dtypes(include=[np.number]).columns:
                stats_dict = self.compute_column_statistics(self.df[col])
                if stats_dict:
                    col_stats[col] = stats_dict
            return pd.DataFrame(col_stats)  # ✅ stats as rows

    def save_to_excel(self, output_file="statistics_summary.xlsx"):
        """
        Save statistics to Excel (handles grouped and ungrouped cases).
        """
        results = self.compute_all()

        try:
            with pd.ExcelWriter(output_file) as writer:
                if self.group_col:
                    for group_name, df_stats in results.items():
                        df_stats.to_excel(writer, sheet_name=str(group_name))
                else:
                    results.to_excel(writer, sheet_name="Overall")
        except PermissionError:
            alt_file = os.path.join("./src/statistical", "statistical_data_new.xlsx")
            results.to_excel(alt_file)
            
        return results



