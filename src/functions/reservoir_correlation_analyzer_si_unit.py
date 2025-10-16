import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class ReservoirCorrelationAnalyzerSI:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_selected = None
        self.corr_matrix = None

        # Empirical → SI conversion map
        self.conversions = {
            "reservoir_pressure_initial (psi)": ("reservoir_pressure_initial (MPa)", 0.00689476),
            "bottomhole_pressure_psi (psi)": ("bottomhole_pressure (MPa)", 0.00689476),
            "permeability_md (mD)": ("permeability (m²)", 9.86923e-16),
            "net_pay_thickness_ft (ft)": ("net_pay_thickness (m)", 0.3048),
            "bubble_point_pressure (psi)": ("bubble_point_pressure (MPa)", 0.00689476),
            "porosity_percent (%)": ("porosity (fraction)", 0.01),
            "choke_size_percent (%)": ("choke_size (fraction)", 0.01),
            "oil_rate_bopd (BOPD)": ("oil_rate (m³/day)", 0.158987),
            "gas_rate_mscf_day (MSCF/day)": ("gas_rate (m³/s)", 0.00032774),
            "water_rate_bwpd (BWPD)": ("water_rate (m³/day)", 0.158987),
            "wellhead_temperature_c (°C)": ("wellhead_temperature (K)", lambda x: x + 273.15)
        }

        # Define base features and targets
        self.base_features = [
            "reservoir_pressure_initial (psi)",
            "bottomhole_pressure_psi (psi)",
            "permeability_md (mD)",
            "net_pay_thickness_ft (ft)",
            "bubble_point_pressure (psi)",
            "porosity_percent (%)",
            "choke_size_percent (%)"
        ]
        self.targets = [
            "oil_rate_bopd (BOPD)",
            "gas_rate_mscf_day (MSCF/day)",
            "water_rate_bwpd (BWPD)"
        ]
        self.selected_features = self.base_features + self.targets

    def convert_to_si(self):
        """Convert all recognized empirical columns to SI units."""
        for col, (new_col, factor) in self.conversions.items():
            if col in self.df.columns:
                if callable(factor):
                    self.df[new_col] = self.df[col].apply(factor)
                else:
                    self.df[new_col] = self.df[col] * factor

    def preprocess(self):
        """Filter relevant columns and drop NaNs."""
        self.convert_to_si()
        si_cols = [v[0] for k, v in self.conversions.items() if k in self.df.columns]
        self.df_selected = self.df[si_cols].dropna()

    def calculate_correlation(self):
        """Compute correlation matrix."""
        if self.df_selected is None:
            self.preprocess()
        if self.df_selected.empty:
            raise ValueError("No data available after preprocessing. Check input or conversions.")
        self.corr_matrix = self.df_selected.corr()
        return self.corr_matrix

    def save_correlation_excel(self, filename="./src/plot/cor/reservoir_correlation_matrix_SI.xlsx"):
        """Save correlation matrix to Excel."""
        if self.corr_matrix is None:
            self.calculate_correlation()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.corr_matrix.to_excel(filename)
        print(f"✅ Correlation matrix (SI) saved to {filename}")

    def plot_heatmap(self, filename="./src/plot/cor/reservoir_heatmap_SI.png", mask=False):
        """Plot correlation heatmap (in SI units)."""
        if self.corr_matrix is None:
            self.calculate_correlation()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(12, 10))

        # Mask upper triangle (optional)
        mask_matrix = np.triu(np.ones_like(self.corr_matrix, dtype=bool)) if mask else None

        sns.heatmap(
            self.corr_matrix,
            mask=mask_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            annot_kws={"size": 12},
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
        )

        plt.title("Correlation Heatmap (Reservoir Parameters + Rates, SI Units)", fontsize=18, pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    def plot_target_correlation(self, target, filename="./src/plot/cor/target_correlation_bar_SI.png"):
        """Plot correlation of all features with a specific target (in SI units)."""
        if self.corr_matrix is None:
            self.calculate_correlation()

        if target not in self.df_selected.columns:
            raise ValueError(f"Target '{target}' not found in dataset.")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Extract correlations with target
        target_corr = self.corr_matrix[target].drop(target)

        # Sort by absolute correlation strength
        target_corr_sorted = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)

        # Plot bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x=target_corr_sorted.values, y=target_corr_sorted.index, palette="coolwarm")

        # Axis labels and title
        plt.xlabel(f"Correlation with {target}", fontsize=14, labelpad=10)
        plt.ylabel("Feature", fontsize=14, labelpad=10)
        plt.title(f"Feature Correlation with {target} (SI Units)", fontsize=16, pad=15)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
