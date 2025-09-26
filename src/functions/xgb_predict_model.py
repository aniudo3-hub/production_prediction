import os
import joblib
import re
import numpy as np
import base64
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import shap
from io import BytesIO
XGB_EXPLAIN_DIR = "./src/explain/xgb"

class XGBModel:
    def __init__(self, target_cols, model_path="src/created_model/xgb/xgb_model.pkl"):
        self.target_cols = target_cols
        self.features = None
        self.model_path = model_path
        self.model = None
        self.bool_cols = ["maintenance_event", "shutdown_flag"]
        self.encoders = {}  # store label encoders for categorical columns

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)

        # --- boolean fix ---
        for b in self.bool_cols:
            if b in df.columns:
                df[b] = df[b].astype(float).fillna(0).astype(int)

        # --- categorical fix ---
        cat_cols = df.select_dtypes(include=["object"]).columns.difference(["well_id", "timestamp"])
        for col in cat_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # transform with existing encoder, handle unseen labels
                df[col] = df[col].map(lambda x: x if x in self.encoders[col].classes_ else "UNK")
                self.encoders[col].classes_ = np.append(self.encoders[col].classes_, "UNK")
                df[col] = self.encoders[col].transform(df[col].astype(str))

        # --- numeric fix ---
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df.groupby("well_id")[num_cols].transform(lambda g: g.ffill().bfill())
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df
    
    def _get_features(self, df: pd.DataFrame):
        """Detect features dynamically from df"""
        exclude = set(self.target_cols + ["well_id", "timestamp"])
        return [c for c in df.columns if c not in exclude]

    def train(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        df = self.preprocess(df)

        if self.features is None:
            self.features = self._get_features(df)

        X = df[self.features].values
        y = df[self.target_cols].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

        base_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
        )

        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

        self.model = model

        # Explain with SHAP
        # result = self.explain_with_shap( X_train, X_test, y_test, index=5, output_prefix=XGB_EXPLAIN_DIR)
        # result = self.explain(df)
        return model

    def predict(self, df: pd.DataFrame, save_dir="src/predictions/xgb", save=True, filename="predictions_xgb.xlsx") -> pd.DataFrame:
        if self.model is None:
            self.model = joblib.load(self.model_path)

        df_out = self.preprocess(df)

        if self.features is None:
            self.features = self._get_features(df_out)

        X = df_out[self.features].values
        y_pred = self.model.predict(X)
    
        for col_i, t in enumerate(self.target_cols):
            df_out[f"pred_{t}"] = y_pred[:, col_i]

        if save:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, filename)
            df_out.to_excel(out_path, index=False)

        return df_out

    def decision_rules(preds: pd.DataFrame):
        actions = []
        for _, row in preds.iterrows():
            if row["water_rate_bwpd_pred"] > 5000:  
                actions.append("High water cut detected – recommend water shutoff treatment")
            elif row["gas_rate_mscf_day_pred"] < 200:
                actions.append("Low gas rate – possible gas lift issue")
            else:
                actions.append("Operating normally")
        return actions

    def explain_with_shap(self, X_train, X_test, y_test, index=0, output_prefix="shap_results"):
        """
        Explain predictions of a trained model (single or multi-output) using SHAP.
        Saves SHAP plots and results to Excel.
        """
        model = self.model
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

        # 🔹 Ensure numeric input + preserve feature names
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
            X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
            feature_names = X_train.columns
        else:  # numpy arrays → wrap into DataFrames with synthetic names
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train = pd.DataFrame(np.asarray(X_train, dtype=np.float64), columns=feature_names)
            X_test = pd.DataFrame(np.asarray(X_test, dtype=np.float64), columns=feature_names)

        # ---- Handle MultiOutput Models ----
        if isinstance(model, MultiOutputRegressor):
            all_results = {}

            for i, estimator in enumerate(model.estimators_):
                target_name = (
                    y_test.columns[i] if hasattr(y_test, "columns") else f"target_{i}"
                )

                # Predictions
                y_pred = estimator.predict(X_test)

                # Handle DataFrame vs ndarray
                if hasattr(y_test, "iloc"):
                    y_true_col = y_test.iloc[:, i] if y_test.ndim > 1 else y_test
                    y_true_single = y_test.iloc[index, i] if y_test.ndim > 1 else y_test.iloc[index]
                else:
                    y_true_col = y_test[:, i] if y_test.ndim > 1 else y_test
                    y_true_single = y_test[index, i] if y_test.ndim > 1 else y_test[index]

                mse = mean_squared_error(y_true_col, y_pred)

                # SHAP explainer for this target
                explainer = shap.Explainer(estimator, X_train)
                shap_values = explainer(X_test)

                # Save plots
                shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
                plt.savefig(f"{output_prefix}_{target_name}_summary.png", bbox_inches="tight")
                plt.close()

                shap.plots.force(shap_values[index], matplotlib=True, show=False)
                plt.savefig(f"{output_prefix}_{target_name}_force_{index}.png", bbox_inches="tight")
                plt.close()

                # Save SHAP values with feature names
                shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
                shap_df.insert(0, "Sample_Index", X_test.index)
                shap_df.to_excel(f"{output_prefix}_{target_name}_shap_values.xlsx", index=False)

                # Collect results
                all_results[target_name] = {
                    "prediction": self.to_serializable(y_pred[index]),
                    "true_value": self.to_serializable(y_true_single),
                    "error": self.to_serializable(abs(y_true_single - y_pred[index])),
                    "shap_values": self.to_serializable(shap_values[index].values)
                }

            return pd.DataFrame(all_results)

        # ---- Single-output Model ----
        else:
            # Predictions
            y_pred = model.predict(X_test)

            if hasattr(y_test, "iloc"):
                y_true_all = y_test
                y_true_single = y_test.iloc[index]
            else:
                y_true_all = y_test
                y_true_single = y_test[index]

            mse = mean_squared_error(y_true_all, y_pred)

            # SHAP explainer
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

            # Save plots
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.savefig(f"{output_prefix}_summary.png", bbox_inches="tight")
            plt.close()

            shap.plots.force(shap_values[index], matplotlib=True, show=False)
            plt.savefig(f"{output_prefix}_force_{index}.png", bbox_inches="tight")
            plt.close()

            # Save SHAP values with feature names
            shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
            shap_df.insert(0, "Sample_Index", X_test.index)
            shap_df.to_excel(f"{output_prefix}_shap_values.xlsx", index=False)

            result = {
                "prediction": self.to_serializable(y_pred[index]),
                "true_value": self.to_serializable(y_true_single),
                "error": self.to_serializable(abs(y_true_single - y_pred[index])),
                "shap_values": self.to_serializable(shap_values[index].values)
            }
            return pd.DataFrame([result])

    def to_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


    def explain(self, input_df: pd.DataFrame, sample_idx=0, target_idx=0):
        """
        Return SHAP explanations for a given sample.

        Args:
            input_df (pd.DataFrame): Input features (1 or more samples).
            sample_idx (int): Index of the sample in input_df to explain.
            target_idx (int): For MultiOutputRegressor, index of the target (default=0).
        """
        model = self.model

        if hasattr(model, "named_steps"):  # Pipeline
            preprocess = model.named_steps["preprocess"]
            transformed = preprocess.transform(input_df)
            if isinstance(model.named_steps["model"], MultiOutputRegressor):
                base_model = model.named_steps["model"].estimators_[target_idx]
            else:
                base_model = model.named_steps["model"]
            feature_names = preprocess.get_feature_names_out()
        else:  # MultiOutputRegressor or plain model
            if isinstance(model, MultiOutputRegressor):
                base_model = model.estimators_[target_idx]
            else:
                base_model = model

            # -------------------------
            # Handle datetime + categorical encoding
            # -------------------------
            X_processed = input_df.copy()

            for col in X_processed.select_dtypes(include=["datetime64", "datetimetz"]).columns:
                X_processed[col + "_year"] = X_processed[col].dt.year
                X_processed[col + "_month"] = X_processed[col].dt.month
                X_processed[col + "_day"] = X_processed[col].dt.day
                X_processed[col + "_hour"] = X_processed[col].dt.hour
                X_processed.drop(columns=[col], inplace=True)

            for col in X_processed.select_dtypes(include=["object"]).columns:
                X_processed[col] = LabelEncoder().fit_transform(X_processed[col])

            transformed = X_processed.values
            feature_names = X_processed.columns

        # -------------------------
        # SHAP explanation
        # -------------------------
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer(transformed)

        # -------------------------
        # Local (Waterfall Plot)
        # -------------------------
        buf_local = BytesIO()
        fig_local = plt.figure()
        shap.plots.waterfall(shap_values[sample_idx], show=False)
        plt.savefig(buf_local, format="png", bbox_inches="tight")
        plt.close(fig_local)
        buf_local.seek(0)
        shap_local_b64 = base64.b64encode(buf_local.read()).decode("utf-8")

        # -------------------------
        # Explanation mapping
        # -------------------------
        explanation = dict(zip(feature_names, shap_values[sample_idx].values.tolist()))

        result = {
            "explanation": explanation,
            "shap_local_b64": shap_local_b64
        }

        return pd.DataFrame([result])

    def evaluate(self, df_out: pd.DataFrame) -> pd.DataFrame:
        results = []
        for col in self.target_cols:
            actual = df_out[col].values
            pred = df_out[f"pred_{col}"].values

            mask = ~np.isnan(actual) & ~np.isnan(pred)
            actual = actual[mask]
            pred = pred[mask]

            mae = mean_absolute_error(actual, pred)
            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - pred) / np.where(actual == 0, 1, actual))) * 100
            r2 = r2_score(actual, pred)

            results.append({
                "target": col,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "R2": r2
            })

        return pd.DataFrame(results)

    # ---------------------- PLOTS ----------------------

    def plot_actual_vs_pred(self, df_out: pd.DataFrame, well_id, save_dir="src/plots/xgb", return_fig=False):
        os.makedirs(save_dir, exist_ok=True)
        df_well = df_out[df_out["well_id"] == well_id]

        for col in self.target_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_well["timestamp"], df_well[col], label=f"Actual {col}", marker="o")
            ax.plot(df_well["timestamp"], df_well[f"pred_{col}"], label=f"Predicted {col}", marker="x")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel(col)
            ax.set_title(f"Well {well_id} - {col}: Actual vs Predicted")
            ax.legend()
            ax.grid(True)

            file_path = os.path.join(save_dir, f"actual_vs_pred_{well_id}_{re.sub(r"[ /]", "_", col)}.png")
            fig.savefig(file_path, bbox_inches="tight")
            print(f"Saved plot: {file_path}")

            if return_fig:
                return fig
            else:
                plt.show(fig)

    def plot_cumulative(self, df_out: pd.DataFrame, well_id, save_dir="src/plots/xgb", return_fig=False):
        os.makedirs(save_dir, exist_ok=True)
        df_well = df_out[df_out["well_id"] == well_id]

        for col in self.target_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_well["timestamp"], df_well[col].cumsum(), label=f"Cumulative Actual {col}", marker="o")
            ax.plot(df_well["timestamp"], df_well[f"pred_{col}"].cumsum(), label=f"Cumulative Predicted {col}", marker="x")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel(f"Cumulative {col}")
            ax.set_title(f"Well {well_id} - {col}: Cumulative Actual vs Predicted")
            ax.legend()
            ax.grid(True)

            file_path = os.path.join(save_dir, f"cumulative_{well_id}_{re.sub(r"[ /]", "_", col)}.png")
            fig.savefig(file_path, bbox_inches="tight")
            print(f"Saved plot: {file_path}")

            if return_fig:
                return fig
            else:
                plt.show(fig)

    def plot_actual_vs_pred_from_file(self, file_path: str, well_id: str, target: str,
                                      save_dir="src/plots/xgb", return_fig=False):
        # Load saved predictions Excel
        df_out = pd.read_excel(file_path)

        if "well_id" not in df_out.columns:
            raise ValueError("Excel file must contain 'well_id' column with predictions")

        os.makedirs(save_dir, exist_ok=True)
        df_well = df_out[df_out["well_id"] == well_id]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_well["timestamp"], df_well[target], label=f"Actual {target}", marker="o")
        ax.plot(df_well["timestamp"], df_well[f"pred_{target}"], label=f"Predicted {target}", marker="x")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(target)
        ax.set_title(f"Well {well_id} - {target}: Actual vs Predicted")
        ax.legend()
        ax.grid(True)

        file_out = os.path.join(save_dir, f"actual_vs_pred_{well_id}_{re.sub(r"[ /]", "_", target)}.png")
        fig.savefig(file_out, bbox_inches="tight")

        if return_fig:
            return fig
        else:
            plt.show(fig)

    def plot_cumulative_from_file(self, file_path: str, well_id: str, target: str,
                                  save_dir="src/plots/xgb", return_fig=False):
        # Load saved predictions Excel
        df_out = pd.read_excel(file_path)

        if "well_id" not in df_out.columns:
            raise ValueError("Excel file must contain 'well_id' column with predictions")

        os.makedirs(save_dir, exist_ok=True)
        df_well = df_out[df_out["well_id"] == well_id]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_well["timestamp"], df_well[target].cumsum(), label=f"Cumulative Actual {target}", marker="o")
        ax.plot(df_well["timestamp"], df_well[f"pred_{target}"].cumsum(), label=f"Cumulative Predicted {target}", marker="x")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(f"Cumulative {target}")
        ax.set_title(f"Well {well_id} - {target}: Cumulative Actual vs Predicted")
        ax.legend()
        ax.grid(True)

        file_out = os.path.join(save_dir, f"cumulative_{well_id}_{re.sub(r"[ /]", "_", target)}.png")
        fig.savefig(file_out, bbox_inches="tight")

        if return_fig:
            return fig
        else:
            plt.show(fig)

    def plot_actual_vs_pred_all(self, df_out: pd.DataFrame, well_id: str, save_dir="output/plots"):
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        wdf = df_out[df_out["well_id"].astype(str) == str(well_id)]
        if wdf.empty:
            return None

        plt.figure(figsize=(12, 6))
        for col in self.target_cols:
            plt.plot(wdf["timestamp"], wdf[col], label=f"actual {col}")
            plt.plot(wdf["timestamp"], wdf[f"pred_{col}"], "--", label=f"pred {col}")
        
        plt.xlabel("Timestamp")
        plt.ylabel("Rate")
        plt.title(f"Well {well_id} Actual vs Predicted")

        plt.legend()
        out = save_dir / f"actual_vs_pred_{well_id}.png"
        plt.savefig(out)
        plt.show()
        return out

    def plot_cumulative_all(self, df_out: pd.DataFrame, well_id: str, save_dir="output/plots"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        wdf = df_out[df_out["well_id"].astype(str) == str(well_id)]
        if wdf.empty:
            return None

        plt.figure(figsize=(12, 6))
        for col in self.target_cols:
            plt.plot(wdf["timestamp"], np.nancumsum(wdf[col]), label=f"actual_cum {col}")
            plt.plot(wdf["timestamp"], np.nancumsum(np.nan_to_num(wdf[f"pred_{col}"])), "--", label=f"pred_cum {col}")
        
        plt.xlabel("Timestamp")
        plt.ylabel("Rate")
        plt.title(f"Well {well_id} Actual vs Predicted")

        plt.legend()
        out = save_dir / f"cumulative_{well_id}.png"
        plt.savefig(out)
        plt.show()
        return out
    