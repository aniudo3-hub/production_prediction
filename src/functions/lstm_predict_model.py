import io
from fastapi.responses import StreamingResponse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class WellLSTMModel:
    def __init__(
        self,
        model_dir: str = "output/model",
        predicted_excel: str = "predictions_lstm.xlsx",
        lookback: int = 30,
        horizon: int = 1,
        batch_size: int = 64,
        epochs: int = 50,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = None
        self.scalers = None
        self.X_cols = None
        self.onehot_cols = None

        # paths
        self.scaler_path = self.model_dir / "scalers.joblib"
        self.model_path = self.model_dir / "lstm_model.keras"
        self.predicted_excel = predicted_excel
        # self.predicted_excel = self.model_dir / "predictions_lstm.xlsx"

        # columns
        self.target_cols = [
            "oil_rate_bopd (BOPD)",
            "gas_rate_mscf_day (MSCF/day)",
            "water_rate_bwpd (BWPD)",
        ]
        self.cat_cols = ["completion_type", "lift_type", "region", "shift", "well_id", "field"]
        self.bool_cols = ["maintenance_event", "shutdown_flag"]

    # -----------------
    # preprocessing
    # -----------------
    def preprocessd(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)

        # boolean fix
        for b in self.bool_cols:
            if b in df.columns:
                df[b] = df[b].astype(float).fillna(0).astype(int)

        # numeric fill
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df.groupby("well_id")[num_cols].apply(lambda g: g.ffill().bfill())
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)

        # --- boolean fix ---
        for b in self.bool_cols:
            if b in df.columns:
                df[b] = df[b].astype(float).fillna(0).astype(int)

        # --- numeric fix ---
        num_cols = df.select_dtypes(include=[np.number]).columns

        # forward/backward fill per well_id with transform (keeps index aligned)
        df[num_cols] = df.groupby("well_id")[num_cols].transform(lambda g: g.ffill().bfill())

        # fill any remaining NaNs with median
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df


    def one_hot_encode(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df.copy()
        onehot_df = pd.get_dummies(df[self.cat_cols].astype(str), prefix=self.cat_cols)
        df = pd.concat([df.drop(columns=self.cat_cols), onehot_df], axis=1)
        return df, onehot_df.columns.tolist()

    def fit_scalers(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Dict[str, object]:
        X_scaler = StandardScaler().fit(X_df.values)
        y_scaler = MinMaxScaler().fit(y_df.values)
        return {"X_scaler": X_scaler, "y_scaler": y_scaler}

    def create_sequences(self, X: np.ndarray, Y: np.ndarray):
        Xs, Ys = [], []
        for i in range(len(X) - self.lookback - self.horizon + 1):
            Xs.append(X[i : i + self.lookback, :])
            Ys.append(Y[i + self.lookback + self.horizon - 1])
        return np.array(Xs), np.array(Ys)

    # -----------------
    # model
    # -----------------
    def build_model(self, input_shape: Tuple[int, int], output_dim: int):
        model = Sequential([
            LSTM(128, input_shape=input_shape),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(output_dim, activation="linear")
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train(self, df: pd.DataFrame):
        df = self.preprocess(df)
        df_enc, onehot_cols = self.one_hot_encode(df)

        X_cols = [c for c in df_enc.columns if c not in self.target_cols + ["timestamp"]]
        y_cols = self.target_cols

        X_df, y_df = df_enc[X_cols], df_enc[y_cols]

        self.scalers = self.fit_scalers(X_df, y_df)
        X_scaled = self.scalers["X_scaler"].transform(X_df)
        y_scaled = self.scalers["y_scaler"].transform(y_df)

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)

        model = self.build_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(str(self.model_dir / "best.h5"), save_best_only=True)
        ]
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  callbacks=callbacks,
                  verbose=2)

        self.model = model
        self.X_cols = X_cols
        self.onehot_cols = onehot_cols

        model.save(self.model_path)
        joblib.dump({"scalers": self.scalers, "X_cols": X_cols, "onehot_cols": onehot_cols}, self.scaler_path)

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path)
        meta = joblib.load(self.scaler_path)
        self.scalers, self.X_cols, self.onehot_cols = meta["scalers"], meta["X_cols"], meta["onehot_cols"]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self.load()

        df = self.preprocess(df)
        df_enc, _ = self.one_hot_encode(df)

        for col in self.X_cols:
            if col not in df_enc.columns:
                df_enc[col] = 0

        X_df = df_enc[self.X_cols]
        X_scaled = self.scalers["X_scaler"].transform(X_df)

        Xs, idx_map = [], []
        for i in range(len(X_scaled) - self.lookback - self.horizon + 1):
            Xs.append(X_scaled[i : i + self.lookback, :])
            idx_map.append(i + self.lookback + self.horizon - 1)

        y_pred_scaled = self.model.predict(np.array(Xs), batch_size=self.batch_size)
        y_pred = self.scalers["y_scaler"].inverse_transform(y_pred_scaled)

        df_out = df.copy().reset_index(drop=True)
        for t in self.target_cols:
            df_out[f"pred_{t}"] = np.nan

        for idx, row in zip(idx_map, y_pred):
            for t, val in zip(self.target_cols, row):
                df_out.at[idx, f"pred_{t}"] = val

        df_out.to_excel(self.predicted_excel, index=False)

        return df_out
    
    def calculate_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate error metrics between actual and predicted columns.
        Assumes df has columns: target and pred_target for each in target_cols.
        """
        results = []

        for col in self.target_cols:
            actual = df[col].values
            pred = df[f"pred_{col}"].values

            # remove NaNs for fair comparison
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

    # -----------------
    # plotting
    # -----------------

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
            fig.savefig(file_path, dpi=300, bbox_inches="tight")

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
            fig.savefig(file_path, dpi=300, bbox_inches="tight")

            if return_fig:
                return fig
            else:
                plt.close(fig)

    def plot_actual_vs_pred_all(self, df_out: pd.DataFrame, well_id: str, save_dir="output/plots"):
        save_dir = Path(save_dir)
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
        plt.savefig(out, dpi=300, bbox_inches="tight")
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
        
        plt.title(f"Well {well_id} Actual vs Predicted")
        plt.xlabel("Timestamp")
        plt.ylabel("Rate")

        plt.legend()
        out = save_dir / f"cumulative_{well_id}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()
        return out
    
    def plot_actual_vs_pred_from_file(self, file_path: str, well_id: str, target: str,
                                      save_dir="src/plots/lstm", return_fig=False):
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
        fig.savefig(file_out, dpi=300, bbox_inches="tight")

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
        fig.savefig(file_out, dpi=300, bbox_inches="tight")

        if return_fig:
            return fig
        else:
            plt.show(fig)

    def _plot_to_response(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")


