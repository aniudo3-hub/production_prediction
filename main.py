from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
from src.functions.lstm_predict_model import WellLSTMModel
from src.functions.xgb_predict_model import XGBModel
from src.functions.gen_data import NigerDeltaDataTransformer
from src.functions.statistics_calculator import StatisticsCalculator
import io
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os

app = FastAPI(
    title="Well Production Using XGB/LSTM Model API",
    description="API for predicting oil production using FastAPI and Swagger UI",
    version="1.0.0",
    docs_url="/swagger",      # Swagger UI URL
    redoc_url=None,           # Disable ReDoc
    openapi_url="/openapi.json"  # OpenAPI JSON URL
)

NIGER_DELTA_EXCEL = "./src/resource/niger_delta.xlsx"
INPUT_EXCEL = "./src/resource/oilfield_sample_data_with_units.xlsx"
niger_delta_to_back_date = "./src/resource/niger_delta_to_back_date.xlsx"
XGB_MODEL_PATH = "./src/created_model/xgb/xgb_model.pkl"
LSTM_MODEL_PATH = "./src/created_model/lstm/lstm_model.h5"
LSTM_MODEL_DIR = "./src/created_model/lstm"
XGB_MODEL_DIR = "./src/created_model/xgb"
XGB_PLOT_DIR = "./src/plot/xgb"
LSTM_PLOT_DIR = "./src/plot/lstm"

LSTM_EXCEL_DIR = "./src/prediction/lstm"
XGB_EXCEL_DIR = "./src/prediction/xgb"
LSTM_PREDICTION_EXCEL = "./src/prediction/lstm/predictions_lstm.xlsx"
XGB_PREDICTION_EXCEL = "./src/prediction/xgb/predictions_xgb.xlsx"

STATISTICAL_DIR = "./src/statistical"
STATISTICAL_EXCEL = "./src/statistical/statistical_data.xlsx"
# save_dir="src/predictions/xgb"
os.makedirs(STATISTICAL_DIR, exist_ok=True)
class PlotRequest(BaseModel):
    well_id: str
    cumulative: bool = False

class PlotTargetRequest(PlotRequest):
    target: str


TARGET_COLS = [
    "oil_rate_bopd (BOPD)",
    "gas_rate_mscf_day (MSCF/day)",
    "water_rate_bwpd (BWPD)"
]


@app.get("/gen_data_statistics")
async def gen_data():
    df = pd.read_excel(NIGER_DELTA_EXCEL)
    stat = StatisticsCalculator(df)

    stat.save_to_excel(STATISTICAL_EXCEL)

    return {"success": "Statistics data created"}

@app.get("/train_lstm")
async def train():
    df = pd.read_excel(NIGER_DELTA_EXCEL)
    model = WellLSTMModel(LSTM_MODEL_DIR, LSTM_PREDICTION_EXCEL)

    model.train(df)
    return {"status": "trained", "model_path": str(model.model_path)}

@app.get("/predict_lstm")
async def predict():
    df = pd.read_excel(NIGER_DELTA_EXCEL)
    model = WellLSTMModel(LSTM_MODEL_DIR, LSTM_PREDICTION_EXCEL)
    df_out = model.predict(df)

    errors = model.calculate_errors(df_out)

    return {"status": "predicted", "saved_file": str(model.predicted_excel), "error": errors}


def _plot_to_response(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/plot_lstm_target")
async def plot(request: PlotTargetRequest):
    
    model = WellLSTMModel(LSTM_MODEL_DIR, LSTM_PREDICTION_EXCEL)
    if request.target.lower()  == "oil":
        target = TARGET_COLS[0]
    elif request.target.lower()  == "gas":
        target = TARGET_COLS[1]
    elif request.target.lower()  == "water":
        target = TARGET_COLS[2]

    if request.cumulative:
        fig = model.plot_cumulative_from_file(LSTM_PREDICTION_EXCEL, request.well_id, target,
                                  save_dir=LSTM_PLOT_DIR, return_fig=True)
    else:
        fig = model.plot_actual_vs_pred_from_file(LSTM_PREDICTION_EXCEL, request.well_id, target,
                                  save_dir=LSTM_PLOT_DIR, return_fig=True)
    return _plot_to_response(fig)

@app.post("/plot_lstm_all")
async def plot(request: PlotRequest):
    model = WellLSTMModel(LSTM_MODEL_DIR, LSTM_PREDICTION_EXCEL)
    
    df = pd.read_excel(model.predicted_excel)
    if request.cumulative:
        out = model.plot_cumulative_all(df, request.well_id, save_dir=LSTM_PLOT_DIR)
    else:
        out = model.plot_actual_vs_pred_all(df, request.well_id, save_dir=LSTM_PLOT_DIR)

    if out is None:
        return JSONResponse(content={"error": "well not found"}, status_code=404)
    return FileResponse(out)


@app.get("/train_xgb")
async def train():
    df = pd.read_excel(NIGER_DELTA_EXCEL)
    xgb_model = XGBModel(TARGET_COLS, model_path=XGB_MODEL_PATH)

    xgb_model.train(df)
    return {"message": "XGB model trained and saved."}


@app.get("/predict_xgb")
async def predict():
    df = pd.read_excel(NIGER_DELTA_EXCEL)
    xgb_model = XGBModel(TARGET_COLS, model_path=XGB_MODEL_PATH)

    filename="predictions_xgb.xlsx"
    df_out = xgb_model.predict(df,save_dir= XGB_EXCEL_DIR,  save=True, filename=filename)
    errors = xgb_model.evaluate(df_out)
    
    return {"message": "Predictions saved", "file_path": f"{XGB_EXCEL_DIR}/{filename}", "errors": errors}


@app.post("/plot_xgb_target")
async def plot(request: PlotRequest):
    model = XGBModel(TARGET_COLS, model_path=XGB_MODEL_PATH)
    
    if request.target.lower()  == "oil":
        target = TARGET_COLS[0]
    elif request.target.lower()  == "gas":
        target = TARGET_COLS[1]
    elif request.target.lower()  == "water":
        target = TARGET_COLS[2]

    if request.cumulative:
        fig = model.plot_cumulative_from_file(XGB_PREDICTION_EXCEL, request.well_id, target,
                                  save_dir=XGB_PLOT_DIR, return_fig=True)
    else:
        fig = model.plot_actual_vs_pred_from_file(XGB_PREDICTION_EXCEL, request.well_id, target,
                                  save_dir=XGB_PLOT_DIR, return_fig=True)
        
    return _plot_to_response(fig)

@app.post("/plot_xgb_all/")
async def plot(request: PlotRequest):
    model = XGBModel(TARGET_COLS, model_path=XGB_MODEL_PATH)
    df = pd.read_excel(XGB_PREDICTION_EXCEL)

    if request.cumulative:
        out = model.plot_cumulative_all(df, request.well_id, save_dir=LSTM_PLOT_DIR)
    else:
        out = model.plot_actual_vs_pred_all(df, request.well_id, save_dir=LSTM_PLOT_DIR)

    if out is None:
        return JSONResponse(content={"error": "well not found"}, status_code=404)
    return FileResponse(out)
