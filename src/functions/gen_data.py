import pandas as pd
import numpy as np

class NigerDeltaDataTransformer:
    def __init__(self, df: pd.DataFrame, start_date="2018-01-01", freq="M"):
        self.df = df.copy()
        self.wells = df["well_id"].unique()
        self.start_date = pd.to_datetime(start_date)
        self.freq = freq

        # --- Find end date from df ---
        if "timestamp" in df.columns:
            self.end_date = pd.to_datetime(df["timestamp"]).max()
        else:
            self.end_date = pd.Timestamp.today()

        # --- Static properties ---
        self.pb_map = {well: np.random.uniform(1500, 3500) for well in self.wells}
        self.porosity_map = {well: np.random.uniform(0.15, 0.30) * 100 for well in self.wells}
        self.perm_map = {well: np.random.uniform(50, 2000) for well in self.wells}
        self.netpay_map = {well: np.random.uniform(30, 200) for well in self.wells}
        self.completion_map = {well: np.random.choice(["Open Hole", "Cased Hole", "Perforated"]) for well in self.wells}
        self.lift_map = {well: np.random.choice(["Gas Lift", "ESP", "Natural Flow", "Rod Pump"]) for well in self.wells}
        self.choke_base_map = {well: np.random.uniform(20, 40) for well in self.wells}

        # Initial reservoir conditions
        self.res_press_init_map = {well: np.random.uniform(4500, 6500) for well in self.wells}
        self.qo_init_map = {well: np.random.uniform(5000, 20000) for well in self.wells}
        self.gor_map = {well: np.random.uniform(200, 1500) for well in self.wells}

        # Regional assignments
        self.regions = {
            "Western Niger Delta": ["Forcados", "Escravos", "Warri", "Jones Creek"],
            "Central Niger Delta": ["Cawthorne Channel", "Imo River", "Nkali", "Obagi"],
            "Eastern Niger Delta": ["Bonny", "Oguta", "Brass", "Ebocha"],
            "Offshore Niger Delta": ["Bonga", "Agbami", "Akpo", "Erha", "Usan"]
        }
        self.region_map = {}
        self.field_map = {}

        for well in self.wells:
            region = np.random.choice(list(self.regions.keys()))
            field = np.random.choice(self.regions[region])
            self.region_map[well] = region
            self.field_map[well] = field

    def _reservoir_pressure_decline(self, well, n_points, timestep=1):
        p_init = self.res_press_init_map[well]
        decline_rate = np.random.uniform(0.0005, 0.002)
        time = np.arange(n_points) * timestep
        pressure = p_init * np.exp(-decline_rate * time)
        noise = np.random.normal(0, 30, n_points)
        return (pressure + noise).clip(min=1000)

    def transform(self):
        all_data = []

        for well in self.wells:
            # Generate timeline from start_date to end_date
            timestamps = pd.date_range(self.start_date, self.end_date, freq=self.freq)
            n_points = len(timestamps)

            pres = self._reservoir_pressure_decline(well, n_points)
            pwf = pres * np.random.uniform(0.6, 0.8)
            qo0 = self.qo_init_map[well]
            denom = max((pres.max() - pwf.max()), 1)

            oil_rate = (qo0 * (pres - pwf) / denom).clip(min=100).round(0)
            gas_rate = (oil_rate * self.gor_map[well] / 1000).round(0)
            water_rate = np.linspace(np.random.uniform(200, 1000),
                                     np.random.uniform(5000, 15000),
                                     n_points).round(0)

            df_well = pd.DataFrame({
                "timestamp": timestamps,
                "well_id": well,
                "region": self.region_map[well],
                "field": self.field_map[well],
                "bubble_point_pressure (psi)": self.pb_map[well],
                "porosity_percent (%)": round(self.porosity_map[well], 1),
                "permeability_md (mD)": round(self.perm_map[well], 1),
                "net_pay_thickness_ft (ft)": round(self.netpay_map[well], 1),
                "completion_type": self.completion_map[well],
                "lift_type": self.lift_map[well],
                "reservoir_pressure_initial (psi)": pres,
                "bottomhole_pressure_psi": pwf,
                "oil_rate_bopd (BOPD)": oil_rate,
                "gas_rate_mscf_day (MSCF/day)": gas_rate,
                "water_rate_bwpd (BWPD)": water_rate,
            })

            # Derived properties
            df_well["water_cut_percent (%)"] = (
                df_well["water_rate_bwpd (BWPD)"] /
                (df_well["oil_rate_bopd (BOPD)"] + df_well["water_rate_bwpd (BWPD)"]) * 100
            ).round(1)

            df_well["choke_size_percent (%)"] = max(10, min(64, self.choke_base_map[well] + np.random.normal(0, 3)))
            df_well["liquid_rate_bpd (BPD)"] = df_well["oil_rate_bopd (BOPD)"] + df_well["water_rate_bwpd (BWPD)"]
            df_well["gor_scf_bbl (scf/bbl)"] = (df_well["gas_rate_mscf_day (MSCF/day)"] /
                                                df_well["oil_rate_bopd (BOPD)"]).round(1)
            df_well["drawdown_psi"] = (df_well["reservoir_pressure_initial (psi)"] -
                                       df_well["bottomhole_pressure_psi"]).round(0)
            df_well["productivity_index_bbl_day_psi (bbl/day/psi)"] = (
                df_well["oil_rate_bopd (BOPD)"] / df_well["drawdown_psi"].replace(0, np.nan)
            ).fillna(0).round(2)

            all_data.append(df_well)

        return pd.concat(all_data, ignore_index=True)

    def save_to_excel(self, path="niger_delta_realistic.xlsx"):
        self.transform().to_excel(path, index=False)
