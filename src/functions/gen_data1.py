import pandas as pd
import numpy as np

class NigerDeltaDataTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.wells = df["well_id"].unique()

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
        df = self.df.copy()

        # --- Add region & field ---
        df["region"] = df["well_id"].map(self.region_map)
        df["field"] = df["well_id"].map(self.field_map)

        # --- Static properties ---
        df["bubble_point_pressure (psi)"] = df["well_id"].map(self.pb_map).round(0)
        df["porosity_percent (%)"] = df["well_id"].map(self.porosity_map).round(1)
        df["permeability_md (mD)"] = df["well_id"].map(self.perm_map).round(1)
        df["net_pay_thickness_ft (ft)"] = df["well_id"].map(self.netpay_map).round(1)
        df["completion_type"] = df["well_id"].map(self.completion_map)
        df["lift_type"] = df["well_id"].map(self.lift_map)

        # --- Initialize dynamic columns ---
        df["reservoir_pressure_initial (psi)"] = 0
        df["bottomhole_pressure_psi"] = 0
        df["oil_rate_bopd (BOPD)"] = 0
        df["gas_rate_mscf_day (MSCF/day)"] = 0
        df["water_rate_bwpd (BWPD)"] = 0

        for well in self.wells:
            mask = df["well_id"] == well
            n_points = mask.sum()

            if n_points == 0:
                continue   # ✅ Skip empty wells safely

            # Pressure decline
            pres = self._reservoir_pressure_decline(well, n_points)
            df.loc[mask, "reservoir_pressure_initial (psi)"] = pres

            # Flowing BHP
            pwf = pres * np.random.uniform(0.6, 0.8)
            df.loc[mask, "bottomhole_pressure_psi"] = pwf

            # Oil rate tied to pressure decline
            qo0 = self.qo_init_map[well]
            denom = max((pres.max() - pwf.max()), 1)  # avoid divide by zero
            df.loc[mask, "oil_rate_bopd (BOPD)"] = (
                qo0 * (pres - pwf) / denom
            ).clip(min=100).round(0)

            # Gas rate
            gor = self.gor_map[well]
            df.loc[mask, "gas_rate_mscf_day (MSCF/day)"] = (
                df.loc[mask, "oil_rate_bopd (BOPD)"] * gor / 1000
            ).round(0)

            # Water rate increasing
            df.loc[mask, "water_rate_bwpd (BWPD)"] = np.linspace(
                np.random.uniform(200, 1000),
                np.random.uniform(5000, 15000),
                n_points
            ).round(0)

        # Water cut
        df["water_cut_percent (%)"] = (
            df["water_rate_bwpd (BWPD)"] /
            (df["oil_rate_bopd (BOPD)"] + df["water_rate_bwpd (BWPD)"]) * 100
        ).round(1)

        # Choke size
        df["choke_size_percent (%)"] = df.apply(
            lambda row: max(10, min(64, self.choke_base_map[row["well_id"]] + np.random.normal(0, 3))),
            axis=1
        ).round(1)

        # Liquid rate & GOR
        df["liquid_rate_bpd (BPD)"] = df["oil_rate_bopd (BOPD)"] + df["water_rate_bwpd (BWPD)"]
        df["gor_scf_bbl (scf/bbl)"] = (df["gas_rate_mscf_day (MSCF/day)"] / df["oil_rate_bopd (BOPD)"]).round(1)

        # Productivity index
        df["drawdown_psi"] = (df["reservoir_pressure_initial (psi)"] - df["bottomhole_pressure_psi"]).round(0)
        df["productivity_index_bbl_day_psi (bbl/day/psi)"] = (
            df["oil_rate_bopd (BOPD)"] / df["drawdown_psi"].replace(0, np.nan)
        ).fillna(0).round(2)

        return df

    def save_to_excel(self, path="niger_delta_realistic.xlsx"):
        self.transform().to_excel(path, index=False)
