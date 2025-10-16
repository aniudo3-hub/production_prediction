import pandas as pd
import numpy as np

class EmpiricalToSIConverter:
    """
    Reads empirical oilfield data from Excel,
    converts to SI units (renaming columns),
    and saves all columns (converted + non-converted) to a new Excel file.
    """

    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None

    # === Conversion mapping: empirical → SI ===
    conversions = {
        "(ft)": {"factor": 0.3048, "si_unit": "(m)"},                   # feet → meters
        "(psi)": {"factor": 6894.76, "si_unit": "(Pa)"},                # psi → pascal
        "(mD)": {"factor": 9.869e-16, "si_unit": "(m²)"},               # millidarcy → m²
        "(L/hr)": {"factor": 2.7778e-7, "si_unit": "(m³/s)"},           # L/hr → m³/s
        "(mm/s)": {"factor": 0.001, "si_unit": "(m/s)"},                # mm/s → m/s
        "(BOPD)": {"factor": 1.84e-6, "si_unit": "(m³/s)"},             # bbl oil/day → m³/s
        "(MSCF/day)": {"factor": 0.000327, "si_unit": "(m³/s)"},        # MSCF/day → m³/s
        "(BWPD)": {"factor": 1.84e-6, "si_unit": "(m³/s)"},             # BWPD → m³/s
        "(BPD)": {"factor": 1.84e-6, "si_unit": "(m³/s)"},             # BWPD → m³/s
        "(scf/bbl)": {"factor": 0.1781, "si_unit": "(m³/m³)"},          # scf/bbl → m³/m³
        "(bbl/day/psi)": {"factor": 2.67e-10, "si_unit": "(m³/s/Pa)"},  # productivity index → m³/s/Pa
        "(bbl/kW)": {"factor": 0.158987, "si_unit": "(m³/kW)"},         # bbl/kW → m³/kW
        "(bbl/ft)": {"factor": 0.158987 / 0.3048, "si_unit": "(m³/m)"}, # bbl/ft → m³/m
        "(°C)": {"offset": 273.15, "si_unit": "(K)"},                   # °C → K
        "(%)": {"factor": 0.01, "si_unit": "(fraction)"},               # % → fraction
        "(Hz)": {"factor": 1.0, "si_unit": "(Hz)"},                     # Hz → Hz
        "(A)": {"factor": 1.0, "si_unit": "(A)"},                       # Ampere → same
        "(bool)": {"factor": 1.0, "si_unit": "(bool)"},                 # Boolean → same
    }

    def convert_series(self, series: pd.Series, unit: str):
        """Apply the correct conversion to a pandas Series."""
        data = pd.to_numeric(series, errors='coerce')
        conv = self.conversions[unit]
        if "offset" in conv:
            return data + conv["offset"]
        else:
            return data * conv["factor"]

    def convert_dataframe(self):
        """Convert all convertible columns and rename them."""
        df_converted = self.df.copy()

        for col in self.df.columns:
            converted = False
            for unit, conv in self.conversions.items():
                if unit in col:
                    new_col = col.replace(unit, conv["si_unit"])
                    df_converted[new_col] = self.convert_series(self.df[col], unit)
                    converted = True
                    print(f"   → {col} converted to {new_col}")
                    df_converted.drop(columns=[col], inplace=True)
                    break  # stop checking once matched
            if not converted:
                print(f"   • {col} kept as-is (no recognized unit)")
        return df_converted

    def run(self):
        """Execute the full pipeline."""
        self.df = pd.read_excel(self.input_file)

        df_si = self.convert_dataframe()

        df_si.to_excel(self.output_file, index=False)


