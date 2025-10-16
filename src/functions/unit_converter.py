import pandas as pd

class UnitConverter:
    """
    Converts well data columns from empirical units to SI units.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

        # Conversion factors (empirical → SI)
        self.conversion_factors = {
            "(ft)": 0.3048,                  # feet to meters
            "(psi)": 6894.76,                # psi to pascal
            "(mD)": 9.86923e-16,             # millidarcy to m²
            "(BOPD)": 0.158987,              # barrel/day to m³/day
            "(MSCF/day)": 28.3168,           # thousand scf/day to m³/day
            "(BWPD)": 0.158987,              # barrel/day to m³/day
            "(bbl/day/psi)": 0.158987 / 6894.76,  # productivity index
            "(bbl/kW)": 0.158987,            # barrel/kW to m³/kW
            "(bbl/ft)": 0.158987 / 0.3048,   # barrel/ft to m³/m
            "(scf/bbl)": 0.1781076,          # scf/bbl to m³/m³
            "(mm/s)": 1.0,                   # already SI
            "(L/hr)": 2.77778e-7,            # L/hr to m³/s
            "(%)": 0.01,                     # percent to fraction
            "(°C)": 1.0,                     # already SI
            "(Hz)": 1.0,                     # already SI
            "(A)": 1.0,                      # already SI
        }

    def convert_units(self, df: pd.DataFrame):
        converted_cols = []

        for col in df.columns:
            for unit, factor in self.conversion_factors.items():
                if unit in col:
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce") * factor
                        new_col = col.replace(unit, self._si_unit(unit))
                        df.rename(columns={col: new_col}, inplace=True)
                        converted_cols.append((col, new_col))
                    except Exception as e:
                        print(f"Error converting {col}: {e}")
        return df, converted_cols

    def _si_unit(self, unit: str) -> str:
        """Return corresponding SI unit symbol."""
        si_units = {
            "(ft)": "(m)",
            "(psi)": "(Pa)",
            "(mD)": "(m²)",
            "(BOPD)": "(m³/day)",
            "(MSCF/day)": "(m³/day)",
            "(BWPD)": "(m³/day)",
            "(bbl/day/psi)": "(m³/day/Pa)",
            "(bbl/kW)": "(m³/kW)",
            "(bbl/ft)": "(m³/m)",
            "(scf/bbl)": "(m³/m³)",
            "(mm/s)": "(mm/s)",
            "(L/hr)": "(m³/s)",
            "(%)": "(fraction)",
            "(°C)": "(°C)",
            "(Hz)": "(Hz)",
            "(A)": "(A)",
        }
        return si_units.get(unit, unit)

    def run(self):
        df = pd.read_excel(self.input_path)
        converted_df, converted_cols = self.convert_units(df)
        converted_df.to_excel(self.output_path, index=False)
        return self.output_path, converted_cols


