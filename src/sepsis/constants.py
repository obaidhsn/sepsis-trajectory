"""Column definitions for the PhysioNet/CinC Challenge 2019 sepsis data.

Every .psv file has one row per ICU hour and the columns below. EtCO2 is
dropped because it is entirely missing at Hospital A (see docs/data.md).
"""

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
DROPPED = ["EtCO2"]

LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

# Time-varying clinical variables = graph nodes (C = 33).
DYNAMIC = VITALS + LABS

# Static / timeline features. Unit1/Unit2 are NaN for some patients.
# ICULOS (hours since ICU admission) is causal and allowed; total length of
# stay is NOT used because septic records are truncated (label leakage).
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

LABEL = "SepsisLabel"

# SepsisLabel is set to 1 from 6 h before clinical onset, so
# t_sepsis = first positive hour + LABEL_SHIFT.
LABEL_SHIFT = 6

# Organ-system groups used to build the clinical-prior graph.
ORGAN_GROUPS = {
    "cardiovascular": ["HR", "SBP", "MAP", "DBP", "TroponinI", "Lactate"],
    "respiratory": ["O2Sat", "Resp", "FiO2", "SaO2", "PaCO2", "pH", "BaseExcess", "HCO3"],
    "thermo_inflammatory": ["Temp", "WBC"],
    "renal": ["BUN", "Creatinine", "Potassium", "Chloride", "Calcium", "Magnesium", "Phosphate", "HCO3"],
    "hepatic": ["AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total"],
    "haematologic": ["Hct", "Hgb", "Platelets", "PTT", "Fibrinogen", "WBC"],
    "metabolic": ["Glucose", "Lactate", "pH", "BaseExcess"],
}
