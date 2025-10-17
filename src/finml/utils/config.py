from dataclasses import dataclass

@dataclass
class Paths:
    data_raw: str = "data/raw"
    data_interim: str = "data/interim"
    data_processed: str = "data/processed"
    data_models: str = "data/models"
    reports_fig: str = "data/reports/figures"
    powerbi_ds: str = "powerbi/datasets"
