"""Build the two small input tables of the pipeline from the authors' original spreadsheets.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Run once, on the authors' computer (the sources are on the project drive):

    python scripts/build_input_tables.py

Outputs (committed to the repository, so this script is not needed to reproduce the paper):
    data/legacy_dissertation.csv  results reported in the master's dissertation (Cardoso 2026)
    data/lab_porosity.csv         laboratory porosity of the samples, with the source of every value
"""
from pathlib import Path
import csv
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
DISS = Path(r"D:/01_Mestrado_Doutorado/mestrado documento final")
SRC_POR = DISS / "segundo topico/02_Planilhas_e_Dados/tabela resultados .xlsx"
SRC_TRA = DISS / "terceiro topico/02_Planilhas_e_Dados/tabela mestre.xlsx"

ID = {  # dissertation sample code -> id used here
    "BC22_TQ-01_40": "TQ01", "BC22_TQ-02_40": "TQ02", "BC23_TQ-05_40": "TQ05", "BC23_TQ-06_40": "TQ06",
    "BC23_TQ-07_40": "TQ07", "DS23_CF4_3mm": "CF4", "DS23_CF5_3mm": "CF5", "DS23_CF7_4mm": "CF7",
    "DS23_CF10_M": "CF10", "VN01_AMOSTRAX": "VN01", "VN02_AMOSTRA01": "VN02", "VN04_AMOSTRAX2": "VN04",
}
ID.update({f"GT24_{i:02d}_40": f"GT{i:02d}" for i in range(1, 13)})


def rows(path):
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    head = [str(h).strip() if h is not None else "" for h in next(it)]
    for r in it:
        if r[0] is not None and str(r[0]).strip() in ID:
            d = dict(zip(head, r))
            d["_raw"] = r
            yield d


def num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return ""


def perc(v):
    return "" if v is None else int(str(v).strip().upper().startswith("PERCOLATES"))


legacy = {}
for r in rows(SRC_POR):
    sid = ID[str(r["ID da Amostra"]).strip()]
    legacy[sid] = {
        "id": sid, "voxel_um_reported": num(r["Voxel Size (µm)"]),
        "phi_total_pct": num(r["Porosidade Total [%]"]), "phi_eff_pct": num(r["Porosidade Efetiva [%]"]),
        "isolated_pct": num(r["Fração de Poros Isolados [%]"]), "fractal_dim": num(r["Dimensão Fractal"]),
        "euler": num(r["Característica de Euler"]),
    }
for r in rows(SRC_TRA):
    sid = ID[str(r["ID da Amostra"]).strip()]
    raw = r["_raw"]
    legacy[sid].update({
        "clusters": num(r["distinct pore clusters"]),
        "perc_x": perc(r["percolation for X-axis"]), "perc_y": perc(r["percolation for Y-axis"]),
        "perc_z": perc(r["percolation for Z-axis"]),
        "F_x": num(raw[5]),
        # the spreadsheet repeats the 'in y' header for z; take the columns by position
        "F_y": num(raw[9]), "F_z": num(raw[13]),
        "tau_x": num(r["tortuosity for the X-axis"]), "tau_y": num(r["tortuosity for the Y-axis"]),
        "tau_z": num(r["tortuosity for the Z-axis"]),
        "k_lbm_z_mD": num(r["FINAL ABSOLUTE PERMEABILITY (Z-Direction) [mD]"]),
    })

out = ROOT / "data/legacy_dissertation.csv"
fields = list(next(iter(legacy.values())).keys())
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for sid in sorted(legacy):
        w.writerow(legacy[sid])
print("wrote", out, len(legacy), "rows")

# Laboratory porosity. Glass-bead columns: Silva (2024), Table 3 (same specimens that were scanned).
# Limestone plugs: Vila Nova (2020), Table 7, helium porosimeter, plugs VN01, VN02 and VN04.
SILVA = "Silva (2024), Table 3"
NOVA = "Vila Nova (2020), Table 7"
lab = []
for sid, vals in {"CF4": (38.70, 38.07, 39.05, 37.14, 32.92), "CF5": (40.07, 40.10, 39.04, 38.93, 33.38),
                  "CF7": (40.70, 40.33, 40.87, 39.51, 33.52), "CF10": (40.00, 39.95, 38.92, 39.17, 31.68)}.items():
    for m, v in zip(("direct", "gravimetric", "indirect", "helium", "uct_vgstudio"), vals):
        lab.append({"id": sid, "method": m, "porosity_pct": v, "source": SILVA})
for sid, v in {"VN01": 16.55, "VN02": 16.17, "VN04": 6.67}.items():
    lab.append({"id": sid, "method": "helium", "porosity_pct": v, "source": NOVA})
out = ROOT / "data/lab_porosity.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "method", "porosity_pct", "source"])
    w.writeheader()
    w.writerows(lab)
print("wrote", out, len(lab), "rows")
