"""Collect the acquisition settings recorded by the scanner (.xtekct files) into data/acquisition.csv.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Run once on the authors' computer (the .xtekct files are on the project drive); the CSV is versioned, so
this script is not needed to reproduce the article. Samples without a scanner file (the limestone plugs
re-scanned in 2024) get empty fields.
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"D:/02_Projetos_e_Eventos/Tomografia")
CTDATA = Path(r"D:/01_Mestrado_Doutorado/Correções da dissertação/Jean Firmino CTData")

FILES = {
    "CF4": CTDATA / "DS23_CF4_3mm_CTData/DS23_CF4_3mm.xtekct",
    "CF5": CTDATA / "DS23_CF5_3mm_CTData/DS23_CF5_3mm.xtekct",
    "CF7": CTDATA / "DS23_CF7_4mm_CTData/DS23_CF7_4mm.xtekct",
    "CF10": CTDATA / "DS23_CF10_M_CTData/DS23_CF10_M.xtekct",
}
for i in range(1, 13):
    d = next((RAW / "GT24 Civil UFPE").glob(f"GT24_{i:02d} *"))
    FILES[f"GT{i:02d}"] = d / f"GT24_{i:02d}_40.xtekct"
for name, folder in (("TQ01", "BC22/BC22_TQ-01 [2022-08-26 11.21.34]"), ("TQ02", "BC22/BC22_TQ-02 [2022-08-26 10.02.24]"),
                     ("TQ05", "BC23/BC23_TQ-05 [2023-05-02 16.16.58]"), ("TQ06", "BC23/BC23_TQ-06 [2023-03-30 10.27.27]"),
                     ("TQ07", "BC23/BC23_TQ-07 [2023-05-02 16.45.27]")):
    d = RAW / "Traquitos - Rochas Vulcânicas" / folder
    FILES[name] = d / f"{folder.split('/')[1].split(' ')[0]}_01.xtekct"

KEYS = {"XraykV": "voltage_kV", "XrayuA": "current_uA", "Filter_Material": "filter_material",
        "Filter_ThicknessMM": "filter_mm", "Projections": "projections", "VoxelSizeX": "voxel_mm",
        "SrcToObject": "source_object_mm", "SrcToDetector": "source_detector_mm",
        "VoxelsX": "voxels_x", "VoxelsY": "voxels_y", "VoxelsZ": "voxels_z"}


def read(p: Path) -> dict:
    out = {}
    for line in p.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out.setdefault(k.strip(), v.strip())
    return out


rows = []
for sid in ["CF4", "CF5", "CF7", "CF10", "VN01", "VN02", "VN04"] + [f"GT{i:02d}" for i in range(1, 13)] + \
        ["TQ01", "TQ02", "TQ05", "TQ06", "TQ07"]:
    r = {"id": sid, "source_file": ""}
    p = FILES.get(sid)
    if p is not None and p.exists():
        m = read(p)
        r["source_file"] = p.name
        for k, v in KEYS.items():
            r[v] = m.get(k, "")
    rows.append(r)
fields = ["id", "source_file"] + list(KEYS.values())
with open(ROOT / "data/acquisition.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
print("wrote data/acquisition.csv", len(rows))
