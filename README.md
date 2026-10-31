# poreconn — resolved pore connectivity, topology and transport from X-ray micro-CT

Code of the article *Pore connectivity and image-based permeability at routine micro-CT resolution: glass-bead packs, reef limestones, coquinas and trachytes* (Cardoso et al., submitted to *Transport in Porous Media*).
DOI of this code: https://doi.org/10.5281/zenodo.23840902 · DOI of the data: https://doi.org/10.5281/zenodo.23840904.

`poreconn` takes grey-level micro-CT volumes of interest (VOIs) of 24 specimens — four glass-bead packs,
three reef limestones and twelve coquinas of the Tambaba Formation (Paraíba Basin) and five trachytes of
the Pernambuco Basin (NE Brazil) — and computes, with tested open-source code:

- a partial-volume (ISO-50) segmentation with bounding thresholds, two air references (outside the plug and
  inside large pores) carried through every analysis, and Otsu for comparison;
- imaging diagnostics: edge-spread function, median-filter sensitivity, enclosed-air control, two-glass bound and
  synthetic bead packs;
- a porosity budget against the laboratory porosity (threshold-free mass balance, the threshold that reproduces
  the laboratory porosity and the percolation of the image at that threshold);
- porosity, connected and isolated pore fractions, percolation along three axes (6-connected pore phase);
- the Euler characteristic density χ_V, Crofton surface density S_V, chord lengths and the critical
  diameter of morphological drainage;
- the loss of connectivity when the voxel is coarsened (block averaging + same threshold);
- sub-volume statistics: representative volume and cross-validated prediction of percolation from
  porosity and from topology;
- the formation factor (finite-volume diffusion, conjugate gradients), permeability by a two-relaxation-time
  D3Q19 lattice Boltzmann solver, by SNOW-2 pore networks (PoreSpy + OpenPNM) and by Kozeny–Carman and
  Katz–Thompson estimates;
- the radial porosity profile of the bead columns (wall effect) and its comparison with laboratory porosity;
- every table, figure and number quoted in the article (`outputs/report/key_numbers.json`).

## Install

Python 3.12 (PoreSpy 2.4.3 pins scikit-image < 0.25, which has no wheels for Python ≥ 3.13):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Linux/macOS: .venv/bin/python
```

`requirements-lock.txt` lists the exact versions used for the article.

## Data

Download the data record (https://doi.org/10.5281/zenodo.23840904) and extract it next to this folder so that the volumes sit in
`data/voi/` (`<sample>_v<k>_grey.tif`, uint16, axis order z, y, x, z = core axis, with a `_meta.json`
sidecar recording the raw slices, crop box, binning and voxel size). The laboratory porosity
(`data/lab_porosity.csv`) and the results reported in the dissertation (`data/legacy_dissertation.csv`)
are part of this repository.

The `extract` stage, which cuts the VOIs from the full reconstructions, needs the raw data of the
laboratory (path in `config.yaml`, or the environment variable `PORECONN_RAW`); every later stage reads
only `data/voi/`.

## Reproduce

```bash
python -m pytest -q tests                                   # unit tests with analytical answers (~1 min)
python -m poreconn.run --stages analysis                    # all stages after extract (several hours)
python -m poreconn.run --stages report,figures              # tables, key numbers and figures only
python -m poreconn.run --stages topology --samples GT01     # one stage, one sample
python -m poreconn.run --quick --stages segment,topology    # smoke test on small VOIs (outputs_quick/)
```

| Stage | What it does | Main output |
|---|---|---|
| `extract` | cuts the VOIs from the raw reconstructions (raw data needed) | `data/voi/` |
| `air` | grey value of the air around each rock plug, 20-100 px and > 100 px from it (raw data needed) | VOI `_meta.json` |
| `segment` | 3-D median filter, air and solid references, ISO-50 / 0.4 / 0.6 / internal-air / Otsu thresholds | `outputs/segment/` |
| `qa` | cupping, internal-air offset and air-reference checks of the segmentation | `outputs/qa/qa.csv` |
| `topology` | connectivity, percolation, χ_V, S_V, chords, pore size, d_c | `outputs/topology/topology.csv` |
| `coarsen` | connectivity after 2×, 3×, 4× block averaging, every VOI at both air references, with the native critical diameter | `outputs/coarsen/coarsen.csv` |
| `rev` | statistics of sub-volumes of 50–300 voxels, at both air references | `outputs/rev/rev.csv` |
| `predict` | percolation of sub-volumes from porosity and topology: pooled and within-specimen AUC, permutation nulls, fixed-effects coefficients, phi50 | `outputs/predict/` |
| `transport` | formation factor, pore-network, Kozeny–Carman and Katz–Thompson permeability of the main VOIs; rocks also at T40, T60 and T50int | `outputs/transport/transport*.csv` |
| `lbm` | lattice Boltzmann vs pore networks vs diffusion on the same domains, at both air references; Lambda sweep and forced whole-VOI runs as functions | `outputs/lbm/lbm*.csv` |
| `estimates` | Kozeny–Carman (whole pore phase and spanning cluster), Katz–Thompson and networks on the whole VOI binned by 2, on the domains of the LBM runs | `outputs/lbm/estimates*.csv` |
| `wall` | radial porosity of the bead columns (raw data needed) | `outputs/wall/wall.csv` |
| `grains` | size and shape of the beads (bubbles filled, seeded watershed of the solid; all VOIs of a column; seed sensitivity) | `outputs/grains/` |
| `imaging` | imaging diagnostics: edge-spread function, median-filter sensitivity, two-glass bound, enclosed-air control, VOI footprint, synthetic bead packs | `outputs/imaging/` |
| `budget` | porosity budget against the laboratory porosity: pooled image porosity, mass-balance porosity, threshold that reproduces the laboratory porosity and percolation of the main VOI there | `outputs/budget/porosity_budget.csv` |
| `validation` | all solvers on synthetic media, and the window-size study of the pore networks | `outputs/validation/` |
| `report` | tables, statistics, key numbers | `outputs/report/` |
| `figures` | figures (PDF + 600-dpi PNG) | `outputs/figures/` |

Every stage writes one file per sample and rebuilds its summary table from all of them, so a stage can be
re-run for some samples (`--samples`) without losing the others.

## Layout

```
config.yaml            samples, paths and parameters
poreconn/              package (one module per stage; morph.py, fd.py, lbm.py, pnm.py are the solvers)
scripts/               build_input_tables.py (builds data/*.csv from the authors' spreadsheets; run once)
tests/                 unit tests: Euler characteristic, percolation, Crofton S_V, diffusion vs a direct
                       sparse solve, LBM vs square-duct and slit solutions, viscosity independence
data/                  lab_porosity.csv, legacy_dissertation.csv; voi/ (from the data record)
outputs/               generated (not versioned)
```

## Conventions

- Pore phase 6-connected (face neighbours) everywhere: labelling, percolation, Euler characteristic,
  diffusion, pore networks and LBM (diagonal lattice links through edge-only contacts are closed).
- The faces of a VOI are symmetry planes for every transport solver: no-flux lateral faces for diffusion,
  mirrored flow axis and specular lateral faces for the LBM, and mirror completion of the pores and throats
  cut by the faces for the pore networks (`pnm.mirror_geometry`; the networks as extracted, whose faces act
  as walls, are kept for comparison).
- Units: lengths in mm in the tables (voxel size in `config.yaml`), permeability in m² and mD.
- Every number in the article is read from `outputs/report/key_numbers.json`; none is typed by hand.

## Licence and citation

Code: MIT (`LICENSE`). Data and outputs: CC BY 4.0 (`LICENSE-data.txt`). Please cite the article and this
code (`CITATION.cff`). The micro-CT data were acquired at the Laboratório de Tomografia Computadorizada de
Raios X (LTC-RX), Departamento de Energia Nuclear, UFPE, within publicly funded research projects, and are
released courtesy of LTC-RX.
