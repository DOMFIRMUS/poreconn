"""Stage `air`: (re)measure the grey value of the air around each rock plug for the existing VOIs.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Reads five raw slices spread over the slice range of each VOI and updates the outside-air fields of its
`_meta.json` (extract.window_air): the median of the air in the annulus 20-100 px beyond the plug surface
(the reference used by the segmentation) and the median of the air farther out (far field, for the QA).
Needs the raw reconstructions; the VOIs themselves are not touched. Bead columns are skipped: the tube
fills the field of view, so their air reference is the pore-air mode inside the VOI.
"""
from __future__ import annotations

from .config import Config, read_json, write_json
from .extract import AIR_NA, list_slices, voi_paths, window_air
from .segment import replicates


def run(cfg: Config, only=None, log=print) -> None:
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        if s.group == "granular":
            for k in replicates(cfg, s.id):
                fmeta = voi_paths(cfg, s.id, k)[1]
                m = read_json(fmeta)
                m.update(AIR_NA)
                write_json(fmeta, m)
            continue
        files = list_slices(cfg.p("raw_root") / s.raw)
        for k in replicates(cfg, s.id):
            fmeta = voi_paths(cfg, s.id, k)[1]
            m = read_json(fmeta)
            geo = {"cy": m["centre_y"], "cx": m["centre_x"], "r": m["sample_radius_px"]}
            old = m.get("air_outside_median")
            m.update(window_air(files, m["slice_index_start"], m["shape"][0] * m["bin"], geo))
            write_json(fmeta, m)
            log(f"  {s.id} v{k}: outside air {m['air_outside_median']:.0f} (was {float(old or 'nan'):.0f}), "
                f"far field {m['air_far_median']:.0f}, {m['air_outside_pixels_per_slice']} px per slice, "
                f"margin {m['air_outside_margin_px']:g} px")
