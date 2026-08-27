"""Spectrum -> sRGB, kept consistent with the colour-science pipeline already used by
ppap.py / R9_koumyou_v2.py (CIE 1931 2-degree observer, D65 illuminant)."""
from __future__ import annotations

import numpy as np
import colour


def spectrum_to_srgb(wl_nm, R):
    R = np.clip(np.asarray(R, dtype=float), 0.0, 1.0)
    sd = colour.SpectralDistribution(R, name='R')
    sd.wavelengths = np.asarray(wl_nm, dtype=float)

    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
    illuminant = colour.SDS_ILLUMINANTS['D65']

    XYZ = colour.sd_to_XYZ(sd, cmfs, illuminant)
    RGB = colour.XYZ_to_sRGB(XYZ / 100)
    rgb255 = np.clip(np.round(RGB * 255), 0, 255).astype(int)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb255)
    return hex_color, RGB, XYZ
