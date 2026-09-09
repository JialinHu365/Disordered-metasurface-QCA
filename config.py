# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:54:05 2026

@author: hjialin
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class SpectralRange:
    wl_min: int = 350
    wl_max: int = 751
    resolution: int = 3
    unit: str = 'nm'
    
    def array(self):
        return np.arange(self.wl_min, self.wl_max, self.resolution)
    
    
    
@dataclass(frozen=True)
class SampleParams:
    radius_nm: int    
    packing_fraction: float = 0.2
    filling_fraction: float = 0.05
    layer_position: int = 150
    configuration_model: str = 'RSA'
    structure: str='homogeneous'
    thickness: int = 180
    chi: str = '0.50'
    
    
@dataclass(frozen=True)
class IncidenceParams:
    theta_deg: int = 0
    polarization_mode: str = 'TE'
    

@dataclass(frozen=True)
class IterationsParams:
    write_result: bool = True

    
    
@dataclass(frozen=True)
class SimulationConfig:
    wavelength_range: SpectralRange
    sample: SampleParams
    incidence: IncidenceParams
    iteration_params: IterationsParams

    def to_dict(self):
        return asdict(self)