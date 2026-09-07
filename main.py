#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:07:20 2026

@author: hjialin
"""
import numpy as np
import matplotlib.pyplot as plt
from config import SpectralRange, SampleParams, IncidenceParams, SimulationConfig, IterationsParams
from theory import fct_run
from dataclasses import replace
import argparse


def read_ref_data(data_file):
    with open(data_file) as file:
        data = file.readlines()
    for i in range(41):
        data[i] = data[i].split()
    data = np.array(data,dtype=float)
    return data



config = SimulationConfig(
    wavelength_range = SpectralRange(350, 751, 3),
    sample = SampleParams(
        radius_nm=20,
        filling_fraction=0.05,
        packing_fraction=0.2,
        # layer_position=150,
        # structure='homogeneous',
        configuration_model='RSA',
        # thickness=180,
        ), 
    incidence = IncidenceParams(
        theta_deg=0,
        polarization_mode='TE'
        ), 
    iteration_params = IterationsParams(
        # write_result=True
        ))



def Figure(n_fig):
    fig,ax = plt.subplots(2,3, figsize=(27,8))
    wavelength_range_ref = np.arange(350, 751,10)   
    
    if n_fig == 'Figure3':
        case = [(0,'TE'), (60,'TE'),(60,'TM')]
    elif n_fig == 'Figure4' or n_fig == 'Figure5':
        case = [('RSA', 0.2), ('RSA', 0.45), ('SHU', 0.05)]
    
    fs = config.sample.filling_fraction
    chi = config.sample.chi
    layer_position = config.sample.layer_position
    radius = config.sample.radius_nm
    wavelength_range = config.wavelength_range.array()
    angle = config.incidence.theta_deg
    polarization = config.incidence.polarization_mode
    
    
    for i,parameters in enumerate(case):
        if n_fig == 'Figure3' or n_fig == 'Figure4':
            if n_fig == 'Figure3':
                y_max = 0.4
                config_new = replace(config, 
                                     incidence=replace(config.incidence, theta_deg=parameters[0], polarization_mode=parameters[1]))
                p = config.sample.packing_fraction
                distance_factor = 2*np.sqrt(p/fs)
                distance_factor_str = str(int(round(distance_factor)))
                file_ref = (f'./numerical_ref/(TiO2_TiO2+NPs Ag_TiO2)_RSA_fs={fs}_r={radius}_h={layer_position}_{parameters[0]}° '+parameters[1]+f'(d={distance_factor_str}r).txt')
                figure_title =  (f'{parameters[0]}°  {parameters[1]}')
            
            elif n_fig == 'Figure4':
                y_max = 0.6
                config_new = replace(config, 
                                     sample = replace(config.sample, configuration_model=parameters[0],packing_fraction=parameters[1]))
                distance_factor = 2*np.sqrt(parameters[1]/fs)
                distance_factor_str = str(int(round(distance_factor)))
                if parameters[0] == 'SHU':
                    file_ref = (f'./numerical_ref/(TiO2_TiO2+NPs Ag_TiO2)_{parameters[0]}_[chi={chi}]_fs={fs}_r={radius}_h={layer_position}_{angle}° '+polarization+'.txt')
                    figure_title = (f'{parameters[0]}  \u03c7=0.50')
                else:
                    file_ref = (f'./numerical_ref/(TiO2_TiO2+NPs Ag_TiO2)_{parameters[0]}_fs={fs}_r={radius}_h={layer_position}_{angle}° '+polarization+f'(d={distance_factor_str}r).txt')
                    figure_title =  (f'{parameters[0]}  p={parameters[1]}')
            result_ref = read_ref_data(file_ref)
            
            R, T = fct_run(config_new, wavelength_range)
            
            ax[0,i].errorbar(wavelength_range_ref, result_ref[:,2], yerr=result_ref[:,3], fmt='o', ms=6, color='blue', label='Num', zorder=1, capsize=5)
            ax[1,i].errorbar(wavelength_range_ref, result_ref[:,10], yerr=result_ref[:,11], fmt='o', ms=6, color='blue', label='Num', zorder=1, capsize=5)
            
            ax[0,i].plot(wavelength_range, R, ls='-', lw=2.5, color='black', label='QCA', zorder=5)
            ax[1,i].plot(wavelength_range, T, ls='-', lw=2.5, color='black', label='QCA', zorder=5)
            
            ax[0,i].set_ylabel('Reflectance')
            ax[1,i].set_ylabel('Transmission')
            ax[0,i].tick_params(axis='x',labelbottom=False)
            ax[0,i].set_ylim(0, y_max)
            ax[1,i].set_ylim(0, 1)
            ax[0,i].legend()
            ax[1,i].legend()    
            ax[0,i].set_title(figure_title)
            
            
        elif n_fig == 'Figure5':
            config_layer_150 = replace(config, 
                                 sample = replace(config.sample, configuration_model=parameters[0],packing_fraction=parameters[1],structure='sub_Ag+Air'))
            config_layer_110 = replace(config, 
                                 sample = replace(config.sample, 
                                                  configuration_model=parameters[0],
                                                  packing_fraction=parameters[1],
                                                  structure='sub_Ag+Air',
                                                  layer_position=110))
            distance_factor = 2*np.sqrt(parameters[1]/fs)
            distance_factor_str = str(int(round(distance_factor)))
            if parameters[0] == 'SHU':
                file_ref_layer_150 = (f'./numerical_ref/(Sub Ag_TiO2+NPs Ag_Air)_{parameters[0]}_[chi={chi}]_fs={fs}_r={radius}_h={150}_{angle}° '+polarization+'.txt')
                file_ref_layer_110 = (f'./numerical_ref/(Sub Ag_TiO2+NPs Ag_Air)_{parameters[0]}_[chi={chi}]_fs={fs}_r={radius}_h={110}_{angle}° '+polarization+'.txt')
                figure_title = (f'{parameters[0]}  \u03c7=0.50')
            else:
                file_ref_layer_150 = (f'./numerical_ref/(Sub Ag_TiO2+NPs Ag_Air)_{parameters[0]}_fs={fs}_r={radius}_h={150}_{angle}° '+polarization+f'(d={distance_factor_str}r).txt')
                file_ref_layer_110 = (f'./numerical_ref/(Sub Ag_TiO2+NPs Ag_Air)_{parameters[0]}_fs={fs}_r={radius}_h={110}_{angle}° '+polarization+f'(d={distance_factor_str}r).txt')
                figure_title =  (f'{parameters[0]}  p={parameters[1]}')
            result_ref_layer_150 = read_ref_data(file_ref_layer_150)
            result_ref_layer_110 = read_ref_data(file_ref_layer_110)
            
            R_layer_150, T_layer_150 = fct_run(config_layer_150, wavelength_range)
            R_layer_110, T_layer_110 = fct_run(config_layer_110, wavelength_range)
            
            ax[0,i].errorbar(wavelength_range_ref, result_ref_layer_150[:,2], yerr=result_ref_layer_150[:,3], fmt='o', ms=6, color='blue', label='Num', zorder=1, capsize=5)
            ax[1,i].errorbar(wavelength_range_ref, result_ref_layer_110[:,2], yerr=result_ref_layer_110[:,3], fmt='o', ms=6, color='blue', label='Num', zorder=1, capsize=5)
            
            ax[0,i].plot(wavelength_range, R_layer_150, ls='-', lw=2.5, color='black', label='QCA', zorder=5)
            ax[1,i].plot(wavelength_range, R_layer_110, ls='-', lw=2.5, color='black', label='QCA', zorder=5)
            
            ax[0,i].set_ylabel('Reflectance')
            ax[1,i].set_ylabel('Reflectance')
            ax[0,i].set_ylim(0, 1)
            ax[1,i].set_ylim(0, 1)
            ax[0,i].tick_params(axis='x',labelbottom=False)
            ax[0,i].legend()
            ax[1,i].legend()    
            ax[0,i].set_title(figure_title)
    fig_name = (f'./figure/{n_fig}.svg')
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "figure", 
        choices=["Figure3", "Figure4","Figure5"],
        help="Figure to reproduce")
    
    args = parser.parse_args()
    
    Figure(args.figure)
    

        
    
        
