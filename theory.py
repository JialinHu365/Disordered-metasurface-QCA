# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:59:59 2026

@author: hjialin
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.special import jv,hankel1
import os


    
    
    
'''
Polarizability 
'''
        
    
def index_refraction(wavelength_range):        
    index_Ag_list = []
    index_TiO2_list = []

    doc_Ag = './refraction_index/n_Ag.txt'    
    doc_TiO2 = './refraction_index/n_TiO2.txt'

    with open(doc_Ag,'r') as file_Ag:
        data_Ag = file_Ag.readlines()   
    with open(doc_TiO2,'r') as file_TiO2:
        data_TiO2 = file_TiO2.readlines()    
      
        
    for i in range(len(data_Ag)):
        data_Ag[i] = data_Ag[i].split()
        index_Ag_list.append(complex(float(data_Ag[i][1]),float(data_Ag[i][2])))
            
    
    index_Ag_interp = interp1d(np.arange(350,751,10), index_Ag_list,fill_value='extrapolate')
    index_Ag_list = index_Ag_interp(wavelength_range)            
        
    for i in range(len(data_TiO2)):
        data_TiO2[i] = data_TiO2[i].split()

        if float(data_TiO2[i][0]) in wavelength_range:
            index_TiO2_list.append(float(data_TiO2[i][1]))   

    n_Ag_list = np.array(index_Ag_list)   
    n_TiO2_list = np.array(index_TiO2_list)     
  
    return n_Ag_list, n_TiO2_list





def fct_polarizability_Mie(radius, wavelength_range, n_particle, n_host):
    """
    Calculate the electric dipole polarizability of a spherical particle using Mie theory.

    Parameters
    ----------
    radius : float
        Particle radius.
    wavelength_range : ndarray
        Wavelength range.
    n_particle : TYPE
        Complex refractive index of the particle over the wavelength range.
    n_host : TYPE
        Complex refractive index of the host medium over the wavelength range.

    Returns
    -------
    polarizability_Mie : ndarray
        Electric dipole polarizability obtained from the first-order electric Mie coefficient.
    """
    
    def psi(x,n=1):
        psi = np.sqrt(np.pi*x/2)*jv(n+1/2,x)
        return psi
    
    def psi_d(x,n=1):
        psi_d = np.sqrt(np.pi/(2*x))*((1+n)*jv(n+1/2,x)-x*jv(n+3/2,x))
        return psi_d


    def xi(x,n=1):
        xi = np.sqrt(np.pi*x/2)*hankel1(n+1/2,x)
        return xi

    def xi_d(x,n=1):
        xi_d = np.sqrt(np.pi/(2*x))*((1+n)*hankel1(n+1/2,x)-x*hankel1(n+3/2,x))
        return xi_d

       
    
    kb = 2*np.pi*n_host/wavelength_range
    x = kb*radius
    m = n_particle/n_host
    y = m*x
    a1 = (m*psi(y)*psi_d(x)-psi(x)*psi_d(y))/(m*psi(y)*xi_d(x)-xi(x)*psi_d(y))
    polarisability_Mie = 1j*a1*6*np.pi/np.power(kb,3)        
    return polarisability_Mie    



'''
Dressed polarizability
'''

def fct_period(integrate_xx,integrate_yy,integrate_zz):
    delta_r = 0.0005
    def dominant_period_index(signal):
        n_data = len(signal)

        fft_values = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(n_data, d=delta_r)

        magnitude = np.abs(fft_values)

        # Exclude the zero-frequency component.
        peak_index = np.argmax(magnitude[1:]) + 1
        if 1<= peak_index < len(magnitude)-1:
            y1 = magnitude[peak_index-1]
            y2 = magnitude[peak_index]
            y3 = magnitude[peak_index+1]
            delta = 0.5*(y1-y3)/(y1-2*y2+y3)
            refinded_index = peak_index+delta
        else:
            refinded_index = peak_index
        df = frequencies[1] - frequencies[0]
        refinded_freq = refinded_index*df
        period = 1/refinded_freq

        period_index = round(period / delta_r)
        n_period_max = int(n_data/period_index)
        if n_period_max<1:
            
            n_period_max = 1
        return period_index, n_period_max
    period_xx_index_real, n_period_xx_real = dominant_period_index(integrate_xx.real)
    period_yy_index_real, n_period_yy_real = dominant_period_index(integrate_yy.real)
    period_zz_index_real, n_period_zz_real = dominant_period_index(integrate_zz.real)

    period_xx_index_imag, n_period_xx_imag = dominant_period_index(integrate_xx.imag)
    period_yy_index_imag, n_period_yy_imag = dominant_period_index(integrate_yy.imag)
    period_zz_index_imag, n_period_zz_imag = dominant_period_index(integrate_zz.imag)
   
    result_period_index = (period_xx_index_real, period_yy_index_real, period_zz_index_real,period_xx_index_imag, period_yy_index_imag, period_zz_index_imag)
    result_n_period = (n_period_xx_real, n_period_yy_real, n_period_zz_real, n_period_xx_imag, n_period_yy_imag, n_period_zz_imag)
    return result_period_index, result_n_period



    

def fct_polarizability_tensor(angle, fs, radius, distance_factor, chi, configuration_model, wavelength_range, resolution, n_host, n_particle, structure, write=False):
    '''
    Calculate the dressed polarizability tensor including particle correlations.
    
    Parameters
    ----------
    angle : float
        Incident angle in degrees.
    fs : float
        Surface filling fraction.
    radius : float
        Particle radius in nm.
    distance_factor : float 
        Minimum center-to-center distance normalized by the particle radius.
    chi : str 
        Stealthiness parameter used for the SHU model.
    configuration_model : str 
        Particle-correlation model: 'SHU', 'alea', or 'hole'.
    wavelength_range : ndarray 
        Wavelength range in nm. 
    resolution : int 
        Resolution identifier used for saving the result. 
    n_host : ndarray 
        Refractive index of the host medium. 
    n_particle : ndarray 
        Refractive index of the particle. 
    structure : str 
        Optical structure used to determine the propagation angle. 
    save : bool, optional 
        If True, save the calculated dressed polarizability. Default is False.

    Returns
    -------
    tensor_polarizability : ndarray 
        Dressed polarizability tensor for each wavelength.

    '''
    def fct_integrale_xx_0(r,kb):
        return  np.exp(1j*kb*r)*(1+(1/(1j*kb*r)+1/np.power(kb*r,2)))/4 

    def fct_integrale_zz_0(r,kb):
        return  np.exp(1j*kb*r)*(1-(1/(1j*kb*r)+1/np.power(kb*r,2)))/2 #np.exp(1j*kb*r)/2#


    def fct_integrale_xx(r,kb,theta):
        return np.exp(1j*kb*r)*(jv(1,kb*r*np.sin(theta))/(kb*r*np.sin(theta))-(1/(1j*kb*r)+1/np.power(kb*r,2))*(3*jv(1,kb*r*np.sin(theta))/(kb*r*np.sin(theta))-2*jv(0,kb*r*np.sin(theta))))/2 #np.exp(1j*kb*r)*(jv(1,kb*r*np.sin(theta))/(kb*r*np.sin(theta)))/2#


    def fct_integrale_yy(r,kb,theta):
        return np.exp(1j*kb*r)*(jv(0,kb*r*np.sin(theta))-jv(1,kb*r*np.sin(theta))/(kb*r*np.sin(theta))-(1/(1j*kb*r)+1/np.power(kb*r,2))*(jv(0,kb*r*np.sin(theta))-3*jv(1,kb*r*np.sin(theta))/(kb*r*np.sin(theta))))/2 #np.exp(1j*kb*r)*(jv(0,kb*r*np.sin(theta))-jv(1,kb*r*np.sin(theta))/(kb*r*np.sin(theta)))/2#


    def fct_integrale_zz(r,kb,theta):
        return np.exp(1j*kb*r)*(jv(0,kb*r*np.sin(theta))*(1- (1/(1j*kb*r)+1/np.power(kb*r,2))))/2    
        
    
    def cumulative_integral(integrand, g_r, r_values, index_lim, n_integration_steps):
        return np.array([
            simpson(
                integrand[:index_lim + j] * g_r[:index_lim + j],
                x=r_values[:index_lim + j]
            )
            for j in range(n_integration_steps)
        ])
    
    # Convert radius and wavelength from nm to µm.
    radius_um = radius*1e-3   
    wavelength_range = wavelength_range*1e-3
    
    polarizability_Mie = fct_polarizability_Mie(radius_um, wavelength_range,n_particle, n_host)
    
    number_density = fs/(np.pi*np.power(radius_um,2))
  
    rmin = distance_factor*radius*1e-3 
    dr =  0.0005 
    r_max = 7.5
    integration_start = 10001
    n_integration_steps = 5000
    
    r_range = np.arange(0,r_max,dr)

    identity_matrix = np.diag([1,1,1])
    
    tensor_polarizability = np.zeros((len(wavelength_range),3),dtype=complex)
    
    k_host = 2*np.pi/wavelength_range*n_host
    
    theta = np.deg2rad(angle)
    
    if configuration_model == 'hole':
        index_rmin = np.argwhere(r_range>=rmin)[0][0]
        g2_extended = np.ones_like(r_range[index_rmin:])
    else:
        if configuration_model == 'SHU':
            g2_file = (f'./Results/g2/{configuration_model}_[chi={chi}]_g2_fs={fs}_r={radius}.npy')
        elif configuration_model == 'RSA':  
            if np.isclose(distance_factor, round(distance_factor)):
                distance_factor_str = str(int(round(distance_factor)))
            else:
                distance_factor_str = f"{distance_factor:.3f}".rstrip("0").rstrip(".")
            g2_file = (f'./Results/g2/{configuration_model}_g2_fs={fs}_r={radius}_({distance_factor_str}r).npy')
        g2 = np.load(g2_file)  
        g2_extended = np.ones_like(r_range)
        g2_extended[:len(g2)] = g2
        g2_extended = g2_extended[1:]
        index_rmin = 1    
    if angle == 0:
        fct_xx = np.array([[fct_integrale_xx_0(r_range[j], k_host[i]) for j in range(index_rmin,len(r_range))] for i in range(len(wavelength_range))])
        fct_zz = np.array([[fct_integrale_zz_0(r_range[j], k_host[i]) for j in range(index_rmin,len(r_range))] for i in range(len(wavelength_range))])
        fct_yy = fct_xx
    else:
        if structure == 'sub_Ag+Air':
            theta_list = np.arcsin(np.sin(theta)/n_host)
        else:
            theta_list = np.ones_like(wavelength_range)*theta
        fct_xx = np.array([[fct_integrale_xx(r_range[j], k_host[i], theta_list[i]) for j in range(index_rmin,len(r_range))] for i in range(len(wavelength_range))])
        fct_yy = np.array([[fct_integrale_yy(r_range[j], k_host[i], theta_list[i]) for j in range(index_rmin,len(r_range))] for i in range(len(wavelength_range))])
        fct_zz = np.array([[fct_integrale_zz(r_range[j], k_host[i], theta_list[i]) for j in range(index_rmin,len(r_range))] for i in range(len(wavelength_range))])
    
    index_lim = integration_start-index_rmin
    for i in range(len(wavelength_range)): 
        integrate_xx = cumulative_integral(fct_xx[i,:], g2_extended, r_range[index_rmin:], index_lim, n_integration_steps)
        integrate_yy = cumulative_integral(fct_yy[i,:], g2_extended, r_range[index_rmin:], index_lim, n_integration_steps)
        integrate_zz = cumulative_integral(fct_zz[i,:], g2_extended, r_range[index_rmin:], index_lim, n_integration_steps)
        

        period_index, n_period = fct_period(integrate_xx, integrate_yy, integrate_zz)
        mean_xx_real = np.mean(integrate_xx.real[:period_index[0]*n_period[0]])
        mean_xx_imag = np.mean(integrate_xx.imag[:period_index[3]*n_period[3]])
        mean_yy_real = np.mean(integrate_yy.real[:period_index[1]*n_period[1]])
        mean_yy_imag = np.mean(integrate_yy.imag[:period_index[4]*n_period[4]])
        mean_zz_real = np.mean(integrate_zz.real[:period_index[2]*n_period[2]])
        mean_zz_imag = np.mean(integrate_zz.imag[:period_index[5]*n_period[5]])
        mean_xx = complex(mean_xx_real, mean_xx_imag)
        mean_yy = complex(mean_yy_real, mean_yy_imag)
        mean_zz = complex(mean_zz_real, mean_zz_imag)
        
        interaction_matrix = identity_matrix - np.diag([mean_xx,mean_yy,mean_zz])*polarizability_Mie[i]*number_density*np.power(k_host[i],2)
        matrice_inv = np.linalg.inv(interaction_matrix)
 
        tensor_polarizability[i,0] = matrice_inv[0,0] 
        tensor_polarizability[i,1] = matrice_inv[1,1] 
        tensor_polarizability[i,2] = matrice_inv[2,2] 
    if write:
        if configuration_model == 'SHU':
            tensor_file = (f'./Results/polarizability_tensor/resolution={resolution}/{configuration_model}_[chi={chi}]_fs={fs}_r={radius}_{angle}°')
        else:     
            if np.isclose(distance_factor, round(distance_factor)):
                distance_factor_str = str(int(round(distance_factor)))
            else:
                distance_factor_str = f"{distance_factor:.3f}".rstrip("0").rstrip(".")
            tensor_file = (f'./Results/polarizability_tensor/resolution={resolution}/{configuration_model}_fs={fs}_r={radius}_{angle}°_({distance_factor_str}r)')
        np.save(tensor_file, tensor_polarizability)
    return  tensor_polarizability



'''
Reflection & Transmission Coefficients
'''
def fct_r_t (fs, radius, angle, configuration_model, distance_factor, chi, polarization_mode, wavelength_range, resolution, n_host, n_particle, structure, write=False):
    """
    Calculate the reflection and transmission coefficients of the
    correlated particle layer for TE and TM polarizations.

    Parameters
    ----------
    fs : float
        Surface filling fraction.
    radius : float
        Particle radius in nm.
    angle : float
        Incident angle in degrees.
    configuration_model : str
        Particle-correlation model.
    distance_facteur : float
        Minimum center-to-center distance normalized by the particle radius.
    chi : str
        Stealthiness parameter used for the SHU model.
    polarization_mode : 
        
    wavelength_range : ndarray
        Wavelength range in nm.
    resolution : int
        Resolution identifier used for loading and saving data.
    n_host : ndarray
        Refractive index of the host medium.
    n_particle : ndarray
        Refractive index of the particle.
    write : bool, optional
        If True, save the TE and TM coefficients. The default is False.

    Returns
    -------
   r_TE, t_TE, r_TM, t_TM : ndarray
        Complex reflection and transmission coefficients for TE and TM
        polarizations.
    """
    
    # Convert radius and wavelength from nm to µm.
    radius_um = radius*1e-3
    wavelength_range = wavelength_range*1e-3
    
    k_host = 2*np.pi/wavelength_range*n_host
    polarizability_Mie = fct_polarizability_Mie(radius_um, wavelength_range, n_particle, n_host)
    
    if configuration_model == 'SHU':
        tensor_file =  (f'./Results/polarizability_tensor/resolution={resolution}/{configuration_model}_[chi={chi}]_fs={fs}_r={radius}_{angle}°.npy')
    else:
        if np.isclose(distance_factor, round(distance_factor)):
            distance_factor_str = str(int(round(distance_factor)))
        else:
            distance_factor_str = f"{distance_factor:.3f}".rstrip("0").rstrip(".")
        tensor_file = (f'./Results/polarizability_tensor/resolution={resolution}/{configuration_model}_fs={fs}_r={radius}_{angle}°_({distance_factor_str}r).npy')
    
    tensor_polarizability = np.load(tensor_file)
    number_density = fs/(np.pi*np.power(radius_um,2))
    theta = np.deg2rad(angle)
    
    if structure == 'sub_Ag+Air':
        theta = np.arcsin(np.sin(theta)/n_host)
    else:
        theta = np.ones_like(wavelength_range)*theta
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    gamma_xx = tensor_polarizability[:,0]
    gamma_yy = tensor_polarizability[:,1]
    gamma_zz = tensor_polarizability[:,2]
    
    prefactor = (number_density*1j*k_host*polarizability_Mie/(2*cos))
    
    if polarization_mode == 'TE':
        r = prefactor*gamma_yy
        t = 1 + r
    elif polarization_mode == 'TM':
        r = prefactor*(np.power(cos,2)*gamma_xx-np.power(sin,2)*gamma_zz)
        t = 1 + prefactor*(np.power(cos,2)*gamma_xx+np.power(sin,2)*gamma_zz)
    else:
        raise ValueError("polarization must be 'TE' or 'TM'.")
    
    if write:
        base_path = (
            f'./Results/reflection_transmission_coefficients/resolution={resolution}/'
            )
        
        if configuration_model == 'SHU':
            base_name = (
                f'{configuration_model}_[chi={chi}]_fs={fs}_r={radius}_{angle}°'
                )
        else:   
            if np.isclose(distance_factor, round(distance_factor)):
                distance_factor_str = str(int(round(distance_factor)))
            else:
                distance_factor_str = f"{distance_factor:.3f}".rstrip("0").rstrip(".")
            base_name = (
                f'{configuration_model}_fs={fs}_r={radius}_{angle}°_{distance_factor_str}r'
                )
        np.save(base_path+base_name+'_'+polarization_mode, np.column_stack((r, t)))
    return r,t



def fct_r_t_Fresnel(theta1, polariz, n1, n2):
    """
    Calculate Fresnel reflection and transmission coefficients.

    Parameters
    ----------
    theta1 : float
        Incident angle in rad.
    polariz : str
        Polarization state, either 'TE' or 'TM'.
    n1 : float or ndarray
        Refractive index of the incident medium.
    n2 : float or ndarray
        Refractive index of the transmitted medium.

    Returns
    -------
    r, t : complex or ndarray
        Fresnel reflection and transmission coefficients.
    """
    

    cos_theta2 = np.sqrt(1 - np.power(n1/n2*np.sin(theta1),2))
        
    if polariz == 'TE':
        r = (n1*np.cos(theta1) - n2*cos_theta2)/(n1*np.cos(theta1) + n2*cos_theta2)
        t = 2*n1*np.cos(theta1)/(n1*np.cos(theta1) + n2*cos_theta2)
    elif polariz == 'TM':
        r = (n1*cos_theta2 - n2*np.cos(theta1))/(n1*cos_theta2 + n2*np.cos(theta1))
        t = 2*n1*np.cos(theta1)/(n1*cos_theta2 + n2*np.cos(theta1))
    else:
        raise ValueError("polarization must be 'TE' or 'TM'.")
    return r,t




def fct_run(config, wavelength_range):
    """
    Run the complete calculation workflow using the specified configuration and wavelength range.

    """
    
    radius = config.sample.radius_nm
    p = config.sample.packing_fraction
    fs = config.sample.filling_fraction
    chi = config.sample.chi
    angle = config.incidence.theta_deg
    h = config.sample.layer_position
    configuration_model = config.sample.configuration_model
    structure = config.sample.structure
    polarization_mode = config.incidence.polarization_mode
    write = config.iteration_params.write_result
    resolution = config.wavelength_range.resolution
    thickness = config.sample.thickness
    n_Ag, n_TiO2 = index_refraction(wavelength_range)
    
    if structure == 'sub_Ag+Air':
        base_name = 'Sub Ag_TiO2+NPs Ag_Air'
    elif structure=='homogeneous':
        base_name = 'TiO2_TiO2+NPs Ag_TiO2'
    elif structure == 'sub_Ag':
        base_name = 'Sub Ag_TiO2+NPs Ag_TiO2'
    
    
    if p < fs:
        raise ValueError('packing fraction must be larger than filling fraction')
    
    distance_factor = 2*np.sqrt(p/fs)
    
    
    print(f'Generating polarizability tensor for structure {base_name} configuration {configuration_model} with fs={fs} p={p} incident angle={angle} {polarization_mode} ...')
    polarizability_tensor = fct_polarizability_tensor(angle, fs, radius, distance_factor, chi, configuration_model, wavelength_range, resolution, n_TiO2, n_Ag, structure, write=True)
    
    print('Generating reflection and transmission coefficients of metasurface...')
    r_coh, t_coh = fct_r_t(fs, radius, angle, configuration_model, distance_factor, chi, polarization_mode, wavelength_range, resolution, n_TiO2, n_Ag, structure, write=True)
      
    print('Generating reflection and transmission spectrums')
   
  
    if structure == 'homogeneous':
        r_st = r_coh
        t_st = t_coh
    else:
        angle_rad = np.deg2rad(angle)
        k_host = 2*np.pi/wavelength_range*n_TiO2
        t_st = np.zeros_like(wavelength_range)
        if structure == 'sub_Ag':
            theta = np.ones_like(wavelength_range)*angle_rad
            r_TiO2_Ag, t_TiO2_Ag = fct_r_t_Fresnel(theta, polarization_mode,n_TiO2,n_Ag)
            r_st = r_coh + np.power(t_coh,2)*r_TiO2_Ag/(np.exp(-2*1j*h*np.cos(theta)*k_host)-r_TiO2_Ag*r_coh)
                 
        elif structure == 'sub_Ag+Air':     
            theta = np.arcsin(np.sin(angle_rad)/n_TiO2)
            n_Air = np.ones_like(wavelength_range) 
            theta_2 = theta
            delta_L = thickness-h
            r_TiO2_Ag, t_TiO2_Ag = fct_r_t_Fresnel(theta, polarization_mode,n_TiO2,n_Ag)
            prefactor = r_coh + np.power(t_coh,2)*r_TiO2_Ag/(np.exp(-2*1j*h*np.cos(theta)*k_host)-r_TiO2_Ag*r_coh)
            r_Air_TiO2, t_Air_TiO2 = fct_r_t_Fresnel(angle_rad, polarization_mode, n_Air, n_TiO2)
            r_TiO2_Air, t_TiO2_Air = fct_r_t_Fresnel(theta, polarization_mode, n_TiO2, n_Air)
            prefactor2 = np.exp(-2*1j*k_host*delta_L*np.cos(theta_2))/prefactor - r_TiO2_Air
            r_st = r_Air_TiO2 + t_TiO2_Air*t_Air_TiO2/prefactor2
    
    R = np.power(abs(r_st),2) 
    T = np.power(abs(t_st),2) 
            
    if write:
        if configuration_model == 'SHU': 
            file_spectrum = (f'./Results/Spectrum/resolution={resolution}/('+ base_name +f')_{configuration_model}_fs={fs}_r={radius}_h={h}_{angle}°_{polarization_mode}_(chi={chi})_W={thickness}nm.npy')
        else: 
            if np.isclose(distance_factor, round(distance_factor)):
                distance_factor_str = str(int(round(distance_factor)))
            else:
                distance_factor_str = f"{distance_factor:.3f}".rstrip("0").rstrip(".")
            file_spectrum = (f'./Results/spectrum/resolution={resolution}/('+base_name+ f')_{configuration_model}_fs={fs}_r={radius}_h={h}_{angle}°_{polarization_mode}_(d={distance_factor_str}r)_W={thickness}nm.npy')
 
        np.save(file_spectrum, np.column_stack((R,T)))

    return R, T
