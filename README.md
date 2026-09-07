### Code for Multiple scattering model for the coherent optical response of correlated disordered plasmonic metasurfaces



This repository contains the code used to reproduce the theoretical calculations presented in:

Multiple scattering model for the coherent optical response of correlated disordered plasmonic metasurfaces

J.HU, K.VYNCK

Advanced Optical Materials



The code provides a simplified implementation of the theoretical model used in the paper and is intended primarily for reproducing the reported results.





##### Requirements

The code was developped and tested with:

Python 3.8.20

NumPy 1.24.4

SciPy 1.10.1

Matplotlib



The required packages can be installed using:

pip install numpy scipy matplotlib





##### Code structure

The main files are:

###### config.py

&#x09;Contains the physical and numerical parameters used in the calculations

###### theory.py

&#x09;Contains the main theoretical expressions and numerical procédures, including the calculation of \[g2(r), coherent reflection/transmission coefficient of the particle monolayer and  reflection/transmission coefficient of multiple-layered structure]

###### main.py

&#x09;Main script used to reproduce the calculations presented in the paper.





##### Usage

The parameters can be modified in *config.py*.



The figures presented in the paper can be reproduced using the functions

provided in the *main.py* script.



To reproduce Figure 3-5:



python main.py Figure("Figure3")

python main.py Figure("Figure4")

python main.py Figure("Figure5")



The required input data and calculation parameters are automatically loaded

from the configuration file.



The calculated results are saved in:

`results/`

The figures are saved in:

`figure/`







##### Theory data

The complete theoretical data corresponding to Figures 3–5 are provided in the `theory_data/` directory for reference:

`theory\_data/`

├──`Figure3/`

├──`Figure4/`

├──`Figure5/`



These folders contain the calculated quantities used in the theoretical analysis, including the pair-correlation function g2, polarizability tensor, and reflection and transmission coefficients.



The figure-generation functions do not directly read data from `theory\_data/`. During reproduction, the required input data, such as the precomputed g2 and particle coordinates, are loaded from the `results/` directory. The subsequent quantities are then recalculated and the corresponding outputs are also saved in `results/`.







##### Calculation workflow

The main calculation follows the sequence:



Input parameters

&#x09;↓

Generation of particle configurations

&#x09;↓

Calculation of the pair-correlation function g2(r)

&#x09;↓

Calculation of the polarizability tensor

&#x09;↓

Calculation of the reflection/transmission coefficients of the particle monolayer

&#x09;↓

Calculation of the reflection/transmission coefficients of the multiple-layered structure

&#x09;↓

Optical Spectrum



The implementation follows the equations described in the main text and Supporting Information of the paper.











##### Reproducibility

For calculations involving randomly generated particle configurations, a fixed random seed (seed = 0) is used by default to ensure reproducibility.



The numerical parameters used to generate the results reported in the paper are provided in *config.py*.



Due to the stochastic nature of the configuration generation, results may differ slightly if the random seed or the number of realizations is modified.









##### Units

Unless otherwise specified:

wavelength: nm

particle radius: nm

angle: degree

surface filling fraction/packing fraction: dimensionless









##### Notes

This repository contains a simplified version of the research code. Only the components required to reproduce the calculations reported in the paper are included.



The code is provided for research and reproducibility purposes.





