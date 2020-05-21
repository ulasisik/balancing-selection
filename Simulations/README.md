# Simulations

We performed simulations for a neutral(NE) and three different selection scenarios, including incomplete sweep(IS), 
overdominance(OD) and negative frequency dependent selection(FD). Simulations were performed using SLiM. 
For selection scenarios, we used 21 different selection start times, starting from 20k to 40k years ago. 
We divide the time of onset of selection into 3 categories: 
- **Recent** includes the selection start times ranging from 20k to 26k years ago (i.e. 20, 21,..., 26kya)
- **Medium** includes the selection start times ranging from 27k to 33k years ago (i.e. 27, 28,..., 33kya)
- **Old** includes the selection start times ranging from 34k to 40k years ago (i.e. 34, 35,..., 40kya)

## Running simulations
Before running simulations, it is important to specify the paths for SLiM, simulation outputs and temporary slim files.
To do so, you should change `DIRSLIM`, `DIRDATA`, and `DIRTMP` values located in `Params.txt` file. By modifying `Params.txt`,
you can also change parameters of demographic model, mutation/recombination rate, sample size, etc. Parameters related to
selection (i.e. dominance and selection coefficients, time of onset of selection) and number of simulations can be 
modified from doXX(NE/IS/OD/FD).sh scripts.

To run simulations, user should run doXX(NE/IS/OD/FD).sh scripts. This scripts will create temporary SLiM scripts in
`DIRTMP`, and the outputs will be stored in `DIRDATA`. The output files will be in MS file format containing 198 
randomly sampled haploid chromosomes.

__Note__: The output files will be named in the format of `<CLASS>_<SELECTION.TIME>_<SIM.NUMBER>.txt` for selection scenarios,
and `<CLASS>_<SIM.NUMBER>.txt` for neutral scenario. So, for example, the first overdominance simulation 
for 20k years old selection will be named as `OD_20_1.txt`. It is __IMPORTANT__ not to change this naming pattern
since the BaSe module assumes this naming pattern, and changing file names may cause some functions not to work as expected.

## Simulating selection

### Incomplete Sweep & Overdominance

In SLiM, fitness effect of a mutation(i.e. _B_) is calculated by the selection coefficient (_s_) and dominance coefficient (_h_) of the mutation (the latter used only if the individual is heterozygous):

- w<sub>_AA_</sub> = _w_ 
- w<sub>_BB_</sub> = _w_ * (1.0 + _s_)
- w<sub>_AB_</sub> = _w_ * (1.0 + _s_ * _h_)

#### Incomplete Sweeep

For incomplete sweep, we set dominance coefficient _h_ = 0.5 and used selection coefficient _s_ values given at the below table for each time of onset of selection:

| Selection Start Time (kya)  | Selection Coefficient (_s_) |
| ------------- | ------------- |
|20| 0.019|
|21| 0.018|
|22| 0.017|
|23| 0.016|
|24| 0.016|
|25| 0.015|
|26| 0.014|
|27| 0.014|
|28| 0.013|
|29| 0.012|
|30| 0.011|
|31| 0.011|
|32| 0.010|
|33| 0.0098|
|34| 0.0085|
|35| 0.0077|
|36| 0.0075|
|37| 0.0073|
|38| 0.0070|
|39| 0.0067|
|40| 0.0064|

#### Overdominance

To simulate overdominance model, we set _s_ = 0.001 and used following dominance coefficient _h_ values:

| Selection Start Time (kya)  | Dominance Coefficient (_h_) |
| ------------- | ------------- |
|20| 20|
|21| 19|
|22| 18|
|23| 17|
|24| 16|
|25| 14|
|26| 14|
|27| 13|
|28| 12|
|29| 11|
|30| 10|
|31| 10|
|32| 9.5|
|33| 9|
|34| 8.5|
|35| 8|
|36| 7.5|
|37| 7|
|38| 6.5|
|39| 6|
|40| 6|

### Negative Frequency Dependent Selection

In order to simulate negative frequency dependent selection, we introduced the _fitness()_ callback as it can assign different fitness to the same polymorphism. In each generation, the relative fitness of the target polymorphism (mut) is calculated as follows:

- Relative Fitness<sub>i</sub>(mut) = _A_ - _B_ * Freq<sub>i</sub>(mut)

We used following _A_ and _B_ values to simulate negative frequency dependent selection:

| Selection Start Time (kya)  | _A_ | _B_ |
| ------------- | ------------- | ------------- |
|20| 1.030| 0.060|
|21| 1.029| 0.058|
|22| 1.028| 0.056|
|23| 1.027| 0.054|
|24| 1.026| 0.052|
|25| 1.025| 0.050|
|26| 1.024| 0.048|
|27| 1.023| 0.046|
|28| 1.022| 0.044|
|29| 1.021| 0.042|
|30| 1.020| 0.040|
|31| 1.019| 0.038|
|32| 1.018| 0.036|
|33| 1.017| 0.034|
|34| 1.016| 0.032|
|35| 1.015| 0.030|
|36| 1.014| 0.028|
|37| 1.013| 0.026|
|38| 1.012| 0.024|
|39| 1.011| 0.022|
|40| 1.010| 0.020|



