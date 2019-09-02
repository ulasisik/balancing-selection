# Simulations

Simulations were performed using SLiM. For selection scenarios, we used 21 different selection start times, starting from 20k to 40k years ago. We divide the time of onset of selection into 3 categories: 
- **Recent** includes the selection start times ranging from 20k to 26k years ago (i.e. 20, 21,..., 26kya)
- **Medium** includes the selection start times ranging from 27k to 33k years ago (i.e. 27, 28,..., 33kya)
- **Old** includes the selection start times ranging from 34k to 40k years ago (i.e. 34, 35,..., 40kya)


## Incomplete Sweep & Overdominance

In SLiM, fitness effect of a mutation(i.e. _B_) is calculated by the selection coefficient (_s_) and dominance coefficient (_h_) of the mutation (the latter used only if the individual is heterozygous):

- w<sub>_AA_</sub> = _w_ 
- w<sub>_BB_</sub> = _w_ * (1.0 + _s_)
- w<sub>_AB_</sub> = _w_ * (1.0 + _s_ * _h_)

### Incomplete Sweeep

For incomplete sweep, we set dominance coefficient _h_ = 0.5 and used selection coefficient _s_ values given at the below table for each time of onset of selection:

| Selection Start Time (kya)  | Selection Coefficient (_s_) |
| ------------- | ------------- |
|20| 0.019|
|21|0.018|
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

### Overdominance


