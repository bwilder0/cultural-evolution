# cultural-evolution
This code implements the simulation of cultural evolution for the paper “Inferring individual-level processes in cultural evolution from population-level patterns".

Dependencies
============

The code is all written for Python 2.7. The only external dependency is Numpy. For faster execution, the simulation is also fully compatible with PyPy.

Files
=====

1.  doSim.py: The main simulation loop

2.  transmission.py: Function to implement the different modes of transmission

3.  getDiscrete.py: Given a starting value of a discrete trait, returns a transmitted value (possibly perturbed by mutation)

4.  getContinuous.py: Returns a possibly mutated continuous trait

5.  histogram.py: Implementation of histogram function (c.f. numpy.histogram). Included for PyPy compatibility.

Usage
=====

Input
-----

To run the simulation, call the function doSim in doSim.py. doSim takes a single argument which is a tuple with all of the simulation parameters. This tuple should have entries

1.  pWithin: probability of transmission

2.  startingPop: the size of the population

3.  pMutate: probability of mutation in each transmission event

4.  numSims: Number of simulation runs to execute.

5.  pOblique: Probability that mixed transmission chooses the oblique mode (as opposed to horizontal).

6.  mutationStd: Standard deviation of noise added in mutation of continuous traits.

7.  discreteTraitBins: Number of variants for discrete traits.

8.  suffix: a string which is appended to the name of the output file.

9.  conformity: boolean value of whether transmission includes conformity.

10. conformityB: parameter of conformity strength.

Example call: doSim((0.5, 100, 0.05, 5, 0.2, 0.1, 5, ’test’, False, 0.2))

Output
------

doSim returns a string which is the name of the file that the results were written to. This file is a Python pickle file, which contains a dictionary with two fields: “params" contains another dictionary with the parameter values used for the run, and “results" contains the results from the simulation. Results are collected at each timestep in the array “coarseSteps"

Results is a dictionary which is keyed by one of the following statistics:

-   frequencies: number of individuals with each variant

-   mean: mean of the entire population (for continuous traits)

-   var: variance of the entire population (for continuous traits)

-   turnoverDiscrete: mean turnoever rate of the most frequent trait

The entry for each of these statistics is a dictionary which is keyed by the mode of transmission. Each mode maps to an array with dimension (numSims, len(coarseSteps)), with entry *i**j* corresponding to the value of the statistic in simulation *i* at time point *j*.
