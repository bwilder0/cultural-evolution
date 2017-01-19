'''

'''

from getDiscrete import getDiscrete
from getContinuous import getContinuous
from transmission import transmission
from collections import Counter
try:
    import numpy as np
except:
    import numpypy as np
import pickle
import random
from histogram import histogram

#import entropy_estimators as ee
def doSim(params):
    traits = ['traitHorizontalDiscrete', 'traitHorizontalContinuous', 'traitObliqueDiscrete', 'traitObliqueContinuous',
        'traitRandomDiscrete', 'traitRandomContinuous', 'traitVerticalDiscrete', 'traitVerticalContinuous', 'traitMixedDiscrete', 'traitMixedContinuous']
    discreteTraits = ['traitHorizontalDiscrete', 'traitObliqueDiscrete', 'traitRandomDiscrete', 'traitVerticalDiscrete', 'traitMixedDiscrete']
    continuousTraits = ['traitHorizontalContinuous', 'traitObliqueContinuous', 'traitRandomContinuous', 'traitVerticalContinuous', 'traitMixedContinuous']
    numOneExtinct = 0
    #load parameters passed in    
    pWithin = params[0]
    startingPop = params[1]
    pMutate = params[2]
    numSims = params[3]
    pOblique = params[4]
    mutationStd = params[5]
    discreteTraitBins = params[6]
    suffix = str(params[7])
    conformity = params[8]
    conformityB = params[9]

    trackContinuous = False    
    trackRunLengths = False
    trackDiscrete = False
            
    conformityThreshold = .3
    if conformity:
        continuousTraits = []
        traits = discreteTraits
        
    #how long to run the simulation
    timeMax = 200
    #when to collect statistics
    coarseSteps = [199]
    if trackDiscrete:
        coarseSteps = list(range(timeMax))
    
    #define unspecified parameters
    meanChildren = 1.5
    continuousLower = 1
    continuousUpper = 2
    maxAge = 5
    pDeath = .15
    
    #record which simulations had a group die
    simsWithExtinction = []
    #record which simulations had the entire population die
    simsAllDead = []
        
    #how to bin continuous traits
    binSizes = [2, 5, 10]
    #calculate what the the histogram bins should be 
    binBounds = {}
    for numBins in binSizes:
        binBounds[numBins] = histogram([continuousLower,continuousUpper], numBins)[1]
    
    #bins to use when histograming the discrete trait
    discreteHistBins = histogram(list(range(1,discreteTraitBins+1)), discreteTraitBins)[1]
    
    #allow iteration over traits
    
    #preallocate all of the statistics arrays
    #the general format: each of these dictionaries has keys for each trait. 
    #For discrete traits: Each of those maps to a
    #Numpy nd array. For things that only record a scalar value (shannon, JD divergence, number of traits present)
    #that array has one dimension for the simulation number and the other for the time step (which of the 'coarse steps' the simulation is on)
    #frequencies has a histogram for each simulation/timestep point, so there's another dimension representing which element of the histogram
    #. So, 0 there corresponds to the most frequent variant, 1 the second most frequent, etc.
    #For continuous traits: each trait keys to another dictionary, where the keys are the bin sizes. Each value in those dictionaries
    #are as described for the discrete traits. The mean/var don't have binning, so they act like the discrete scalar traits.  
    frequencies = {}
    mean = {}
    var = {}

    if trackContinuous:
        continuousValues = {}
    if trackDiscrete:
        discreteValues = {}
    #for each discrete trait, allocate the 2d array
    for t in range(len(discreteTraits)):
        frequencies[discreteTraits[t]] = np.zeros((numSims, len(coarseSteps), discreteTraitBins), dtype=np.float)
        if trackDiscrete:
            discreteValues[discreteTraits[t]] = np.zeros((numSims, len(coarseSteps), startingPop), dtype=np.float)
    for t in range(len(continuousTraits)):
#        mean and var just have an array
        mean[continuousTraits[t]] = np.zeros((numSims, len(coarseSteps)), dtype=np.float)
        var[continuousTraits[t]] = np.zeros((numSims, len(coarseSteps)), dtype=np.float)
#        everything else gets a dictionary mapping the bin sizes to arrays
        frequencies[continuousTraits[t]] = {}
#        store the trait values for each individual for the continuous traits so entropy can be estimate later (scipy doesn't work with pypy)
        if trackContinuous:        
            continuousValues[continuousTraits[t]] = np.zeros((numSims, len(coarseSteps), startingPop), dtype=np.float)
        for numBins in binSizes:
#            create the appropriate array for each bin size
            frequencies[continuousTraits[t]][numBins] = np.zeros((numSims, len(coarseSteps), numBins), dtype=np.float)
#        allocate an array for the estimate of the entropy of the true distribution

    #This records the most frequent variant at each time step. It's just overwritten every simulation.
    #The dictionary maps each trait to an array that has one entry for each group. There are 4 elements in
    #each one: 1) the current most frequent variant 2) the timestep that became the most frequent
    # 3) the total of the all the durations of most frequent traits so far and 4) the number of distinct
    #most frequent traits so far. At the end 3/4 is the mean turnover rate
    #The continuous traits do the same thing with an extra dictionary to map bin sizes to those arrays
    mostFreq = {}
    for t in range(len(discreteTraits)):
        mostFreq[discreteTraits[t]] = [0]*4
    for t in range(len(continuousTraits)):
        mostFreq[continuousTraits[t]] = {}
        for numBins in binSizes:
            mostFreq[continuousTraits[t]][numBins] = [0]*4
    
    
    #initialize data structures to hold the mean turnover time for each simulation
    turnoverMeansDiscrete = {}
    for t in range(len(discreteTraits)):
            turnoverMeansDiscrete[discreteTraits[t]] = [0]*numSims

    turnoverMeansContinuous = {}
    for numBins in binSizes:
        turnoverMeansContinuous[numBins] = {}
        for t in range(len(continuousTraits)):
            turnoverMeansContinuous[numBins][continuousTraits[t]] = [0]*numSims
    
    if trackRunLengths:
        runLengths = {}
        for t in discreteTraits:
            runLengths[t] = np.zeros((numSims, timeMax))
        for t in continuousTraits:
            runLengths[t] = {}
            for numBins in binSizes:
                runLengths[t][numBins] = np.zeros((numSims, timeMax))    
    
    #run numSims simulations
    for sim in range(numSims):
        print(str(sim))
        
        
        #initialize the groups
        nextId = 0
        group = {'age': [],
            'traitHorizontalDiscrete': [],
            'traitHorizontalContinuous': [],
            'traitObliqueDiscrete': [],
            'traitObliqueContinuous': [],
            'traitRandomDiscrete': [],
            'traitRandomContinuous': [],
            'traitVerticalDiscrete': [],
            'traitVerticalContinuous': [],
            'extinct': [],
            'location': []}
            
        if conformity:
            group['conformityB'] = conformityB
        else:
            group['conformityB'] = 0

        #set the ages uniformly
        group['age'] = [random.randint(1, maxAge) for x in range(startingPop)]
        
        #initialize each trait to a homogeneous random value
        for tr in discreteTraits:
            val = 1
            group[tr] = [val for x in range(startingPop)]
            #initialize the most frequent value of the trait
            mostFreq[tr][0] = val
            mostFreq[tr][1] = 0
            mostFreq[tr][2] = 0
            mostFreq[tr][3] = 0
            
        for tr in continuousTraits:
            group[tr] = [1.5]*startingPop
            #initialize the most frequent category for each bin
            for numBins in binSizes:
                mostFreq[tr][numBins][0] = histogram(group[tr], binBounds[numBins])[0].argmax()
                mostFreq[tr][numBins][1] = 0
                mostFreq[tr][numBins][2] = 0
                mostFreq[tr][numBins][3] = 0

        #not extinct
        group['extinct'] = 0
        #id
        group['id'] = nextId
        nextId += 1
        #end group initialization
        
        timeStep = 0
        #main simulation loop
        while timeStep < timeMax:
            #age
            group['age'] = [x + 1 for x in group['age']]
    
            #death and reproduction
            #death
            #find individuals at the max age
            toDie = [x for x in range(len(group['age'])) if group['age'][x] > maxAge]
            #select other individuals with probability pDeath
            for j in range(len(group['age'])):
                if group['age'][j] <= maxAge and random.random() < pDeath:
                    toDie.append(j)

            #remove them from all the age vectors
            group['age'] = [group['age'][x] for x in range(len(group['age'])) if x not in toDie]
            #remove them from all the trait vectors
            for tr in traits:
               group[tr] = [group[tr][x] for x in range(len(group[tr])) if x not in toDie]

            #reproduction
            parents = [x for x in range(len(group['age'])) if group['age'][x] == 2 or group['age'][x]  == 3]
            #if there's no agents of reproductive age, move on
            if(len(parents) == 0):
                continue

            #add children                
            toAdd = len(toDie)
            for k in range(toAdd):
                parent = random.choice(parents)
                #add discrete traits
                for tr in discreteTraits:
                    group[tr].append(getDiscrete(group[tr][parent], pMutate, discreteTraitBins))
                #add continuous traits
                for tr in continuousTraits:
                    group[tr].append(getContinuous(group[tr][parent], pMutate, mutationStd, continuousLower, continuousUpper))

                #set their age
                group['age'].append(1)
            #end adding children
            #end death and reproduction
    
            #remove empty groups
            if len(group['age']) == 0:
                numOneExtinct += 1
                simsAllDead.append(sim)
                print('all groups dead')
                break
            
                    
            #transmission
            group = transmission(group, pWithin, pMutate, mutationStd, discreteTraitBins, pOblique, discreteTraits, continuousTraits, continuousLower, continuousUpper, conformity, group['conformityB'], conformityThreshold, discreteHistBins)

            #end within group transmission
            #collecting statistics
            #save the frequency distributions at designated times
            if timeStep in coarseSteps:
                coarseTimeStep = coarseSteps.index(timeStep)
                #frequency distribution and Shannon diversity index
                for t in range(len(discreteTraits)):
                    if trackDiscrete:
                        discreteValues[discreteTraits[t]][sim, coarseTimeStep] = group[discreteTraits[t]]
                    #get a frequency count
                    freq = histogram(group[discreteTraits[t]], discreteHistBins)[0]
                    frequencies[discreteTraits[t]][sim, coarseTimeStep] = freq    
                #end discrete traits
                        
                #do it all again for each bin size of continuous traits
                for t in range(len(continuousTraits)):
                    #first, store the actual values for this trait for later entropy estimation
                    if trackContinuous:
                        continuousValues[continuousTraits[t]][sim, coarseTimeStep] = group[continuousTraits[t]]
                    #binned frequencies
                    for numBins in binSizes:
                        #get a frequency count
                        freq = histogram(group[continuousTraits[t]], binBounds[numBins])[0]
                        frequencies[continuousTraits[t]][numBins][sim, coarseTimeStep] = freq  
                    #mean and variance
                    mean[continuousTraits[t]][sim, coarseTimeStep] = np.mean(group[continuousTraits[t]])
                    var[continuousTraits[t]][sim, coarseTimeStep] = np.var(group[continuousTraits[t]])
                #end continuous traits
                #end group
            #end coarse grained statistics
            #most frequent value of each trait
            for tr in discreteTraits:
                count = Counter(group[tr]).most_common(1)[0]
                greatest = count[0]
                if trackRunLengths:
                    runLengths[tr][sim, timeStep] = greatest
                if greatest != mostFreq[tr][0]:
                    #set the current most frequent trait
                    mostFreq[tr][0] = greatest
                    #add on the duration of this run
                    mostFreq[tr][2] += timeStep - mostFreq[tr][1]
                    #reset the clock
                    mostFreq[tr][1] = timeStep
                    #add to the total number of runs
                    mostFreq[tr][3] += 1

                for tr in continuousTraits:
                    for numBins in binSizes:
                        greatest = histogram(group[tr], binBounds[numBins])[0].argmax()
                        if trackRunLengths:
                            runLengths[tr][numBins][sim, timeStep] = greatest
                        if greatest != mostFreq[tr][numBins][0]:
                            #set the current most frequent trait
                            mostFreq[tr][numBins][0] = greatest
                            #add on the duration of this run
                            mostFreq[tr][numBins][2] += timeStep - mostFreq[tr][numBins][1]
                            #reset the clock
                            mostFreq[tr][numBins][1] = timeStep
                            #add to the total number of runs
                            mostFreq[tr][numBins][3] += 1

            #end most frequent
            timeStep += 1
        #end main loop
        
        #make the last update to the most frequent trait measure
        for tr in discreteTraits:
            #add on the duration of this run
            mostFreq[tr][2] += timeStep - mostFreq[tr][1]
            #add to the total number of runs
            mostFreq[tr][3] += 1
        for tr in continuousTraits:
            for numBins in binSizes:
                #add on the duration of this run
                mostFreq[tr][numBins][2] += timeStep - mostFreq[tr][numBins][1]
                #add to the total number of runs
                mostFreq[tr][numBins][3] += 1
        #find the mean turnover time- this is just dividing the appropriate array elements from mostFreq
        for t in range(len(discreteTraits)):
            turnoverMeansDiscrete[discreteTraits[t]][sim] = float(mostFreq[discreteTraits[t]][2])/mostFreq[discreteTraits[t]][3]
        for t in range(len(continuousTraits)):
            for numBins in binSizes:
                turnoverMeansContinuous[numBins][continuousTraits[t]][sim] = float(mostFreq[continuousTraits[t]][numBins][2])/mostFreq[continuousTraits[t]][numBins][3]
    #end all simulations
        
    #write results out to file
    
    #store the complete set of paramters used for this run    
    params = {'startingPop' : startingPop,
    'meanChildren' : meanChildren,
    'pDeath' : pDeath,
    'pWithin' : pWithin,
    'pMutate' : pMutate,
    'pOblique' : pOblique,
    'mutationStd' : mutationStd,
    'k' : discreteTraitBins,
    'continuousLower' : continuousLower,
    'continuousUpper' : continuousUpper,
    'maxAge' : maxAge,
    'timeMax' : timeMax,
    'numSims' : numSims,
    'coarseSteps' : coarseSteps,
    'binSizes' : binSizes,
    'conformity' : conformity,
    'conformityB' : conformityB,
    }
    
    #all of the statistics that we collected 
    #make everything into python lists because numpypy can't pickle yet
    for tr in discreteTraits:
        frequencies[tr] = frequencies[tr].tolist()
        if trackDiscrete:
            discreteValues[tr] = discreteValues[tr].tolist()
    for tr in continuousTraits:
        if trackContinuous:
            continuousValues[tr] = continuousValues[tr].tolist()
        for numBins in binSizes:
            frequencies[tr][numBins] = frequencies[tr][numBins].tolist()
    
    results = {'frequencies' : frequencies,
    'mean' : mean,
    'var' : var,
    'turnoverDiscrete' : turnoverMeansDiscrete,
    'simsWithExtinction' : simsWithExtinction,
    }
    if trackContinuous:
        results['continuousValues'] = continuousValues
    if trackDiscrete:
        results['discreteValues'] = discreteValues
        
    if trackRunLengths:
        for t in discreteTraits:
            runLengths[t] = runLengths[t].tolist()
        for t in continuousTraits:
            for numBins in binSizes:
                runLengths[t][numBins] = runLengths[t][numBins].tolist()
        results['runLengths'] = runLengths
    
    #combined them both in a single dictionary
    dataDump = {'params' : params, 'results' : results}
    #name the file by the paramters
    file = str(pWithin) + '-' + str(pDeath) + '-' + str(pMutate) + '-' + str(startingPop) + '-'  + '-' + str(pOblique) + '-' + str(mutationStd) + '-' + str(conformity) \
    + '-' + str(conformityB) + '-' + suffix

    #use pickle to dump it out    
    with open(file, 'wb') as f:    
        pickle.dump(dataDump, f)
    #somtimes use a return value for debugging purposes
    return file