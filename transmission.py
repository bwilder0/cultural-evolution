from getDiscrete import getDiscrete
from getContinuous import getContinuous
import random
try:
    import numpy as np
except:
    import numpypy as np
def transmission(group, pTransmission, pMutate, mutationStd, discreteTraitBins, pMixed, discreteTraits, continuousTraits, continuousLower, continuousUpper, conformity, conformityS, conformityThreshold, discreteHistBins):
    #set a random ordering of the individuals

    order = list(range(len(group['age'])))
    random.shuffle(order)
    
    conformityWeights = {}
    conformitySet = {}
    for t in discreteTraits:
        conformityWeights[t] = [None]*5
        conformitySet[t] = [False]*5
    

    for t in discreteTraits + continuousTraits:
        if t == 'traitVerticalDiscrete':
            continue
        if conformity:
            freq = np.zeros((6))
            for val in group[t]:
                freq[val] += 1
            k = (freq != 0).sum()
            if k == 1:
                trait = np.argmax(freq)
                for i in order:
                    if random.random() < pTransmission:
                        group[t][i] = getDiscrete(trait, pMutate, discreteTraitBins)
                continue
                    
        for i in range(len(order)):
            agent = order[i]
            if random.random() < pTransmission:
            #process: for each trait, calculate a source group and peer group, and then do a random choice and a transmission call
                if t == 'traitHorizontalDiscrete' or t == 'traitHorizontalContinuous':
                    sourceAgents = [x for x in range(len(group['age'])) if group['age'][x] == group['age'][agent]]
                elif t == 'traitObliqueDiscrete' or t == 'traitObliqueContinuous':
                    sourceAgents = [x for x in range(len(group['age'])) if group['age'][x] > group['age'][agent]]
                elif t == 'traitRandomDiscrete' or t == 'traitRandomContinuous':
                    sourceAgents = list(range(len(group['age'])))
                elif t == 'traitMixedDiscrete' or t == 'traitMixedContinuous':
                    if random.random() < pMixed:
                        sourceAgents = [x for x in range(len(group['age'])) if group['age'][x] > group['age'][agent]]
                    else:
                        sourceAgents = [x for x in range(len(group['age'])) if group['age'][x] == group['age'][agent]]
                else:
                    sourceAgents = None
                    continue
                    
                if len(sourceAgents) > 0:
                    if conformity:
                        freq = np.zeros((6))
                        for val in group[t]:
                            freq[val] += 1
                        k = (freq != 0).sum()
                        freq = freq/sum(freq)
                        freq += conformityS*(k*freq - 1)
                        freq[freq < 0] = 0
                        freq = freq/sum(freq)
                        freqCum = np.cumsum(freq)
                        r = random.random()
                        for i in range(1, freqCum.shape[0]):
                            if freqCum[i] > r:
                                traitChoice = i
                                break
                        group[t][agent] = getDiscrete(traitChoice, pMutate, discreteTraitBins)                        
                    else:
                        source = random.choice(sourceAgents)
                        if t in discreteTraits:
                            group[t][agent] = getDiscrete(group[t][source], pMutate, discreteTraitBins)
                        else:
                            group[t][agent] = getContinuous(group[t][source], pMutate, mutationStd, continuousLower, continuousUpper)
    return group