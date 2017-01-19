import random

def getDiscrete(sourceVal, pMutate, numTraits):
    newVal = sourceVal
    if random.random() < pMutate:
        newVal = random.randint(1,numTraits)
        while newVal == sourceVal:
            newVal = random.randint(1,numTraits)
    return newVal
