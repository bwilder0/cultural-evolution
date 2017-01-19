import random
def getContinuous(sourceVal, pMutate, mutationStd, lowerBound, upperBound):
    newVal = sourceVal
    if random.random() < pMutate:
        newVal = sourceVal + random.normalvariate(0,1)*mutationStd
        while newVal > upperBound or newVal < lowerBound:
#            print((newVal, upperBound, lowerBound, newVal > upperBound, newVal < lowerBound))
            newVal = sourceVal + random.normalvariate(0,1)*mutationStd
    return(newVal)