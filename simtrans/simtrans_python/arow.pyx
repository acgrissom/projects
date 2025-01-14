# Cython port of Vlachos, 2013:
# export PYTHONPATH="hvector/build/lib.linux-x86_64-2.7/:$PYTHONPATH"
from _mycollections import mydefaultdict
from mydouble import mydouble, counts
import cPickle as pickle
import gzip
from operator import itemgetter

import random
import math
import numpy

from nltk import FreqDist

# cost-sensitive multiclass classification with AROW the instances consist of a
# dictionary of labels to costs and feature vectors (Huang-style)


cdef class Instance:
    cpdef public float maxCost
    cpdef public featureVector #mydefaultdict
    cpdef public costs
    cpdef public list worstLabels
    cpdef public list correctLabels
    cpdef public name
    cpdef public label
    @staticmethod
    def removeHapaxLegomena(instances):
        print "Counting features"
        feature2counts = mydefaultdict(mydouble)
        for instance in instances:
            for element in instance.featureVector:
                feature2counts[element] += 1

        print "Removing hapax legomena"
        cdef list newInstances = []
        for instance in instances:
            newFeatureVector = mydefaultdict(mydouble)
            for element in instance.featureVector:
                # if this feature was encountered more than once
                if feature2counts[element] > 1:
                    newFeatureVector[element] = instance.featureVector[element]
            newInstances.append(Instance(newFeatureVector, instance.costs))
        return newInstances

    def __init__(self, featureVector, costs=None, name=""):
        self.featureVector = featureVector
        self.name = name

        # we assume that the label with the lowest cost has a cost of zero and
        # the rest increment on that find out which are the correct answer,
        # assuming it has a cost of zero

        self.costs = costs
        if self.costs != None:
            minCost = float("inf")
            self.maxCost = float("-inf")
            self.worstLabels = []
            self.correctLabels = []
            for label, cost in self.costs.items():
                if cost < minCost:
                    minCost = cost
                    self.correctLabels = [label]
                elif cost == minCost:
                    self.correctLabels.append(label)
                if cost > self.maxCost:
                    self.maxCost = cost
                    self.worstLabels = [label]
                elif cost == self.maxCost:
                    self.worstLabels.append(label)

            if minCost>0:
                for label in self.costs:
                    self.costs[label] -= minCost
                self.maxCost -= minCost

    def __str__(self):
        retString = ""
        labels = []
        for label,cost in self.costs.items():
            labels.append(label+":"+str(cost))
        retString += ",".join(labels)

        retString += "\t"

        features = []
        for feature in self.featureVector:
            features.append(feature + ":" + str(self.featureVector[feature]))

        retString += " ".join(features)

        return retString


cdef class Prediction:
    cpdef public dict label2score
    cpdef public float score
    cpdef public label
    cpdef public list featureValueWeights
    cpdef public dict label2prob
    cpdef public dict currentWeightVectors
    cpdef public dict currentVarianceVectors
    cpdef public list featureVector
    cpdef public float entropy
    def __init__(self):
        self.label2score = {}
        self.score = float("-inf")
        self.label = None
        self.featureValueWeights = []
        self.label2prob = {}
        self.entropy = 0.0

cdef class AROW:
    cpdef public probabilities
    cpdef public dict currentWeightVectors
    cpdef public dict currentVarianceVectors
    cpdef public list probWeightVectors
    
    def __init__(self, seed = 0):
        self.probabilities = False
        self.currentWeightVectors = {}
        self.currentVarianceVectors = {}
        random.seed(0)
        numpy.random.seed(0)

    # This predicts always using the current weight vectors
    def predict(self, instance, verbose=False, probabilities=False):
        # always add the bias
        instance.featureVector["biasAutoAdded"] = 1.0

        prediction = Prediction()

        for label, weightVector in self.currentWeightVectors.items():
            score = instance.featureVector.dot(weightVector)
            prediction.label2score[label] = score
            if score > prediction.score:
                prediction.score = score
                prediction.label = label

        if verbose:
            for feature in instance.featureVector:
                # keep the feature weights for the predicted label
                prediction.featureValueWeights.append([feature, instance.featureVector[feature], self.currentWeightVectors[prediction.label][feature]])
            # order them from the most positive to the most negative
                prediction.featureValueWeights = sorted(prediction.featureValueWeights, key=itemgetter(2))
        if probabilities:
            # if we have probabilistic training
            if self.probabilities:
                probPredictions ={}
                for label in self.probWeightVectors[0].keys():
                    # smoothing the probabilities with add 0.01 of 1 out of the vectors
                    probPredictions[label] = 0.01/len(self.probWeightVectors)
                # for each of the weight vectors obtained get its prediction
                for probWeightVector in self.probWeightVectors:
                    maxScore = float("-inf")
                    maxLabel = None
                    for label, weightVector in probWeightVector.items():
                        score = instance.featureVector.dot(weightVector)
                        if score > maxScore:
                            maxScore = score
                            maxLabel = label
                    # so the winning label adds one vote
                    probPredictions[maxLabel] += 1

                # now let's normalize:
                for label, score in probPredictions.items():
                    prediction.label2prob[label] = float(score)/len(self.probWeightVectors)

                # Also compute the entropy:
                for prob in prediction.label2prob.values():
                    if prob > 0:
                        prediction.entropy -= prob * math.log(prob, 2)
                # normalize it:
                prediction.entropy /= math.log(len(prediction.label2prob),2)
                #print prediction.label2prob
                #print prediction.entropy
            else:
                print "Need to obtain weight samples for probability estimates first"

        return prediction

    # This is just used to optimize the params
    # if probabilities is True we return the ratio for the average entropies, otherwise the loss
    def batchPredict(self, instances, probabilities=False):
        totalCost = 0
        sumCorrectEntropies = 0
        sumIncorrectEntropies = 0
        sumLogProbCorrect = 0
        totalCorrects = 0
        totalIncorrects = 0
        sumEntropies = 0
        for instance in instances:
            prediction = self.predict(instance, False, probabilities)
            # This is without probabilities, with probabilities we want the average entropy*cost
            if probabilities:
                if instance.costs[prediction.label] == 0:
                    sumLogProbCorrect -= math.log(prediction.label2prob[prediction.label],2)
                    totalCorrects += instance.maxCost
                    sumEntropies += instance.maxCost*prediction.entropy
                    sumCorrectEntropies += instance.maxCost*prediction.entropy
                else:
                    maxCorrectProb = 0.0
                    for correctLabel in instance.correctLabels:
                        if prediction.label2prob[correctLabel] > maxCorrectProb:
                            maxCorrectProb = prediction.label2prob[correctLabel]
                    #if maxCorrectProb > 0.0:
                    sumLogProbCorrect -= math.log(maxCorrectProb, 2)
                    #else:
                    #    sumLogProbCorrect = float("inf")
                    totalIncorrects += instance.maxCost
                    sumEntropies += instance.maxCost*(1-prediction.entropy)
                    sumIncorrectEntropies += instance.maxCost*prediction.entropy

            else:
                # no probs, just keep track of the cost incurred
                if instance.costs[prediction.label] > 0:
                    totalCost += instance.costs[prediction.label]

        if probabilities:
            avgCorrectEntropy = sumCorrectEntropies/float(totalCorrects)
            print avgCorrectEntropy
            avgIncorrectEntropy = sumIncorrectEntropies/float(totalIncorrects)
            print avgIncorrectEntropy
            print sumLogProbCorrect
            return sumLogProbCorrect
        else:
            return totalCost

    # the parameter here is for AROW learning
    def train(self, instances, averaging=True, shuffling=True, rounds = 10, param = 1):
        # we first need to go through the dataset to find how many classes

        # Initialize the weight vectors in the beginning of training"
        # we have one variance and one weight vector per class
        self.currentWeightVectors = {}
        self.currentVarianceVectors = {}
        if averaging:
            averagedWeightVectors = {}
            updatesLeft = rounds*len(instances)
        for label in instances[0].costs:
            self.currentWeightVectors[label] = mydefaultdict(mydouble)
            # remember: this is sparse in the sense that everething that doesn't have a value is 1
            # everytime we to do something with it, remember to add 1
            self.currentVarianceVectors[label] = {}
            # keep the averaged weight vector
            if averaging:
                averagedWeightVectors[label] = mydefaultdict(mydouble)

        # in each iteration
        for r in range(rounds):
            # shuffle
            if shuffling:
                random.shuffle(instances)
            errorsInRound = 0
            costInRound = 0
            # for each instance
            for instance in instances:
                prediction = self.predict(instance)

                # so if the prediction was incorrect
                # we are no longer large margin, since we are using the loss from the cost-sensitive PA
                if instance.costs[prediction.label] > 0:
                    errorsInRound += 1
                    costInRound += instance.costs[prediction.label]

                    # first we need to get the score for the correct answer
                    # if the instance has more than one correct answer then pick the min
                    minCorrectLabelScore = float("inf")
                    minCorrectLabel = None
                    for label in instance.correctLabels:
                        score = instance.featureVector.dot(self.currentWeightVectors[label])
                        if score < minCorrectLabelScore:
                            minCorrectLabelScore = score
                            minCorrectLabel = label

                    # Calculate the confidence values
                    # first for the predicted label
                    zVectorPredicted = mydefaultdict(mydouble)
                    zVectorMinCorrect = mydefaultdict(mydouble)
                    for feature in instance.featureVector:
                        # the variance is either some value that is in the dict or just 1
                        if feature in self.currentVarianceVectors[prediction.label]:
                            zVectorPredicted[feature] = instance.featureVector[feature] * self.currentVarianceVectors[prediction.label][feature]
                        else:
                            zVectorPredicted[feature] = instance.featureVector[feature]
                        # then for the minCorrect:
                        if feature in self.currentVarianceVectors[minCorrectLabel]:
                            zVectorMinCorrect[feature] = instance.featureVector[feature] * self.currentVarianceVectors[minCorrectLabel][feature]
                        else:
                            zVectorMinCorrect[feature] = instance.featureVector[feature]

                    confidence = zVectorPredicted.dot(instance.featureVector) + zVectorMinCorrect.dot(instance.featureVector)

                    beta = 1.0/(confidence + param)

                    # the loss is the scaled margin loss also used by Mejer and Crammer 2010
                    loss = prediction.score - minCorrectLabelScore  + math.sqrt(instance.costs[prediction.label])

                    alpha = loss * beta

                    # update the current weight vectors
                    self.currentWeightVectors[prediction.label].iaddc(zVectorPredicted, -alpha)
                    self.currentWeightVectors[minCorrectLabel].iaddc(zVectorMinCorrect, alpha)
                    if averaging:
                        averagedWeightVectors[prediction.label].iaddc(zVectorPredicted, -alpha * updatesLeft)
                        averagedWeightVectors[minCorrectLabel].iaddc(zVectorMinCorrect, alpha * updatesLeft)

                    # update the diagonal covariance
                    for feature in instance.featureVector.iterkeys():
                        # for the predicted
                        if feature in self.currentVarianceVectors[prediction.label]:
                            self.currentVarianceVectors[prediction.label][feature] -= beta * pow(zVectorPredicted[feature],2)
                        else:
                            # Never updated this covariance before, add 1
                            self.currentVarianceVectors[prediction.label][feature] = 1 - beta * pow(zVectorPredicted[feature],2)
                        # for the minCorrect
                        if feature in self.currentVarianceVectors[minCorrectLabel]:
                            self.currentVarianceVectors[minCorrectLabel][feature] -= beta * pow(zVectorMinCorrect[feature],2)
                        else:
                            # Never updated this covariance before, add 1
                            self.currentVarianceVectors[minCorrectLabel][feature] = 1 - beta * pow(zVectorMinCorrect[feature],2)

                if averaging:
                    updatesLeft-=1

            print "Training error rate in round " + str(r) + " : " + str(float(errorsInRound)/len(instances))

        if averaging:
            for label in self.currentWeightVectors:
                self.currentWeightVectors[label].iaddc(averagedWeightVectors[label], 1.0/float(rounds*len(instances)))

        # Compute the final training error:
        finalTrainingErrors = 0
        finalTrainingCost = 0
        for instance in instances:
            prediction = self.predict(instance)
            if instance.costs[prediction.label] > 0:
                finalTrainingErrors +=1
                finalTrainingCost += instance.costs[prediction.label]

        finalTrainingErrorRate = float(finalTrainingErrors)/len(instances)
        print "Final training error rate=" + str(finalTrainingErrorRate)
        print "Final training cost=" + str(finalTrainingCost)

        return finalTrainingCost

    def probGeneration(self, scale=1.0, noWeightVectors=100):
        # initialize the weight vectors
        print "Generating samples for the weight vectors to obtain probability estimates"
        self.probWeightVectors = []
        for i in xrange(noWeightVectors):
            self.probWeightVectors.append({})
            for label in self.currentWeightVectors:
                self.probWeightVectors[i][label] = mydefaultdict(mydouble)

        for label in self.currentWeightVectors:
            # We are ignoring features that never got their weight set
            for feature in self.currentWeightVectors[label]:
                # note that if the weight was updated, then the variance must have been updated too, i.e. we shouldn't have 0s
                weights = numpy.random.normal(self.currentWeightVectors[label][feature], scale * self.currentVarianceVectors[label][feature], noWeightVectors)
                # we got the samples, now let's put them in the right places
                for i,weight in enumerate(weights):
                    self.probWeightVectors[i][label][feature] = weight

        print "done"
        self.probabilities = True

    # train by optimizing the c parametr
    @staticmethod
    def trainOpt(instances, rounds = 10, paramValues=[0.01, 0.1, 1.0, 10, 100], heldout=0.2, optimizeProbs=False):
        print "Training with " + str(len(instances)) + " instances"

        # this value will be kept if nothing seems to work better
        bestParam = 1
        lowestCost = float("inf")
        bestClassifier = None
        trainingInstances = instances[:int(len(instances) * (1-heldout))]
        testingInstances = instances[int(len(instances) * (1-heldout)) + 1:]
        for param in paramValues:
            print "Training with param="+ str(param) + " on " + str(len(trainingInstances)) + " instances"
            # Keep the weight vectors produced in each round
            classifier = AROW()
            classifier.train(trainingInstances, True, True, rounds, param)
            print "testing on " + str(len(testingInstances)) + " instances"
            # Test on the dev for the weight vector produced in each round
            devCost = classifier.batchPredict(testingInstances)
            print "Dev cost:" + str(devCost) + " avg cost per instance " + str(devCost/float(len(testingInstances)))

            if devCost < lowestCost:
                bestParam = param
                lowestCost = devCost
                bestClassifier = classifier

        # OK, now we got the best C, so it's time to train the final model with it
        # Do the probs
        # So we need to pick a value between
        if optimizeProbs:
            print "optimizing the scale parameter for probability estimation"
            bestScale = 1.0
            lowestEntropy = float("inf")
            steps = 20
            for i in xrange(steps):
                scale = 1.0 - float(i)/steps
                print "scale= " +  str(scale)
                bestClassifier.probGeneration(scale)
                entropy = bestClassifier.batchPredict(testingInstances, True)
                print "entropy sums: " + str(entropy)

                if entropy < lowestEntropy:
                    bestScale = scale
                    lowestEntropy = entropy


        # Now train the final model:
        print "Training with param="+ str(bestParam) + " on all the data"

        finalClassifier = AROW()
        finalClassifier.train(instances, True, True, rounds, bestParam)
        if optimizeProbs:
            print "Adding weight samples for probability estimates with scale " + str(bestScale)
            finalClassifier.probGeneration(bestScale)

        return finalClassifier

    # save function for the parameters:
    def save(self, filename):
        model_file = open(filename, "w")
        # prepare for pickling
        pickleDict = {}
        for label in self.currentWeightVectors:
            pickleDict[label] = {}
            for feature in self.currentWeightVectors[label]:
                pickleDict[label][feature] = self.currentWeightVectors[label][feature]
        pickle.dump(pickleDict, model_file)
        model_file.close()
        # Check if there are samples for probability estimates to save
        if self.probabilities:
            pickleDictProbVectors = []
            for sample in self.probWeightVectors:
                label2vector = {}
                for label, vector in sample.items():
                    label2vector[label] = {}
                    for feature in vector:
                        label2vector[label][feature] = vector[feature]
                pickleDictProbVectors.append(label2vector)
            probVectorFile = gzip.open(filename + "_probVectors.gz", "wb")
            pickle.dump(pickleDictProbVectors, probVectorFile, -1)
            probVectorFile.close()
        # this is just for debugging, doesn't need to be loaded as it is not used for prediction
        # Only the non-one variances are added
        pickleDictVar = {}
        covariance_file = open(filename + "_variances", "w")
        for label in self.currentVarianceVectors:
            pickleDictVar[label] = {}
            for feature in self.currentVarianceVectors[label]:
                pickleDictVar[label][feature] = self.currentVarianceVectors[label][feature]
        pickle.dump(pickleDictVar, covariance_file)
        covariance_file.close()


    # load a model from a file:
    def load(self, filename):
        model_weights = open(filename, 'r')
        weightVectors = pickle.load(model_weights)
        model_weights.close()
        for label, weightVector in weightVectors.items():
            self.currentWeightVectors[label] = mydefaultdict(mydouble, weightVector)

        try:
            with gzip.open(filename + "_probVectors.gz", "rb") as probFile:
                print "loading probabilities"
                pickleDictProbVectors = pickle.load(probFile)
                self.probWeightVectors = []
                for sample in pickleDictProbVectors:
                    label2Vectors = {}
                    for label,vector in sample.items():
                        label2Vectors[label] = mydefaultdict(mydouble, vector)
                    self.probWeightVectors.append(label2Vectors)

                probFile.close()
                self.probabilities = True
        except IOError:
            print 'No weight vectors for probability estimates'
            self.probabilities = False



if __name__ == "__main__":
    import sys
    import random
    random.seed(13)
    numpy.random.seed(13)

    feature_1 = mydefaultdict(mydouble)
    feature_2 = mydefaultdict(mydouble)
    feature_3 = mydefaultdict(mydouble)

    cost_1 = {}
    cost_2 = {}
    cost_3 = {}

    feature_1[0] = 2.0
    cost_1["1"] = 0.2
    cost_1["2"] = 0.5
    cost_1["3"] = 1.0
    feature_2[1] = 3.0
    cost_2["2"] = 0.1
    cost_2["1"] = 0.5
    cost_2["3"] = 1.0
    feature_2[3] = 4.0
    cost_3["3"] = 0.0
    cost_3["2"] = 0.5
    cost_3["1"] = 1.0

    instances = []
    instances.append(Instance(feature_1, cost_1, "1e1"))
    instances.append(Instance(feature_2, cost_2, "2e2"))
    instances.append(Instance(feature_3, cost_3, "3e3"))

    classifier_p = AROW()
    classifier_p.train(instances)
    classifier_p.probGeneration()

    for ii in instances:
        pred = classifier_p.predict(ii, probabilities=True)
        print("Prediction %s: %s (%s)" % \
                  (ii.name, FreqDist(pred.label2prob).max(),
              str(pred.label2prob)))

def old():
    import sys
    import random
    random.seed(13)
    numpy.random.seed(13)
    dataLines = open(sys.argv[1]).readlines()

    instances = []
    classifier_p = AROW()
    print "Reading the data"
    for line in dataLines:
        details = line.split()
        costs = {}
        featureVector = mydefaultdict(mydouble)

        if details[0] == "-1":
            costs["neg"] = 0
            costs["pos"] = 1
        elif details[0] == "+1":
            costs["neg"] = 1
            costs["pos"] = 0

        for feature in details[1:]:
            featureID, featureVal = feature.split(":")
            featureVector[featureID] = float(featureVal)
        instances.append(Instance(featureVector, costs))
        #print instances[-1].costs

    random.shuffle(instances)
    #instances = instances[:100]
    # Keep some instances to check the performance
    testingInstances = instances[int(len(instances) * 0.75) + 1:]
    trainingInstances = instances[:int(len(instances) * 0.75)]

    print "training data: " + str(len(trainingInstances)) + " instances"
    #classifier_p.train(trainingInstances, True, True, 10, 10)

    classifier_p = AROW.trainOpt(trainingInstances, 10, [0.01, 0.1, 1.0, 10, 100], 0.1, True)

    cost = classifier_p.batchPredict(testingInstances)
    avgCost = float(cost)/len(testingInstances)
    print "Avg Cost per instance " + str(avgCost) + " on " + str(len(testingInstances)) + " testing instances"

    avgRatio = classifier_p.batchPredict(testingInstances, True)
    print "entropy sums: " + str(avgRatio)

    # Save the parameters:
    #print "saving"
    #classifier_p.save(sys.argv[1] + ".arow")
    #print "done"
    # load again:
    #classifier_new = AROW()
    #print "loading model"
    #classifier_new.load(sys.argv[1] + ".arow")
    #print "done"

    #avgRatio = classifier_new.batchPredict(testingInstances, True)
    #print "entropy sums: " + str(avgRatio)
