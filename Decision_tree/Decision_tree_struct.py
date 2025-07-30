# -*- coding: UTF-8 -*-
import platform
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from math import log
import operator

# Auto-select appropriate font for the system
def get_font():
    system = platform.system()
    if system == 'Windows':
        return FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    elif system == 'Linux':
        return FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=14)
    else:
        return FontProperties(size=14)

font = get_font()


"""
Function Description: Calculate the empirical entropy (Shannon entropy) of a given dataset

Parameters:
    dataSet - dataset
Returns:
    shannonEnt - empirical entropy (Shannon entropy)
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)                        # Return the number of rows in the dataset
    labelCounts = {}                                # Dictionary to save the count of each label
    for featVec in dataSet:                            # Count each feature vector
        currentLabel = featVec[-1]                    # Extract label information
        if currentLabel not in labelCounts.keys():    # If the label is not in the count dictionary, add it
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1                # Label count
    shannonEnt = 0.0                                # Empirical entropy (Shannon entropy)
    for key in labelCounts:                            # Calculate Shannon entropy
        prob = float(labelCounts[key]) / numEntires    # Probability of selecting this label
        shannonEnt -= prob * log(prob, 2)            # Calculate using formula
    return shannonEnt                                # Return empirical entropy (Shannon entropy)

"""
Function Description: Create test dataset

Parameters:
    None
Returns:
    dataSet - dataset
    labels - feature labels
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-20
"""
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        # dataset
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['Age', 'Has Job', 'Has House', 'Credit Rating']        # feature labels
    return dataSet, labels                             # Return dataset and classification attributes

"""
Function Description: Split dataset according to given feature

Parameters:
    dataSet - dataset to be split
    axis - feature to split the dataset
    value - value of the feature to return
Returns:
    None
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def splitDataSet(dataSet, axis, value):       
    retDataSet = []                                        # Create list for returned dataset
    for featVec in dataSet:                             # Traverse dataset
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                # Remove axis feature
            reducedFeatVec.extend(featVec[axis+1:])     # Add qualified data to returned dataset
            retDataSet.append(reducedFeatVec)
    return retDataSet                                      # Return split dataset

"""
Function Description: Choose the best feature

Parameters:
    dataSet - dataset
Returns:
    bestFeature - index of the feature with maximum information gain (best feature)
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-20
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                    # Number of features
    baseEntropy = calcShannonEnt(dataSet)                 # Calculate Shannon entropy of dataset
    bestInfoGain = 0.0                                  # Information gain
    bestFeature = -1                                    # Index of best feature
    for i in range(numFeatures):                         # Traverse all features
        # Get all features of the i-th column in dataSet
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                         # Create set {}, elements cannot be repeated
        newEntropy = 0.0                                  # Empirical conditional entropy
        for value in uniqueVals:                         # Calculate information gain
            subDataSet = splitDataSet(dataSet, i, value)         # subDataSet split subset
            prob = len(subDataSet) / float(len(dataSet))           # Calculate probability of subset
            newEntropy += prob * calcShannonEnt(subDataSet)     # Calculate empirical conditional entropy using formula
        infoGain = baseEntropy - newEntropy                     # Information gain
        # print("The gain of the %d-th feature is %.3f" % (i, infoGain))            # Print information gain of each feature
        if (infoGain > bestInfoGain):                             # Calculate information gain
            bestInfoGain = infoGain                             # Update information gain, find maximum information gain
            bestFeature = i                                     # Record index of feature with maximum information gain
    return bestFeature                                             # Return index of feature with maximum information gain


"""
Function Description: Count the element that appears most in classList (class label)

Parameters:
    classList - class label list
Returns:
    sortedClassCount[0][0] - element that appears most in classList (class label)
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        # Count occurrences of each element in classList
        if vote not in classCount.keys():classCount[vote] = 0   
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        # Sort by dictionary value in descending order
    return sortedClassCount[0][0]                                # Return element that appears most in classList

"""
Function Description: Create decision tree

Parameters:
    dataSet - training dataset
    labels - classification attribute labels
    featLabels - store selected optimal feature labels
Returns:
    myTree - decision tree
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-25
"""
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]            # Get classification labels (loan approval: yes or no)
    if classList.count(classList[0]) == len(classList):            # If all classes are identical, stop further division
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:                                    # Return most frequent class label when all features are traversed
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                # Choose optimal feature
    bestFeatLabel = labels[bestFeat]                            # Label of optimal feature
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                                    # Generate tree based on optimal feature label
    del(labels[bestFeat])                                        # Delete used feature label
    featValues = [example[bestFeat] for example in dataSet]        # Get all attribute values of optimal feature in training set
    uniqueVals = set(featValues)                                # Remove duplicate attribute values
    for value in uniqueVals:                                    # Traverse features, create decision tree.  
        subLabels = labels[:]               
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree

"""
Function Description: Get the number of leaf nodes in decision tree

Parameters:
    myTree - decision tree
Returns:
    numLeafs - number of leaf nodes in decision tree
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def getNumLeafs(myTree):
    numLeafs = 0                                                # Initialize leaves
    firstStr = next(iter(myTree))                                # In python3, myTree.keys() returns dict_keys, not list, so cannot use myTree.keys()[0] to get node attributes, can use list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                # Get next dictionary
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                # Test if this node is a dictionary, if not a dictionary, this node is a leaf node
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

"""
Function Description: Get the depth of decision tree

Parameters:
    myTree - decision tree
Returns:
    maxDepth - depth of decision tree
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def getTreeDepth(myTree):
    maxDepth = 0                                                # Initialize decision tree depth
    firstStr = next(iter(myTree))                                # In python3, myTree.keys() returns dict_keys, not list, so cannot use myTree.keys()[0] to get node attributes, can use list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                # Get next dictionary
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                # Test if this node is a dictionary, if not a dictionary, this node is a leaf node
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth            # Update depth
    return maxDepth

"""
Function Description: Draw node

Parameters:
    nodeTxt - node name
    centerPt - text position
    parentPt - arrow position for annotation
    nodeType - node format
Returns:
    None
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                            # Define arrow format
    font = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=14)
      # Set font
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    # Draw node
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontproperties=font)

"""
Function Description: Annotate directed edge attribute values

Parameters:
    cntrPt, parentPt - used to calculate annotation position
    txtString - annotation content
Returns:
    None
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                            # Calculate annotation position                   
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

"""
Function Description: Draw decision tree

Parameters:
    myTree - decision tree (dictionary)
    parentPt - annotation content
    nodeTxt - node name
Returns:
    None
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        # Set node format
    leafNode = dict(boxstyle="round4", fc="0.8")                                            # Set leaf node format
    numLeafs = getNumLeafs(myTree)                                                          # Get number of leaf nodes in decision tree, determines tree width
    depth = getTreeDepth(myTree)                                                            # Get decision tree depth
    firstStr = next(iter(myTree))                                                            # Next dictionary                                                 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    # Center position
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    # Annotate directed edge attribute values
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        # Draw node
    secondDict = myTree[firstStr]                                                            # Next dictionary, continue drawing child nodes
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        # y offset
    for key in secondDict.keys():                               
        if type(secondDict[key]).__name__=='dict':                                            # Test if this node is a dictionary, if not a dictionary, this node is a leaf node
            plotTree(secondDict[key],cntrPt,str(key))                                        # Not a leaf node, recursively call to continue drawing
        else:                                                                                # If it's a leaf node, draw leaf node and annotate directed edge attribute values                                             
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

"""
Function Description: Create drawing panel

Parameters:
    inTree - decision tree (dictionary)
Returns:
    None
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                                                    # Create fig
    fig.clf()                                                                                # Clear fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                # Remove x, y axes
    plotTree.totalW = float(getNumLeafs(inTree))                                            # Get number of leaf nodes in decision tree
    plotTree.totalD = float(getTreeDepth(inTree))                                            # Get decision tree depth
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                # x offset
    plotTree(inTree, (0.5,1.0), '')                                                            # Draw decision tree
    plt.show()                                                                                 # Show drawing result     

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)  
    createPlot(myTree)