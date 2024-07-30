import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, BlogCatalog, Flickr, Facebook, Gowalla
from dhg.models import GCN, GIN, HyperGCN, GraphSAGE
from dhg.random import set_seed
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
from scipy import sparse
import random
import pandas as pd
import collections
import networkx as nx
from scipy.sparse import csr_matrix
import pickle
import math
import cmath
from sklearn import decomposition
from rdp_accountant import compute_rdp, get_privacy_spent
from sklearn import preprocessing, model_selection
from sklearn.cluster import KMeans
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter


warnings.filterwarnings('ignore', category=FutureWarning)


def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[1][train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()   #计算梯度
    optimizer.step()  #更新梯度
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[1][idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
        return res
    else:
        res = evaluator.test(lbls, outs)
        predict = np.argmax(outs, axis=-1)
        recall0 = recall_score(lbls, predict, average='macro')
        precision0 = precision_score(lbls, predict, average='macro')
        f0 = f1_score(lbls, predict, average='macro')

        print("accuracy:", res["accuracy"], "f1:", f0, "recall:", recall0, "precision", precision0)

        return res, f0, recall0, precision0



def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def collectingData(data):
    feature = data["features"].numpy()
    labels = data["labels"].numpy()
    edges = data["edge_list"]

    np.save("BlogCatalog/feature.npy", feature)
    np.save("BlogCatalog/edges.npy", edges)
    np.save("BlogCatalog/labels.npy", labels)

    return None

def getSamplingMatrix(samplingGlobalAdjWithReduceNode, sampling_idx_range):
    adjLen = len(samplingGlobalAdjWithReduceNode)
    samplingMatrix = np.zeros((adjLen, adjLen))

    for idx in sampling_idx_range:
        currentList = samplingGlobalAdjWithReduceNode[idx]
        for listIdx in currentList:
            samplingMatrix[idx, listIdx] = 1

    return samplingMatrix

def getSamplingAdj1(adjList, sampling_idx_range): #using for global adj and take the node out of sampling nodes
    newNodeAdjList = []
    for listIndex in adjList:
        withInFlag = listIndex in sampling_idx_range
        if withInFlag:
            newNodeAdjList.append(listIndex)
    return newNodeAdjList

def getSamplingAdj(adjList, sampling_idx_range):   #the index of sampling_idx_range is the current node index
    newNodeAdjList = []
    for listIndex in adjList:
        withInFlag = listIndex in sampling_idx_range
        if withInFlag:
            newNodeAdjList.append(sampling_idx_range.index(listIndex))
    return newNodeAdjList

def getSamplingGlobalAdj(graph, sampling_idx_range):  #pos the sampling nodes in global adj
    adjLen = len(graph)
    samplingGlobalAdj = collections.defaultdict(list)
    for idx in range(adjLen):
        withInFlag = idx in sampling_idx_range
        if withInFlag:
            currentList = graph[idx]
            newCurrentList = getSamplingAdj1(currentList, sampling_idx_range)
            samplingGlobalAdj[idx] = newCurrentList
        else:
            samplingGlobalAdj[idx] = []
    samplingMatrix = getSamplingMatrix(samplingGlobalAdj, sampling_idx_range)
    return samplingMatrix

def loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, classNum, saveName):
    saveName = saveName + '-TrainLabelIndex'
    trainsl = samplingTrainsetLabel.tolist()
    label = trainsl

    samplingTrainIndex111 = []
    # -----getting the train index from the file---------#
    # nd = np.genfromtxt(saveName, delimiter=',', skip_header=True)
    # samplingIndex = np.array(nd).astype(int)
    # for i in range(classNum):
    #     currentSamplingIndex = samplingIndex[:, i]
    #     samplingTrainIndex111 += currentSamplingIndex.tolist()

    # --------getting train index and save train index--------#
    samplingClassList = []
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(label) if x == k]
        samplingIndex = random.sample(range(0, len(labelIndex)), sampleNumEachClass)
        samplingFixedIndex = np.array(labelIndex)[samplingIndex].tolist()
        samplingClassList.append(samplingFixedIndex)
        samplingTrainIndex111 += samplingFixedIndex

    fileCol = {}
    for i in range(classNum):
        colName = 'TrainLabelIndex' + str(i)
        fileCol[colName] = samplingClassList[i]

    dataframe = pd.DataFrame(fileCol)  # save the samplingIndex of every client
    dataframe.to_csv(saveName, index=False, sep=',')  #

    return samplingTrainIndex111
#

#----------------------for federated average-----------------------#
def getSamplingIndex(nodesNum, samplingRate, testNodesNum, valNodesNum, trainLabel, labels):
    totalSampleNum = nodesNum - testNodesNum - valNodesNum  # get the train num according to the fixed testset and valset
    samplingNum = int(samplingRate * totalSampleNum)  # get
    testAndValIndex = [i for i in range(totalSampleNum, nodesNum)]

    #-----analysis the label distributation result of test and val------#
    classNum = 6
    classDict = {}
    classDictIndex = {}
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(trainLabel) if x == k]
        labelCount = len(labelIndex)
        classDict[k] = labelCount
        classDictIndex[k] = labelIndex

    lastSamplingNum = int(samplingNum / (classNum + 5))
    beforeSamplingNum = int((samplingNum - lastSamplingNum) / 5)
    finalSamplingNum = samplingNum - beforeSamplingNum * 5

    samplingNumList = [beforeSamplingNum] * 6
    samplingNumList[1] = finalSamplingNum

    samplingIndex = []
    for k in range(classNum):
        currentSamplingNum = samplingNumList[k]
        currentSamplingIdex = random.sample(classDictIndex[k], currentSamplingNum)
        samplingIndex += currentSamplingIdex

    return samplingIndex

def dataOverlapSplitting(samplingIndex, nodesNum, testNodesNum, valNodesNum, graph,
                  trainLabel, labels, sampleNumEachClass, classNum, saveName):
    train_sampling_idx_range = np.sort(samplingIndex)
    totalSampleNum = nodesNum - testNodesNum - valNodesNum
    sampling_idx_range = train_sampling_idx_range.tolist() + [i for i in range(totalSampleNum, nodesNum)]
    samplingLabels = labels[sampling_idx_range]
    trainLabel = trainLabel.numpy()
    samplingTrainsetLabel = trainLabel[train_sampling_idx_range]
    samplingTrainFixedIndex = loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, classNum,
                                                 saveName)  # getting the sampling train data

    samplingMatrix = getSamplingGlobalAdj(graph, sampling_idx_range)  # get the global adj matrix with reduced node

    count = 0
    samplingAdj = collections.defaultdict(
        list)  # getting current sampling adj which is used for training is this client
    for index in sampling_idx_range:
        currentList = graph[index]
        newCurrentList = getSamplingAdj(currentList, sampling_idx_range)
        samplingAdj[count] = newCurrentList
        count += 1

    return samplingAdj, samplingTrainFixedIndex, sampling_idx_range, samplingLabels, samplingMatrix


def dataSplitting(nodesNum, testNodesNum, valNodesNum, graph,
                  trainLabel, labels, samplingRate,
                  sampleNumEachClass, classNum, saveName):    #the val and test data keep and random sampling train data
    totalSampleNum = nodesNum - testNodesNum - valNodesNum  # get the train num according to the fixed testset and valset
    samplingNum = int(samplingRate * totalSampleNum)  # get
    labels = labels.numpy()
    trainLabel = trainLabel.numpy()

    #----getting sampling index from file----------------#
    # nd = np.genfromtxt(saveName + '-TrainIndex', delimiter=',', skip_header=True)
    # samplingIndex = np.array(nd).astype(int).tolist()

    #-----saving sampling index---------------------------#
    samplingIndex = random.sample(range(0, totalSampleNum), samplingNum)  # get random samplingIndex
    print('samplingIndex:', samplingIndex)
    # samplingIndex = getSamplingIndex(nodesNum, samplingRate, testNodesNum, valNodesNum, trainLabel, labels)
    dataframe = pd.DataFrame({'samplingIndex': samplingIndex})  # save the train samplingIndex of every client
    dataframe.to_csv(saveName + '-TrainIndex', index=False, sep=',')

    train_sampling_idx_range = np.sort(samplingIndex)  # sort the sampling index
    sampling_idx_range = train_sampling_idx_range.tolist() + [i for i in range(totalSampleNum, nodesNum)] #getting the whole graph node index
    samplingLabels = labels[sampling_idx_range]
    samplingTrainsetLabel = trainLabel[train_sampling_idx_range]  # the new graph train set label????

    samplingTrainFixedIndex = loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, classNum, saveName)   #getting the sampling train data

    samplingMatrix = getSamplingGlobalAdj(graph, sampling_idx_range)  # get the global adj matrix with reduced node

    count = 0
    samplingAdj = collections.defaultdict(list)    #getting current sampling adj which is used for training is this client
    for index in sampling_idx_range:
        currentList = graph[index]
        newCurrentList = getSamplingAdj(currentList, sampling_idx_range)
        samplingAdj[count] = newCurrentList
        count += 1

    return samplingAdj, samplingNum, samplingTrainFixedIndex, sampling_idx_range, samplingLabels, samplingMatrix

def getTheCutSampling(samplingAdj, saveName):
    saveName = saveName + '-cut'

    #------get cut index from file----------------------#
    # nd = np.genfromtxt(saveName, delimiter=',', skip_header=True)
    # samplingCutIndexs = np.array(nd).astype(int).tolist()

    #-----------save cut index --------------------------#
    adjLen = len(samplingAdj)
    samplingNum = int(adjLen * 0.7)  # cut 30% nodes from train set  #BlogCatalog 0.5
    samplingCutIndexs = random.sample(range(0, adjLen), samplingNum)
    dataframe = pd.DataFrame({'samplingCutIndex': samplingCutIndexs})  # save the samplingIndex of every client
    dataframe.to_csv(saveName, index=False, sep=',')  #

    for cutIdx0 in samplingCutIndexs:
        cutRow0 = samplingAdj[cutIdx0]
        if (cutRow0 != []):
            cutIdx1 = cutRow0[len(cutRow0) - 1]
            cutPos0 = len(cutRow0) - 1
            cutRow0.pop(cutPos0)  # remove the last element

            cutRow1 = samplingAdj[cutIdx1]
            if cutRow1 != []:
                if cutIdx0 in cutRow1:
                    cutPos1 = cutRow1.index(cutIdx0)
                    cutRow1.pop(cutPos1)

            samplingAdj[cutIdx0] = cutRow0
            samplingAdj[cutIdx1] = cutRow1

    return samplingAdj

def get_graph1(data):
    currentGraph = {}
    nodesNum = data["num_vertices"]
    for i in range(nodesNum):
        currentGraph[i] = []

    for edge in data["edge_list"]:
        currentNIdx = edge[0]
        currentGraph[currentNIdx].append(edge[1])

    graph = collections.defaultdict(list)
    for i in range(nodesNum):
        graph[i] = currentGraph[i]
    return graph

def get_graph(data):
    nodesNum = data["num_vertices"]
    graphList = []
    nodeIdx = 0
    row = []
    for edge in data["edge_list"]:
        currentNIdx = edge[0]
        if currentNIdx != nodeIdx:
            nodeIdx = currentNIdx
            graphList.append(row)
            row = []
        row.append(edge[1])
    graphList.append(row)
    graph = collections.defaultdict(list)
    for i in range(nodesNum):
        graph[i] = graphList[i]
    return graph

def get_edge_list(samplingAdj):
    nodesNum = len(samplingAdj)
    edge_list = []
    for i in range(nodesNum):
        rowNodes = samplingAdj[i]
        for j in rowNodes:
            node = tuple([i, j])
            edge_list.append(node)
    return edge_list

def getOverlapClientData(data, samplingIndex, sampleNumEachClass, classNum, saveName):
    X, lbl = data["features"], data["labels"]

    graph = get_graph1(data)

    nodesNums = data["num_vertices"]
    testNodesNum = int(nodesNums * 0.4)
    valNodesNum = int(nodesNums * 0.2)
    trainNum = nodesNums - testNodesNum - valNodesNum
    idx_train = range(0, trainNum)
    trainLabel = lbl[idx_train]

    samplingAdj, samplingTrainFixedIndex, sampling_idx_range, \
    samplingLabels, samplingMatrix = dataOverlapSplitting(samplingIndex, nodesNums, testNodesNum, valNodesNum, graph,
                  trainLabel, lbl, sampleNumEachClass, classNum, saveName)

    samplingAdj = getTheCutSampling(samplingAdj, saveName)

    X = X[sampling_idx_range, :]
    edge_list = get_edge_list(samplingAdj)

    samplingAdjNum = len(samplingAdj)
    idx_test = range(samplingAdjNum - testNodesNum, samplingAdjNum)  # get the last 2800 indexes as test set
    idx_val = range(samplingAdjNum - testNodesNum - valNodesNum,
                    samplingAdjNum - testNodesNum)  # sampling 420 indexes as train set, each class has 60 labels
    idx_train = samplingTrainFixedIndex  # samplingTrainFixedIndex  # get 1400 indexes as val set
    # range(0, samplingAdjNum - testNodesNum - valNodesNum)

    test_index = [sampling_idx_range[idx] for idx in idx_test]

    train_mask = sample_mask(idx_train, samplingAdjNum)
    val_mask = sample_mask(idx_val, samplingAdjNum)
    test_mask = sample_mask(idx_test, samplingAdjNum)

    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)

    dataDict = {"num_classes": data["num_classes"],
                "num_vertices": samplingAdjNum,
                "num_edges": data["num_edges"],
                "dim_features": X.shape[1],
                "features": X,
                "edge_list": edge_list,
                "labels": torch.tensor(samplingLabels),
                "sampling_idx_range": sampling_idx_range,
                "idx_test": test_index}

    return dataDict, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum

def get_all_data(data, clientNum, clientList, currentTestIdx):
    X, lbl = data["features"], data["labels"]

    graph = get_graph1(data)

    all_sample_list = []
    for i in range(clientNum):
        eachClient = clientList[i]
        all_sample_list += eachClient.data["sampling_idx_range"]
    unique_set = set(all_sample_list)
    all_sample_list = list(unique_set)

    count = 0
    samplingAdj = collections.defaultdict(
            list)
    for index in all_sample_list:
        currentList = graph[index]
        newCurrentList = getSamplingAdj(currentList, all_sample_list)
        samplingAdj[count] = newCurrentList
        count += 1
    X = data["features"][all_sample_list, :]
    edge_list = get_edge_list(samplingAdj)
    lbls = data["labels"][all_sample_list]
    communitieLen = len(all_sample_list)

    all_test_idx = getSamplingAdj(currentTestIdx, all_sample_list)

    all_data = {"num_classes": data["num_classes"],
                "num_vertices": communitieLen,
                "num_edges": len(edge_list),
                "dim_features": X.shape[1],
                "features": X,
                "edge_list": edge_list,
                "labels": lbls,
                "sampling_idx_range": all_sample_list}
    return all_data, all_test_idx

def getClientData(data, samplingRate, sampleNumEachClass, classNum, saveName):
    X, lbl = data["features"], data["labels"]

    graph = get_graph1(data)

    nodesNums = data["num_vertices"]
    testNodesNum = int(nodesNums * 0.4)
    valNodesNum = int(nodesNums * 0.2)
    trainNum = nodesNums - testNodesNum - valNodesNum
    idx_train = range(0, trainNum)
    trainLabel = lbl[idx_train]

    samplingAdj, samplingNum, samplingTrainFixedIndex,\
    sampling_idx_range, samplingLabels, samplingMatrix = dataSplitting(nodesNums, testNodesNum, valNodesNum, graph,
                  trainLabel, lbl, samplingRate,
                  sampleNumEachClass, classNum, saveName)

    samplingAdj = getTheCutSampling(samplingAdj, saveName)

    X = X[sampling_idx_range, :]
    edge_list = get_edge_list(samplingAdj)

    samplingAdjNum = len(samplingAdj)
    idx_test = range(samplingAdjNum - testNodesNum, samplingAdjNum)  # get the last 2800 indexes as test set
    idx_val = range(samplingAdjNum - testNodesNum - valNodesNum,
                    samplingAdjNum - testNodesNum)  # sampling 420 indexes as train set, each class has 60 labels
    idx_train = samplingTrainFixedIndex #samplingTrainFixedIndex  # get 1400 indexes as val set
    #range(0, samplingAdjNum - testNodesNum - valNodesNum)

    test_index = [sampling_idx_range[idx] for idx in idx_test]

    train_mask = sample_mask(idx_train, samplingAdjNum)
    val_mask = sample_mask(idx_val, samplingAdjNum)
    test_mask = sample_mask(idx_test, samplingAdjNum)

    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)

    dataDict = {"num_classes": data["num_classes"],
                "num_vertices": samplingAdjNum,
                "num_edges": data["num_edges"],
                "dim_features": X.shape[1],
                "features": X,
                "edge_list": edge_list,
                "labels": torch.tensor(samplingLabels),
                "sampling_idx_range": sampling_idx_range,
                "idx_test": test_index
                }

    return dataDict, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum

def loadFoursquare(data_str):
    edgesfilePath = data_str + '/following.npy'
    userlabelPath = data_str + '/multilabel2id.pkl'
    userFeaturesPath = data_str + '/userattr.npy'
    user = data_str + '/user'

    features = np.load(userFeaturesPath)
    features = np.float32(features)

    edges = np.load(edgesfilePath)

    f = open(userlabelPath, 'rb')
    labels = pickle.load(f)

    users = []
    for userNode in range(len(features)):
        if userNode in labels.keys():
            users.append(userNode)

    currentEdges = []
    for edge in edges:
        startNode = edge[0]
        endNode = edge[1]
        if startNode in labels.keys() and endNode in labels.keys():
            currentEdges.append(tuple([users.index(startNode), users.index(endNode)]))

    userlabels = []
    for userid in range(len(users)):
        userlabels.append(sum(labels[users[userid]])-1)

    currentFeatures = features[users]
    currentNodeNums = len(currentFeatures)
    current_num_edges = len(currentEdges)

    dataDict = {"num_classes": 9,
                "num_vertices": currentNodeNums,
                "num_edges": current_num_edges,
                "dim_features": features.shape[1],
                "features": torch.tensor(currentFeatures),
                "edge_list": currentEdges,
                "labels": torch.tensor(userlabels)}

    return dataDict

def getSamplingData(data):
    globalLabelStatistic = {}
    globalLabelIndex = {}
    classNum = 6  # flickr_9 facebook_4 BlogCatalog_6
    labels = data["labels"].numpy().tolist()
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(labels) if x == k]
        labelCount = len(labelIndex)
        globalLabelStatistic[k] = labelCount
        globalLabelIndex[k] = labelIndex

    #sampling 300 from class 1 and construct new edges__Blogcatalog
    samplingNum = 300
    samplingClassIndex = random.sample(globalLabelIndex[1], samplingNum)
    newSamplingNode = []
    newSamplingLabels = []
    newSamplingFeatures = []
    labels = data["labels"].numpy()
    for k in range(classNum):
        if k == 1:
            newSamplingNode += samplingClassIndex
            newSamplingLabels += list(labels[samplingClassIndex])
            newSamplingFeatures += data["features"][samplingClassIndex]
        else:
            newSamplingNode += globalLabelIndex[k]
            newSamplingLabels += list(labels[globalLabelIndex[k]])
            newSamplingFeatures += data["features"][globalLabelIndex[k]]

    currentGlobalNode = []
    for i in range(len(labels)):
        if i in newSamplingNode:
            currentGlobalNode.append(i)
    currentGlobalLabels = labels[currentGlobalNode]

    newSamplingEdges = []
    edges = data["edge_list"]
    for edge in edges:
        start = edge[0]
        end = edge[1]
        if start in currentGlobalNode and end in currentGlobalNode:
            newStartIndex = currentGlobalNode.index(start)
            newEndIndex = currentGlobalNode.index(end)
            newSamplingEdges.append(tuple([newStartIndex, newEndIndex]))
    np.save('BlogCatalog/newSampling_edge_list.npy', newSamplingEdges)
    np.save('BlogCatalog/newSamplingNode.npy', currentGlobalNode)
    np.save('BlogCatalog/newSamplingLabels.npy', currentGlobalLabels)
    # np.save('BlogCatalog/newSamplingFeatures.npy', newSamplingFeatures)

    dataDict = {"num_classes": classNum,
                "num_vertices": len(newSamplingNode),
                "num_edges": len(newSamplingEdges),
                "dim_features": data["dim_features"],  # data["dim_features"], #facebook_feature_dim,
                "features": newSamplingFeatures,
                "edge_list": newSamplingEdges,
                "labels": newSamplingLabels}

    return dataDict

def loadSamplingData(data):
    classNum = 6
    edges = np.load('BlogCatalog/newSampling_edge_list.npy')
    samplingNode = np.load('BlogCatalog/newSamplingNode.npy')
    samplingLables = np.load('BlogCatalog/newSamplingLabels.npy')
    # np.save('BlogCatalog/newSamplingFeatures.npy', newSamplingFeatures)

    newSamplingFeatures = data["features"][samplingNode]

    # edges_list = [tuple([edge[0], edge[1]]) for edge in edges]

    dataDict = {"num_classes": classNum,
                "num_vertices": len(samplingNode),
                "num_edges": len(samplingLables),
                "dim_features": data["dim_features"],  # data["dim_features"], #facebook_feature_dim,
                "features": newSamplingFeatures,
                "edge_list": edges,
                "labels": torch.tensor(samplingLables)}

    return dataDict

def getCutedGradient(grad, clip):
    gradShape = np.array(grad).shape
    norm2 = np.linalg.norm(grad, ord=2, axis=1, keepdims=True)
    norm2 = norm2 / clip
    cutedGrad = []
    for i in range(gradShape[0]):
        currentNorm = norm2[i]

        if currentNorm > 1:
            currentGrad = grad[i] / norm2[i]
        else:
            currentGrad = grad[i]

        cutedGrad.append(np.array(currentGrad).tolist())

    return cutedGrad

def getNoise(sigma, sensitivity, shape):  # what the mean of batchsize?

    noise = torch.normal(0, sigma * sensitivity, size=shape)#

    return noise

def pca_project(param, sigma, sensitivity, clip):
    param = param.detach().numpy()
    mean = np.average(param, axis=0)
    pca = decomposition.PCA(n_components=3)   #n_components=0.99
    pca.fit(param)   #training
    X = pca.transform(param)   #return the result of dimensionality reduction

    shape = np.array(X).shape
    noise = torch.normal(0, sigma * sensitivity, size=shape)
    XN = X + np.array(noise)                               #perturb the result

    a0 = np.matrix(X)
    b0 = np.matrix(pca.components_)                     #inverse matrix
    inverseMatrix0 = a0 * b0 + mean                     #get the inverse transformation
    redisual0 = param - inverseMatrix0

    a1 = np.matrix(XN)
    b1 = np.matrix(pca.components_)                 # inverse matrix
    inverseMatrix1 = a1 * b1 + mean                 # get the inverse transformation
    theFinalNM = inverseMatrix1 + redisual0
    theFinalNM = np.array(theFinalNM)
    return theFinalNM

def getSigma(eps, delta, sensitivity):  #Gaussian mechanism

    sigma = math.sqrt((2*(sensitivity**2)*math.log(1.25/delta))/(eps**2))

    return sigma

def getProjectedNoiseGrad(grad, clip, eps, delta, sensitivity):
    cutedGrad = getCutedGradient(grad, clip)
    sigma = getSigma(eps, delta, sensitivity)
    noisedGrad = pca_project(cutedGrad, sigma, clip)
    return noisedGrad

def getGaussianSigma(eps, delta):
    sigma = math.sqrt((2 * math.log(1.25 / delta)) / (eps ** 2))

    return sigma

def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size

def scale(input, clipVal):
    norms = torch.norm(input, dim=1)

    scale = torch.clamp(clipVal / norms, max=1.0)

    pp = input * scale.view(-1, 1)

    return pp


def getHistogramNoiseGradient(sigma, grad, min, interval, index, splitCount, sensitivity):
    grad = np.array(grad)
    size = grad.shape


    for i in range(size[1]):
        start = min[i] + interval[i] * index[i]
        end = min[i] + interval[i] * (index[i] + 1)
        # sensitivity = end - start
        maxCount = splitCount[i][index[i]]
        noise = torch.normal(0, sigma * sensitivity/2708, size=[maxCount, 1])#
        count = 0
        for j in range(size[0]):
            if grad[j][i] >= start and grad[j][i] < end:
                grad[j][i] += noise[count]
                count += 1

    return grad

def partial_noise(grad, importPos, eps, delta, sensitivity):
    grad = np.array(grad)
    size = grad.shape
    sigma = getGaussianSigma(eps / 200, delta)

    maxCount = np.sum(importPos)

    noise = torch.normal(0, sigma * sensitivity / 2708, size=[maxCount, 1])  #
    count = 0

    for i in range(size[0]):
        for j in range(size[1]):
            if importPos[i][j] == 1:
                grad[i][j] += noise[count]
                count += 1

    return grad

def all_noise(grad, eps, delta, sensitivity):
    sigma = getGaussianSigma(eps / 200, delta)
    size = grad.shape

    for i in range(size[1]):

        noise = torch.normal(0, sigma * sensitivity / 2708, size=[size[0], 1])  #

        for j in range(size[0]):
            grad[j][i] += noise[j]

    return grad

def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break
    return cur_sigma, previous_eps

def get_sigma_(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma

    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps

def getSigmaAndEps(batchsize, n_training, n_epoch, eps, delta, rgp):
    sampling_prob = batchsize / n_training
    steps = int(n_epoch / sampling_prob)
    sigma, eps = get_sigma_(sampling_prob, steps, eps, delta, rgp=rgp)
    noise_multiplier0 = noise_multiplier1 = sigma
    print('noise scale for gradient embedding: ', noise_multiplier0, 'noise scale for residual gradient: ',
          noise_multiplier1, '\n rgp enabled: ', rgp, 'privacy guarantee: ', eps)
    return noise_multiplier0, eps

def get_noise():
    # steps = args.n_epoch
    # sampling_prob = batchsize / n_training
    # steps = int(n_epoch / sampling_prob)
    sampling_prob = 1
    steps = 200
    eps = 8
    delta = 1e-5
    sigma, eps = get_sigma_(sampling_prob, steps, eps, delta, rgp=True)
    clip0 = 1
    batchsize = 50
    noise_multiplier0 = noise_multiplier1 = sigma
    theta_noise = torch.normal(0, noise_multiplier0 * clip0 / batchsize, size=[100, 100],
                               device="cpu")

def getGlobalAdjMatrix(clientList):
    globalMatrix = 0
    for client in clientList:
        globalMatrix += client.samplingMatrix
    return globalMatrix

def getOverlapNodes1(clientList, clientNum):
    overlapNodes = []
    for i in range(clientNum):
        for j in range(i + 1, clientNum):
            samplingIndex_i = clientList[i]
            samplingIndex_j = clientList[j]
            for ind in samplingIndex_i:
                if ind in samplingIndex_j and ind not in overlapNodes:
                    overlapNodes.append(ind)
    return overlapNodes

def getOverlapNodes1(clientList):
    sample_list = []
    for i in range(clientNum):
        net = clientList[i]
        sampling_idx_range = net.data['sampling_idx_range']
        sample_list += sampling_idx_range
    counter = Counter(sample_list)
    overlapNodes = [item for item, count in counter.items() if count > 1]
    return overlapNodes

def getOverlapNodes(globalMatrix):
    overlapNodes = []
    globalMatrixShape = globalMatrix.shape
    for idx in range(globalMatrixShape[0]):
        eachRow = globalMatrix[idx]
        # rowIndexs = [i for i, x in enumerate(eachRow) if x != 1]
        rowIndexs = np.where(eachRow > 1)

        for rowIdx in rowIndexs[0]:
            if rowIdx not in overlapNodes:
                overlapNodes.append(rowIdx)
    return overlapNodes

def getFusNodesEmd(clientList, overlapNodes, clientOuts, globalNodeEmbeddings):
    fusOut = {}
    a = 1
    for pid, outs in enumerate(clientOuts):
        fusOut[pid] = torch.zeros_like(outs.data)
    for i in range(clientNum):
        net = clientList[i]
        sampling_idx_range = net.data['sampling_idx_range']
        for idx in sampling_idx_range:
            nodeIndex = sampling_idx_range.index(idx)
            currentNodeEmd = clientOuts[i][nodeIndex]
            curNodeEmd = currentNodeEmd.detach().numpy()
            if idx in overlapNodes:
                globalIdx = overlapNodes.index(idx)
                fusOut[i][nodeIndex] = torch.tensor((1-a) * curNodeEmd + a * globalNodeEmbeddings[globalIdx])
            else:
                fusOut[i][nodeIndex] = torch.tensor(curNodeEmd + 0.0)

    return fusOut

def getGlobalOverlapNodesEmd(clientList, overlapNodes, clientOuts):
    globalNodeEmbeddings = []
    for idx in overlapNodes:
        mean = 0
        count = 0
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmb = clientOut[nodeIndex]
                mean += currentNodeEmb.detach().numpy()
                count += 1
        # print(count)
        if count == 0:count = 1
        mean = mean / count
        # mean = mean.detach().numpy()
        expDis = []
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmd = clientOut[nodeIndex]
                curNodeEmd = currentNodeEmd.detach().numpy()
                dist = np.linalg.norm(curNodeEmd - mean)
                # dist = torch.norm(currentNodeEmd - mean)
                try:
                    expDis.append(math.exp(dist))
                except OverflowError:
                    expDis.append(math.exp(700))
                # print('dist:', dist)
                # print('math.exp(dist):', cmath.exp(dist))
                # expDis.append(math.exp(dist))

        finalNodeEmb = 0
        count = 0
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmd = clientOut[nodeIndex]
                curNodeEmd = currentNodeEmd.detach().numpy()
                finalNodeEmb += (expDis[count] / sum(expDis)) * curNodeEmd
                # finalNodeEmb += (expDis[count] / sum(expDis)) * currentNodeEmd
                count += 1
        globalNodeEmbeddings.append(finalNodeEmb)
    # fusOuts = getFusNodesEmd(clientList, overlapNodes, clientOuts, globalNodeEmbeddings)
    # return globalNodeEmbeddings, fusOuts
    return globalNodeEmbeddings

def setGlobalNodeEmdForLocalNodes0(clientList, clientOuts, fusOuts):
    for i in range(clientNum):
        net = clientList[i]
        sampling_idx_range = net.data['sampling_idx_range']
        for idx in sampling_idx_range:
            nodeIndex = sampling_idx_range.index(idx)
            clientOuts[i][nodeIndex].copy_(torch.Tensor(fusOuts[i][nodeIndex]))

    return clientOuts

def setGlobalNodeEmdForLocalNodes(clientList, clientOuts, overlapNodes, globalNodeEmbeddings):      ##设置不同客户端的重叠节点的增量节点
    for idx in overlapNodes:
        globalIdx = overlapNodes.index(idx)
        for i in range(clientNum):
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                clientOuts[i][nodeIndex] = clientOuts[i][nodeIndex].clone().detach().requires_grad_()
                clientOuts[i][nodeIndex].copy_(torch.Tensor(globalNodeEmbeddings[globalIdx]))
                # currentNodeEmd = clientOuts[i][nodeIndex]
                # curNodeEmd = currentNodeEmd.detach().numpy()
                # clientOuts[i][nodeIndex].copy_(torch.Tensor(curNodeEmd + globalNodeEmbeddings[globalIdx]))
                # print("updated data", clientOuts[i][nodeIndex])
                # clientOuts[i][nodeIndex].copy_((1-a) * clientOuts[i][nodeIndex].clone().detach() + a * torch.Tensor(globalNodeEmbeddings[globalIdx]))
                # clientOuts[i][nodeIndex].data = torch.Tensor(globalNodeEmbeddings[globalIdx])#torch.Tensor(globalNodeEmbeddings[globalIdx])
                # print("each client data:", clientOuts[i][nodeIndex].data)
    return clientOuts

def getLeftSamplingNodes(totalSampleNum, samplingOverlappedNodesIndex):
    totalIndex = list(range(0, totalSampleNum))
    totalCopyIndex = deepcopy(totalIndex)

    for currentIndex in samplingOverlappedNodesIndex:
        totalIndex.remove(currentIndex)

    return totalIndex, totalCopyIndex

def random_dataset_overlap_0(data, clientNum, overlapRate):   #
    trainIdx, testIdx = model_selection.train_test_split(
        range(0, data["num_vertices"]), train_size=0.6, test_size=0.2, stratify=range(0, data["num_vertices"])
    )
    trainIdx = list(trainIdx.index.values)
    testIdx = list(testIdx.index.values)
    valIdx = [idx for idx in range(0, data["num_vertices"]) if idx not in trainIdx and idx not in testIdx]

    #固定验证机20%和测试集20%，训练集60%
    nodesNums = data["num_vertices"]
    testNodesNum = int(nodesNums * 0.2)
    valNodesNum = int(nodesNums * 0.2)
    trainNum = nodesNums - testNodesNum - valNodesNum
    eachSamplingNum = int(trainNum / clientNum)
    overlapNodes = int(eachSamplingNum * overlapRate)
    everySplitLen = eachSamplingNum - overlapNodes
    labels = data['labels']
    trainLabel = labels[trainIdx]

    # sampleNumEachClass = [3, 3, 3, 3, 3, 3, 3, 2, 2]  #flickr_5%
    # sampleNumEachClass = [3, 3, 3, 3, 3, 2]  #Blogcatalog_5%
    # sampleNumEachClass = [6, 6, 6, 6, 6, 4]  # Blogcatalog_10%
    # sampleNumEachClass = [9, 9, 9, 9, 9, 6]  # Blogcatalog_15%

    sampleNumEachClass = [3, 3, 3, 3, 3, 3, 3, 3, 1]  # Flickr_5%
    # sampleNumEachClass = [6, 6, 6, 6, 6, 6, 6, 6, 2]  #Flickr_10%
    # sampleNumEachClass = [9, 9, 9, 9, 9, 9, 9, 9, 3]  #Flickr_15%

    # sampleNumEachClass = [19, 19, 19, 17]  #Facebook_5%
    # sampleNumEachClass = [38, 38, 38, 35]  # Facebook_10%
    # sampleNumEachClass = [56, 56, 56, 56]  #Facebook_15%

    samplingClassList = []
    samplingOverlappedNodesIndex = []
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(trainLabel) if x == k]
        samplingIndex = random.sample(range(0, len(labelIndex)), sampleNumEachClass[k])
        samplingFixedIndex = np.array(labelIndex)[samplingIndex].tolist()
        samplingClassList.append(samplingFixedIndex)
        samplingOverlappedNodesIndex += samplingFixedIndex

    currentTotalIndex, orginalTotalIndex = getLeftSamplingNodes(trainNum, samplingOverlappedNodesIndex)

    splitNodes = []
    for i in range(clientNum):
        splitIndex = random.sample(currentTotalIndex, everySplitLen)
        splitNodes.append(splitIndex + samplingOverlappedNodesIndex)
        for currentIndex in splitIndex:
            currentTotalIndex.remove(currentIndex)

    return splitNodes




def random_dataset_overlap(data, clientNum, overlapRate):   #
    #5%10%15%
    nodesNums = data["num_vertices"]
    classNum = data["num_classes"]
    testNodesNum = int(nodesNums * 0.4)
    valNodesNum = int(nodesNums * 0.2)
    trainNum = nodesNums - testNodesNum - valNodesNum
    eachSamplingNum = int(trainNum / clientNum)
    overlapNodes = int(eachSamplingNum * overlapRate)
    everySplitLen = eachSamplingNum - overlapNodes

    labels = data['labels']
    idx_train = range(0, trainNum)
    trainLabel = labels[idx_train]
    samplingClassList = []
    samplingOverlappedNodesIndex = []
    # sampleNumEachClass = [3, 3, 3, 3, 3, 3, 3, 2, 2]  #flickr_5%
    # sampleNumEachClass = [3, 3, 3, 3, 3, 2]  #Blogcatalog_5%
    # sampleNumEachClass = [6, 6, 6, 6, 6, 4]  # Blogcatalog_10%
    # sampleNumEachClass = [9, 9, 9, 9, 9, 6]  # Blogcatalog_15%

    # sampleNumEachClass = [3, 3, 3, 3, 3, 3, 3, 3, 1]  #Flickr_5%
    sampleNumEachClass = [6, 6, 6, 6, 6, 6, 6, 6, 2]  #Flickr_10%
    # sampleNumEachClass = [9, 9, 9, 9, 9, 9, 9, 9, 3]  #Flickr_15%

    # sampleNumEachClass = [19, 19, 19, 17]  #Facebook_5%
    # sampleNumEachClass = [38, 38, 38, 35]  # Facebook_10%
    # sampleNumEachClass = [56, 56, 56, 56]  #Facebook_15%

    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(trainLabel) if x == k]
        samplingIndex = random.sample(range(0, len(labelIndex)), sampleNumEachClass[k])
        samplingFixedIndex = np.array(labelIndex)[samplingIndex].tolist()
        samplingClassList.append(samplingFixedIndex)
        samplingOverlappedNodesIndex += samplingFixedIndex

    currentTotalIndex, orginalTotalIndex = getLeftSamplingNodes(trainNum, samplingOverlappedNodesIndex)

    splitNodes = []
    for i in range(clientNum):
        splitIndex = random.sample(currentTotalIndex, everySplitLen)
        splitNodes.append(splitIndex + samplingOverlappedNodesIndex)
        for currentIndex in splitIndex:
            currentTotalIndex.remove(currentIndex)

    return splitNodes

def random_flickr_overlap(data):
    return None

def random_facebook_overlao(data):
    return None

klayer = 2
class GlobalGCN(nn.Module):
    def __init__(self, data: dict,
                 test_idx: list):
        super().__init__()
        self.data = data
        self.test_idx = test_idx
        hid_channels = 16

        self.globalModel = GCN(data["dim_features"], hid_channels, data["num_classes"], 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.X, self.lbls = self.data["features"], self.data["labels"]
        self.A = Graph(self.data["num_vertices"], self.data["edge_list"])
        self.testNodesNum = int(self.data["num_vertices"] * 0.4)
        self.test_mask = self.getTestMask()
        self.acc = []
        self.f1 = []
        self.pre = []
        self.recall = []

    def getTestMask(self):
        # idx_test = []
        # for i in range(len(self.test_idx)):
        #     idx_test += self.test_idx[i]
        # idx_test = range(self.data["num_vertices"] - self.testNodesNum, self.data["num_vertices"])
        test_mask = sample_mask(self.test_idx, self.data["num_vertices"])
        test_mask = torch.tensor(test_mask)
        return test_mask

    def test(self, globalParam, epoch, iteration):
        # test
        print(f"global--test...")
        for pid, param in enumerate(list(self.globalModel.parameters())):
            param.data = torch.tensor(globalParam[pid], dtype=torch.float32) #+2

        res, f0, recall0, precision0 = infer(self.globalModel, self.X, self.A, self.lbls, self.test_mask, test=True)
        self.acc.append(res['accuracy'])
        self.f1.append(f0)
        self.recall.append(recall0)
        self.pre.append(precision0)

        if epoch == iteration - 1:
            acc_avg = sum(self.acc[iteration - 1 - 10:iteration - 1])
            f1_avg = sum(self.f1[iteration - 1 - 10:iteration - 1])
            recall_avg = sum(self.recall[iteration - 1 - 10:iteration - 1])
            pre_avg = sum(self.pre[iteration - 1 - 10:iteration - 1])
            print('avg acc:', acc_avg / 10, 'avg f2:', f1_avg / 10, 'avg recall:', recall_avg / 10, 'avg pre:',
                  pre_avg / 10)

            # idx_test = []
            # for i in range(len(self.test_idx)):
            #     idx_test += self.test_idx[i]

            self.globalModel.eval()
            outs = self.globalModel(self.X, self.A)
            tsne = TSNE(n_components=2)
            x_tsne = tsne.fit_transform(outs[klayer-1].detach().numpy())
            fig = plt.figure()
            preLabel = np.argmax(outs[klayer-1].detach().numpy(), axis=1)

            dataframe = pd.DataFrame(
                {'x0': x_tsne[:, 0], 'x1': x_tsne[:, 1],
                 'c': preLabel})  # save datavg
            dataframe.to_csv('flickr/Community3/3_FedAvg_scatter.csv', index=False, sep=',')

            plt.scatter(x_tsne[:, 0][self.test_idx], x_tsne[:, 1][self.test_idx], c=preLabel[self.test_idx], label="t-SNE")
            fig.savefig('flickr/Community3/3_FedAvg_scatter.png')
            plt.show()

        print(res)


class ClientGCN:
    def __init__(self, data: dict,
                       train_idx: torch.Tensor,
                       test_mask: torch.Tensor,
                       val_mask: torch.Tensor,
                       samplingMatrix: np.array,
                       trainNum: int
                       ):
        self.data = data
        self.samplingMatrix = samplingMatrix
        self.net = GCN(data["dim_features"], 16, data["num_classes"])
        # print(list(self.net.parameters()))
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01, weight_decay=5e-4)
        self.train_idx = train_idx
        self.X, self.lbls = self.data["features"], self.data["labels"]
        self.A = Graph(self.data["num_vertices"], self.data["edge_list"])
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.best_state = None
        self.best_epoch = 0
        self.best_val = 0
        self.trainNum = trainNum

    def loadGlobalAdjAndUpdate(self):
        globalAdjPath = 'BlogCatalog/globalAdj.npy'
        globalNodeIndexPath = 'BlogCatalog/nodeIndex.npy'
        globalAdj = np.load(globalAdjPath)
        globalNodeIndex = list(np.load(globalNodeIndexPath))
        sampling_idx_range = self.data["sampling_idx_range"]   #sampling index of current client
        # globalIndexOfSamplingIdx = [globalNodeIndex.index(samplingIndex) for samplingIndex in sampling_idx_range]
        # globalAdjOfSamplingIndex = globalAdj[globalIndexOfSamplingIdx]

        edges = []
        for node in globalNodeIndex:
            if node in sampling_idx_range:
                idx = globalNodeIndex.index(node)
                currentAdj = globalAdj[idx]
                edgesIdx = np.nonzero(currentAdj)
                for edgeIdx in edgesIdx[0]:
                    adjacentNodeIndex = globalNodeIndex[edgeIdx]
                    if adjacentNodeIndex in sampling_idx_range:
                        start = sampling_idx_range.index(node)
                        end = sampling_idx_range.index(adjacentNodeIndex)
                        edges.append([start, end])

        self.A = Graph(self.data["num_vertices"], edges)

    def updateGraph1(self, globalData):
        sampling_idx_range = self.data["sampling_idx_range"]
        edges = []
        for edge in globalData["edge_list"]:
            if edge[0] in sampling_idx_range and edge[1] in sampling_idx_range:
               start = sampling_idx_range.index(edge[0])
               end = sampling_idx_range.index(edge[1])
               edges.append(tuple([start, end]))
        self.A = Graph(self.data["num_vertices"], edges)

    def updateGraph(self, globalMatrix):
        edges = []
        sampling_idx_range = self.data["sampling_idx_range"]
        for idx in sampling_idx_range:
            eachRow = globalMatrix[idx]
            start = sampling_idx_range.index(idx)
            rowIndexs = [i for i, x in enumerate(eachRow) if x != 0]  # the index which not equal zero

            count = 0
            for rowIdx in rowIndexs:
                withInFlag = rowIdx in sampling_idx_range
                if withInFlag:
                    # if count % 20 == 0:
                        end = sampling_idx_range.index(rowIdx)
                        edges.append(tuple([start, end]))
                        count += 1

        self.A = Graph(self.data["num_vertices"], edges)


    ##----------将输出改成节点增强--------------------#
    def getTrainLayerOutput_0(self):
        self.net.train()
        self.st = time.time()
        self.optimizer.zero_grad()  # 清空过往的梯度
        layerOut = self.net.train_layer_0(self.X, self.A)
        return layerOut

    def getTrainLayerOutput_1(self, X):
        layerOut = self.net.train_layer_1(X, self.A)
        return layerOut

    def getTrainOutput(self, clientId, epoch):
        self.net.train()
        self.st = time.time()
        self.optimizer.zero_grad()  # 清空过往的梯度
        outs = self.net(self.X, self.A)
        return outs

    def trainBackward(self, clientId, outs, epoch):
        outs, lbls = outs[self.train_idx], self.lbls[self.train_idx]
        self.loss = F.cross_entropy(outs, lbls)
        self.loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新梯度
        print(f"clientId:{clientId}, Epoch: {epoch}, Time: {time.time() - self.st:.5f}s, Loss: {self.loss.item():.5f}")
        return self.loss.item()
    ###---------------------------------------------------#


    def train(self, clientId, epoch):
        self.net.train()
        self.st = time.time()
        self.optimizer.zero_grad()   #清空过往的梯度
        outs = self.net(self.X, self.A)
        outs, lbls = outs[1][self.train_idx], self.lbls[self.train_idx]

        # saveName = 'BlogCatalog/client' + str(clientId) + '-labels.npy'
        # np.save(saveName, self.lbls)

        self.loss = F.cross_entropy(outs, lbls)
        self.loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新梯度
        print(f"clientId:{clientId}, Epoch: {epoch}, Time: {time.time() - self.st:.5f}s, Loss: {self.loss.item():.5f}")
        return self.loss.item()

    def val(self, clientId, epoch):
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(self.net, self.X, self.A, self.lbls, self.val_mask)
            if val_res > self.best_val:
                print(f"clientId:{clientId}, update best: {val_res:.5f}")
                self.best_epoch = epoch
                self.best_val = val_res
                self.best_state = deepcopy(self.net.state_dict())
            return val_res

    def test(self, clientId):
        print(f"clientId:{clientId}, best val: {self.best_val:.5f}")
        # test
        print(f"clientId:{clientId}, test...")
        self.net.load_state_dict(self.best_state)
        res = infer(self.net, self.X, self.A, self.lbls, self.test_mask, test=True)
        print(f"clientId:{clientId}, final result: epoch: {self.best_epoch}")
        print(res)

    def saveFinalEmbedding(self, saveName):
        self.net.eval()
        outs = self.net(self.X, self.A)
        outs = outs[0].detach().numpy()
        #------save  embedding---------------#
        fileOuts = saveName + "-outs.npy"
        np.save(fileOuts, outs)
        #------save sampling index-----------#
        fileIndex = saveName + "-samplingIndex.npy"
        np.save(fileIndex, self.data["sampling_idx_range"])
        print(saveName + "  save success")

    def per_sample_clip(self, clipping, norm):
        grad_samples = [x for x in self.net.parameters()]
        grad_samples = [grad_samples[0], grad_samples[1]]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in self.net.parameters():
            if len(param.shape) > 1:
                param.data = param.detach().mean(dim=0)

    def clip_gradients(self, dp_mechanism, dp_clip):
        if dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.per_sample_clip(dp_clip, norm=1)
        elif dp_mechanism == 'Gaussian' or dp_mechanism == 'MA':
            # Gaussian use 2 norm
            self.per_sample_clip(dp_clip, norm=2)

    def add_noise(self, dp_mechanism, lr, dp_clip, ):
        sensitivity = cal_sensitivity(lr, self.args.dp_clip, len(self.idxs_sample))
        state_dict = self.net.state_dict()
        if dp_mechanism == 'Laplace':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
        elif dp_mechanism == 'Gaussian':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        # elif self.args.dp_mechanism == 'MA':
        #     sensitivity = cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
        #     for k, v in state_dict.items():
        #         state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
        #                                                            size=v.shape)).to(self.args.device)
        net.load_state_dict(state_dict)

    def get_all_gaussion_noise(self, nodeNums, epoch):
        eps = 20
        delta = 0.00001

        lr = 0.1
        eps = eps / epoch
        delta = delta / epoch
        dp_mechanism = 'Gaussian'
        clipVal = 4

        sigma = getGaussianSigma(eps, delta)
        sensitivity = cal_sensitivity(lr, clipVal, nodeNums)

        for pid, param in enumerate(list(self.net.parameters())):
            if len(param.shape) > 1:
                noise = torch.normal(0, sigma * sensitivity, size=param.shape)  #
                # self.clip_gradients(dp_mechanism, clipVal)    #pp = scale(param, clipVal)
                pp = scale(param, clipVal)
                param.data = pp + noise

    def get_partial_gaussion_noise(self, nodeNums, epoch):
        eps = 20
        delta = 0.00001

        lr = 0.1
        eps = eps / epoch
        delta = delta / epoch
        dp_mechanism = 'Gaussian'
        clipVal = 2

        sigma = getGaussianSigma(eps, delta)
        sensitivity = cal_sensitivity(lr, clipVal, nodeNums)

        for pid, param in enumerate(list(self.net.parameters())):
            if len(param.shape) > 1:
                his = getHistogramSplit(param)
                pp = scale(param, clipVal)
                pp = getHistogramNoiseGradient(sigma, sensitivity, pp, his)
                param.data = pp

    def get_projected_gaussion_noise(self, nodeNums, epoch):
        eps = 20
        delta = 0.00001

        lr = 0.1
        eps = eps / epoch
        delta = delta / epoch
        dp_mechanism = 'Gaussian'
        clipVal = 2

        sigma = getGaussianSigma(eps, delta)
        sensitivity = cal_sensitivity(lr, clipVal, nodeNums)

        for pid, param in enumerate(list(self.net.parameters())):
            if len(param.shape) > 1:
                pp = scale(param, clipVal)
                pp = pca_project(pp, sigma, sensitivity, clipVal)
                param.data = torch.tensor(pp)

    def cal_sensitivity(self, lr, clip, dataset_size):
        # dataset_size = 1000
        return 2 * lr * clip / dataset_size

    def Laplace(self, epsilon):
        # print('epsilon',epsilon)
        return 1 / epsilon

    def calcalate_gaussian_noise_scale(self, eps, delta):
        return np.sqrt(2 * np.log(1.25 / delta)) / eps


    def calculate_lap_noise_scale(self, eps, epoch):
        epsilon_single_query = eps / epoch
        # print(eps, epoch)
        return self.Laplace(epsilon=epsilon_single_query)

    def getLap(self, sensitivity, noise_scale, shape):
        return np.random.laplace(loc=0, scale=sensitivity * noise_scale, size=shape)

    def getGaussian(self, sensitivity, noise_scale, shape):
        return np.random.normal(loc=0, scale=sensitivity * noise_scale, size=shape)

        # -----lap  noise---------#
    def perturb_all_ldp(self, param, nodeNums):  # add all noise
        if len(param.shape) > 1:
            eps = 0.8/200
            lr = 0.1   #0.1
            clipVal = 4
            sensitivity = self.cal_sensitivity(lr, clipVal, nodeNums)
            noise_scale = self.calcalate_gaussian_noise_scale(eps, 1e-5)
            # print("sen:", sensitivity)
            # print("noise:", noise_scale)

            # pp = scale(param, clipVal)
            gaussian_noise = self.getGaussian(sensitivity, noise_scale, param.shape)# 这边的噪声尺度是不是要重新算，不重新算是不是不合适，和
            param.data = param.data + torch.tensor(gaussian_noise, dtype=torch.float32)

        return param

    def get_different_lap_noise_data(self, his, hisC, param, sensitivity, eps, clipVal, epoch):
        impactRate = []
        dataCount = 0

        for count in hisC:
            dataCount += count

        pp = scale(param, clipVal)
        for i in range(len(hisC)):
            pos = his[i]
            count = hisC[i]
            rate = float(count)/dataCount
            impactRate.append(rate)
            if rate == 0:
                continue
            current_eps = eps * rate
            current_noise_scale = self.calculate_noise_scale(current_eps, epoch)

            lap_noise = self.getLap(sensitivity, current_noise_scale, param.shape)
            pos0 = np.array(pos)[:, 0]
            pos1 = np.array(pos)[:, 1]
            pp[pos0, pos1] += torch.tensor(lap_noise[pos0, pos1], dtype=torch.float32)
            # for k in range(count):
            #     currentPos = pos[k]
            #     pp[currentPos[0]][currentPos[1]] += lap_noise[k]
        return pp

    def perturb_data_differLdp(self, param, epoch, nodeNums):
        eps = 80
        lr = 0.1
        clipVal = 2
        quentile = 5

        sensitivity = self.cal_sensitivity(lr, clipVal, nodeNums)
        # noise_scale = self.calculate_noise_scale(eps, epoch)

        if len(param.shape) > 1:
            his, hisC = getHistogramSplit(param, quentile)
            pp = self.get_different_lap_noise_data(his, hisC, param, sensitivity, eps, clipVal, epoch)
            param.data = pp

        return param

    def perturb_partial_data(self, param, row, col, nodeNums):
        eps = 2/200
        lr = 0.1
        clipVal = 4
        sensitivity = self.cal_sensitivity(lr, clipVal, nodeNums)
        noise_scale = self.calcalate_gaussian_noise_scale(eps, 1e-5)

        # pp = scale(param, clipVal)
        # pp = pp.detach().numpy()
        maxCount = len(row)
        gaussian_noise = self.getGaussian(sensitivity, noise_scale, [maxCount, 1])

        importPos = np.zeros(shape=(param.shape[0], param.shape[1]), dtype=np.int16)

        for i in range(len(row)):
            importPos[row[i]][col[i]] = 1

        count = 0
        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                if importPos[i][j] == 1:
                    param.data[i][j] += gaussian_noise[count]
                    # param.data[i, j] += torch.tensor(gaussian_noise[count], dtype=torch.float32)
                    count += 1

        #
        # gaussian_noise = self.getGaussian(sensitivity, noise_scale, param.shape)  # 这边的噪声尺度是不是要重新算，不重新算是不是不合适，和
        # pp[row, col] += torch.tensor(gaussian_noise[row, col], dtype=torch.float32)

        return param

    def getHistogramSplit(self, param):
        param = np.array(param)
        size = param.shape

        max = np.amax(param, axis=0)  # get the max val of each column
        min = np.amin(param, axis=0)
        splitNum = 5
        interval = (max - min) / splitNum
        splitCount = []

        for i in range(size[1]):
            eachColIntervalCount = []
            for j in range(splitNum):
                curCount = 0
                for k in range(size[0]):
                    start = min[i] + interval[i] * j
                    end = min[i] + interval[i] * (j + 1)
                    if param[k][i] >= start and param[k][i] < end:
                        curCount += 1
                eachColIntervalCount.append(curCount)
            splitCount.append(eachColIntervalCount)

        splitCount = np.array(splitCount)
        maxSplit = np.amax(splitCount, axis=1)
        index = []
        for i in range(len(maxSplit)):
            currentIndex = splitCount[i].tolist().index(maxSplit[i])
            index.append(currentIndex)

        return max, min, interval, index, splitCount

    def getHistogramMappingGradient(self, param, eps, delta, sensitivity):
        if len(param.shape) > 1:
            param = param.detach().numpy()
            sigma = getGaussianSigma(eps / 200, delta)
            max, min, interval, index, splitCount = getHistogramSplit(param)
            # sensitivity = np.amax(np.array(max), axis=1)
            noisedGrad = getHistogramNoiseGradient(sigma, param, min, interval, index, splitCount, sensitivity)

            noisedGrad = torch.tensor(noisedGrad, dtype=torch.float32)
        else:
            noisedGrad = param

        return noisedGrad

    def get_center_index(self, k, labels_):
        # index = []
        # for k in range(300):
        #     labelIndex = [i for i, x in enumerate(labels_) if x == k]
        #     index.append(labelIndex)
        labelIndex = [i for i, x in enumerate(labels_) if x == k]
        return labelIndex

    def k_means(self, param):
        param = param.reshape(-1, 1)
        km = KMeans(n_clusters=10)
        y_pred = km.fit(param)

        # index = []
        # for k in range(300):
        #     labelIndex = [i for i, x in enumerate(km.labels_) if x == k]
        #     index.append(labelIndex)

        return y_pred, km.cluster_centers_, km.labels_  #index,

    def expMechanism(self, scores, eps, sensitivity):
        probability0 = []
        probability1 = []
        scores = scores.flatten()
        for score in scores:
            probability0.append(np.exp(0.5 * eps * score / sensitivity))

        sum = np.sum(probability0)
        # 归一化处理
        for i in range(len(probability0)):
            probability1.append(probability0[i] / sum)

        choose = np.random.choice(scores, 1, p=probability1)[0]

        # print("center:", choose)

        return choose

    def getExpSensitivity(self, centers):
        centers = centers.flatten()
        max = centers[0]
        min = centers[0]

        for i in range(len(centers)):
            if max < centers[i]:
                max = centers[i]

            if min > centers[i]:
                min = centers[i]

        sensitivity = max - min

        return sensitivity

    def clusterNoise(self, nodeNums, param, expEps):
        # print(param.shape)
        if len(param.shape) > 1 and param.shape[0] * param.shape[1] > 1000:
            clipVal = 4
            pp = scale(param, clipVal)

            currentParamData = pp.data.detach().numpy()

            y_pred, centers, labels = self.k_means(currentParamData)  #index,
            sensitivity = self.getExpSensitivity(centers)
            choose = self.expMechanism(centers, expEps, sensitivity)
            centers = list(centers.flatten())
            centerIndex = centers.index(choose)  #get chosed class
            currentIndex = self.get_center_index(centerIndex, labels)

            # print("centerIndex:", centerIndex)
            # currentIndex = index[centerIndex]
            row = np.array(currentIndex) / param.shape[1]
            col = np.array(currentIndex) % param.shape[1]
            row_int = row.astype(int)
            print(len(row_int))

            pp = self.perturb_partial_data(param, row_int, col, nodeNums)

            param.data = pp

        return param


if __name__ == "__main__":
    set_seed(2023)# set_seed(2023) #BlogCatalog #Flickr
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    # data = BlogCatalog()
    #
    # data = Facebook()
    data = Flickr()
    facebook_feature_dim = 4714
    dataDict = {"num_classes": data["num_classes"],
                "num_vertices": data["num_vertices"],
                "num_edges": data["num_edges"],
                "dim_features": data["dim_features"], #data["dim_features"], #data["dim_features"], #facebook_feature_dim, #data["dim_features"], #facebook_feature_dim,
                "features": data["features"],
                "edge_list": data["edge_list"],
                "labels": data["labels"]}

    #--------------splitting data and create client net----------------#
    train_link_opinion = False
    train_local = False
    train_fedavg = False
    train_node_augmentation = True

    clientList = []
    clientNum = 6
    samplingRate = [0.3, 0.4, 0.5, 0.5, 0.6, 0.7]
    classNum = data["num_classes"]
    sampleNumEachClass = 10                #Facebook 200  others_10
    filePath = 'flickr/'              #"BlogCatalog/"  'flickr/'   #BlogCatalog   #facebook_200
    overlapRate = 0.1
    clientSamplingIndex = random_dataset_overlap(data, clientNum, overlapRate)
    all_test_idx = []
    all_data = []
    for i in range(clientNum):
        saveName = filePath + 'client' + str(i)
        #------------fix----overlapNodes-----------------#
        samplingIndex = clientSamplingIndex[i]
        clientData, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum = getOverlapClientData(data, samplingIndex, sampleNumEachClass, classNum, saveName)
        #---------random---------------------------------#
        # clientData, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum = getClientData(data, samplingRate[i], sampleNumEachClass, classNum, saveName)
        client = ClientGCN(clientData, train_mask, val_mask, test_mask, samplingMatrix, trainNum)
        clientList.append(client)
        all_test_idx += clientData["idx_test"]

    all_data,  all_test_idx = get_all_data(data, clientNum, clientList, all_test_idx)


    unique_set = set(all_test_idx)
    all_test_idx = list(unique_set)

    # --------------update graph----------------------#
    if train_link_opinion:
        globalMatrix = getGlobalAdjMatrix(clientList)
        # overlapNodes = getOverlapNodes(globalMatrix) #list(range(0, data["num_vertices"]))#getOverlapNodes(globalMatrix)
        overlapNodes = getOverlapNodes1(clientList)
        # print('overlapNodes:', len(overlapNodes))
        for i in range(clientNum):
            client = clientList[i]
            client.updateGraph(globalMatrix)
            # net.updateGraph1(data["edge_list"])

    #-----training with nodes augmentation--------#
    if train_node_augmentation:
        globalMatrix = getGlobalAdjMatrix(clientList)    #w/o
        # overlapNodes = getOverlapNodes(globalMatrix)
        overlapNodes = getOverlapNodes1(clientList)
        # overlapNodes = list(range(0, data["num_vertices"]))
        aggParams = {0: 0, 1: 0, 2: 0, 3: 0}
        clientLoss = {
            0: [], 1: [], 2: [], 3: [], 4: [], 5: []
        }
        clientAcc = {
            0: [], 1: [], 2: [], 3: [], 4: [], 5: []
        }
        iteration = 200
        for epoch in range(iteration):
            #-----------------training------------------------#
            clientOuts = []
            for i in range(clientNum):
                client = clientList[i]
                outs = client.getTrainOutput(i, epoch)
                clientOuts.append(outs[1])

            #---------------node augmentation---------------------#
            globalNodeEmbeddings = getGlobalOverlapNodesEmd(clientList, overlapNodes, clientOuts)
            outs = setGlobalNodeEmdForLocalNodes(clientList, clientOuts, overlapNodes, globalNodeEmbeddings)

            for i in range(clientNum):
                client = clientList[i]
                eachLossItem = client.trainBackward(i, outs[i], epoch)
                clientLoss[i].append(eachLossItem)

            #-------------validate---------------------------------#
            for i in range(clientNum):
                val = clientList[i].val(i, epoch)
                clientAcc[i].append(val)

            for i in range(clientNum):
               for pid, param in enumerate(list(clientList[i].net.parameters())):
                   aggParams[pid] += param.detach().numpy()#param #clientList[i].clusterNoise(clientList[i].trainNum, param, 2)
                       #perturb_all_ldp(param, clientList[i].trainNum)
                       #clusterNoise(clientList[i].trainNum, param, 2)
                       #perturb_all_ldp(param, clientList[i].trainNum)
            #2、--------getting average parameters-------------#
            for id, aggParam in aggParams.items():
                aggParams[id] = aggParam/clientNum
            #3、-----------setting aggerated parameters for each client-----------#
            for i in range(clientNum):
                for pid, param in enumerate(list(clientList[i].net.parameters())):
                    param.data = torch.tensor(aggParams[pid])#aggParams[pid]
                    #clientList[i].perturb_all_ldp(aggParams[pid], clientList[i].trainNum)  #aggParams[id]
                    # perturb_all_ldp(aggParams[pid], clientList[i].trainNum)
                    #clientList[i].clusterNoise(clientList[i].trainNum, aggParams[pid], 2)  #aggParams[id]
                    #clientList[i].perturb_all_ldp(param, clientList[i].trainNum)
                    #clientList[i].clusterNoise(clientList[i].trainNum, param, 2)
                    #clientList[i].getHistogramMappingGradient(param, 2, 0.00001, 4)
                    #clientList[i].clusterNoise(clientList[i].trainNum, param, 2)
                    #clientList[i].perturb_all_ldp(param, clientList[i].trainNum)
                    #clientList[i].clusterNoise(200, clientList[i].trainNum, param, 2)
                    #clientList[i].perturb_all_ldp(param, 200, clientList[i].trainNum)
                    # #clientList[i].clusterNoise(200, clientList[i].trainNum, param, 2) #aggParams[pid]
            #4、----------clear aggParams--------------------------#
            for key in aggParams.keys():
                aggParams[key] = 0

    if train_local:
        aggParams = {0: 0, 1: 0, 2: 0, 3: 0}
        iteration = 200
        for epoch in range(iteration):
            #1、-------training and aggerating------------------#
            for i in range(clientNum):
               clientList[i].train(i, epoch)
               clientList[i].val(i, epoch)
               #####################################################
               # nodeNums = len(clientList[i].X)
               # clientList[i].get_all_gaussion_noise(nodeNums, 200)
               # clientList[i].get_projected_gaussion_noise(nodeNums, 200)

               if train_fedavg:
                   for pid, param in enumerate(list(clientList[i].net.parameters())):
                       aggParams[pid] += param.detach().numpy()#param   #param.detach().numpy()
                       #clientList[i].perturb_all_ldp(param, clientList[i].trainNum)
                       #clusterNoise(clientList[i].trainNum, param, 2)
                       #clientList[i].perturb_all_ldp(param, clientList[i].trainNum) #param
            #2、--------getting average parameters-------------#
            if train_fedavg:
                for id, aggParam in aggParams.items():
                    aggParams[id] = aggParam/clientNum

                #3、-----------setting aggerated parameters for each client-----------#
                u = 0.5
                for i in range(clientNum):
                    for pid, param in enumerate(list(clientList[i].net.parameters())):
                        param.data = torch.tensor(aggParams[pid])
                        # param.data = torch.tensor(aggParams[pid] + u * (
                        #         aggParams[pid] - param.detach().numpy())) #aggParams[pid] #finalAggParam#aggParams[pid]

                #4、----------clear aggParams--------------------------#
                for key in aggParams.keys():
                    aggParams[key] = 0

        ###-------plot-------
    plt.show()

    print("\ntrain finished!")
    for i in range(clientNum):
        clientList[i].test(i)

    # print("\nsave final embedding")
    # for i in range(clientNum):
    #     saveName = filePath + "client" + str(i) + "local embedding"
    #     clientList[i].saveFinalEmbedding(saveName)
