#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:41:51 2022

@author: filippi_j
"""

import src.coleolabels as receptors
import src.affichage as drawIm

import src.segmentation as segmentation
import src.train as train
import src.report as report
import json
import numpy as np

traindatadir="trainDatabase/e0/"
testdatadir="testDatabase/31CanomalusF/"

savedModelFileName = 'AI/trained_model.h5'
reportFileName = 'report'

inSet = json.load(open(traindatadir+"trainingSete0.json"))

# plot image set
drawIm.drawImageSet(inSet,receptors.labels,datadir=traindatadir, title="Observation ")
 
# learn images
model = train.learnLabelledSet(inSet,datadir=traindatadir,model_v=2,nEpochs=45,flag_balanced=True)

# save model

model.save(savedModelFileName)

print("model saved at ",savedModelFileName)


# reload model
storedModel = train.loadImageLabeller('AI/trained_model.h5')
# open activation set (empty)
imageSet = json.load(open(traindatadir+"testSete0.json"))
# find receptors
segmentedSet = segmentation.segmentImageSet(imageSet,datadir=traindatadir)
# classify receptors
outSet = train.classifyImageSet(segmentedSet,storedModel,receptors.labels,datadir=traindatadir)
# plot classified receptrs
drawIm.drawImageSet(outSet,receptors.labels,datadir=traindatadir, title="Prediction ")

# save text report
report.file_report(inSet, receptors.labels, reportFileName)

# on new dataset (never seen)
testdatadir="testDatabase/"
imageSet = json.load(open(testdatadir+"testSet.json"))
segmentedSet = segmentation.segmentImageSet(imageSet,datadir=testdatadir)
outSet = train.classifyImageSet(segmentedSet,storedModel,receptors.labels,datadir=testdatadir)
drawIm.drawImageSet(outSet,receptors.labels,datadir=testdatadir, title="Prediction ")

