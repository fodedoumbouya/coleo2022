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
import json
import numpy as np

tdatadir="trainDatabase/e0/"

inSet = json.load(open(tdatadir+"trainingSete0.json"))

# plot image set
drawIm.drawImageSet(inSet,receptors.labels,datadir=tdatadir, title="Observation ")
 
# learn images
model = train.learnLabelledSet(inSet,datadir=tdatadir)

# save model
model.save('AI/trained_model.h5')

# reload model
storedModel = train.loadImageLabeller('AI/trained_model.h5')

# open activation set (empty)
imageSet = json.load(open(tdatadir+"testSete0.json"))

# find receptors
segmentedSet = segmentation.segmentImageSet(imageSet,datadir=tdatadir)

# classify receptors
outSet = train.classifyImageSet(segmentedSet,storedModel,receptors.labels,datadir=tdatadir)

# plot classified receptrs
drawIm.drawImageSet(outSet,receptors.labels,datadir=tdatadir, title="Prediction ")
   