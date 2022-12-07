# -*- coding: utf-8 -*-
import json
import coleolabels as receptors

def file_report(dataset, labels, fileNamePrefix):
    file = open(fileNamePrefix+'.txt', 'w')
    
    for i,pkey in enumerate(dataset.keys()):
        labels_count, total = report_data(dataset[pkey], labels, fileNamePrefix)
        stats = labels_count
        file.write(f'{pkey}:\n{stats}\nTotal: {total}\n\n')

    file.close()

def report_data(data, labels, fileNamePrefix):
    labels_count = {}
    for key in labels:
        labels_count[key] = 0
    total_circles = 0
    for circle in range(len(data)):
        label = data[circle]['label']
        if label not in labels:
            label = "non classé"
        labels_count[label] += 1
        total_circles += 1
    return labels_count, total_circles

traindatadir="../trainDatabase/e0/"
inSet = json.load(open(traindatadir+"trainingSete0.json"))
file_report(inSet, receptors.labels, "report")