# -*- coding: utf-8 -*-
def file_report(dataset, labels, fileNamePrefix):
    total_labels_set = {}
    total_circles_set = 0
    total_stats = ''
    for key in labels:
        total_labels_set[key] = 0
    
    file = open(fileNamePrefix+'.md', 'w')
    for i,pkey in enumerate(dataset.keys()):
        stats = ''
        labels_count, total = report_data(dataset[pkey], labels, fileNamePrefix)
        for label in labels_count:
            label_nb = labels_count[label]
            total_labels_set[label] += label_nb
            stats += f'* {label}: **{label_nb}** ' + \
                f'*({round(label_nb/total*100, 2)}%)*\n'
        file.write(f'### {pkey}\n{stats}**Total: {total}**\n\n')
        total_circles_set += total
    for label in total_labels_set:
        label_nb = total_labels_set[label]
        total_stats += f'* {label}: **{label_nb}** ' + \
            f'*({round(label_nb/total_circles_set*100, 2)}%)*\n'
    file.write(f'### Total\n{total_stats}**Total: {total_circles_set}**')
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