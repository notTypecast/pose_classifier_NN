"""
Final data preprocessing for file dataset-numerical.csv
Does the following:
-> Normalizes data for each column.
-> Splits data into train and test data sets, correctly balancing all classes between sets.
Creates files dataset-train.csv and dataset-test.csv
"""
import csv
from random import shuffle

data = []

with open("dataset-numerical.csv", "r") as csvf:
    reader = csv.reader(csvf, delimiter=";")
    data.append(next(reader))

    for row in reader:
        data.append([float(x) for x in row])


# normalize data
normalized_data = []
for _ in range(len(data)-1):
    normalized_data.append([])

for col in range(len(data[0]) - 1):
    xmax = data[1][col]
    xmin = data[1][col]

    for row in data[2:]:
        if row[col] > xmax:
            xmax = row[col]

        if row[col] < xmin:
            xmin = row[col]

    colrange = xmax - xmin

    for i, row in enumerate(data[1:]):
        normalized_data[i].append((row[col] - xmin) / colrange)

# add class as-is (do not normalize since its an output)
for i, row in enumerate(data[1:]):
    normalized_data[i].append(row[-1])

classes = {}

# split data by class
for row in normalized_data:
    if row[-1] not in classes:
        classes[row[-1]] = []

    classes[row[-1]].append(row)

train_data = []
test_data = []

for class_ in classes:
    train_points = int(0.7 * len(classes[class_]))
    test_points = len(classes[class_]) - train_points

    for i in range(train_points):
        train_data.append(classes[class_][i])

    for i in range(test_points):
        test_data.append(classes[class_][train_points + i])

with open("dataset-train.csv", "w") as train_csvf, open("dataset-test.csv", "w") as test_csvf:
    train_writer = csv.writer(train_csvf, delimiter=";")
    for row in train_data:
        train_writer.writerow(row)

    test_writer = csv.writer(test_csvf, delimiter=";")
    for row in test_data:
        test_writer.writerow(row)
