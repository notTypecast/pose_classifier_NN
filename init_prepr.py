"""
Initial data preprocessing for file dataset-HAR-PUC-Rio.csv
Does the following:
-> Replaces all non-numerical values with corresponding numerical values. 
For each class in non-numerical value column, assigns an integer (0 to first class met, and so on).
-> Replaces all comma decimal separators with dots.
Creates file dataset-numerical.csv
"""
import csv

data = []

with open("dataset-HAR-PUC-Rio.csv", "r") as csvf:
    reader = csv.reader(csvf, delimiter=";")

    for row in reader:
        data.append(row)

users = {}
genders = {}
classes = {}

new_data = [data.pop(0)]

for row in data:
    new_data.append([])
    if row[0] not in users:
        users[row[0]] = len(users)

    new_data[-1].append(users[row[0]])

    if row[1] not in genders:
        genders[row[1]] = len(genders)

    new_data[-1].append(genders[row[1]])

    new_data[-1].append(row[2])
    new_data[-1].append(row[3].replace(",", "."))
    new_data[-1].append(row[4])
    new_data[-1].append(row[5].replace(",", "."))

    new_data[-1].extend(row[6:18])

    if row[18] not in classes:
        classes[row[18]] = len(classes)

    new_data[-1].append(classes[row[18]])

with open("dataset-numerical.csv", "w") as csvf:
    writer = csv.writer(csvf, delimiter=";")

    for row in new_data:
        writer.writerow(row)