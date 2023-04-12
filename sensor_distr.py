"""
Creates plots to view distribution of sensor data.
Reads file dataset-numerical.csv.
"""
import csv
import matplotlib.pyplot as plt

vals = []
for _ in range(12):
    vals.append([])

with open("dataset-numerical.csv", "r") as csvf:
    reader = csv.reader(csvf, delimiter=";")
    next(reader)

    for row in reader:
        for i in range(12):
            vals[i].append(int(row[6 + i]))

for sensor in range(0, 10, 3):
    fig, ax = plt.subplots(1, 3, sharey=True)

    for i in range(3):
        ax[i].scatter(vals[sensor + i], range(len(vals[sensor + i])))
    
    sensor_n = int(sensor/3) + 1
    fig.suptitle(f"Sensor {sensor_n}")
    plt.savefig(f"img/sensor{sensor_n}.png")