import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from openpyxl import load_workbook

workbook = load_workbook(filename="stats.xlsx")

#print(workbook.sheetnames)

sheets = workbook['Train']

new_point = [76.2, 2495, 14.20, -56.60, 17.10, 2.4]

headings = [sheets.cell(row=1, column=i).value for i in range(1, sheets.max_column+1)]
#print(headings)

pitch_ranges = {
    "fseam": ([7, 12]),
    "slider": (14, 19),
    "changeup": (21, 26),
    "curve": (28, 33),
    "sinker": (35, 40),
    "cutter": (42, 47),
    "splitter": (49, 54),
    "knuckle": (56, 61),
    "tseam": (63, 68),
    "breaking": (70, 75),
    "offspeed": (77, 82)}

results = {}


for pitch, pitch_range in pitch_ranges.items():
    results[pitch] = []
    for i in range(2, sheets.max_row + 1):
        row_data = []
        for j in range(pitch_range[0], pitch_range[1] + 1):
            cell_value = sheets.cell(row=i, column=j).value
            if cell_value is not None or cell_value == 0:
                row_data.append(cell_value)
        if row_data:  # Only append non-empty rows
            results[pitch].append(row_data)



fseam = results["fseam"]
slider = results["slider"]
changeup = results["changeup"]
curve = results["curve"]
sinker = results["sinker"]
cutter = results["cutter"]
splitter = results["splitter"]
knuckle = results["knuckle"]
tseam = results["tseam"]
breaking = results["breaking"]
offspeed = results["offspeed"]

points = {'4-seam fastball': fseam,
          'slider': slider,
          'changeup': changeup,
          'curve': curve,
          'sinker': sinker,
          'cutter': cutter,
          'splitter': splitter,
          'knuckle': knuckle,
          '2-seam fastball': tseam,
          'breaking': breaking,
          'offspeed': offspeed}

print(new_point)


def euclidean_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Both points must have the same number of features. P length:", len(p), "Q length: ", len(q))
    else:
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


class KNearestNeighbors:
    def __init__(self, k=6):
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []

        for category, point_list in self.points.items():
            for point in point_list:
                try:
                    distance = euclidean_distance(point, new_point)
                    distances.append([distance, category])
                except ValueError as e:
                    print(f"Error: {e}")
                    print(f"Category: {category}")
                    print(f"Point: {point}")
                    print(f"New Point: {new_point}")

        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result


clf = KNearestNeighbors()
clf.fit(points)
prediction = clf.predict(new_point)
print("Category:", prediction)
