import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import openpyxl
from openpyxl import load_workbook

workbook = load_workbook(filename="stats.xlsx")

sheets = workbook['Train']

new_point = [83.4, 2310, -3.5, -32.90, 7.80, 1.5]

headings = [sheets.cell(row=1, column=i).value for i in range(1, sheets.max_column + 1)]
# print(headings)

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
}

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

points = {'4-seam fastball': fseam,
          'slider': slider,
          'changeup': changeup,
          'curve': curve,
          'sinker': sinker,
          'cutter': cutter,
          'splitter': splitter,
          'knuckle': knuckle,
          '2-seam fastball': tseam,
          }

color_map = {
            '4-seam fastball': '#e41a1c',
            'slider': '#377eb8',
            'changeup': '#4daf4a',
            'curve': '#984ea3',
            'sinker': '#ff7f00',
            'cutter': '#ffff33',
            'splitter': '#a65628',
            'knuckle': '#f781bf',
            '2-seam fastball': '#999999'
        }

print(new_point)


def euclidean_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Both points must have the same number of features. P length:", len(p), "Q length: ", len(q))
    else:
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


class KNearestNeighbours:
    def __init__(self, k=10):
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []
        k = self.k
        for category, point_list in self.points.items():
            for point in point_list:
                try:
                    distance = euclidean_distance(point, new_point)
                    distances.append([distance, category, point])
                except ValueError as e:
                    print(f"Error: {e}")
                    print(f"Category: {category}")
                    print(f"Point: {point}")
                    print(f"New Point: {new_point}")
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]

        distances.sort()
        for distance, category, neighbour in distances[:k]:
            print(f"Distance: {distance}, Category: {category}, Point: {neighbour}")
            nearest_neighbours = [neighbour for _, category, neighbour in distances[:k]]
            
        # VISUALISATION
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.grid(True, color="#323232")
        ax.set_facecolor("black")
        ax.figure.set_facecolor("#121212")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.tick_params(axis="z", colors="white")

        

        ax.scatter(new_point[0], new_point[1], new_point[4], color=color_map.get(result), marker="*", s=500,
                   zorder=150)

        for distance, category, neighbour in distances[:k]:
            ax.scatter(neighbour[0], neighbour[1], neighbour[4], color=color_map.get(category), marker="o", s=100,
                       zorder=100)

            ax.plot([new_point[0], neighbour[0]], [new_point[1], neighbour[1]], zs=[new_point[4], neighbour[4]],
                    color='red', linestyle="--", linewidth=1)

        ax.set_xlabel('Velocity', color='white')
        ax.set_ylabel('Spinrate', color='white')
        ax.set_zlabel('Break', color='white')
        plt.title('KNN Nearest Neighbours Pitch Type Results', color='white')
        for category, color in color_map.items():
            ax.scatter([], [], [], c=color, label=category)

        ax.legend(fontsize=5, title='Categories', loc='upper right')


        return result


clf = KNearestNeighbours()
clf.fit(points)
prediction = clf.predict(new_point)
print("Category preditcion:", prediction)
plt.show()
