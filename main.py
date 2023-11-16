import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import openpyxl
from openpyxl import load_workbook

workbook = load_workbook(filename="stats-trial.xlsx")

sheets = workbook['Train']

trial = workbook['Trial']

headings = [sheets.cell(row=1, column=i).value for i in range(1, sheets.max_column + 1)]

new_point = [trial.cell(row=2, column=i).value for i in range(7, 13)]


pitch_ranges = {
    "fseam": (7, 12),
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
          'sinker': sinker,
          'cutter': cutter,
          '2-seam fastball': tseam,
          'splitter': splitter,
          'changeup': changeup,
          'slider': slider,
          'curve': curve,
          'knuckle': knuckle,
          }

color_map = {
            '4-seam fastball': '#de2a33',
            'sinker': '#ffff33',
            'cutter': '#e8208e',
            '2-seam fastball': '#ffad00',
            'splitter': '#f3ffe3',
            'changeup': '#008080',
            'slider': '#86d8f7',
            'curve': '#b19cd9',
            'knuckle': '#98FF98'
        }


def euclidean_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Both points must have the same number of features. P length:", len(p), "Q length: ", len(q))
    else:
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


class KNearestNeighbours:
    def __init__(self, k=15):
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

    
    
ranges = [(7, 12), (14, 19), (21, 26), (28, 33), (35, 40), (42, 47), (49, 54), (56, 61), (63, 68)]
unknown_pitch = []
    
for start, end in ranges:
    for j in range(2, trial.max_row + 1):
        column_values = []  # Reset column_values for each row
        for start, end in ranges:
            for i in range(start, end + 1):
                cell_value = trial.cell(row=j, column=i).value
                if cell_value is not None or cell_value == 0:
                    column_values.append(cell_value)
        if column_values:
            unknown_pitch.extend(column_values)
            

clf = KNearestNeighbours()
clf.fit(points)

total_predictions = 0
correct_predictions = 0


num_variables = len(unknown_pitch) // 6
for i in range(num_variables):
    globals()[f'unknown_pitch {i+1}'] = unknown_pitch[i*6: (i+1)*6]
for i in range(num_variables):
    print(f'unknown_pitch {i+1}: {globals()[f"unknown_pitch {i+1}"]}')
    guess_pitch = unknown_pitch[i*6: (i+1)*6]
    prediction = clf.predict(guess_pitch)
    print("Category prediction:", prediction)
    plt.show()

"""
    
    col = i+1
    if col > 6:
        col = 0
        
    print(col)
    total_predictions += 1
    if col == (7, 12) and prediction == "4-seam fastball":
        correct_predictions += 1
    elif col == (14, 19) and prediction == "slider":
        correct_predictions += 1
    elif col == (21, 26) and prediction == "changeup":
        correct_predictions += 1
    elif col == (28, 33) and prediction =="curve":
        correct_predictions +=1
    elif col == (35, 40) and prediction == "sinker":
        correct_predictions += 1
    elif col == (42, 47) and prediction == "cutter":
        correct_predictions += 1
    elif col == (49, 54) and prediction == "splitter":
        correct_predictions += 1
    elif col == (56, 61) and prediction == "knuckle":
        correct_predictions += 1
    elif col == (63, 68) and prediction == "tseam":
        correct_predictions += 1
    else:
        print("incorrect")
    col += 1
    

print("Correct predictions: ", correct_predictions)
print("Total predictions: ", total_predictions)

"""

