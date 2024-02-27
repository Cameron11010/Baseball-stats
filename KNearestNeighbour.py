import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# Calculating the nearest neighbour as long as there data has 6 features
def euclidean_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Both points must have the same number of features. P length:", len(p), "Q length: ", len(q))
    else:
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


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


class KNearestNeighbours:
    def __init__(self, k=12):
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point, unknown_pitch_number, display):
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
        if display == True:

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
            plt.title(('KNN Nearest Neighbours Pitch Type Results', f"Unknown Pitch {unknown_pitch_number}"),
                      color='white')
            for category, color in color_map.items():
                ax.scatter([], [], [], c=color, label=category)

            ax.legend(fontsize=5, title='Categories', loc='upper right')

        return result
