import numpy as np


class CentroidTracker:
    def __init__(self, maxDissappeared=50):
        self.nextID = 0
        self.objects = {}
        self.dissappeared = {}
        self.maxDissappeared = maxDissappeared

    def register(self, centroid):
        self.objects[self.nextID] = centroid
        self.dissappeared[self.nextID] = 0
        self.nextID += 1

    def get_count(self):
        return self.nextID

    def remove(self, id):
        del self.objects[id]
        del self.dissappeared[id]

    def update(self, rects):
        if len(rects) == 0:
            for id in list(self.dissappeared.keys()):
                self.dissappeared[id] += 1

                if self.dissappeared[id] > self.maxDissappeared:
                    self.remove(id)
            return self.objects

        inputCentroids = np.asarray(
            [((x + x + w) // 2, (y + y + h) // 2) for (x, y, w, h) in rects])

        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            centroids = np.array(list(self.objects.values()))

            distances = np.linalg.norm(
                centroids[:, np.newaxis] - inputCentroids, axis=-1)

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                id = objectIDs[row]
                self.objects[id] = inputCentroids[col]
                self.dissappeared[id] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(len(centroids))) - usedRows
            unusedCols = set(range(len(inputCentroids))) - usedCols

            for row in unusedRows:
                id = objectIDs[row]
                self.dissappeared[id] += 1

                if self.dissappeared[id] > self.maxDissappeared:
                    self.remove(id)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects
