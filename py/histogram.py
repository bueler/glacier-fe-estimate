from sys import argv
from figures import histogramPhirat

ratkey = 'Phirat[q=4.0]'

def ratvals(filename):
    from csv import DictReader
    import numpy as np
    with open(filename, newline='') as f:
        reader = DictReader(f)
        return np.array([float(row[ratkey]) for row in reader], dtype=np.float64)

histogramPhirat(argv[2], ratvals(argv[1]), leftlog10=-23.5, rightlog10=-17.5)
