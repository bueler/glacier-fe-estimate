# module provides writeout()

import matplotlib.pyplot as plt

def writeout(outname, show=False):
    if show:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname, bbox_inches='tight')
