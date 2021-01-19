from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

def plotMeanAndStd(time, meanTraj, stdTraj, indices = None):
    colorCycle = cycle(np.linspace(0, 1, 10))
    colorMap = cm.get_cmap('rainbow')

    legendHandles = []

    if (not indices):
        indices = range(meanTraj.shape[1])



    for i in indices:
        plt.figure()

        curve = meanTraj[:,i]
        curve_std = stdTraj[:, i]

        lowerBound = curve - 2 * curve_std
        upperBound = curve + 2 * curve_std

        x = time
        color = colorMap(next(colorCycle))

        plt.fill_between(x, lowerBound, upperBound, color=color, alpha=0.5)

        label = 'Joint %d' % (i)
        newHandle = plt.plot(x, curve, color=color, label=label)
        legendHandles.append(newHandle[0])

    plt.legend(handles=legendHandles)

    plt.xlabel('time')
    plt.ylabel('q')

    plt.autoscale(enable=True, axis='x', tight=True)