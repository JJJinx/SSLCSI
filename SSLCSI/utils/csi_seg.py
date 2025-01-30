import ruptures as rpt


def csi_segmentation(signal):
    """
        input signal, has the shape [T,C]

        return a list of segmentation point's index
    """

    # detection
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=10)

    # # display
    # rpt.display(signal_amp,[0,45,1500], result)
    # plt.show()
    return result