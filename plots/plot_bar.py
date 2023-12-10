from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import ipdb

def plot_bar(data_lists, name_list, x_name=None, y_name=None,  **kwargs):
    """
    plot a stacked bar. Each element in data_lists should correspond to a row

    param data_lists: [[y1_1,y2_1,y3_1,...], [y1_2,y2_2,y3_2,...], ...]
    param name_list: [name_for_stack1, name_for_stack2, ...]

    """

    if x_name is None:
        x_name = [str(i+1) for i in range(len(data_lists[0]))]
    else:
        while len(x_name) < len(data_lists[0]):
            iter=0
            x_name.append(str(iter))
            iter+=1

    fig = plt.figure(dpi=150)
    fig.clf()
    bottom = np.zeros(len(data_lists[0]))

    for data_list,name in zip(data_lists,name_list):
        plt.bar(x_name, data_list, width=0.5, label = name, bottom=bottom)
        bottom += data_list

    if y_name is not None:
        plt.ylabel(y_name)

    plt.legend(loc='upper right')

    return fig
