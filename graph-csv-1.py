import pandas as pd
import matplotlib.pyplot as plt
import globals

def load_and_graph_csv(path_to_csv):
    data = pd.read_csv(path_to_csv)

    data.set_index('Time(s)', inplace=True)
    fig, axs = plt.subplots(3, 4, figsize=(15, 9))

    axs = axs.flatten()

    for i, column in enumerate(data.columns):
        axs[i].plot(data.index, data[column])
        axs[i].set_title(column)
        axs[i].set_xlabel('time')
        axs[i].grid(True)

    for ax in axs[len(data.columns):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

csv_location = f'{globals.DIR_BIOGEARS_BIN}\csv-data\\none-old\\DefaultTemplateMaleResults.csv'

load_and_graph_csv(csv_location)
