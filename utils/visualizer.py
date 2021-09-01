import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_client_meta(summary_path: str, clients_meta: dict):
    df = pd.DataFrame.from_dict(clients_meta).fillna(0).astype(int)
    df = df.sort_index()
    sns.set(rc={'figure.figsize': (11, 11)})
    sns.heatmap(df, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Clients', fontsize=14)
    plt.ylabel('Class Index', fontsize=14)
    plt.savefig(summary_path+"/client_meta.png")
