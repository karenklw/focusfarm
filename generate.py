import argparse
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import rules
import plot


def generate_df(n, seed=None):
    """
    Returns a dataframe of n sessions per player type (a.k.a. a time series with n * 4 sequences).
    """
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    df_a = pd.DataFrame(rules.generate_type_a_sessions(n, rng))
    df_b = pd.DataFrame(rules.generate_type_b_sessions(n, rng))
    df_c = pd.DataFrame(rules.generate_type_c_sessions(n, rng))
    df_d = pd.DataFrame(rules.generate_type_d_sessions(n, rng))

    transformed_a = rules.transform_time_series_df(df_a, 'A')
    transformed_b = rules.transform_time_series_df(df_b, 'B')
    transformed_c = rules.transform_time_series_df(df_c, 'C')
    transformed_d = rules.transform_time_series_df(df_d, 'D')

    # add offset to 'Session' to make them unique inside combined df
    transformed_b['Session'] += transformed_a['Session'].max() + 1
    transformed_c['Session'] += transformed_b['Session'].max() + 1
    transformed_d['Session'] += transformed_c['Session'].max() + 1

    combined_df = pd.concat([transformed_a, transformed_b, transformed_c, transformed_d], ignore_index=True)

    combined_df.info()

    return combined_df

def draw(combined_df, n, folder):
    plt.close('all')

    set2_colors = sns.color_palette("Set2", 4) # can change the colours by changing the name (but I like the pastel colours so :p)
    types = ['A', 'B', 'C', 'D']
    color_mapping = dict(zip(types, set2_colors))
        
    merged_df = plot.get_merged(combined_df)

    bp = plot.draw_boxplot(merged_df, color_mapping)
    bp.savefig(os.path.join(folder, f"boxplot_{n}.png"))

    brp = plot.draw_barplots(merged_df, types, color_mapping)
    brp.savefig(os.path.join(folder, f"barplots_{n}.png"))

    sp_focused = plot.draw_stripplot(combined_df, 'TimeFocused')
    sp_focused.savefig(os.path.join(folder, f"stripplot_focused_{n}.png"))

    sp_diverted = plot.draw_stripplot(combined_df, 'TimeDiverted')
    sp_diverted.savefig(os.path.join(folder, f"stripplot_diverted_{n}.png"))

    sp_distracted = plot.draw_stripplot(combined_df, 'TimeDistracted')
    sp_distracted.savefig(os.path.join(folder, f"stripplot_distracted_{n}.png"))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates synthetic dataset using the manually defined business rules of player types."
    )
    parser.add_argument(
        "-n",
        required=True,
        type=int,
        help="number of sessions to generate for each player type (combined dataset will have n*4 sessions)"
    )
    parser.add_argument(
        "-seed",
        required=False,
        type=int,
        help="(optional) number to use as the seed for numpy random Generator"
    )
    parser.add_argument(
        "-plot",
        required=False,
        type=bool,
        default=False,
        help="(optional) set to True to make visualizations of the dataset"
    )
    args = parser.parse_args()

    n = args.n
    seed = args.seed
    showPlot = args.plot

    combined_df = generate_df(n, seed)

    current_date = datetime.now().strftime("%Y%m%d")
    folder = f"{current_date}_{n}"
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"synthetic_{n}.csv")
    combined_df.to_csv(file_path, index=False)

    print(f"Saved dataset of {n} sessions per type ({n*4} total) to {file_path}")


    if showPlot:
        draw(combined_df, n, folder)

    print(f"Saved visualizations to {folder}")

    print("done :D")

    



    