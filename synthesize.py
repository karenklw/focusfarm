import argparse
import numpy as np
import os
import pandas as pd
import sdv as sdv
from sdv.metadata import SingleTableMetadata
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer
import matplotlib.pyplot as plt
import seaborn as sns

import plot


def make_meta(df):
    md = Metadata.detect_from_dataframe(df)
    md.update_column(column_name='Session', sdtype='id')
    md.set_sequence_index(column_name='TimeStep')
    md.set_sequence_key(column_name='Session')

    md.validate()
    return md


def adjust(sd, rng):
    sd = sd.apply(adjust_times, axis=1)
    sd = sd.apply(adjust_diversion, axis=1, args=(rng,))

    return sd

def adjust_times(row):
    """
    Ensure TimeFocused + TimeDiverted + TimeDistracted == 900 for each timestep.
    """
    t1, t2, t3 = row['TimeFocused'], row['TimeDiverted'], row['TimeDistracted']
    
    if 900 - t1 - t2 != t3:
        scale_factor = 900 / (t1 + t2 + t3)
        row['TimeFocused'] = int(round(t1 * scale_factor))
        row['TimeDiverted'] = int(round(t2 * scale_factor))
        row['TimeDistracted'] = 900 - row['TimeFocused'] - row['TimeDiverted']
    
    return row

def adjust_diversion(row, rng: np.random.Generator):
    """
    Ensure FirstDiversion follows logical constraints for each timestep.
    """
    diversion = row['FirstDiversion']
    friends = row['AvgFriendsPerMin']
    num = row['NumDiverted']
    time = row['TimeDiverted']

    # if TimeDiverted == 0 --> NumDiverted == 0 and FirstDiversion == 'none'
    if time == 0:
        row['NumDiverted'] = 0
        row['FirstDiversion'] = 'none'
        return row

    # if TimeDiverted > 0 --> NumDiverted > 0 and FirstDiversion != 'none'
    # FirstDiversion can be 'friends'; adjust probs same as rules.py
    if time > 0 and num == 0:
        if row['Type'] == 'A':
            row['NumDiverted'] = 1
            if row['FirstDiversion'] == 'none':
                row['FirstDiversion'] = rng.choice(["timer", "progress", "friends", "customizations"], p=[0.5, 0.3, 0.15, 0.05])
                diversion = row['FirstDiversion'] # update diversion for next check

        elif row['Type'] == 'B':
            row['NumDiverted'] = rng.integers(1, max(2, time // 300 + 1))
            if row['FirstDiversion'] == 'none':
                row['FirstDiversion'] = rng.choice(["timer", "progress", "friends", "customizations"], p=[0.25, 0.25, 0.4, 0.1])
                diversion = row['FirstDiversion'] # ''
        
        elif row['Type'] == 'C':
            row['NumDiverted'] = rng.integers(1, 4)
            if row['FirstDiversion'] == 'none':
                row['FirstDiversion'] = rng.choice(["timer", "progress", "friends", "customizations"], p=[0.1, 0.1, 0.4, 0.4])
                diversion = row['FirstDiversion'] # ''
        
        else:
            row['NumDiverted'] = rng.integers(1, 6)
            if row['FirstDiversion'] == 'none':
                row['FirstDiversion'] = rng.choice(["timer", "progress", "friends", "customizations"], p=[0.25, 0.25, 0.25, 0.25])
                diversion = row['FirstDiversion'] # ''
    
    # if AvgFriendsPerMin == 0.0 --> FirstDiversion != 'friends'
    # FirstDiversion can't be 'friends'; adjust probs same as rules.py
    if friends == 0.0 and diversion == 'friends':
        if row['Type'] == 'A':
            row['FirstDiversion'] = rng.choice(["timer", "progress", "customizations"], p=[0.55, 0.4, 0.05])
        
        elif row['Type'] == 'B':
            row['FirstDiversion'] = rng.choice(["timer", "progress", "customizations"], p=[0.4, 0.5, 0.1])
        
        elif row['Type'] == 'D':
            row['FirstDiversion'] = rng.choice(["timer", "progress", "customizations"], p=[0.33, 0.34, 0.33])
        
        else: # edge case: type C shouldn't have 0.0 friends unless synthesizer models session that starts with 1 friend, and that 1 friend leaves
            row['FirstDiversion'] = rng.choice(["timer", "progress", "customizations"], p=[0.1, 0.1, 0.8])
    
    return row


def balance(df, k: int, rng: np.random.Generator):
    ret = []
    num_avail = k
    
    for type_ in df['Type'].unique():
        type_df = df[df['Type'] == type_]

        unique_sessions = type_df['Session'].drop_duplicates()
        num_avail = min(num_avail, len(unique_sessions)) # sometimes PAR produces a really unbalanced dataset

    for type_ in df['Type'].unique():
        type_df = df[df['Type'] == type_]

        sampled_sessions = type_df['Session'].drop_duplicates().sample(n=num_avail, random_state=rng)
        sampled_df = type_df[type_df['Session'].isin(sampled_sessions)]
        ret.append(sampled_df)
    
    balanced_df = pd.concat(ret, ignore_index=True)
    
    return balanced_df


def draw(df, n, folder):
    plt.close('all')

    set2_colors = sns.color_palette("Set2", 4) # can change the colours by changing the name (but I like the pastel colours so :p)
    types = ['A', 'B', 'C', 'D']
    color_mapping = dict(zip(types, set2_colors))
        
    merged_df = plot.get_merged(df)

    bp = plot.draw_boxplot(merged_df, color_mapping)
    bp.savefig(os.path.join(folder, f"PAR_boxplot_{n}.png"))

    brp = plot.draw_barplots(merged_df, types, color_mapping)
    brp.savefig(os.path.join(folder, f"PAR_barplots_{n}.png"))

    sp_focused = plot.draw_stripplot(df, 'TimeFocused')
    sp_focused.savefig(os.path.join(folder, f"PAR_stripplot_focused_{n}.png"))

    sp_diverted = plot.draw_stripplot(df, 'TimeDiverted')
    sp_diverted.savefig(os.path.join(folder, f"PAR_stripplot_diverted_{n}.png"))

    sp_distracted = plot.draw_stripplot(df, 'TimeDistracted')
    sp_distracted.savefig(os.path.join(folder, f"PAR_stripplot_distracted_{n}.png"))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize data using PAR and generate visualizations."
    )
    parser.add_argument(
        "-f",
        required=True,
        type=str,
        help="filepath to the input seed sample dataset"
    )
    parser.add_argument(
        "-n",
        required=True,
        type=int,
        help="number of sequences to sample from synthesizer"
    )
    parser.add_argument(
        "-p",
        required=False,
        type=str,
        default=None,
        help="filepath to saved synthesizer"
    )
    parser.add_argument(
        "-seed",
        required=False,
        type=int,
        help="(optional) number to use as the seed for np.random.Generator and state for random sampling when creating balanced dataset"
    )
    parser.add_argument(
        "-plot",
        required=False,
        type=bool,
        default=False,
        help="(optional) set to True to make visualizations of the dataset"
    )
    args = parser.parse_args()

    f = args.f
    n = args.n
    pkl = args.p
    seed = args.seed
    showPlot = args.plot

    
    sample = pd.read_csv(f)
    sample_md = make_meta(sample)

    if pkl is None:
        synth = PARSynthesizer(
            sample_md,
            enforce_min_max_values=True, # default = True
            enforce_rounding=True, # default = True
            context_columns=['Type'],
            epochs=128, # default = 128
            cuda=True, # default = True (only works if CUDA-capable GPU available)
            verbose=True
        )
        synth.fit(sample)

        synth.save(f'{os.path.dirname(f)}/synthesizer.pkl')
        print(f"Saved synthesizer to {os.path.dirname(f)}/synthesizer.pkl")
    
    # if p option is set, load synthesizer
    else:
        synth = PARSynthesizer.load(filepath=pkl)
        synth.fit(sample) # reproducible only if using the same synthesizer fitted on the same dataset

        synth.save(f'{os.path.dirname(f)}/synthesizer.pkl')
        print(f"Saved synthesizer to {os.path.dirname(f)}/synthesizer.pkl")
    
    synthetic_data = synth.sample( 
        num_sequences=n*4, # try to ensure PAR produces enough sequences to sample n from (doesn't gaurantee, see *balance*)
        sequence_length=None # default = None (auto determine sequence lengths)
    )

    rng = np.random.default_rng(seed)
    adjusted = adjust(synthetic_data, rng)
    balanced = balance(adjusted, n, rng)

    balanced.to_csv(f'{os.path.dirname(f)}/PAR_data_{n}.csv', index=False)
    print(f"Saved adjusted+balanced PAR-synthesized dataset to {os.path.dirname(f)}/PAR_data_{n}.csv")

    if showPlot:
        draw(balanced, n, os.path.dirname(f))
        print(f"Saved visualizations to {os.path.dirname(f)}/")

    print("done :D")
    