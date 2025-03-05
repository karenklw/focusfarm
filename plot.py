import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_merged(combined_df):
    sequence_lengths = combined_df.groupby('Session').size().reset_index(name='SequenceLength')
    merged_df = pd.merge(sequence_lengths, combined_df[['Session', 'Type']], on='Session', how='left').drop_duplicates()

    return merged_df


def draw_boxplot(merged_df, color_mapping):
    plt.close('all')

    plt.figure(figsize=(10, 6))
    bp = merged_df.boxplot(column='SequenceLength', by='Type', patch_artist=True, medianprops=dict(color='black', linewidth=1.5))

    sorted_types = sorted(merged_df['Type'].unique())

    box_patches = bp.findobj(plt.matplotlib.patches.PathPatch)

    for patch, typ in zip(box_patches, sorted_types):
        patch.set_facecolor(color_mapping[typ])

    yticks = bp.get_yticks() 
    bp.set_yticks(yticks)
    bp.set_yticklabels([int(y * 15) for y in yticks]) # display in 15-min intervals

    plt.title('')
    plt.suptitle('')
    plt.xlabel('Player Type')
    plt.ylabel('Session Length (minutes)')

    return plt


def draw_barplots(merged_df, types, color_mapping):
    plt.close('all')

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for i, typ in enumerate(types):
        ax = axes[i]
        type_data = merged_df[merged_df['Type'] == typ]
        
        sequence_counts = type_data['SequenceLength'].value_counts().sort_index()

        sns.barplot(x=sequence_counts.index, y=sequence_counts.values, color=color_mapping[typ], ax=ax)

        ax.set_title(f'Counts of Session Lengths for Type {typ}')
        ax.set_xlabel('Session Length (minutes)')
        ax.set_ylabel('Count')

        ax.set_xticks(range(0, 17))
        ax.set_xticklabels([(x + 1) * 15 for x in ax.get_xticks()])
        ax.set_ylim(0, 60) # FIXME: find dynamic way to draw y-axis, but 60 is ok for n=200
        ax.set_yticks(range(0, 60, 5))

    plt.tight_layout()  

    return plt


def draw_stripplot(combined_df, y_type):
    plt.close('all')
    
    total_time = combined_df.groupby('Session')[y_type].sum().reset_index(name=f'Total{y_type}')
    merged = pd.merge(total_time, combined_df[['Session', 'Type']], on='Session', how='left').drop_duplicates()

    plt.figure(figsize=(10, 6))
    sns.stripplot(x='Type', y=f'Total{y_type}', data=merged, jitter=True, alpha=0.7, palette='Set2', hue='Type')

    plt.ylim(0, 15000)
    # plt.title(f'Distribution of Total {y_type} for Each Player Type')
    plt.title('')
    plt.xlabel('Player Type')
    plt.ylabel(f'Total {y_type} (seconds)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    return plt
