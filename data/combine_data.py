import os
import json
import numpy as np
import pandas as pd
import mne


"""
This file contains functions to combine the key and EEG data into a single DataFrame.
It also contains a function to convert the DataFrame to a FIF file to be used with MNE-Python.
"""



def create_fif(data_df: pd.DataFrame) -> mne.io.RawArray:
    """
    Converts a DataFrame to a FIF file. The DataFrame should contain EEG data and stimulus information.
    
    Arguments:
        data_df (pd.DataFrame): DataFrame containing EEG data and stimulus information.
    
    Returns:
        mne.io.RawArray: MNE RawArray object containing the EEG data and stimulus information.
    """
    assert 'stimulus' in data_df.columns, "DataFrame must contain 'stimulus' column."
    assert 'sys_time' in data_df.columns, "DataFrame must contain 'sys_time' column."

    # Unicorn hybrid black properites
    SFREQ = 250
    CH_NAMES = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

    channel_df_names = [f'ch{i}' for i in range(1, 9)]  # ch1, ..., ch8
    channel_data = data_df[channel_df_names].values.T  # shape (n_channels, n_samples)
    print(channel_data.dtype)
    channel_data *= 1e-6  # Convert V to µV

    times = data_df['sys_time'].values  # timestamps in seconds
    times = times - times[0]  # normalize to start at 0

    # Create an MNE representation
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
    info.times = times
    raw = mne.io.RawArray(channel_data, info)

    # Create MNE events (event format [sample, _, event_id])
    stimuli = data_df['stimulus'].values
    events = [[i, 0, stim] for i, stim in enumerate(stimuli) if stim != 0]  
    events = np.array(events)

    # Create a new Raw object for the stimulus channel
    stim_channel = np.zeros((1, raw.n_times))
    for event in events:
        stim_channel[0, event[0]] = event[2]
    stim_info = mne.create_info(['stimulus'], sfreq=SFREQ, ch_types=['stim'])
    stim_channel = mne.io.RawArray(stim_channel, stim_info)

    # Add the stimulus channel to the channel data
    raw.add_channels([stim_channel], force_update_info=False)

    # Set the montage to standard 10-20 system for better visualization
    raw.set_montage('standard_1020')

    return raw


def temporal_alignment(key_data: pd.DataFrame, eeg_data: pd.DataFrame, event_map: dict, delay: float=0.0) -> pd.DataFrame:
    """
    Combine key and EEG data into a single DataFrame by aligning the timestamps.

    Arguments:
        key_data (pd.DataFrame): DataFrame containing key data with 'timestamp' and 'key' columns.
        eeg_data (pd.DataFrame): DataFrame containing EEG data with 'sys_time' column.
        event_map (dict): Dictionary mapping keys to event IDs.
        delay (float): Delay in seconds to account for the latency of the EEG system.
    
    Returns:
        pd.DataFrame: Merged DataFrame containing EEG data and stimulus information.
    """
    assert -0.3 <= delay <= 0.0, "A useful delay must be between 0 and -300ms"
    assert 'key' in key_data.columns, "Key data must contain 'key' column."
    assert 'ch1' in eeg_data.columns, "EEG data must contain the eeg sample information."

    # Create a new column 'stimulus' initialized with zeros
    merged_data = eeg_data.copy()
    merged_data['stimulus'] = 0

    # get the timestamps of the EEG data
    eeg_system_time = eeg_data['sys_time'].values

    # filter the key data to only include keys that are in the event_map
    key_data = key_data[key_data['key'].isin(event_map.keys())]

    # Iterate over key events and align with EEG timestamps
    for _, row in key_data.iterrows():

        # get the key stroke event id
        event_id = event_map[row['key']]

        # Convert the key timestamp to the corresponding EEG timestamp
        key_time = row['timestamp'] + delay

        # Find the closest EEG timestamp
        closest_idx = (np.abs(eeg_system_time - key_time)).argmin()

        # set the stimulus channel to the event id at the closest index
        merged_data.loc[closest_idx, 'stimulus'] = event_id
    
    return merged_data


def create_datasets(key_data_dir: str, eeg_data_dir: str, csv_data_dir: str, fif_data_dir: str, event_map: dict, delay: float=0.0):
    """
    Create datasets from key and EEG data by merging them and saving as CSV and FIF files.

    Arguments:
        key_data_dir (str): Directory containing key data.
        eeg_data_dir (str): Directory containing EEG data.
        csv_data_dir (str): Directory to save the merged CSV files.
        fif_data_dir (str): Directory to save the FIF files.
        event_map (dict): Dictionary mapping keys to event IDs.
        delay (float): Delay in seconds to account for the latency of the EEG system.
    """
    assert os.path.exists(key_data_dir), f"Key log directory {key_data_dir} does not exist."
    assert os.path.exists(eeg_data_dir), f"EEG log directory {eeg_data_dir} does not exist."
    
    subjects = sorted(os.listdir(key_data_dir))
    print(f"Subjects found: {subjects}")

    for sub in subjects:

        key_sub_dir = os.path.join(key_data_dir, sub)
        eeg_sub_dir = os.path.join(eeg_data_dir, sub)
        fif_sub_dir = os.path.join(fif_data_dir, sub)
        csv_sub_dir = os.path.join(csv_data_dir, sub)
        os.makedirs(fif_sub_dir, exist_ok=True)
        os.makedirs(csv_sub_dir, exist_ok=True)

        key_files = sorted([f for f in os.listdir(key_sub_dir) if f.endswith('.csv')])
        eeg_files = sorted([f for f in os.listdir(eeg_sub_dir) if f.endswith('.csv')])
        assert len(key_files) == len(eeg_files), f"Number of key files and EEG files does not match for subject {sub}."

        for key_file, eeg_file in zip(key_files, eeg_files):
            
            # load the data
            key_data = pd.read_csv(os.path.join(key_sub_dir, key_file))
            eeg_data = pd.read_csv(os.path.join(eeg_sub_dir, eeg_file))
            
            # merge the data
            merged_data = temporal_alignment(key_data, eeg_data, event_map, delay=delay)

            # save the merged data as csv file
            merged_data.to_csv(os.path.join(csv_sub_dir, key_file), index=False)

            # convert the merged data to fif format
            fif_data = create_fif(merged_data)

            # save the fif data
            fif_path = os.path.join(fif_sub_dir, key_file.replace('.csv', '.fif'))
            fif_data.save(fif_path, overwrite=True)


def main():
    
    DELAY = 0.0  # delay in seconds to account for the latency of the EEG system

    # load the event ID mapping (key-stroke to id)
    with open('data/event_ids1.json') as f:
        event_map = json.load(f)

    # create the datasets using the filtered data
    create_datasets(
        "data/key_recordings/",
        "data/eeg_recordings/",
        "data/datasets/csv/",
        "data/datasets/fif/",
        event_map,
        delay=DELAY
    )

    # create the dataset using the raw data
    create_datasets(
        "data/key_recordings/",
        "data/eeg_raw_recordings/",
        "data/datasets_raw/csv/",
        "data/datasets_raw/fif/",
        event_map,
        delay=DELAY
    )

if __name__ == "__main__":
    main()


