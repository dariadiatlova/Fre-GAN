import numpy as np
import pandas as pd
import re


def get_file_names(tsv_filepath: str):
    """
    Function takes the path the tsv file and returns the column with filenames, changed from .mp3 to .wav.
    :param tsv_filepath: str
    :return: ndarray
    """
    df = pd.read_csv(tsv_filepath, sep='\t')
    mp3_filenames = np.array(df.path)
    pattern = re.compile(".*(?=.mp3)")
    wav_filenames = [pattern.search(filename).group() + ".wav" for filename in mp3_filenames]
    return np.array(wav_filenames)
