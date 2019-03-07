"""This function cleans the data from original data productivity.csv.
   It is useful for further analysis of data.
"""

import pandas as pd
import numpy as np
from bld.project_paths import project_paths_join as ppj


def save_data(sample):
    """
    Save cleaned data as .csv file.
    """
    sample.to_csv(ppj("OUT_DATA", "productivity_clean.csv"), sep=",")



if __name__ == "__main__":
    data = pd.read_csv(ppj("IN_DATA", "productivity.csv"))
    data['PUB_CAP'] = data['P_CAP'] + data['HWY'] + data['WATER'] + data['UTIL']
    data['LNPUB_CAP'] = np.log(data.PUB_CAP)
    data['LNPC'] = np.log(data.PC)
    data['LNEMP'] = np.log(data.EMP)
    data['LNGSP'] = np.log(data.GSP)
    save_data(data)

data