import pandas as pd
import logging
import os
import glob

log = logging.getLogger(__name__)


def read_audio_info_from_csv(csvfile):
    csvData = pd.read_csv(csvfile)
    print (csvData)
    log.debug("filename {} csvdata {}".format(csvfile,csvData.iloc[0, :]))

    return csvData

def write_audio_info_to_csv(filename,csvDatadict):
    csvData = pd.DataFrame(csvDatadict)
    csvData.to_csv(filename)

def append_audio_info_to_csv(filename,csvDatadict):
    csvData = pd.DataFrame(csvDatadict)
    csvData.to_csv(filename,mode='a', header=False)

def remove_files_in_dir(directory):
    files = glob.glob('./'+directory+'/**', recursive=True)
    for f in files:
        try:
                os.remove(f)
        except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

def create_dir_if_not_exists(directory):
    # Check whether the specified path exists or not
    isExist = os.path.exists(directory)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(directory)