# This is a handy script that enables you to parse through a directory to look 
# for files of specific format and then copy them to a specific folder.
# In this case we are looking for and copying epub files

import glob, os, shutil

sourceFolder="/path/to/source"
destinationFolder="/path/to/destination"


files = glob.iglob(os.path.join(sourceFolder, "*.epub"))
for file in files:
    if os.path.isfile(file):
        shutil.copy2(file, destinationFolder)
