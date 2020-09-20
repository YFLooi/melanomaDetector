#Ref:
# https://stackabuse.com/reading-and-writing-xml-files-in-python/
# https://docs.python.org/2/library/xml.etree.elementtree.html

from xml.dom import minidom
import xml.etree.ElementTree as ET
from pathlib import Path
import glob
import os #package that allows direct manipulation of folder's files
from os import listdir
from os.path import isfile, isdir, join


#source folder
sourceFolder = 'C:\\Users\\deSni\\.deeplearning4j\\data\\melanomaChallenge\\labelled_2Classes\\train\\Annotations'
#destination folder
destinationFolder = 'C:\\Users\deSni\\.deeplearning4j\\data\\melanomaChallenge\\labelled_2Classes\\train\\fixedAnnotations'

#Check to see if sourceFolder and destinationFolder paths are correct
if os.path.isdir(sourceFolder):
    print ("sourceFolder exists\n")
else:
    print ("sourceFolder does not exist\n")

if os.path.isdir(destinationFolder):
    print ("destinationFolder exists\n")
else:
    print ("destinationFolder does not exist\n")


#Grabs all files ending in .xml in sourcePath
#All files sorted by name, as in Windows. Hence, 1st entry is in top left of folder
pathsToTargetFiles = glob.glob(f'{sourceFolder}\\*.xml')

iterCount = 0
# loop though each file path in pathsToTargetFiles
for path in pathsToTargetFiles:
    print(f'Current iteration number: {iterCount}')
    #Ref: https://stackoverflow.com/questions/678236/how-to-get-the-filename-without-the-extension-from-a-path-in-python
    fileName = os.path.basename(pathsToTargetFiles[iterCount])
    print(f'Name of current xml file: {fileName}')

    tree = ET.parse(f'{pathsToTargetFiles[iterCount]}')
    root = tree.getroot()

    for name in root.iter('name'):    
        currentText = name.text
        print(f'Current text in <name>: {currentText}\n\n')

        if currentText != "melanoma":
            #Specify new text here
            newText = 'not_melanoma'
            name.text = str(newText)

    #Write resulting xml file to destinationFolder
    tree.write(f'{destinationFolder}\\{fileName}')

    #Update counter at end of loop
    iterCount += 1
    
    #Use to pause at first iteration, if testing
    #break

