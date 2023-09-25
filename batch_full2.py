import os
from full2 import fullpipeline


def getInfoTxt(txtfile):
    inputfile = open(txtfile)

    for line in inputfile:
        if "bscan width" in line.lower():
            width = inputfile.readline()
            degline = width.split(' ')
            if degline.__len__() > 0:
                inputfile.close()
                return degline[0]

    inputfile.close()


# ROOT DIRECTORY TO SCAN FOR FOLDERS WITH INFO/README.TXT FILES FROM XML-SORTING SCRIPT
rootDir = "F:/IMMAGINI MNTSN/"

print('going through subfolders of ' + rootDir)
for root, subfolders, files in os.walk(rootDir):
    for folder in subfolders:
        if os.path.isfile(os.path.join(root, folder, "README.txt")):

            deg = getInfoTxt(os.path.join(root, folder, "README.txt"))
            if type(deg) == str:
                print(folder + ' has a README.txt and the width of bscans is found. Processing..')
                print('calling full2.fullpipeline(' + os.path.join(root, folder) + ', ' + deg + ')')
                try:
                    fullpipeline(os.path.join(root, folder), int(deg))
                except NameError:
                    print('there was a problem processing the folder ' + os.path.join(root, folder))
            else:
                print(folder + ' has a README.txt but the width of bscans was not found in it. Cannot process.')

        elif os.path.isfile(os.path.join(root, folder, "INFO.txt")):

            deg = getInfoTxt(os.path.join(os.path.join(root, folder, "INFO.txt")))
            if type(deg) == str:
                print(folder + ' has a INFO.txt and the width of bscans is found. Processing..')
                print('calling full2.fullpipeline(' + os.path.join(root, folder) + ', ' + deg + ')')
                try:
                    fullpipeline(os.path.join(root, folder), int(deg))
                except NameError:
                    print('there was a problem processing the folder ' + os.path.join(root, folder) )

            else:
                print(folder + ' has a INFO.txt but the width of bscans was not found in it. Cannot process.')
