#!/usr/bin/env python

# STEP 1 START
import sys
import os
import importlib
import pathlib
import numpy as np
import zipfile
from io import TextIOWrapper

# expected to be /Algorithms/ScalableAlgorithms/PythonScripts
currentDir = pathlib.Path(__file__).absolute()

parentDir = currentDir.parent.parent.parent
if parentDir not in set(sys.path):
    sys.path.append(str(parentDir))
    print(sys.path)
from ModuleDir import functions

importlib.reload(functions)
dir(functions)
# STEP 1 END

# STEP 2 EDITED IN functions.py

# STEP 3 START
fieldDict = functions.fieldDictBuild()
# STEP 3 END

# STEP 4 START
path = f"{parentDir}/Data/"
fileList = os.listdir(path)
print(parentDir)
print(path)
print(fileList)

# STEP 11 START
q = 4 # Number of features
A = np.zeros(shape=(q, q)) # 4 x 4 Matrix of 0s
z = np.matrix(np.zeros(shape=(q, 1))) # 4 x 1 Vector of 0s
sumSqrs = 0 # Sum of squares
n = 0 # Number of records
# STEP 11 END
for filename in fileList:
    try:
        shortYear = int(filename[6:8])
        if shortYear < 11:
            1 / 0
        year = 2000 + shortYear

        file = path + filename
        print(filename)
        fields = fieldDict[shortYear]

        # STEP 5 START
        fEduc = fields["education"]
        sInc, eInc = fields["income"]
        sBMI, eBMI = fields["bmi"]
        fGH = fields["genhlth"]
        # STEP 5 END

        # STEP 6 START
        with zipfile.ZipFile(file) as zf:
            zipFileList = zf.namelist()
            assert len(zipFileList) == 1
            with zf.open(zipFileList[0]) as f:
                reader = TextIOWrapper(f, encoding="latin-1")
                for record in reader:
                    education = functions.getEducation(record[fEduc - 1])
                    income = functions.getIncome(record[sInc - 1 : eInc])
                    bmi = functions.convertBMI(record[sBMI - 1 : eBMI], shortYear)
                    # STEP 6 END

                    # STEP 7 EDITED IN functions.py
                    # STEP 8 EDITED IN functions.py

                    # STEP 9 START
                    y = functions.getHlth(record[fGH - 1])
                    # STEP 9 END

                    # STEP 10 START
                    try:
                        if education < 9 and income < 9 and 0 < bmi < 99 and y != -1:
                            x = np.matrix([1, income, education, bmi]).T # Vector of 4 x 1
                        else:
                            1 / 0

                        # STEP 12 START
                        A += x * x.T # 4 x 4 matrix for X^T X
                        z += x * y # 4 x 1 vector for X^T Y
                        sumSqrs += y**2 # Y^T Y
                        n += 1 # row counter
                        # STEP 12 END

                        # STEP 13 START
                        if n % 10_000 == 0 and n != 0:
                            b = np.linalg.inv(A) * z # Slope coefficient β = (X^T X)^(-1) X^T Y
                            print("\t".join(str(float(bi[0, 0])) for bi in b))
                        # STEP 13 END
                    except ZeroDivisionError:
                        pass
                    # STEP 10 END

    except (ValueError, ZeroDivisionError):
        pass
# STEP 4 END

# STEP 14 START
ybar = z[0, 0] / n
varEst = (sumSqrs - n * ybar**2) / (n - 1) # SST / (N-1)
# STEP 14 END

# STEP 15 START
rAdj = 1 - ((sumSqrs - b.T * z) / (n - q)) / varEst # 1 - (SSE / (n-q)/ SST / (n-1))
# STEP 15 END

# STEP 16 START
bList = [str(round(float(bi[0, 0]), 3)) for bi in b]
print('\t'.join(('n', 'b0', 'inc', 'edu', 'bmi', 'r^2_adj')))
print(n, "\t".join(bList), round(float(rAdj[0, 0]), 3), sep="\t")
# STEP 16 END