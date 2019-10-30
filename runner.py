"""
Author: Pranav Srivastava
file: runner.py
"""

from subprocess import call

# values for the run can be set below
list_files = ['Part1.py']
k = [1]

#EXAMPLE VALUES
#list_files = ['Part1.py']
#k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

remark = "scaled"

for fileName in list_files:
    for valuK in k:
        call(["python", "Part3_b.py", str(valuK), str(fileName), str(remark)])