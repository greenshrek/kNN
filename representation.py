"""
Author: Pranav Srivastava
file: representation.py
This file is used to plots the results in graph.
Input to this file is test_results.txt 
"""

from matplotlib import pyplot as plt
import ast 
import statistics



def drawgraph():
    fnam = "test_results.txt"
    problInst = "Part2_b.py"
    category = "scaled"


    file = open(fnam, 'r')
    data = []
    x = []
    y = []
    for line in file:
        print(type(line))
        data = line
        data = ast.literal_eval(data)
        if data[2] == problInst and data[3] == category:
            x.append(data[0])
            y.append(data[1])
    print(x)
    print(y)
    title = 'problem instance: '+ str(problInst)+'\n'+  'category: '+ str(category)
    plt.title(title)
    plt.plot(x,y)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

drawgraph()