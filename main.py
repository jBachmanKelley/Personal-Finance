# This is a python script to read, analyze, and display personal finance material from a Transactions CSV
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import array as arr
import math

"""
    Script Goals:
        - Read in my Transcript CSV Data
        - Analyze the different transactions 
        - Print Statistics"""
class Node:
    def __init__(self, value=None, index=-1):
        self.value = value
        self.index = index
        self.next = None


class SList:
    def __init__(self, head):
        self.head = Node(head)

    def insert(self, value):
        end = self.head
        while end.next != None:
            end = end.next
        end.next = Node(value, end.index + 1)

    def getValue(self, value, mode):
        temp = self.head
        while temp.value != value and temp.next is not None:
            temp = temp.next
        if temp.value == value and mode == 0:
            return temp.index
        elif temp.value == value and mode == 1:
            return temp.value
        else:
            self.insert(value)
            return self.getValue(value, mode)

    def __len__(self):
        end = self.head
        while end.next != None:
            end = end.next
        return end.index + 1


class Analyzer:
    def __init__(self):
        self.price = np.genfromtxt('Transactions.csv', delimiter=',', skip_header=1, usecols=(0))
        self.cat = np.genfromtxt('Transactions.csv', delimiter=',', skip_header=1, usecols=(1), dtype=str)
        self.date = np.genfromtxt('Transactions.csv', delimiter=',', skip_header=1, usecols=(2), dtype=str)
        self.clean_data = np.zeros((len(self.price), 3))
        self.classes = SList('Head')
        self.transpose()
        self.stats()
        self.plot()

    def transpose(self):
        # Alter the Classes to  Integer values
        for i in range(self.clean_data.shape[0]):
            self.clean_data[i][0] = -1 * math.floor(self.price[i] * 100)/ 100
            self.clean_data[i][1] = self.convert_to_class(self.cat[i])

    def convert_to_class(self, value):
        return self.classes.getValue(value, 0)

    def __main__(self):
        return

    def stats(self):
        # Sum By Category
        sums = [0.0] * len(self.classes)
        for x in self.clean_data:
            sums[int(x[1])] += x[0]

        # Print Values
        i = 0
        for s in sums:
            print(f'Class {i:2} Sum: ${s:8.2f}')
            i += 1

    def plot(self):
        x = self.date
        y = self.clean_data[:, 0]
        plt.scatter(x, y, c=self.clean_data[:, 1])
        plt.title('Transactions By Date')
        plt.xlabel('Date: MM/DD/YYY')
        plt.xticks(rotation=295, fontsize=6)
        plt.ylabel('Expense')
        plt.set_cmap('tab10')
        plt.show()



ob1 = Analyzer()
