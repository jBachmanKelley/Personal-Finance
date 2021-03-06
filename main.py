# This is a python script to read, analyze, and display personal finance material from a Transactions CSV
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
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
        self.color = None
        self.next = None
        if index != -1:
            self.color = plt.cm.get_cmap('viridis').colors[index]



class SList:
    def __init__(self, head):
        self.head = Node(head)

    def insert(self, value):
        end = self.head
        while end.next is not None:
            end = end.next
        end.next = Node(value, end.index + 1)

    def get_value(self, value):
        temp = self.head
        while temp.value != value and temp.next is not None:
            temp = temp.next
        if temp.value == value:
            return temp.index
        else:
            self.insert(value)
            return self.get_value(value)

    def get_label(self, index):
        temp = self.head
        while temp.index != index and temp.next is not None:
            temp = temp.next
        if temp.index == index:
            return temp.value
        else:
            raise ValueError

    def __len__(self):
        end = self.head
        while end.next is not None:
            end = end.next
        return end.index + 1

    def __iter__(self):
        self.iterator = 1
        return self

    def __next__(self):
        temp = self.head
        for i in range(self.iterator):
            temp = temp.next
        if temp is None:
            raise StopIteration
        else:
            self.iterator += 1
            return temp


class Analyzer:
    # noinspection PyTypeChecker
    def __init__(self):

        self.price = np.genfromtxt('Transactions.csv', delimiter=',', skip_header=1, usecols=0)
        self.cat = np.genfromtxt('Transactions.csv', delimiter=',', skip_header=1, usecols=1, dtype=str)
        self.date = np.genfromtxt('Transactions.csv', delimiter=',', skip_header=1, usecols=2, dtype=str)
        self.clean_data = np.zeros((len(self.price), 3))
        self.classes = SList('Head')
        self.stat_str = ''  # Make stat_str a dict for each class
        self.transpose()
        self.stats()
        self.plot()

    def transpose(self):
        # Alter the Classes to  Integer values
        for i in range(self.clean_data.shape[0]):
            self.clean_data[i][0] = -1 * math.floor(self.price[i] * 100) / 100
            self.clean_data[i][1] = self.classes.get_value(self.cat[i])

    def __main__(self):
        return

    def stats(self):
        # Sum By Category
        stats = [[0] * 3 for _ in  range(len(self.classes))]  # [size, sum, mean]
        for x in self.clean_data:
            x_class = int(x[1])
            stats[x_class][0] += 1
            stats[x_class][1] += x[0]
            stats[x_class][2] = stats[x_class][0] / stats[x_class][1]

        # Print Values
        i = 0
        for s in stats:
            self.stat_str += f'Class: {self.classes.get_label(i):2}\n    Size: {s[0]}\n    Sum: ${s[1]:8.2f}\n'
            i += 1

    def plot(self):
        x = self.date
        y = self.clean_data[:, 0]
        legend_classes = []
        for c in self.classes:
            temp = mpatches.Patch(color=c.color, label=c.value)
            legend_classes.append(temp)
        plt.legend(handles=legend_classes)
        plt.scatter(x, y, c=self.clean_data[:, 1], cmap='viridis')
        plt.suptitle('Transactions By Date')
        plt.title('Updated: 12/07/2020', fontsize=10)  # Put a time keeper variable to track when last updated
        plt.xlabel('Date: MM/DD/YYY')
        plt.xticks(rotation=295, fontsize=6)
        self.update_stat_box()
        plt.ylabel('Expense')
        plt.show()

    def update_stat_box(self):
        plt.text(55, 200, self.stat_str, bbox={'facecolor': 'green', 'alpha': 0.4, 'pad': 15},
                 verticalalignment='top', horizontalalignment='left')


ob1 = Analyzer()

"""
    Notes For Improvement:
    - Use Text.contains(self, mouseevent) to create buttons and such 
    - Make a Dynamic statistics box 
"""