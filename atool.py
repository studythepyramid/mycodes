
"""
How to add one new element to a list every n elements.

https://stackoverflow.com/questions/31040525/insert-element-in-python-list-after-every-nth-element?answertab=scoredesc#tab-top

## 1
>>> from itertools import chain

>>> n = 2
>>> ele = 'x'
>>> lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

>>> list(chain(*[lst[i:i+n] + [ele] if len(lst[i:i+n]) == n else lst[i:i+n] for i in xrange(0, len(lst), n)]))
[0, 1, 'x', 2, 3, 'x', 4, 5, 'x', 6, 7, 'x', 8, 9, 'x', 10]

## 2
>>> letters = ['a','b','c','d','e','f','g','h','i','j']
>>> new_list = []
>>> n = 3
>>> for start_index in range(0, len(letters), n):
...     new_list.extend(letters[start_index:start_index+n])
...     new_list.append('x')
...
>>> new_list.pop()
'x'
>>> new_list
['a', 'b', 'c', 'x', 'd', 'e', 'f', 'x', 'g', 'h', 'i', 'x', 'j']
"""


def addto_every_n(origList, n, foo='O'):
    i = n
    while i < len(origList):
        print(i, len(origList))

        origList.insert(i, 'x')
        i += (n+1)
    
    return origList


def new_with_every_n(origList, n, foo='O'):

    lst = []
    count = 0

    for a in origList :
        lst.append(a)
        count += 1

        if(count % n == 0):
            lst.append(foo)

    return lst


def printLongList(llist, cols, demark='\t'):
    newList = new_with_every_n(llist, cols, foo="\n")

    print(demark.join( str(e) for e in newList))
    pass


la = list('abcdefghijk')
n = 2


addto_every_n( la, 2)
new_with_every_n( la, 2)


##  textwrap, pprint

# historical_data.__dict__
attrHD = dir(historical_data)

from pprint import pprint
import textwrap

text = ",\t ".join(attrHD)
# wtext= textwrap.wrap(text, width=66)
wtext= textwrap.fill(text, width=66)

print(wtext)
# pprint(",  ".join(attrHD), width=80)
# print("\t".join(attrHD))


"""
cmd.Cmd().columnize can do good list

https://stackoverflow.com/questions/1524126/how-to-print-a-list-more-nicely
"""
import cmd
cli = cmd.Cmd()
cli.columnize(attrHD, displaywidth=100)

def columnize_list(alist):
    cli.columnize(alist)
    pass



