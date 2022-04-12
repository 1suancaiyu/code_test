#!/usr/bin/python
# -*- coding: UTF-8 -*-

fo = open("scnu.txt", 'r', encoding='UTF-8')
where = input("What are you trying to find ?")
for line, n in zip(fo.readlines(), range(10)):
    print(line)
    if where in line:
        print("It's here >>>>> ",line)
fo.close()

