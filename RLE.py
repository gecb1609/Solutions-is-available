# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 21:36:22 2020

@author: subhabrata.c
"""

def encode(input_string):
    count = 1
    prev = ''
    lst = []
    for character in input_string:
        if character != prev:
            if prev:
                entry = (prev,count)
                print('Entry is :',entry)
                lst.append(entry)
                print("lst is :",lst)
                #print lst
            count = 1
            prev = character
            print("prev is :",prev)
            print("charactor is :", character)
            
        else:
            count += 1
            print("count is :",count)
    else:
        try:
            entry = (character,count)
            lst.append(entry)
            return (lst, 0)
        except Exception as e:
            print("Exception encountered {e}".format(e=e)) 
            return (e, 1)

def decode(lst):
    q = ""
    for character, count in lst:
        q += character * count
    return q
 
#Method call
value = encode("aaaaahhhhhhmmmmmmmuiiiiiiiaaaaaa")
if value[1] == 0:
    print("Encoded value is {}".format(value[0]))
    decode(value[0])