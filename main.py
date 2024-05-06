import datetime
import sys

from Covid_data import data_cleaning, data_representation  # This is a sample Python script.
import pandas as pd

# data = pd.read_csv('data/covid_19_india.csv')
# data_cleaning(data)
# data_representation(data)
#
# # Press the green button in the gutter to run the script.
# # if __name__ == '__main__':
# #     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
#
# remove the duplicates
# remove the null values
# keep only selected columns
# Data visualize

# print('Twinkle')  # print
# print(sys.version)    # python version
# print(datetime.datetime.now())  # print date time.
# now = datetime.datetime.now()
# print(now.strftime("%Y-%m-%d %H:%M:%S")) # print date time in specific format
# area of circle
# from math import pi
# r= 1.1
# area = pi*r*r
# print(area)
#
# #Input function
# fname = input('enter your first name: ')
# lname = input('enter your last name: ')
# print('hello ' +lname+' '+fname)

# #prints the filename extension
# file_name = input('enter filename')
# print(file_name.split('.')[-1])

#prints first and last colors of the list
# color_list = ["Red","Green","White" ,"Black"]
# print(color_list[0], color_list[-1])

#prints date in order
# tuple_qw = (12,5,1998)
# print('%i/%i/%i' % tuple_qw)

#Write a Python program that accepts an integer (n) and computes the value of n+nn+nnn
# n = int(input("enter an integer"))
# n1 = int('%s%s' % (n,n))
# n2 = int('%s%s%s' % (n,n,n))
# print(n+n1+n2)

# Write a Python program that prints the calendar for a given month and year.
# import calendar
# print(calendar.month(int(input('enter the year')),int(input('enter the month'))))

# Write a Python program to get the volume of a sphere with radius six.
# from math import pi
# r = 6
# volume = 4/3*(pi*(r*r*r))
# print(volume)

# Write a Python program to calculate the difference between a given number and 17. If the number is greater than 17, return twice the absolute difference.
# def difference(n):
#     if n >= 17:
#         return n - 17
#     else:
#         return abs((n-17)*2)
# print(difference(int(input('enter value'))))

# Write a Python program to get a newly-generated string from a given string where "Is" has been added to the front. Return the string unchanged if the given string already begins with "Is".
# def string_add(s):
#     if len(s)>=2 and s[:2] == 'Is':
#         return s
#     else:
#         return 'Is'+s
# print(string_add(input('enter value string')))