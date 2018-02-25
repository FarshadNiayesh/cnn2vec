import random
import os


def human_format(num):
    """ from https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python """ # noqa
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude]) # noqa


def format_number(num):
    dec = random.randint(0, 3)
    sep = "," if random.uniform(0, 1) < 0.5 else ""
    templ = "{sep}.{dec}f".format(sep=sep, dec=dec)
    choose = random.uniform(0, 1)
    if choose < 0.05:
        to_return = human_format(num)
    elif choose < 0.1:
        to_return = format(num, templ).replace(',', '_')
        to_return = to_return.replace('.', ',').replace('_', '.')
    elif choose < 0.15:
        if num < 0:
            to_return = format(num, templ).replace('-', '-0')
        else:
            to_return = "0"+format(num, templ)
    else:
        to_return = format(num, templ)
    return to_return


number = -10e6
max_num_per_file = 1000  # small number to be splited in many batches
counter = 1
filenamebase = os.path.join(os.getcwd(), 'corpus', 'numbers_{}.txt')
filename = filenamebase.format(str(counter))
file = open(filename, 'w+')
while number < 10e6:
    if counter % max_num_per_file == 0:
        print('New file {} !'.format(counter), end="\r")
        filename = filenamebase.format(str(counter))
        file = open(filename, 'w+')
    file.write(format_number(number)+" ")
    number += random.uniform(0, 10)
    counter += 1
file.close()
