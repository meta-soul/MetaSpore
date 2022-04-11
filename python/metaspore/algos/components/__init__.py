import os
# using a loop to check the current path
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    __import__(module[:-3], locals(), globals(),level=1)    # level =1 means relative
del module