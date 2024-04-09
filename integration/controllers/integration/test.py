import random
space = 7
import datetime

savePath = 'runs/{}_SAC'.format(datetime.datetime.now().strftime("%Y%m%d__%H:%M:%S"))
print(savePath)

for i in range(0,100):
    print(random.randint(-1,1))