import random
space = 7
import datetime
import numpy as np
import matplotlib.pyplot as plt


# savePath = 'runs/{}_SAC'.format(datetime.datetime.now().strftime("%Y%m%d__%H:%M:%S"))
# print(savePath)
#
# for i in range(0,100):
#     print(random.randint(-1,1))

x0 = [309.989,369.991,89.990,309.995,279.987,319.990,109.989,329.988,-150.007]
x = [313.092,374.356,90.780,311.531,281.399,320.374,110.612,332.244,-150.719]
y0 = [388.158,-441.840,-501.837,118.162,338.156,58.160,-521.833,-221.838,-441.834]
y= [392.049,-447.049,-506.227,118.748,339.864,58.230,-524.771,-223.353,-443.937]
z0 = [400.987,520.984,360.985,290.995,260.987,240.992,240.981,310.991,240.985]
z = [398.099,519.888,357.755,286.659,256.783,236.505,237.034,306.942,236.782]

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
plt.show()
ax.scatter3D(x0, y0, z0, marker = "o",label='Original point')
ax.scatter3D(x, y, z, marker = "o", label='DRV points')
ax.legend()
plt.show()