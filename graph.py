import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
count = 0
for line in open('sample.txt', 'r'):
    count += 1
    y.append(float(line))
    x.append(count)

# y_mean = [np.mean(y)]*len(x)
fig, ax = plt.subplots()
#calculate equation for quadratic trendline
z = np.polyfit(x, y, 3)
p = np.poly1d(z)

#add trendline to plot
# plt.plot(x, p(x))

plt.title("Fitness")
plt.xlabel('training game')
plt.ylabel("average fitness")
# plt.plot(x, y, marker='o', c='g')
data_line = ax.scatter(x, y, label='Data', marker='o', color='orange')
# mean_line = ax.plot(x, y_mean, label='Mean', linestyle='--')
trend_line = ax.plot(x, p(x), label='Trend', linestyle='-', color='green')
legend = ax.legend(loc='upper right')

plt.show()
