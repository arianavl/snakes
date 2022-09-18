import matplotlib.pyplot as plt

x = []
y = []
count = 0
for line in open('sample.txt', 'r'):
    count += 1
    y.append(int(line))
    x.append(count)

plt.title("Fitness")
plt.xlabel('training game')
plt.ylabel("average fitness")
plt.yticks(y)
plt.plot(x, y, marker='o', c='g')

plt.show()
