import numpy as np
from numpy import array
import matplotlib.pyplot as plt

data = np.array([array([-1.68943594,  0.84345508, -0.28201664]), array([-1.47943594,  1.05345508, -0.07201664]), array([-1.47943594,  0.84345508, -0.28201664]), array([-1.26943594,  1.05345508, -0.07201664]), array([-1.26943594,  0.84345508, -0.28201664]), array([-1.05943594,  1.05345508, -0.07201664])])
x_values = [d[0] for d in data]
y_values = [d[1] for d in data]
z_values = [d[2] for d in data]

# Create a plot for each component
plt.plot(x_values, label='Weight 1')
plt.plot(y_values, label='Weight 2')
plt.plot(z_values, label='Bias ')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Component Value')
plt.title('Vector Components Over Time')
plt.legend()

plt.show()
