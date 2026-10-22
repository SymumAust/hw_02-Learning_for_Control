import numpy as np
import matplotlib.pyplot as plt

states = np.arange(-4, 5)

policy_sequence = np.array([
    [ 1, 0, 0, 0, 0, 0, 0, 0, -1],  # mu_0
    [ 1, 1, 0,-1,-1,-1,-1,-1, -1],  # mu_1
    [ 1, 1, 1, 1, 1, 0,-1,-1, -1],  # mu_2
    [ 1, 1, 1, 0, 0,-1,-1,-1, -1],  # mu_3
    [ 1, 1, 1, 1, 0,-1,-1,-1, -1]   # mu_4
])

for k, policy in enumerate(policy_sequence):
    plt.plot(states, policy, marker="o", label=f"$\\mu_{k}$")

plt.xlabel("State e")
plt.ylabel("Selected action")
plt.xticks(states)
plt.yticks([-1, 0, 1])
plt.title("Policy Improvement Across Iterations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()