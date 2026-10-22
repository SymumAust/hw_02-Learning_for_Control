
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Problem settings
# --------------------------------------------------
SEED = 2026
GAMMA = 0.95
NUM_TRAJECTORIES = 100
STEPS_PER_TRAJECTORY = 60

states = np.arange(-4, 5)
disturbances = [-1, 0, 1]
probabilities = [0.10, 0.60, 0.30]

# State order: -4, -3, ..., 3, 4
mu0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, -1])


def stage_cost(e, u):
    return e**2 + 0.20 * u**2


def phi(e):
    z = e / 4.0
    return np.array([1.0, z, z**2, z**3])


# --------------------------------------------------
# 2. Exact policy evaluation
# --------------------------------------------------
P = np.zeros((9, 9))
g_mu = np.zeros(9)

for i, e in enumerate(states):
    u = mu0[i]
    g_mu[i] = stage_cost(e, u)

    for w, p in zip(disturbances, probabilities):
        next_e = int(e + u + w)
        P[i, next_e + 4] += p

J_exact = np.linalg.solve(
    np.eye(9) - GAMMA * P, g_mu
)


# --------------------------------------------------
# 3. Generate common trajectories for MC and TD
# --------------------------------------------------
rng = np.random.default_rng(SEED)
trajectories = []

for _ in range(NUM_TRAJECTORIES):
    e = int(rng.choice(states))
    trajectory = []

    for _ in range(STEPS_PER_TRAJECTORY):
        u = int(mu0[e + 4])
        w = int(rng.choice(
            disturbances, p=probabilities
        ))

        next_e = int(e + u + w)
        c = stage_cost(e, u)

        trajectory.append((e, u, c, next_e))
        e = next_e

    trajectories.append(trajectory)


# --------------------------------------------------
# 4. Monte Carlo: one return per trajectory
# --------------------------------------------------
X_mc = []
returns = []

for trajectory in trajectories:
    start_e = trajectory[0][0]

    G0 = sum(
        GAMMA**k * step[2]
        for k, step in enumerate(trajectory)
    )

    X_mc.append(phi(start_e))
    returns.append(G0)

X_mc = np.asarray(X_mc)
returns = np.asarray(returns)

# Ordinary least squares
theta_mc = np.linalg.lstsq(
    X_mc, returns, rcond=None
)[0]

J_mc = np.array([
    phi(e) @ theta_mc for e in states
])


# --------------------------------------------------
# 5. TD(0): update after every transition
# --------------------------------------------------
theta_td = np.zeros(4)
t = 0

for trajectory in trajectories:
    for e, u, c, next_e in trajectory:
        t += 1

        # Bootstrap at every step, including
        # the final transition of each segment.
        delta = (
            c
            + GAMMA * (phi(next_e) @ theta_td)
            - (phi(e) @ theta_td)
        )

        alpha = 0.1 / (1 + t / 1000.0)**0.6

        theta_td += alpha * delta * phi(e)

J_td = np.array([
    phi(e) @ theta_td for e in states
])


# --------------------------------------------------
# 6. RMSE against exact J_mu0
# --------------------------------------------------
rmse_mc = np.sqrt(np.mean((J_mc - J_exact)**2))
rmse_td = np.sqrt(np.mean((J_td - J_exact)**2))

print("Random seed:", SEED)
print("Exact J_mu0:", np.round(J_exact, 6))
print("MC estimate:", np.round(J_mc, 6))
print("TD(0) estimate:", np.round(J_td, 6))
print("MC RMSE:", round(rmse_mc, 6))
print("TD(0) RMSE:", round(rmse_td, 6))


# --------------------------------------------------
# 7. Plot the three value functions
# --------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(
    states, J_exact, marker="o",
    linewidth=2.5, label="Exact J_mu0"
)
plt.plot(
    states, J_mc, marker="s", linestyle="--",
    linewidth=2, label="MC estimate"
)
plt.plot(
    states, J_td, marker="^", linestyle="-.",
    linewidth=2, label="TD(0) estimate"
)

plt.xlabel("State e")
plt.ylabel("Estimated cost-to-go")
plt.title("Part 1: MC and TD(0) Policy Evaluation")
plt.xticks(states)
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig("part1_mc_td_comparison.png", dpi=200)
plt.show()
