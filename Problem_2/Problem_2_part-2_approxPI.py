
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PROBLEM 2, PART 2: MC and TD(0) APPROXIMATE POLICY ITERATION
# ============================================================

# 1. Problem settings
STATES = np.arange(-4, 5)
ACTIONS = [-1, 0, 1]
DISTURBANCES = [-1, 0, 1]
PROBS = [0.10, 0.60, 0.30]

GAMMA = 0.95
NUM_TRAJECTORIES = 100
STEPS_PER_TRAJECTORY = 60

# Use the same five seeds for MC and TD.
SEEDS = [42, 123, 2026, 7, 99]

# Initial policy, with states ordered from -4 to 4.
MU0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, -1])


# ------------------------------------------------------------
# 2. Cost, features, and admissible actions
# ------------------------------------------------------------
def stage_cost(e, u):
    return e**2 + 0.20 * u**2


def phi(e):
    z = e / 4.0
    return np.array([1.0, z, z**2, z**3])


def admissible_actions(e):
    return [u for u in ACTIONS if abs(e + u) <= 3]


# ------------------------------------------------------------
# 3. Exact policy evaluation and exact policy iteration
#    Exact values are used only for diagnostics/reference.
# ------------------------------------------------------------
def transition_matrix_and_cost(policy):
    P = np.zeros((len(STATES), len(STATES)))
    g = np.zeros(len(STATES)

    )

    for i, e in enumerate(STATES):
        u = int(policy[i])
        g[i] = stage_cost(e, u)

        for w, p in zip(DISTURBANCES, PROBS):
            next_e = int(e + u + w)
            P[i, next_e + 4] += p

    return P, g


def exact_evaluation(policy):
    P, g = transition_matrix_and_cost(policy)
    I = np.eye(len(STATES))
    return np.linalg.solve(I - GAMMA * P, g)


def greedy_improvement(J_hat):
    new_policy = []

    for e in STATES:
        candidates = admissible_actions(int(e))
        q_values = []

        for u in candidates:
            expected_future_cost = 0.0

            for w, p in zip(DISTURBANCES, PROBS):
                next_e = int(e + u + w)
                expected_future_cost += p * J_hat[next_e + 4]

            q = (
                stage_cost(e, u)
                + GAMMA * expected_future_cost
            )
            q_values.append(q)

        # Tie-breaking: first admissible action within tolerance.
        min_q = min(q_values)
        best_index = next(
            i for i, q in enumerate(q_values)
            if q <= min_q + 1e-12
        )

        new_policy.append(candidates[best_index])

    return np.array(new_policy, dtype=int)


def compute_exact_pi():
    policies = [MU0.copy()]
    policy = MU0.copy()

    while True:
        J = exact_evaluation(policy)
        improved = greedy_improvement(J)

        if np.array_equal(improved, policy):
            break

        policy = improved
        policies.append(policy.copy())

    return policies, policy


EXACT_POLICIES, MU_STAR = compute_exact_pi()
J_STAR = exact_evaluation(MU_STAR)


def policy_gap(policy):
    # Exact values are diagnostics only, not training targets.
    J_policy = exact_evaluation(policy)
    return float(np.max(J_policy - J_STAR))


EXACT_GAPS = np.array([
    policy_gap(policy) for policy in EXACT_POLICIES
])


# ------------------------------------------------------------
# 4. Generate one batch of trajectories
# ------------------------------------------------------------
def generate_trajectories(policy, rng):
    trajectories = []

    for _ in range(NUM_TRAJECTORIES):
        # Uniformly sample the initial state.
        e = int(rng.choice(STATES))
        trajectory = []

        for _ in range(STEPS_PER_TRAJECTORY):
            u = int(policy[e + 4])
            w = int(rng.choice(DISTURBANCES, p=PROBS))

            next_e = int(e + u + w)
            c = stage_cost(e, u)

            trajectory.append((e, u, c, next_e))
            e = next_e

        trajectories.append(trajectory)

    return trajectories


# ------------------------------------------------------------
# 5. MC evaluation
#    One discounted return per trajectory starting state.
#    Fit theta using ordinary least squares.
# ------------------------------------------------------------
def mc_evaluation(trajectories):
    X = []
    returns = []

    for trajectory in trajectories:
        start_state = trajectory[0][0]

        G = sum(
            GAMMA**k * step[2]
            for k, step in enumerate(trajectory)
        )

        X.append(phi(start_state))
        returns.append(G)

    X = np.asarray(X)
    returns = np.asarray(returns)

    theta = np.linalg.lstsq(X, returns, rcond=None)[0]

    J_hat = np.array([
        phi(e) @ theta for e in STATES
    ])

    return theta, J_hat


# ------------------------------------------------------------
# 6. TD(0) evaluation
#    IMPORTANT:
#    - Reset theta to zero for EVERY evaluation.
#    - Reset t for EVERY evaluation.
#    - Use denominator 10000, as specified.
# ------------------------------------------------------------
def td_evaluation(trajectories):
    theta = np.zeros(4)
    t = 0

    for trajectory in trajectories:
        for e, u, c, next_e in trajectory:
            t += 1

            delta = (
                c
                + GAMMA * (phi(next_e) @ theta)
                - (phi(e) @ theta)
            )

            alpha = 0.1 / (1 + t / 10000.0)**0.6

            theta = theta + alpha * delta * phi(e)

    J_hat = np.array([
        phi(e) @ theta for e in STATES
    ])

    return theta, J_hat


# ------------------------------------------------------------
# 7. Run one complete trial for a method
#    Each trial has mu_0 through mu_4.
#    The RNG is seeded ONCE at the start of the trial.
# ------------------------------------------------------------
def run_trial(method, seed):
    rng = np.random.default_rng(seed)

    policy = MU0.copy()
    policies = [policy.copy()]
    gaps = [policy_gap(policy)]

    # Evaluation of mu_0.
    # For seed 42, this regenerates the same Part 1 data
    # if Part 1 also uses seed 42 and the same sampling order.
    trajectories = generate_trajectories(policy, rng)

    if method == "MC":
        _, J_hat = mc_evaluation(trajectories)
    elif method == "TD":
        _, J_hat = td_evaluation(trajectories)
    else:
        raise ValueError("method must be 'MC' or 'TD'")

    # Improve mu_0 to mu_1.
    policy = greedy_improvement(J_hat)
    policies.append(policy.copy())
    gaps.append(policy_gap(policy))

    # Three additional evaluations and improvements.
    for _ in range(3):
        # Fresh trajectories for the current policy.
        trajectories = generate_trajectories(policy, rng)

        if method == "MC":
            _, J_hat = mc_evaluation(trajectories)
        else:
            # td_evaluation resets theta and t internally.
            _, J_hat = td_evaluation(trajectories)

        policy = greedy_improvement(J_hat)
        policies.append(policy.copy())
        gaps.append(policy_gap(policy))

    return np.array(policies), np.array(gaps)


# ------------------------------------------------------------
# 8. Run five trials for MC and TD using the same seeds
# ------------------------------------------------------------
results = {"MC": [], "TD": []}

for method in ["MC", "TD"]:
    for seed in SEEDS:
        policies, gaps = run_trial(method, seed)

        results[method].append({
            "seed": seed,
            "policies": policies,
            "gaps": gaps
        })


# ------------------------------------------------------------
# 9. Print every trial's policy sequence and policy gaps
# ------------------------------------------------------------
for method in ["MC", "TD"]:
    print(f"\n{'=' * 20} {method} RESULTS {'=' * 20}")

    for trial in results[method]:
        print(f"\nSeed: {trial['seed']}")

        for r in range(5):
            print(
                f"mu_{r} = {trial['policies'][r].tolist()}, "
                f"d(mu_{r}) = {trial['gaps'][r]:.6f}"
            )


# ------------------------------------------------------------
# 10. Calculate mean gaps, sample SDs, and match counts
# ------------------------------------------------------------
summary = {}

for method in ["MC", "TD"]:
    all_gaps = np.array([
        trial["gaps"] for trial in results[method]
    ])

    mean_gaps = np.mean(all_gaps, axis=0)
    sample_sd = np.std(all_gaps, axis=0, ddof=1)

    final_match_count = sum(
        np.array_equal(trial["policies"][-1], MU_STAR)
        for trial in results[method]
    )

    summary[method] = {
        "all_gaps": all_gaps,
        "mean": mean_gaps,
        "sd": sample_sd,
        "final_mean_gap": mean_gaps[-1],
        "final_match_count": final_match_count
    }

    print(f"\n--- {method} SUMMARY ---")
    print("Mean gaps:", np.round(mean_gaps, 6))
    print("Sample SD:", np.round(sample_sd, 6))
    print("Final mean gap:", round(mean_gaps[-1], 6))
    print(
        "Final policies matching mu*:",
        final_match_count, "/ 5"
    )


# ------------------------------------------------------------
# 11. Print exact PI reference
# ------------------------------------------------------------
print("\n========== EXACT PI REFERENCE ==========")

for r, policy in enumerate(EXACT_POLICIES):
    print(
        f"mu_{r} = {policy.tolist()}, "
        f"d(mu_{r}) = {policy_gap(policy):.6f}"
    )

print("Optimal policy:", MU_STAR.tolist())


# ------------------------------------------------------------
# 12. Plot mean gap +/- sample SD and exact PI
# ------------------------------------------------------------
r = np.arange(5)

plt.figure(figsize=(10, 6))

plt.errorbar(
    r,
    summary["MC"]["mean"],
    yerr=summary["MC"]["sd"],
    marker="o",
    capsize=5,
    linewidth=2,
    label="MC mean +/- sample SD"
)

plt.errorbar(
    r,
    summary["TD"]["mean"],
    yerr=summary["TD"]["sd"],
    marker="s",
    capsize=5,
    linewidth=2,
    label="TD(0) mean +/- sample SD"
)

plt.plot(
    np.arange(len(EXACT_GAPS)),
    EXACT_GAPS,
    marker="^",
    linestyle="--",
    linewidth=2,
    label="Exact PI"
)

plt.xlabel("Policy index r")
plt.ylabel("Policy gap d(mu_r)")
plt.title("Problem 2 Part 2: Approximate PI vs Exact PI")
plt.xticks(r)
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()

plt.savefig("part2_policy_gap_comparison.png", dpi=200)
plt.show()
