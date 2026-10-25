
import numpy as np

# ============================================================
# 1. Model parameters
# ============================================================

states = np.arange(-4, 5)        # States: -4, -3, ..., 4
actions = [-1, 0, 1]             # Possible control actions

disturbances = [-1, 0, 1]
probabilities = [0.10, 0.60, 0.30]

gamma = 0.95                     # Discount factor
n_states = len(states)           # Number of states: 9
tie_tol = 1e-12                  # Tolerance for ties


# ============================================================
# 2. Admissible control actions
#    No clipping is used in this modified model.
# ============================================================

def admissible_actions(e):
    return [
        u for u in actions
        if abs(e + u) <= 3
    ]


# ============================================================
# 3. Immediate stage cost
#    g(e, u) = e^2 + 0.20*u^2
# ============================================================

def stage_cost(e, u):
    return e**2 + 0.20 * u**2


# ============================================================
# 4. Exact policy evaluation
#    Solve (I - gamma*P_mu) J_mu = g_mu
# ============================================================

def policy_evaluation(policy):
    P = np.zeros((n_states, n_states))
    g = np.zeros(n_states)

    for i, e in enumerate(states):
        u = policy[i]

        # Check that the policy action is admissible
        assert u in admissible_actions(e)

        # Immediate cost for this state and policy action
        g[i] = stage_cost(e, u)

        # Construct the transition probability matrix
        for w, prob in zip(disturbances, probabilities):
            next_state = e + u + w

            # Verify that the successor is inside the state space
            assert next_state in states

            # Convert state to array index: -4 -> 0, ..., 4 -> 8
            j = int(next_state + 4)

            P[i, j] += prob

    # Solve the Bellman linear system
    J = np.linalg.solve(
        np.eye(n_states) - gamma * P,
        g
    )

    return J, P, g


# ============================================================
# 5. Greedy policy improvement
#    Select the admissible action with minimum Q_mu(e, u)
# ============================================================

def greedy_improvement(J):
    new_policy = []

    for e in states:
        candidates = admissible_actions(e)
        Q_values = []

        for u in candidates:
            expected_future_cost = 0.0

            for w, prob in zip(disturbances, probabilities):
                next_state = e + u + w
                j = int(next_state + 4)

                expected_future_cost += prob * J[j]

            q = (
                stage_cost(e, u)
                + gamma * expected_future_cost
            )

            Q_values.append(q)

        # Minimum candidate action value
        min_Q = min(Q_values)

        # Resolve numerical ties using action order [-1, 0, 1].
        # candidates preserve this order because 'actions' does.
        best_index = next(
            i for i, q in enumerate(Q_values)
            if q <= min_Q + tie_tol
        )

        new_policy.append(candidates[best_index])

    return np.array(new_policy, dtype=int)


# ============================================================
# 6. Policy iteration
# ============================================================

# Initial feasible policy:
# State order is (-4, -3, -2, -1, 0, 1, 2, 3, 4)
# Use zero steering where admissible; use forced boundary actions.
policy = np.array([1, 0, 0, 0, 0, 0, 0, 0, -1])

iteration = 0

while True:

    # Step 1: Evaluate the current policy exactly
    J, P, g = policy_evaluation(policy)

    # Step 2: Improve the policy greedily
    improved_policy = greedy_improvement(J)

    # Step 3: Count how many state actions changed
    changes = np.count_nonzero(improved_policy != policy)

    # Display this iteration
    print(f"\nIteration {iteration}")
    print("Evaluated policy:", policy.tolist())
    print("Value function:", np.round(J, 6).tolist())
    print("Mean state value:", round(float(np.mean(J)), 6))
    print("Improved policy:", improved_policy.tolist())
    print("Number of changed actions:", changes)

    # Step 4: Stop when the policy no longer changes
    if changes == 0:
        break

    # Otherwise, continue with the improved policy
    policy = improved_policy
    iteration += 1


# ============================================================
# 7. Final results
# ============================================================

print("\n========== FINAL RESULTS ==========")
print("States:", states.tolist())
print("Optimal policy:", policy.tolist())
print("Optimal value function:", np.round(J, 6).tolist())
print("Mean optimal state value:", round(float(np.mean(J)), 6))
print("Policy is admissible at every state:",
      all(policy[i] in admissible_actions(e)
          for i, e in enumerate(states)))
