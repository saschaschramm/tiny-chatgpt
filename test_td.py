# S0 -> S1,R1=0
# S1 -> S2,R2=1

num_episodes = 2
n = 2  # 2 step TD
t = 1  # current step

# R0,R1,R2
rewards = [None, 0, 1]
values = [0.09, 0.1]
print("state_values =", values)
gamma = 0.9

while True:
    tau = t - n + 1
    print("--------------------------------")
    print(f"t={t}: Update V(S{tau})")
    if t < num_episodes:
        print(f"S{t}: Observe R_{t+1} and S_{t+1}")

    if tau >= 0:
        g = 0
        str = []
        for i in range(tau + 1, min(tau + n + 1, num_episodes + 1)):
            str.append(f"γ^{i-tau-1} * R_{i}")
            g += pow(gamma, i - tau - 1) * rewards[i]
        print(f"G_{tau} = " + " + ".join(str))
        print(f"G_{tau} = {g}")
        if tau + n < num_episodes:
            # In case tau+n is not the leaf node we have to add the discounted estimate
            g += pow(gamma, n) + values[tau + n]
        delta = g - values[tau]
        print(f"delta_t{tau} = G_{tau} - V{tau+n-1}(S{tau})")
        print(f"delta_t{tau} = {g} - {values[tau]} = {delta}")

    if tau == num_episodes - 1:
        break
    t += 1
