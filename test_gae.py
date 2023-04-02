values = [0.09, 0.1]
rewards = [0, 1]
gamma = 0.9
lam = 1.0
advantages_reversed = []
gae = 0.0
gaes_reversed = []
num_episodes = 2

for t in reversed(range(num_episodes)):
    current_value = values[t]
    if t < num_episodes - 1:
        next_value = values[t+1]
    else:
        next_value = 0.0
    reward = rewards[t]
    delta = reward + gamma * next_value - current_value
    print(f"delta={float(reward)}+{gamma}*{float(next_value):.3f}-{float(current_value):.3f}")
    gae = delta + gamma * lam * gae
    gaes_reversed.append(gae)

gaes = list(reversed(gaes_reversed))
print("gaes", gaes)