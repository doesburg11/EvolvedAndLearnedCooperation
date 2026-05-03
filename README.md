# Two-Timescale Reciprocity Simulation

This project simulates the interaction between two timescales in the evolution of cooperation:

1. **Fast timescale: learning within a lifetime**
2. **Slow timescale: evolutionary selection between generations**

The model is designed to show how learned reciprocal behavior can interact with evolved predispositions.

In simple terms:

```text
behavior now = evolved predisposition + learned social experience
```

The simulation uses a repeated social interaction game inspired by the Prisoner's Dilemma / donation game.

Agents can cooperate or defect. Cooperation costs the actor something but gives a larger benefit to the other agent. Over repeated interactions, agents learn which partners are trustworthy. Across generations, agents with higher total payoff reproduce more successfully.

---

## Main idea

The model separates two processes:

### 1. Learning during a lifetime

During one generation, agents interact many times with local neighbors.

Each agent keeps a learned trust value for each partner:

```python
learned_trust[i, j]
```

This means:

```text
what agent i has learned about agent j during this lifetime
```

If partner `j` cooperates, agent `i` becomes more trusting of `j`.

If partner `j` defects, agent `i` becomes less trusting of `j`.

This is the fast, developmental, or "nurture" layer.

---

### 2. Evolution between generations

At the end of each generation, agents reproduce based on their lifetime payoff.

Agents with higher payoff are more likely to become parents.

Their offspring inherit three traits:

```python
trust_prior
learning_rate
responsiveness
```

These inherited traits are then slightly mutated.

This is the slow, evolutionary, or "nature" layer.

---

## Inherited traits

Each agent has three inherited traits.

### `trust_prior`

The agent's initial tendency to cooperate with an unknown partner.

A high value means the agent starts out more trusting.

A low or negative value means the agent starts out more suspicious.

---

### `learning_rate`

How quickly the agent updates trust after experience.

A high learning rate means the agent quickly changes its opinion of a partner.

A low learning rate means the agent changes slowly.

---

### `responsiveness`

How strongly learned trust affects future behavior.

A high responsiveness means the agent strongly adjusts its cooperation based on past experience.

A low responsiveness means the agent mostly ignores learned trust.

---

## The core decision rule

The most important line in the model is:

```python
score_i = genes["trust_prior"][i] + genes["responsiveness"][i] * learned_trust[i, j]
```

This means:

```text
agent i's decision = inherited trust tendency + learned trust in partner j
```

Then the agent cooperates when:

```python
cooperate_i = score_i > 0.0
```

So an agent's behavior is not purely genetic and not purely learned.

It is the result of both.

---

## Payoff structure

The model uses a donation-game version of the Prisoner's Dilemma.

If agent `i` cooperates with agent `j`:

```text
agent i pays a cost
agent j receives a benefit
```

In the default script:

```python
benefit = 3.0
cost = 1.0
```

So cooperation is socially beneficial, because the recipient gains more than the actor loses.

However, cooperation can still be individually risky, because a defector can receive benefits without paying costs.

---

## Why compare one-shot and repeated interaction?

The script runs two scenarios.

### Scenario 1: Mostly one-shot interaction

```python
lifetime_rounds = 1
```

Agents barely have time to learn who is trustworthy.

Direct reciprocity has little chance to develop.

This usually makes cooperation harder to maintain.

---

### Scenario 2: Repeated interaction

```python
lifetime_rounds = 80
```

Agents repeatedly meet neighbors.

They can learn who cooperates and who defects.

This allows direct reciprocity to matter.

Selection can then favor inherited traits that make reciprocal cooperation work better.

---

## Output

The script prints summary statistics such as:

```text
Final cooperation
Final payoff
Final trust prior
Final learning rate
Final responsiveness
```

It also saves plots in the `output/` folder:

```text
output/one_shot_cooperation.png
output/one_shot_traits.png
output/repeated_cooperation.png
output/repeated_traits.png
```

The most important plot is the cooperation plot.

It shows whether cooperation increases, collapses, or remains unstable over generations.

---

## How to run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy matplotlib
```

Run the simulation:

```bash
python two_timescale_reciprocity.py
```

---

# Difference from real Q-learning

This model uses a simple reinforcement-like trust update.

It is related to reinforcement learning, but it is not full Q-learning.

---

## What this model does

In this model, agents update partner-specific trust:

```python
learned_trust[i, j] += alpha_i * (target_for_i - learned_trust[i, j])
```

Where:

```python
target_for_i = 1.0 if cooperate_j else -1.0
```

So the agent learns:

```text
partner cooperated  -> trust goes up
partner defected    -> trust goes down
```

This is a simple social-learning rule.

The agent is not explicitly learning the value of its own actions.

It is learning whether the partner seems trustworthy.

---

## What real Q-learning does

In real Q-learning, an agent learns the expected value of taking an action in a state.

The basic form is:

```python
Q[state, action] = Q[state, action] + alpha * (
    reward + gamma * max(Q[next_state, next_action]) - Q[state, action]
)
```

A Q-learning agent would learn values such as:

```python
Q[partner, COOPERATE]
Q[partner, DEFECT]
```

This means the agent learns:

```text
how valuable it is for me to cooperate with this partner
how valuable it is for me to defect against this partner
```

That is different from merely learning whether the partner is trustworthy.

---

## Simple trust learning versus Q-learning

| Feature | Current model | Real Q-learning |
|---|---|---|
| Learns about partners? | Yes | Can, if partner identity is part of the state |
| Learns action values? | No | Yes |
| Has Q-values? | No | Yes |
| Has states? | Very limited | Yes |
| Has actions? | Cooperation is chosen by a rule | Actions are selected from learned values |
| Uses reward directly? | No, mostly partner behavior | Yes |
| Uses future expected reward? | No | Yes |
| Has discount factor `gamma`? | No | Yes |
| Has exploration strategy? | Only random mistakes | Usually epsilon-greedy or softmax |
| Learns policy from reward? | Not fully | Yes |

---

## Important conceptual difference

The current model says:

```text
I cooperate if I have enough inherited trust plus learned trust in this partner.
```

Q-learning says:

```text
I choose the action that has produced the best expected reward in this situation.
```

So the current model is better described as:

```text
evolution + simple partner-specific social learning
```

not:

```text
evolution + full reinforcement learning
```

---

## Why use this simpler model?

The simple trust model is useful because it directly shows the biological idea:

```text
evolution shapes learning tendencies
learning shapes behavior during life
behavior affects payoff
payoff affects evolutionary selection
```

That is the two-timescale process.

It is also easier to understand than full Q-learning.

The goal of this script is not to build an optimal RL agent.

The goal is to demonstrate how evolved predispositions and learned reciprocity can interact.

---

## How to extend this to real Q-learning

To make the model closer to true reinforcement learning, replace:

```python
learned_trust[i, j]
```

with a Q-table:

```python
Q[i, j, action]
```

where `action` is one of:

```python
COOPERATE = 0
DEFECT = 1
```

Then each agent would choose actions using an exploration rule, for example epsilon-greedy:

```python
if random_number < epsilon:
    action = random action
else:
    action = best known action
```

After receiving a reward, the agent would update:

```python
Q[i, j, action] += alpha * (reward - Q[i, j, action])
```

For a repeated game with longer-term consequences, the update could include a discount factor:

```python
Q[i, j, action] += alpha * (
    reward + gamma * max_future_value - Q[i, j, action]
)
```

Then evolution could act on inherited RL parameters such as:

```python
initial_Q_bias
learning_rate
exploration_rate
discount_factor
forgiveness_bias
partner_memory_strength
```

That would create a stronger model of:

```text
evolution of reinforcement-learning parameters
+
learning of cooperation during lifetime
```

---

## Relation to cooperation mechanisms

This model mainly includes two cooperation mechanisms:

### Direct reciprocity

Agents condition behavior on previous interactions with the same partner.

In the script, this is represented by:

```python
learned_trust[i, j]
```

---

### Network reciprocity

Agents interact repeatedly with local neighbors instead of random strangers.

In the script, this is represented by:

```python
make_ring_neighbors()
```

---

## Mechanisms not yet included

The current model does not yet include:

### Kin selection

Agents do not know who their relatives are.

To add this, give agents family IDs and add extra cooperation tendency toward kin.

---

### Indirect reciprocity

Agents do not observe reputation.

To add this, create public reputation scores that increase when agents cooperate and decrease when they defect.

---

### Group selection

Groups do not reproduce or die as units.

To add this, divide agents into groups and allow high-performing groups to contribute more offspring to the next generation.

---

## Summary

This model demonstrates a mutual process between evolution and learning.

The fast process is:

```text
agents learn which partners cooperate
```

The slow process is:

```text
selection favors inherited traits that make successful learning and cooperation more likely
```

The model is not full Q-learning.

It is a simpler trust-learning model designed to show the interaction between:

```text
nature: evolved predispositions
nurture: learned social experience
behavior: cooperation or defection
selection: reproductive success
```

---

## Simulation results

The results below come from a single run with default parameters (120 generations, 100 agents, ring topology, `benefit=3.0`, `cost=1.0`).

### Final statistics

| Metric | One-shot (`rounds=1`) | Repeated (`rounds=80`) |
|---|---|---|
| Final cooperation | 0.000 | 0.979 |
| Final payoff | 0.000 | 313.300 |
| Final trust prior | −0.864 | 1.447 |
| Final learning rate | 0.095 | 0.186 |
| Final responsiveness | 1.126 | 2.459 |

---

### One-shot interaction

Without repeated contact agents cannot learn who cooperates, so reciprocity never gets off the ground.

**Cooperation collapses to zero.**

Evolution responds by driving `trust_prior` negative (−0.86): selection favors innate suspicion because unconditional cooperators are exploited. `responsiveness` stays moderate but is effectively irrelevant when there is nothing useful to learn in a single round.

![One-shot cooperation](output/one_shot_cooperation.png)

![One-shot evolved traits](output/one_shot_traits.png)

---

### Repeated interaction

With 80 rounds per generation, cooperation stabilizes near full (~98%).

**The trait trajectories explain the mechanism:**

- `trust_prior` rises to ~1.45 — selection favors agents who start out cooperative, because unconditional cooperators can seed mutual cooperation with neighbors.
- `responsiveness` rises strongly to ~2.46 — agents who amplify what they have learned become sharply conditional: they strongly reward cooperators and punish defectors, reinforcing the reciprocal equilibrium.
- `learning_rate` stays low (~0.19) in both scenarios — fast forgetting is not favored because stable trust relationships are valuable.

The dip around generation 45–55 is a classic invasion event: a defector lineage briefly spreads, trust collapses, and cooperation crashes. The population recovers because reciprocators with high `responsiveness` re-establish cooperation faster than defectors can spread.

![Repeated interaction cooperation](output/repeated_cooperation.png)

![Repeated interaction evolved traits](output/repeated_traits.png)

---

### Core message

The two timescales reinforce each other in the repeated case.

Learning makes cooperation individually rational *within* a lifetime.

Evolution then favors the inherited traits (`trust_prior`, `responsiveness`) that make that learning work most effectively.

In the one-shot case, the fast timescale provides no useful signal, so evolution strips away cooperative predispositions entirely.
