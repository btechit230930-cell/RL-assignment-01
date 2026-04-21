# 🚪 Automatic Door using Reinforcement Learning (Policy Iteration)

## 📌 Overview
This project implements a Reinforcement Learning model using Policy Iteration to control an automatic door.

## 🎯 Objective
The agent decides:
- Open the door
- Close the door
- Wait

based on whether a person is nearby.

## 🧠 RL Components

### States
(person_near, door_state)

### Actions
- Open
- Close
- Wait

### Rewards
- Open when person near → +10
- Open unnecessarily → -5
- Keep closed properly → +5
- Block person → -10
- Wait → -1

## 🔁 Algorithm
Policy Iteration:
1. Policy Evaluation
2. Policy Improvement
3. Repeat until optimal

## ▶️ Run
```bash
python door_policy_iteration.py
```