import numpy as np

class EpsilonGreedyRecommender:
    def __init__(self, n_items, epsilon=0.1):
        self.n_items = n_items
        self.epsilon = epsilon
        self.counts = np.zeros(n_items)  # Number of times each item has been recommended
        self.values = np.zeros(n_items)  # Estimated value (reward) of each item

    def select_item(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_items)  # Exploration: randomly choose an item
        else:
            return np.argmax(self.values)  # Exploitation: choose the item with the highest estimated reward

    def update(self, item, reward):
        self.counts[item] += 1
        self.values[item] += (reward - self.values[item]) / self.counts[item]

n_items = 5  
epsilon = 0.1  
n_users = 1000  

true_item_values = np.random.uniform(0, 1, n_items)

recommender = EpsilonGreedyRecommender(n_items, epsilon)

for user in range(n_users):
    item = recommender.select_item()
    reward = np.random.binomial(1, true_item_values[item])
    recommender.update(item, reward)

print("Item recommendation counts:", recommender.counts)
print("Estimated item values:", recommender.values)
print("True item values:", true_item_values)