import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import subprocess

class RLRecommender(nn.Module):
    """
    Neural Network that scores a candidate given a concatenated vector of the 
    user embedding and the candidate's embedding.
    Input dimension is 2 * embedding_dimension; output is a single score.
    """
    def __init__(self, emb_dim):
        super(RLRecommender, self).__init__()
        input_dim = 2 * emb_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output shape: (batch_size, 1)

class RecommendationAgent:
    """
    This agent uses the RLRecommender network to score each candidate light novel.
    Given a user embedding, it forms a batch by concatenating the user embedding with each candidate's embedding.
    An epsilon-greedy strategy is applied for exploration, and a diversity penalty is applied to recently recommended items.
    """
    def __init__(self, emb_dim, dataset_embeddings, epsilon=0.1, diversity_penalty=0.2, eval_interval=30):
        self.emb_dim = emb_dim
        self.dataset_embeddings = dataset_embeddings
        self.num_items = dataset_embeddings.shape[0]
        self.model = RLRecommender(emb_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon
        self.diversity_penalty = diversity_penalty
        self.history = []
        
        # Logs for evaluation
        self.reward_log = []
        self.loss_log = []
        self.eval_interval = eval_interval  # Run evaluation every `eval_interval` feedback interactions
        self.train_step = 0  # Track training steps

        self.training_steps = 0
        self.rewards_history = []  # Store rewards
        self.loss_history = []  # Store loss values

    def select_actions(self, user_embedding, top_k=3):
        """
        Given a user embedding (tensor of shape (emb_dim,)),
        builds inputs for all candidates (concatenates user embedding with each candidate's embedding),
        then returns the indices of the top_k candidates with the highest scores.
        Epsilon-greedy exploration returns random candidates with probability epsilon.
        """
        # Expand user_embedding to shape (num_items, emb_dim)
        user_batch = user_embedding.unsqueeze(0).repeat(self.num_items, 1)
        # Concatenate with candidate embeddings: shape (num_items, 2*emb_dim)
        input_tensor = torch.cat([user_batch, self.dataset_embeddings], dim=1)
        with torch.no_grad():
            scores = self.model(input_tensor).squeeze()  # shape: (num_items,)
        scores = scores.clone()
        # Apply diversity penalty for the last 5 recommended items
        for idx in self.history[-5:]:
            scores[idx] -= self.diversity_penalty
        # Epsilon-greedy: random exploration with probability epsilon
        if random.uniform(0, 1) < self.epsilon:
            random_indices = random.sample(range(self.num_items), top_k)
            return random_indices
        # Otherwise, select top_k highest-scoring indices
        _, top_indices = torch.topk(scores, top_k)
        top_indices = top_indices.tolist()
        return top_indices

    def train(self, user_embedding, candidate_idx, reward):
        candidate_embedding = self.dataset_embeddings[candidate_idx].unsqueeze(0)
        user_batch = user_embedding.unsqueeze(0)
        input_tensor = torch.cat([user_batch, candidate_embedding], dim=1)
        predicted_score = self.model(input_tensor).squeeze()
        target = torch.tensor(reward, dtype=torch.float32)
        loss = self.criterion(predicted_score, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store training data
        self.training_steps += 1
        self.rewards_history.append(reward)
        self.loss_history.append(loss.item())

        # Save progress every 30 steps
        if self.training_steps % 30 == 0:
            self.save_training_data()

    def save_training_data(self):
        """Saves model and training data for visualization."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rewards_history': self.rewards_history,
            'loss_history': self.loss_history,
            'training_steps': self.training_steps
        }, "agent_stats.pth")
        print(f"Training data saved at step {self.training_steps}")