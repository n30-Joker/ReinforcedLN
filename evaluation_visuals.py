import torch
import matplotlib.pyplot as plt
import seaborn as sns
from rl_recommender import RLRecommender  # Load model class

def load_training_data():
    """Loads saved training data from file."""
    checkpoint = torch.load("agent_stats.pth")
    return checkpoint

def plot_rewards(rewards_history):
    """Plots reward progression."""
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, label="Reward Trend", color="blue")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.title("Reward Progression Over Training")
    plt.legend()
    plt.show()

def plot_loss(loss_history):
    """Plots loss reduction over time."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Loss Trend", color="red")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Reduction Over Time")
    plt.legend()
    plt.show()

def show_model_architecture():
    """Displays the RL Neural Network architecture."""
    from torchsummary import summary
    model = RLRecommender(emb_dim=384)  # Assuming SBERT embedding size
    checkpoint = load_training_data()
    model.load_state_dict(checkpoint['model_state_dict'])
    print(summary(model, (1, model.fc1.in_features)))

if __name__ == "__main__":
    # Load saved data
    data = load_training_data()
    
    # Plot results
    plot_rewards(data['rewards_history'])
    plot_loss(data['loss_history'])
    show_model_architecture()
