# gto.py (extended with training example)

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
from torch.utils.data import Dataset, DataLoader


class OpponentCardPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1326):
        """
        Initializes the OpponentCardPredictor network.

        Args:
            input_size (int): Dimension of the input feature vector.
            hidden_size (int): Number of neurons in hidden layers.
            output_size (int): Number of output classes (fixed to 1326 for two-card combinations).
        """
        super(OpponentCardPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Precompute all unique two-card combinations in canonical order.
        self.all_two_card_combinations = self._compute_two_card_combinations()

    def _compute_two_card_combinations(self):
        """
        Computes and returns a list of all 1,326 unique two-card combinations.
        Cards are represented as strings in the format 'AH', 'TD', etc.
        """
        # Define all 52 cards using ranks and suits.
        ranks = "23456789TJQKA"
        suits = "CDHS"  # Clubs, Diamonds, Hearts, Spades
        cards = [r + s for r in ranks for s in suits]
        # Use itertools.combinations to get all unique two-card sets (order does not matter).
        combinations = list(itertools.combinations(cards, 2))
        return combinations  # List of tuples; length should be 1326

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Probability distribution over 1,326 classes for each sample.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        # Use softmax to convert logits to probabilities.
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def predict_cards(self, features):
        """
        Given an input feature vector (or batch), predicts the opponent's hole card combination.

        Args:
            features (torch.Tensor): Tensor of shape (input_size,) or (batch_size, input_size).

        Returns:
            tuple or list of tuples: The predicted two-card combination(s) as tuples, e.g. ('AH', 'KD').
        """
        self.eval()
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)  # Convert to batch_size=1 if necessary.
            probs = self.forward(features)
            # For each sample, take the index with the maximum probability.
            _, indices = torch.max(probs, dim=-1)
            # Map each predicted index to its corresponding two-card combination.
            predicted_cards = [self.all_two_card_combinations[idx] for idx in indices.cpu().numpy()]
            if len(predicted_cards) == 1:
                return predicted_cards[0]
            return predicted_cards


# --------------------------
# Example Dataset for Training
# --------------------------
class OpponentCardDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (np.array or torch.Tensor): Array of shape (N, input_size).
            labels (np.array or torch.Tensor): Array of shape (N,) containing integer labels in [0, 1325].
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --------------------------
# Training Loop Example
# --------------------------
if __name__ == "__main__":
    # For demonstration, we generate random training data.
    # In practice, load your real dataset here.
    input_size = 10
    num_samples = 1000  # Replace with your dataset size
    # Create random feature vectors (e.g., encoded opponent actions)
    features = np.random.randn(num_samples, input_size)
    # Create random target labels (each an integer in [0, 1325])
    labels = np.random.randint(0, 1326, size=(num_samples,))

    dataset = OpponentCardDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate the model.
    model = OpponentCardPredictor(input_size=input_size, hidden_size=64, output_size=1326)

    # Define an optimizer and loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20  # Adjust the number of epochs as needed
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)  # Outputs shape: (batch_size, 1326)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the model weights (optional)
    torch.save(model.state_dict(), "opponent_card_predictor.pth")

    # Example usage of the trained model
    model.eval()
    dummy_input = torch.randn(input_size)  # Single sample feature vector
    predicted_cards = model.predict_cards(dummy_input)
    print("Predicted opponent cards for dummy input:", predicted_cards)
