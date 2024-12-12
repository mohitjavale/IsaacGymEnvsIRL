import torch
import torch.nn as nn
import torch.nn.functional as F


class ICMFeatureEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        return self.net(x)


class ICMForwardModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state_features, actions):
        x = torch.cat([state_features, actions], dim=-1)
        return self.net(x)


class ICMInverseModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, current_features, next_features):
        x = torch.cat([current_features, next_features], dim=-1)
        return self.net(x)


class ICM(nn.Module):
    def __init__(self, input_dim, action_dim, feature_dim=256, beta=0.2, device="cuda:0"):
        super().__init__()
        self.feature_encoder = ICMFeatureEncoder(input_dim, feature_dim).to(device)
        self.forward_model = ICMForwardModel(feature_dim, action_dim).to(device)
        self.inverse_model = ICMInverseModel(feature_dim, action_dim).to(device)
        self.feature_encoder.train()
        self.forward_model.train()
        self.inverse_model.train()
        self.beta = beta
        self.device = device

        print(input())

    def reset(self):
        """Reset accumulated gradients"""
        pass

    def compute_intrinsic_reward(self, obs, next_obs, actions):
        # Encode states into features
        current_features = self.feature_encoder(obs)
        next_features = self.feature_encoder(next_obs)
        # Forward model prediction
        predicted_next_features = self.forward_model(current_features, actions)
        # Compute forward model loss (intrinsic reward)
        forward_loss = F.mse_loss(predicted_next_features, next_features, reduction='none')
        intrinsic_reward = torch.mean(forward_loss, dim=-1)
        # Inverse model prediction
        predicted_actions = self.inverse_model(current_features, next_features)
        inverse_loss = F.mse_loss(predicted_actions, actions, reduction='none')
        inverse_loss = torch.mean(inverse_loss, dim=-1)
        # Total loss for training
        total_loss = (1 - self.beta) * inverse_loss + self.beta * torch.mean(forward_loss, dim=-1)
        # Test gradient flow explicitly
        # grads = torch.autograd.grad(outputs=total_loss.mean(), inputs=list(self.parameters()), retain_graph=True)

        return intrinsic_reward, total_loss
