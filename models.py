from abc import ABC
import torch
from torch import embedding, nn
from torch.nn import functional as F
from torch.distributions import Normal

from common_net.attentions import MAB, MHAConfig
from common_net.pos_embedding.sinusoidal_positonal import SinusoidalPositionalEmbedding


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module, ABC):
    def __init__(self, n_states: int, n_skills: int, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states: torch.Tensor):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits

class HistoryDiscriminator(nn.Module, ABC):
    """
    Discriminator that takes in the history of states and predicts the skill. The history is represented as a sequence of states, and we can use an attention mechanism to aggregate the information from the sequence. The output is a probability distribution over the skills.
    """
    def __init__(self, n_states: int, n_skills: int, n_hidden_filters=256):
        super(HistoryDiscriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(
            in_features=self.n_states, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()

        self.history_block = MAB(
            embed_dim=self.n_hidden_filters,
            d_ff=self.n_hidden_filters*2,
            mha_config=MHAConfig(
                positional_embedding_cls=SinusoidalPositionalEmbedding
            ),
        )

        self.hidden2 = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.q = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_skills
        )
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states: torch.Tensor, his_mask: torch.Tensor):
        # states: (batch_size, history_length, n_states)
        # his_mask: (batch_size, 1)
        _, history_length, _ = states.shape

        valid_lengths = his_mask.view(-1).long().clamp(min=1, max=history_length)   # (batch_size,)
        history_positions = torch.arange(history_length, device=states.device)      # (history_length,)
        valid_history_mask = history_positions.unsqueeze(0) < valid_lengths.unsqueeze(1)    # (batch_size, history_length)
        causal_mask = torch.tril(
            torch.ones(
                history_length,
                history_length,
                device=states.device,
                dtype=torch.bool,
            )
        )


        x = F.relu(self.hidden1(states))

        # Each history position can only attend to valid entries from its prefix.
        attn_mask = valid_history_mask.unsqueeze(1).expand(-1, history_length, -1)
        attn_mask = attn_mask & causal_mask.unsqueeze(0)

        x = self.history_block(x, attn_mask=attn_mask)  # (batch_size, history_length, n_hidden_filters)

        batch_indices = torch.arange(x.size(0), device=x.device)
        last_valid_indices = valid_lengths - 1  # the index of the last valid entry for each history
        x = x[batch_indices, last_valid_indices]
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits


class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states: int, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states: torch.Tensor):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)


class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states: int, n_actions: int, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states: int, n_actions: int, action_bounds: tuple, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states: torch.Tensor):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std: torch.Tensor = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states: torch.Tensor):
        dist: torch.distributions.Distribution = self(states)
        # Reparameterization trick
        u: torch.Tensor = dist.rsample()
        action: torch.Tensor = torch.tanh(u)
        log_prob: torch.Tensor = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob
