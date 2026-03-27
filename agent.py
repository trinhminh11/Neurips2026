import numpy as np
from models import (
    PolicyNetwork,
    QvalueNetwork,
    ValueNetwork,
    Discriminator,
    HistoryDiscriminator,
)
import torch
from replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax


class SACAgent:
    def __init__(
        self,
        k: int,
        n_states: int,
        n_actions: int,
        action_bounds: tuple,
        n_skills: int = 50,
        batch_size: int = 256,
        n_hiddens: int = 256,
        lr: float = 3e-4,
        alpha: float = 0.1,
        reward_scale: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        mem_size: int = int(1e6),
        seed: int = 0,
    ):
        self.k = k
        self.n_states = n_states
        self.n_skills = n_skills
        self.batch_size = batch_size
        self.p_z = np.tile(np.full((n_skills,), 1 / n_skills), (batch_size, 1))
        self.memory = Memory(mem_size, seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_actions = n_actions
        self.alpha = alpha
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.tau = tau

        torch.manual_seed(seed)
        self.policy_network = PolicyNetwork(
            n_states=self.n_states + self.n_skills,
            n_actions=n_actions,
            action_bounds=action_bounds,
            n_hidden_filters=n_hiddens,
        ).to(self.device)

        self.q_value_network1 = QvalueNetwork(
            n_states=self.n_states + self.n_skills,
            n_actions=n_actions,
            n_hidden_filters=n_hiddens,
        ).to(self.device)

        self.q_value_network2 = QvalueNetwork(
            n_states=self.n_states + self.n_skills,
            n_actions=n_actions,
            n_hidden_filters=n_hiddens,
        ).to(self.device)

        self.value_network = ValueNetwork(
            n_states=self.n_states + self.n_skills, n_hidden_filters=n_hiddens
        ).to(self.device)

        self.value_target_network = ValueNetwork(
            n_states=self.n_states + self.n_skills, n_hidden_filters=n_hiddens
        ).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(
            n_states=self.n_states, n_skills=self.n_skills, n_hidden_filters=n_hiddens
        ).to(self.device)

        self.his_discriminator = HistoryDiscriminator(
            n_states=self.n_states, n_skills=self.n_skills, n_hidden_filters=n_hiddens
        ).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=lr)
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=lr)
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=lr)
        self.his_discriminator_opt = Adam(self.his_discriminator.parameters(), lr=lr)

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state, skill_start=False):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state, skill_start)

    def unpack(self, batch: list[Transition]):
        # batch = Transition(*zip(*batch))

        # states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        # zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        # dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        # actions = torch.cat(batch.action).view(-1, self.n_actions).to(self.device)
        # next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)

        # return states, zs, dones, actions, next_states
        states_ = []
        zs_ = []
        dones_ = []
        actions_ = []
        next_states_ = []

        state_with_history_ = []
        his_masks_ = []

        for transition in batch:
            states_.append(transition.state)
            zs_.append(transition.z)
            dones_.append(transition.done)
            actions_.append(transition.action)
            next_states_.append(transition.next_state)

            _current_state_with_history = []
            for his_transition in transition.traverse():
                _current_state_with_history.append(
                    torch.split(
                        his_transition.state, [self.n_states, self.n_skills], dim=-1
                    )[0]
                )
            _current_state_with_history.append(
                torch.split(
                    transition.next_state, [self.n_states, self.n_skills], dim=-1
                )[0]
            )  # include the next state in the history as well
            his_masks_.append(len(_current_state_with_history))
            if len(_current_state_with_history) < self.k + 1:
                _current_state_with_history += [
                    torch.zeros_like(_current_state_with_history[0])
                ] * (self.k + 1 - len(_current_state_with_history))

            state_with_history_.append(
                torch.stack(_current_state_with_history).to(self.device)
            )

        states = (
            torch.stack(states_)
            .view(self.batch_size, self.n_states + self.n_skills)
            .to(self.device)
        )
        zs = torch.stack(zs_).view(self.batch_size, 1).long().to(self.device)
        dones = torch.stack(dones_).view(self.batch_size, 1).to(self.device)
        actions = torch.stack(actions_).view(-1, self.n_actions).to(self.device)
        next_states = (
            torch.stack(next_states_)
            .view(self.batch_size, self.n_states + self.n_skills)
            .to(self.device)
        )

        state_with_history = torch.stack(state_with_history_).to(
            self.device
        )  # B, k, n_states
        his_masks = (
            torch.Tensor(his_masks_).view(self.batch_size, 1).to(self.device)
        )  # B, 1

        return states, zs, dones, actions, next_states, state_with_history, his_masks

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states, state_with_history, his_masks = (
                self.unpack(batch)
            )

            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(
                states
            )
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.alpha * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            # print("state_with_history shape:", state_with_history.shape)   # should be (batch_size, k, n_states)

            # print(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0].shape)

            # discriminator_input = torch.stack()

            logits = self.his_discriminator(state_with_history, his_masks)


            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = (
                    self.reward_scale * rewards.float()
                    + self.gamma * self.value_target_network(next_states) * (~dones)
                )
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            policy_loss = (self.alpha * log_probs - q).mean()
            logits = self.discriminator(
                torch.split(states, [self.n_states, self.n_skills], dim=-1)[0]
            )
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(
                self.value_network, self.value_target_network
            )

            return -discriminator_loss.item()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(
            target_network.parameters(), local_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1 - self.tau) * target_param.data
            )

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
