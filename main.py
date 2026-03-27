import gymnasium as gym
import numpy as np
from tqdm import tqdm
from wrapper import SkillWrapper
from agent import SACAgent
from logger import Logger

env_name = "Hopper-v5"
n_skills = 20
k = 8
batch_size = 256
n_hiddens = 256+64
lr = 3e-4
alpha = 0.1
reward_scale = 1.0
gamma = 0.99
tau = 0.005
mem_size = int(1e6)
seed = 0

max_episode_len = 5000


interval = 20

def concat_state_latent(state, z, n_skills):
    z_one_hot = np.zeros(n_skills)
    z_one_hot[z] = 1
    return np.concatenate([state, z_one_hot])

def main():
    env = SkillWrapper(gym.make(env_name), n_skills=n_skills, k=k)
    n_states = 11
    n_actions = 3
    action_bounds = (-1, 1)

    agent = SACAgent(
        k=k,
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        n_skills=n_skills,
        batch_size=batch_size,
        n_hiddens=n_hiddens,
        lr=lr,
        alpha=alpha,
        reward_scale=reward_scale,
        gamma=gamma,
        tau=tau,
        mem_size=mem_size,
        seed=seed,
    )
    logger = Logger(
        agent,
        env_name=env_name,
        interval=interval,
        seed=seed,
        n_states=n_states,
        n_actions=n_actions,
        n_skills=n_skills,
        batch_size=batch_size,
        n_hiddens=n_hiddens,
        lr=lr,
        alpha=alpha,
        reward_scale=reward_scale,
        gamma=gamma,
        tau=tau,
        mem_size=mem_size,
    )

    # if not params["train_from_scratch"]:
    #     (
    #         episode,
    #         last_logq_zs,
    #         np_rng_state,
    #         *env_rng_states,
    #         torch_rng_state,
    #         random_rng_state,
    #     ) = logger.load_weights()
    #     agent.hard_update_target_network()
    #     min_episode = episode
    #     np.random.set_state(np_rng_state)
    #     env.np_random.set_state(env_rng_states[0])
    #     env.observation_space.np_random.set_state(env_rng_states[1])
    #     env.action_space.np_random.set_state(env_rng_states[2])
    #     agent.set_rng_states(torch_rng_state, random_rng_state)
    #     print("Keep training from previous run.")

    # else:
    min_episode = 0
    last_logq_zs = 0
    np.random.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    print("Training from scratch.")

    logger.on()
    for episode in tqdm(range(1 + min_episode, max_episode_len + 1)):
        state, _ = env.reset()

        concat_state = concat_state_latent(state["observation"], state["skill"], n_skills)

        episode_reward: float = 0
        logq_zses = []

        if env.spec is None:
            raise ValueError("The environment does not have a spec, which is required to determine the maximum episode length.")
        else:
            if env.spec.max_episode_steps is None:
                raise ValueError("...")
            max_n_steps = min(max_episode_len, env.spec.max_episode_steps)


        for step in range(1, 1 + max_n_steps):
            action = agent.choose_action(concat_state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_concat_state = concat_state_latent(next_state['observation'], state["skill"], n_skills)
            agent.store(concat_state, state["skill"], terminated, action, next_concat_state, state["skill_start"])

            logq_zs = agent.train()

            if logq_zs is None:
                logq_zses.append(last_logq_zs)
            else:
                logq_zses.append(logq_zs)
            episode_reward += reward # type: ignore
            concat_state = next_concat_state
            state = next_state
            if done:
                break

            # logger.log(
            #     episode,
            #     episode_reward,
            #     state["skill"],
            #     sum(logq_zses) / len(logq_zses),
            #     step,
            #     *agent.get_rng_states(),
            # )

        logger.log(
            episode,
            episode_reward,
            state["skill"],
            sum(logq_zses) / len(logq_zses),
            step,
            *agent.get_rng_states(),
        )



if __name__ == "__main__":
    main()
