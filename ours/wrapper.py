import gymnasium as gym
import numpy as np
import shimmy  # noqa: F401


class SkillWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_skills: int,
        k: int,
        fixed_eps_skill: bool = True,
    ):
        """
        Args:
            env: The environment to wrap.
            n_skills: The number of skills.
            k: The number of steps to take before sampling a new skill.
            fixed_eps_skill: Whether to fixed the skill for the entire episode.
        """
        super(SkillWrapper, self).__init__(env)
        self.n_skills = n_skills
        self.k = k
        self.current_skill = None
        self.fixed_eps_skill = fixed_eps_skill

        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "skill": gym.spaces.Discrete(n_skills),
                "skill_start": gym.spaces.Discrete(n=2),
            }
        )

    def observation(self, obs, skill_start: bool = False):
        return {
            "observation": obs,
            "skill": self.current_skill,
            "skill_start": skill_start,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.current_step = 0
        self.current_skill = kwargs.get(
            "initial_skill", self.np_random.choice(self.n_skills)
        )  # uniformly sample a skill at the beginning of each episode
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs, skill_start=True), info

    def step(self, action):
        skill_start = False
        self.current_step += 1
        if not self.fixed_eps_skill and self.current_step % self.k == 0:
            self.current_skill = self.np_random.choice(
                self.n_skills
            )  # uniformly sample a skill every k steps
            skill_start = True

        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            self.observation(obs, skill_start=skill_start),
            reward,
            terminated,
            truncated,
            info,
        )
class DMControlHopperWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DMControlHopperWrapper, self).__init__(env)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6+2+7,), dtype=np.float64)

    def observation(self, obs):
        return np.concatenate([obs["position"], obs["velocity"], obs["touch"]])


def main():
    env = SkillWrapper(DMControlHopperWrapper(gym.make("dm_control/hopper-hop-v0", render_mode="human", max_episode_steps=1000)), n_skills=10, k=3)

    obs, info = env.reset()


    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
