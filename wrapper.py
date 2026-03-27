import gymnasium as gym


class SkillWrapper(gym.Wrapper):
    def __init__(self, env, n_skills: int = 4, k: int = 8):
        super(SkillWrapper, self).__init__(env)
        self.n_skills = n_skills
        self.k = k
        self.current_step = 0
        self.current_skill = None
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "skill": gym.spaces.Discrete(n_skills),
            }
        )

    def observation(self, obs, skill_start: bool = False):
        return {"observation": obs, "skill": self.current_skill, "skill_start": skill_start}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.current_step = 0
        self.current_skill = self.np_random.choice(
            self.n_skills
        )  # uniformly sample a skill at the beginning of each episode
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs, skill_start=True), info

    def step(self, action):
        skill_start = False
        self.current_step += 1
        if self.current_step % self.k == 0:
            self.current_skill = self.np_random.choice(
                self.n_skills
            )  # uniformly sample a skill every k steps
            skill_start = True

        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs, skill_start=skill_start), reward, terminated, truncated, info


def main():
    a = SkillWrapper(gym.make("Hopper-v5", render_mode="human", max_episode_steps=1000))
    print(a.current_skill)


if __name__ == "__main__":
    main()
