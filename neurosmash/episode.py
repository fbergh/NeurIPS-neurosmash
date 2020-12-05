class Episode:
    def __init__(self, environment, t_threshold=500, loss_reward=True, cooldown=False):
        self.env = environment
        self.t_threshold = t_threshold
        self.loss_reward = loss_reward
        self.cooldown = cooldown

    def run(self, agent):
        # Reset environment 
        end, reward, state = self.env.reset()

        # Run entire episode
        t = 0
        while not end:
            if t > self.t_threshold:
                print("Time threshold reached. Stopping early.")
                episode_reward = self.determine_loss_reward(state, t, overtime= True)
                is_win = False
                return is_win, episode_reward
            end, reward, state = self.step(agent, end, reward, state)
            t += 1
        
        # We have won if the reward is 10
        is_win = reward == 10
        # Losses might have rewards > 0 (but still < 10)
        if not is_win and self.loss_reward:
            reward = self.determine_loss_reward(state, t)
        # Store reward at the end of episode
        episode_reward = reward 

        # Additional steps if we want time for things to settle down
        if self.cooldown:
            for i in range(100):
                _ = self.step(reward, state)

        return is_win, episode_reward

    def step(self, agent, end, reward, state):
        action = agent.step(end, reward, state)
        end, reward, state = self.env.step(action)
        return end, reward, state

    def determine_loss_reward(self, state, time_elapsed,overtime = False):
        # overtime means draw
        if overtime:
            return 5
        # Determine the reward the agent get even though it lost
        max_loss_reward = 2
        return max_loss_reward * (time_elapsed/self.t_threshold)