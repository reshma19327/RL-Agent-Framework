class GridWorld:

    def __init__(self, size=5, reward_strategy="step"):

        self.size = size
        self.reward_strategy = reward_strategy

        # Start and goal
        self.start = (0, 0)
        self.goal = (4, 4)

        # Obstacles
        self.obstacles = [
            (1, 1),
            (1, 2),
            (2, 3),
            (3, 1)
        ]

        self.reset()


    def reset(self):

        self.agent_position = self.start

        return self.agent_position


    def step(self, action):

        row, col = self.agent_position

        # Actions
        if action == 0:
            new_position = (row - 1, col)

        elif action == 1:
            new_position = (row + 1, col)

        elif action == 2:
            new_position = (row, col - 1)

        elif action == 3:
            new_position = (row, col + 1)

        else:
            raise ValueError("Invalid action")


        # ----------------------------------------------------
        # Check boundaries
        # ----------------------------------------------------

        if (
            new_position[0] < 0
            or new_position[0] >= self.size
            or new_position[1] < 0
            or new_position[1] >= self.size
        ):

            new_position = self.agent_position

            reward = self.get_reward(
                "boundary"
            )


        # ----------------------------------------------------
        # Check obstacle
        # ----------------------------------------------------

        elif new_position in self.obstacles:

            new_position = self.agent_position

            reward = self.get_reward(
                "obstacle"
            )


        # ----------------------------------------------------
        # Normal movement
        # ----------------------------------------------------

        else:

            old_position = self.agent_position

            self.agent_position = new_position

            reward = self.get_reward(
                "move",
                old_position,
                new_position
            )


        # Update position
        self.agent_position = new_position


        # ----------------------------------------------------
        # Check goal
        # ----------------------------------------------------

        done = self.agent_position == self.goal

        if done:

            reward = 10


        return self.agent_position, reward, done


    # ========================================================
    # Reward Function
    # ========================================================

    def get_reward(
        self,
        event,
        old_position=None,
        new_position=None
    ):

        # ----------------------------------------------------
        # Strategy 1: Sparse Reward
        # ----------------------------------------------------

        if self.reward_strategy == "sparse":

            if event == "move":
                return 0

            elif event == "obstacle":
                return 0

            elif event == "boundary":
                return 0


        # ----------------------------------------------------
        # Strategy 2: Step Penalty
        # ----------------------------------------------------

        elif self.reward_strategy == "step":

            if event == "move":
                return -1

            elif event == "obstacle":
                return -5

            elif event == "boundary":
                return -1


        # ----------------------------------------------------
        # Strategy 3: Reward Shaping
        # ----------------------------------------------------

        elif self.reward_strategy == "shaped":

            if event == "obstacle":
                return -5

            if event == "boundary":
                return -2

            if event == "move":

                old_distance = (
                    abs(old_position[0] - self.goal[0])
                    + abs(old_position[1] - self.goal[1])
                )

                new_distance = (
                    abs(new_position[0] - self.goal[0])
                    + abs(new_position[1] - self.goal[1])
                )

                if new_distance < old_distance:

                    return 1

                elif new_distance > old_distance:

                    return -2

                else:

                    return -1


        raise ValueError(
            "Unknown reward strategy"
        )


    # ========================================================
    # Render Environment
    # ========================================================

    def render(self):

        grid = [
            ["." for _ in range(self.size)]
            for _ in range(self.size)
        ]


        # Obstacles
        for row, col in self.obstacles:

            grid[row][col] = "X"


        # Start
        start_row, start_col = self.start

        grid[start_row][start_col] = "S"


        # Goal
        goal_row, goal_col = self.goal

        grid[goal_row][goal_col] = "G"


        # Agent
        agent_row, agent_col = self.agent_position

        grid[agent_row][agent_col] = "A"


        for row in grid:

            print(" ".join(row))