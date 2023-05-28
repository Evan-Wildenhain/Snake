import gym
from gym import spaces
import numpy as np

class SnakeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.field_size = (25,25)
        self.current_direction = 0


        #4 actions up left down right continue
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=3,shape=self.field_size,dtype=np.uint8)

        self.reset()
    
    def step(self,action):
        prev_head = self.snake_head
        if action == 4:
            action = self.current_direction
        else:
            self.current_direction = action
        
        if action == 0:
            #move up
            self.snake_head = (self.snake_head[0] - 1, self.snake_head[1])
        elif action == 1:
            #move down
            self.snake_head = (self.snake_head[0] +1 , self.snake_head[1])
        elif action == 2:
            #move left
            self.snake_head = (self.snake_head[0], self.snake_head[1] -1)
        elif action == 3:
            #move right
            self.snake_head = (self.snake_head[0], self.snake_head[1]+ 1)
        
        game_over = False

        if (self.snake_head in self.snake_body or
            self.snake_head[0] < 0 or
            self.snake_head[0] >= self.field_size[0] or
            self.snake_head[1] < 0 or
            self.snake_head[1] >= self.field_size[1]):
            game_over = True


        self.eaten = self.snake_head == self.food_pos

        if not game_over:
            self.field[self.snake_head] = 2
            if not self.eaten:
                if len(self.snake_body) == 0:
                    self.field[prev_head] = 0
                else:
                    tail = self.snake_body.pop()
                    self.field[tail] = 0
                    self.snake_body.insert(0,prev_head)
                    self.field[prev_head] = 1
            elif self.eaten:
                if len(self.snake_body) == 0:
                    self.field[prev_head] = 1
                    self.snake_body.append(prev_head)
                else:
                    self.snake_body.insert(0,prev_head)
                    self.field[prev_head] = 1
        
        #print(self.snake_head)
        #print(self.snake_body)
        #print(action)
                
        
        if self.eaten:
            self.food_pos = self.place_food()
            self.field[self.food_pos] = 5
        reward = -10 if game_over else 80 if self.eaten else -1
        return self.field,reward,game_over,{}
        

    def reset(self):
        self.field = np.zeros(self.field_size, dtype = np.uint8)
        #place snake at center
        self.snake_head = (self.field_size[0]//2, self.field_size[1]//2)
        self.snake_body = []
        self.field[self.snake_head] = 2
        self.current_direction = 0
        #place initial food
        self.food_pos = self.place_food()
        self.field[self.food_pos] = 5
        return self.field
    
    def render(self, mode='human'):
        field_str = self.field.astype(str)
        field_str[field_str == '0'] = ' '
        field_str[field_str == '1'] = '#'
        field_str[field_str == '2'] = '@'
        field_str[field_str == '5'] = '$'

        border  = '+' + "-" * (self.field_size[1]*2) + "+"
        print(border)
        for row in field_str:
            print("|" + " ".join(row) + "|")
        print(border)
        print()

    def place_food(self):
        #place food at empty random spot
        empty_spots = np.argwhere(self.field == 0)
        food_spot = empty_spots[np.random.randint(len(empty_spots))]
        return (food_spot[0],food_spot[1])