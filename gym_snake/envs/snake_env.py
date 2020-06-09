from collections import Counter
from collections import deque
import random
from copy import deepcopy  
import numpy as np


import gym
from gym import error, spaces, utils
from gym.envs.classic_control import rendering

from gym.envs.classic_control.rendering import Transform
from gym.utils import seeding

from .node import Node 
from .utils import rotate_90_clockwise, rotate_90_counterclockwise



class SnakeAction(object):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class SnakeCellState(object):
    EMPTY = 0
    WALL = 1
    DOT = 2


class SnakeReward(object):
    ALIVE = -0.1
    DOT = 5
    DEAD = -100
    WON = 100

class SnakeCellDirection(object):
    """Type of the points on the game map."""
    HEAD_L = 100
    HEAD_U = 101
    HEAD_R = 102
    HEAD_D = 103
    BODY_LU = 104
    BODY_UR = 105
    BODY_RD = 106
    BODY_DL = 107
    BODY_HOR = 108
    BODY_VER = 109


class SnakeGame(object):

    transposition_table = {}
    def __init__(self, width, height, head):
        self.n_moves = 0 
        self.width = width
        self.height = height

        self.snake = deque()
        self.empty_cells = {(x, y) for x in range(width) for y in range(height)}
        self.dot = None
        self.init_direc = None
        # Track cell directions
        #{(x,y):SnakeCellDirection}
        self.cell_directions = {}

        self.prev_action = SnakeAction.UP

        # Temporal variable for holding action to be applied
        self.next_action = None

        self.add_to_head(head)

        self.cell_directions[head] = SnakeCellDirection.HEAD_U

        self.generate_dot()
    
    
    def copy(self):
        new_object = deepcopy(self)
        return new_object

        
    def move(self):
        """
        Create and returns starting_node,fruit_node
        """

        self.starting_node = Node(self.head())
        self.starting_node.action =self.prev_action
        
        self.fruit_node = Node(self.dot)

   
    def availabe_tiles_count(self):
        return self.width * self.height

    def _path_move_from_transposition_table(self, starting_node, fruit_node):
        path_from_transposition_table = self._path_from_transposition_table(fruit_node)
        if path_from_transposition_table:
            for index in range(0, len(path_from_transposition_table)):
                node = path_from_transposition_table[index]
                if node.point == starting_node.point:
                    destination_node = path_from_transposition_table[index - 1]
                    return destination_node.action
        self.transposition_table = {}
        return None
    

    def _path_from_transposition_table(self, key):
        try:
            return self.transposition_table[key]
        except KeyError:
            self.transposition_table = {}
            return []

    def possible_actions_for_current_action(self, current_action):

        actions = [SnakeAction.LEFT, SnakeAction.RIGHT, SnakeAction.UP, SnakeAction.DOWN]
        valid_actions = []
        reverse_action = invert_action(current_action)

        for action in actions:
            if action != reverse_action:
                valid_actions.append(action)
        return valid_actions

    def add_to_head(self, cell):
        self.snake.appendleft(cell)
        if cell in self.empty_cells:
            self.empty_cells.remove(cell)
        if self.dot == cell:
            self.dot = None

    def cell_state(self, cell):
        if cell in self.empty_cells:
            return SnakeCellState.EMPTY
        if cell == self.dot:
            return SnakeCellState.DOT
        return SnakeCellState.WALL

    def head(self):
        return self.snake[0]

    def remove_tail(self):
        tail = self.snake.pop()
        self.empty_cells.add(tail)

    def can_generate_dot(self):
        return len(self.empty_cells) > 0

    def generate_dot(self):

        self.dot =random.sample(self.empty_cells, 1)[0]
        self.empty_cells.remove(self.dot)


    # TODO: rewrite using inverse_action function
    def is_valid_action(self, action):
        if len(self.snake) == 1:
            return True
        
        
        """ inverse_prev_action = invert_action(self.prev_action)
        print("Action {}, inverse prev action {}".format(action, inverse_prev_action))
        return  action != inverse_prev_action"""

        horizontal_actions = [SnakeAction.LEFT, SnakeAction.RIGHT]
        vertical_actions = [SnakeAction.UP, SnakeAction.DOWN]

        if self.prev_action in horizontal_actions:
            return action in vertical_actions
        
        return action in horizontal_actions
          

    def next_head(self, action):
        current_head = self.head()
        head_vector = np.array(current_head)

        action_vector = np.array(action_to_vector(action))

        # Apply action
        new_head = head_vector + action_vector

        return tuple(new_head)

    def _new_types(self):
        old_head_type , new_head_type = None, None

        # New head type 
        if self.next_action == SnakeAction.LEFT:
            new_head_type = SnakeCellDirection.HEAD_L
        
        elif self.next_action == SnakeAction.UP:
            new_head_type = SnakeCellDirection.HEAD_U

        elif self.next_action == SnakeAction.RIGHT:
            new_head_type = SnakeCellDirection.HEAD_R
        
        elif self.next_action == SnakeAction.DOWN:
            new_head_type = SnakeCellDirection.HEAD_D
        
        # Old head type 

        if ((self.prev_action == SnakeAction.LEFT and self.next_action == SnakeAction.LEFT) or 
           (self.prev_action == SnakeAction.RIGHT and self.next_action == SnakeAction.RIGHT)):

           old_head_type = SnakeCellDirection.BODY_HOR

        elif ((self.prev_action == SnakeAction.UP and self.next_action == SnakeAction.UP) or 
           (self.prev_action == SnakeAction.DOWN and self.next_action == SnakeAction.DOWN)):

           old_head_type = SnakeCellDirection.BODY_VER
        
        elif ((self.prev_action == SnakeAction.RIGHT and self.next_action == SnakeAction.UP) or 
           (self.prev_action == SnakeAction.DOWN and self.next_action == SnakeAction.LEFT)):

           old_head_type = SnakeCellDirection.BODY_LU
        
        elif ((self.prev_action == SnakeAction.LEFT and self.next_action == SnakeAction.UP) or 
           (self.prev_action == SnakeAction.DOWN and self.next_action == SnakeAction.RIGHT)):

           old_head_type = SnakeCellDirection.BODY_UR

        elif ((self.prev_action == SnakeAction.LEFT and self.next_action == SnakeAction.DOWN) or 
           (self.prev_action == SnakeAction.UP and self.next_action == SnakeAction.RIGHT)):

           old_head_type = SnakeCellDirection.BODY_RD

        elif ((self.prev_action == SnakeAction.RIGHT and self.next_action == SnakeAction.DOWN) or 
           (self.prev_action == SnakeAction.UP and self.next_action == SnakeAction.LEFT)):

           old_head_type = SnakeCellDirection.BODY_DL
        
        return old_head_type, new_head_type


    def step(self, action):

        if not self.is_valid_action(action):
            #print("Forbidden action {}, previous action {} ".format(action,self.prev_action ) )
            action = self.prev_action
        self.prev_action = action

        old_head = self.head()

        # Update head position
        next_head = self.next_head(action)
        next_head_state = self.cell_state(next_head)
        
        self.next_action = action 

        old_head_type, new_head_type = self._new_types()

        # Update old head direction 
        self.cell_directions[old_head] = old_head_type

        # Add new head direction
        self.cell_directions[next_head] = new_head_type


        if next_head_state == SnakeCellState.WALL:
            return SnakeReward.DEAD

        self.add_to_head(next_head)
        self.n_moves+= 1

        if next_head_state == SnakeCellState.DOT:
            if self.can_generate_dot():
                self.generate_dot()
                return SnakeReward.DOT   

            return SnakeReward.WON
        
        self.remove_tail()
        return SnakeReward.ALIVE


class SnakeEnv(gym.Env):
    metadata= {'render.modes': ['human']}

    # TODO: define observation_space
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = None

        # Grid width and height in number of squares
        self.width = 40
        self.height = 40
        self.start = (10, 10)



        # Screen width and height in pixels
        self.width_px = self.height_px = 1000
        self.width_scaling_factor = self.width_px / self.width
        self.height_scaling_factor = self.height_px / self.height

        self.game = SnakeGame(self.width, self.height, self.start)
        self.viewer = None

        self._init_draw_params()


    def _init_draw_params(self):
        self.grid_pad_ratio = .08

        pad_ratio = self.grid_pad_ratio
        
        self._dx1 = pad_ratio * self.width_scaling_factor
        self._dx2 = (1 - pad_ratio) * self.width_scaling_factor + 1

        self._dy1 = pad_ratio * self.height_scaling_factor
        self._dy2 = (1 - pad_ratio) * self.height_scaling_factor + 1

       
    # TODO: define  info
    def step(self, action):
        reward = self.game.step(action)

        done = reward in [SnakeReward.DEAD, SnakeReward.WON]
        observation = self._get_obs()
        info = None

        return observation, reward, done, info

    def reset(self):
        self.game = SnakeGame(self.width, self.height, self.start)
       
        return self._get_obs()

   
    def compute_vertices(self,x,y,direction):
        """
        direction: int
            One of SnakeCellDirection constants
        """

        cell_direction = self.game.cell_directions[(x,y)]
       
        x,y = self.width_scaling_factor*x, self.height_scaling_factor*y

        two_polygons = False

        if cell_direction == SnakeCellDirection.HEAD_L:
            l, t, r, b = (x + self._dx1, y + self._dy1,
                          x + self.width_scaling_factor, y + self._dy2)
        
        elif cell_direction == SnakeCellDirection.HEAD_U:
            l, t, r, b = (x + self._dx1, y + self._dy1,
                          x + self._dx2, y + self.height_scaling_factor)

        elif cell_direction == SnakeCellDirection.HEAD_R:
            l, t, r, b =(x, y + self._dy1,
                         x + self._dx2, y + self._dy2)

        elif cell_direction == SnakeCellDirection.HEAD_D:
            l, t, r, b =(x + self._dx1, y,
                         x + self._dx2, y + self._dy2)

        elif cell_direction == SnakeCellDirection.BODY_LU:
            l, t, r, b =(x, y + self._dy1,
                         x + self._dx1, y + self._dy2)

            l2, t2, r2, b2 =(x + self._dx1, y,
                            x + self._dx2, y + self._dy2)

            two_polygons = True

        elif cell_direction == SnakeCellDirection.BODY_UR:
            l, t, r, b =(x + self._dx1, y,
                         x + self._dx2, y + self._dy2)

            l2, t2, r2, b2 =(x + self._dx1, y + self._dy1,
                             x + self.width_scaling_factor, y + self._dy2)
            
            two_polygons = True

        elif cell_direction == SnakeCellDirection.BODY_RD:
            l, t, r, b =(x + self._dx1, y + self._dy1,
                        x + self.width_scaling_factor, y + self._dy2)

            l2, t2, r2, b2 =(x + self._dx1, y + self._dy1,
                             x + self._dx2, y + self.height_scaling_factor)
            two_polygons = True

        elif cell_direction == SnakeCellDirection.BODY_DL:
            l, t, r, b =(x + self._dx1, y + self._dy1,
                         x + self._dx2, y + self.height_scaling_factor)

            l2, t2, r2, b2 =(x, y + self._dy1,
                             x + self._dx1, y + self._dy2)
            two_polygons = True

        elif cell_direction == SnakeCellDirection.BODY_HOR:
            l, t, r, b=(x, y + self._dy1,
                        x + self.width_scaling_factor, y + self._dy2)

            

        elif cell_direction == SnakeCellDirection.BODY_VER:
            l, t, r, b=(x + self._dx1, y,x + self._dx2,
                        y + self.height_scaling_factor)
        
            
        color_squares = []

        color_square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        color_squares.append(color_square)
        
        if two_polygons:
            color_square2 = rendering.FilledPolygon([(l2,b2), (l2,t2), (r2,t2), (r2,b2)])
            color_squares.append(color_square2)

            

        return color_squares


    def render(self, mode='human', close=False):
       
        width_scaling_factor = self.width_scaling_factor
        height_scaling_factor =self.height_scaling_factor

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.width_px, self.height_px)
            
        k = 0
        n  = len(self.game.snake)
        for x, y in self.game.snake:
            color_squares = self.compute_vertices(
                x,
                y,
                direction = self.game.cell_directions[(x,y)] 
                )

            # Coloring tail
            if k == n-1:
                
                random_color =tuple(np.random.choice(range(256), size=3) /255.0 )
                color_squares[0].set_color(*random_color)
                

            else:
                color_squares[0].set_color(0,1.0,0)
                if len(color_squares) > 1:
                    color_squares[1].set_color(0,1.0,0)
                
            #square = rendering.PolyLine([(l,b), (l,t), (r,t), (r,b)], close = True)
            #square.set_linewidth(4)  
          
            for color_square in color_squares:
                self.viewer.add_onetime(color_square)
                #self.viewer.add_onetime(square)
            k+= 1
            

        if self.game.dot:
            x, y = self.game.dot
            l, r, t, b = x*width_scaling_factor, (x+1)*width_scaling_factor, y*height_scaling_factor, (y+1)*height_scaling_factor
            square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            square.set_color(1,0,0)
            self.viewer.add_onetime(square)

        return self.viewer.render(return_rgb_array=mode=='rgb_array')


    def _get_obs(self):
        """
        return the game 
        """
        

        return self.game

    def close(self,):
        if self.viewer:
            self.viewer.close()
            self.viewer = None 

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
        return [seed]


def action_to_vector(action):
    """
        
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3
    """

    if action ==0:
        return (-1, 0)

    if action == 1:
        return ( 1, 0)

    if action == 2:
        return ( 0, 1)

    if action == 3:
        return (0, -1)
    return None



def vector_to_action(vector):
    """
    vector: tuple
    return action (int)
    """
    #LEFT
    if vector ==(-1, 0):
        return 0
    #RIGHT
    if vector ==( 1, 0):
        return 1
    #UP
    if vector ==( 0, 1) :
        return 2
    #DOWN
    if vector ==( 0, -1) :

        return 3
    return None 



def rotate_action_clockwise(action):
    """
    action: integer

    return tuple
    """

    t = action_to_vector(action)    

    v = rotate_90_clockwise(t)

    return vector_to_action( tuple(v))


def rotate_action_counter_clockwise(action):
    """
    action: integer
    return tuple
    """

    t = action_to_vector(action)    

    v = rotate_90_counterclockwise(t)
    print("rotate_action_counter_clockwise t {}, v {}".format(t,v))

    return vector_to_action( tuple(v))

def invert_action(action):
    """
    action: integer
    return action (integer)
    """

    t = action_to_vector(action)

    new_t = ()
    for i in range(len(t)):
        new_t+= (t[i] * -1,)
    
    int_action = vector_to_action(new_t)
    return int_action



   
if __name__ == "__main__":
    action = 0 
    print("Action {}, inverted action {} == {} (real action)".format(action, invert_action(action),1 ) )
    print("Action {}, double inversion {}".format(action, invert_action(invert_action(action))))


    action = 1
    print("Action {}, inverted action {} == {} (real action)".format(action, invert_action(action),0 ) )
    print("Action {}, double inversion {}".format(action, invert_action(invert_action(action))))



    action = 2
    print("Action {}, inverted action {} == {} (real action)".format(action, invert_action(action),3) )
    print("Action {}, double inversion {}".format(action, invert_action(invert_action(action))))


    action = 3
    print("Action {}, inverted action {} == {} (real action)".format(action, invert_action(action),2) )
    print("Action {}, double inversion {}".format(action, invert_action(invert_action(action))))


    action = 0
    print("Action {}, rotate left {} rotate right {}".format(action,rotate_action_counter_clockwise(action), rotate_action_clockwise(action)))


    for action in range(4):
        inverse = invert_action(action)
        print("Action {}, inverse action {}".format(action, inverse))

    game  = SnakeGame( width = 100, height = 100, head = (10,10))

    valid = game.is_valid_action(3)

    print("Is valid: ", valid)

    