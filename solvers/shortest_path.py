import sys
sys.path.append("..")

from gym_snake.envs.node import Node
from gym_snake.envs.snake_env import action_to_vector
from gym_snake.envs.snake_env import SnakeAction
from gym_snake.envs.snake_env import SnakeCellState
from gym_snake.envs.snake_env import rotate_action_clockwise
from gym_snake.envs.snake_env import rotate_action_counter_clockwise
from gym_snake.envs.snake_env import  invert_action
from gym_snake.queue import Queue





class ShortestPathBFSSolver():


    def __init__(self):
        pass
    
    def move(self, environment):
        self.environment = environment.copy()
        self.environment.move()

        shortest_path_move_from_transposition_table = self.environment._path_move_from_transposition_table(self.environment.starting_node, self.environment.fruit_node)

        if shortest_path_move_from_transposition_table:
            #print(" shortest_path_move_from_transposition_table: ",  shortest_path_move_from_transposition_table)
            return shortest_path_move_from_transposition_table

        shortest_path = self.shortest_path(self.environment, self.environment.starting_node, self.environment.fruit_node)

       
        if shortest_path:
            #print("Shortest path: ", [x.action for x in shortest_path])
            
            self.environment.transposition_table[self.environment.fruit_node] = shortest_path
            first_point = shortest_path[-2]
            return first_point.action
        
        #print("prev action: ", self.environment.prev_action)
        return self.environment.prev_action



    def shortest_path(self, environment, start, end):
        queue = Queue([start])
        visited_nodes = set([start])
        shortest_path = []
        while queue.queue:
            current_node = queue.dequeue()
            if current_node == end:
                shortest_path = current_node._recreate_path_for_node()
                break
            for action in environment.possible_actions_for_current_action(current_node.action):
                # Convert action (int) to tuple 
                a_vector = action_to_vector(action)
                # Apply action to point
                neighbor = (current_node.point[0] + a_vector[0], current_node.point[1] + a_vector[1])
                neighbor_state = environment.cell_state(neighbor)

                if (neighbor_state == SnakeCellState.EMPTY or
                    neighbor_state == SnakeCellState.DOT
                   ):
                
                    child_node = Node(neighbor)
                    child_node.action = action
                    child_node.previous_node = current_node
                    if child_node not in visited_nodes and child_node not in queue.queue:
                        visited_nodes.add(current_node)
                        queue.enqueue(child_node)
        if shortest_path:
            return shortest_path
        else:
            return []
