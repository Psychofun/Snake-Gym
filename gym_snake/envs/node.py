class Node:

   

    def __init__(self, point):
        # Points must be passed as tuples (x,y)
        self.point = point
        self.previous_node = None
        # Actions must be passed integers
        self.action = None

    def _recreate_path_for_node(self):
        nodes = [self]
        prev_node = self.previous_node

        while prev_node:
            nodes.extend([prev_node])
            prev_node = prev_node.previous_node
        
        return nodes

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash(str(self.point[0])+str(self.point[1]))