import numpy as np



def get_rotation_matrix(angle):


    thetha = (angle/180.) * np.pi
    rotation_matrix = np.array([
                            [np.cos(thetha), -np.sin(thetha)],
                            [np.sin(thetha), np.cos(thetha) ]
                            ])
    return rotation_matrix



def rotate_90_clockwise(vector):
    """
    vector:tuple
    """
    v = np.array(vector,dtype= np.float)
   
    m = get_rotation_matrix(-90)

    r = np.matmul(m,v)

    r = r.astype('int') 

    return r



def rotate_90_counterclockwise(vector):
    """
    vector:tuple
    """
    v = np.array(vector)
   
    m = get_rotation_matrix(90)

    r = np.matmul(m,v)
    r = r.astype('int') 
    return r




if __name__ == "__main__":
    l  = (-1,0)
    up  = (0,1)
    r = (1,0)
    down = (0,-1)



    r0 = rotate_90_clockwise(l)

    print("Rotate clockwise l {} must give up {}, r={} ".format(l,up, r0))


    r0 = rotate_90_clockwise(up)

    print("Rotate clockwise up {} must give r {}, r={} ".format(up,r, r0))


    r0 = rotate_90_clockwise(r)

    print("Rotate clockwise r {} must give down {}, r={} ".format(r,down, r0))


    r0 = rotate_90_clockwise(down)

    print("Rotate clockwise down {} must give l {}, r={} ".format(down,l, r0))

    print("\n\n")
    ### COUNTERCLOCKWISE

    r0 = rotate_90_counterclockwise(l)

    print("Rotate counterclockwise l {} must give down {}, r={} ".format(l,down, r0))

    
    r0 = rotate_90_counterclockwise(up)

    print("Rotate clockwise up {} must give l {}, r={} ".format(up,l, r0))


    r0 = rotate_90_counterclockwise(r)

    print("Rotate clockwise r {} must give up {}, r={} ".format(r,up, r0))


    r0 = rotate_90_counterclockwise(down)

    print("Rotate clockwise down {} must give r {}, r={} ".format(down,r, r0))



    