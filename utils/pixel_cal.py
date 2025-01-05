import numpy as np

center = np.array([418,350])
p1 = np.array([474,352])
p2 = np.array([362,351])
p3 = np.array([420, 409])
p4 = np.array([415, 297])

std = 7
def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

def main():

    d1 = distance(center,p1)
    d2 = distance(center,p2)
    d3 = distance(center,p3)
    d4 = distance(center,p4)
    print(d1,d2,d3,d4)
    mean = np.mean([d1,d2,d3,d4])
    print("mean dist:",mean)
    
    r = std/mean
    print("the r is:",r)
    
if __name__ == "__main__":
    main()