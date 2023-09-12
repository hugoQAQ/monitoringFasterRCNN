from copy import deepcopy

class Box:
    def __init__(self):
        self.dimensions = None
        self.ivals = []
        self.element_indexes = [] # record this box is built for what samples
        self.low_bound_indexes = dict() # record which samples visit the low bound for each dimension
        self.high_bound_indexes = dict() # record which samples visit the low bound for each dimension

    def build(self, dimensions, points):
        # a point is a tuple (index, n-dim numpy)
        # index = point[0]
        # value = point[1]
        piter = iter(points)
        self.dimensions = dimensions
        self.ivals = []
        self.element_indexes = []
        self.low_bound_indexes = dict()
        self.high_bound_indexes = dict()

        try:
            point = next(piter)
        except StopIteration:
            return
        else:
            self.element_indexes.append(point[0]) # update index list
            i = 0
            for coord in point[1]:
                if(i >= self.dimensions):
                    break
                self.ivals.append([coord, coord])
                self.low_bound_indexes["n"+str(i+1)] = [point[0]] # update low bound visiting index list
                self.high_bound_indexes["n"+str(i+1)] = [point[0]] # update upper bound visiting index list
                i += 1
            if(len(self.ivals) != self.dimensions):
                raise "IllegalArgument"

        while True:
            try:
                point = next(piter)
            except StopIteration:
                break
            else:
                self.element_indexes.append(point[0]) # update index list
                i = 0
                for coord in point[1]:
                    if(i >= self.dimensions):
                        break
                    ival = self.ivals[i]
                    if(coord < ival[0]):
                        ival[0] = coord
                        self.low_bound_indexes["n"+str(i+1)] = [point[0]] # update the bound and its index
                    elif(coord == ival[0]):
                        low_index_list = self.low_bound_indexes["n"+str(i+1)]
                        low_index_list.append(point[0])

                    if(coord > ival[1]):
                        ival[1] = coord
                        self.high_bound_indexes["n"+str(i+1)] = [point[0]] # update the bound and its index
                    elif(coord == ival[1]):
                        high_index_list = self.high_bound_indexes["n"+str(i+1)]
                        high_index_list.append(point[0])
                    i += 1

    def query(self, point):
        i = 0
        for coord in point:
            if(i >= self.dimensions):
                break
            ival = self.ivals[i]
            if(coord < ival[0] or coord > ival[1]):
                return False
            i += 1
        return True

    def __str__(self):
        return self.ivals.__str__()
    
    def query_delta(self, point, delta):
        i = 0
        for coord in point:
            if(i >= self.dimensions):
                break
            ival = self.ivals[i]
            if(coord < ival[0]*(1+delta) or coord > ival[1]*(1+delta)):
                return False
            i += 1
        return True


def boxes_query(point, boxes):
    for box in boxes:
        if len(box.ivals):
            if box.query(point):
                return True
    return False

def boxes_query_delta(point, boxes, delta):
    for box in boxes:
        if len(box.ivals):
            if box.query_delta(point, delta):
                return True
    return False


                
