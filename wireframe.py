import numpy as np

def unitVector((x,y,z)):
    return np.array([x,y,z])/np.linalg.norm(np.array([x,y,z]))

def translationMatrix(dx=0, dy=0, dz=0):
    """ Return matrix for translation along vector (dx, dy, dz). """

    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [dx,dy,dz,1]])

def translateAlongVectorMatrix(vector):
    """ Return matrix for translation along a vector for a given distance. """

    unit_vector = np.hstack([vector, 1])
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0], unit_vector])


def translateAlongVectorMatrix2(vector, distance):
    """ Return matrix for translation along a vector for a given distance. """

    unit_vector = np.hstack([unitVector(vector) * distance, 1])
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0], unit_vector])

def scaleMatrix(s, cx=0, cy=0, cz=0):
    """ Return matrix for scaling equally along all axes centred on the point (cx,cy,cz). """

    return np.array([[s,0,0,0],
                     [0,s,0,0],
                     [0,0,s,0],
                     [cx*(1-s), cy*(1-s), cz*(1-s), 1]])

def rotateXMatrix(radians):
    """ Return matrix for rotating about the x-axis by 'radians' radians """

    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1,0, 0,0],
                     [0,c,-s,0],
                     [0,s, c,0],
                     [0,0, 0,1]])

def rotateYMatrix(radians):
    """ Return matrix for rotating about the y-axis by 'radians' radians """

    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c,0,s,0],
                     [ 0,1,0,0],
                     [-s,0,c,0],
                     [ 0,0,0,1]])

def rotateZMatrix(radians):
    """ Return matrix for rotating about the z-axis by 'radians' radians """

    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s,0,0],
                     [s, c,0,0],
                     [0, 0,1,0],
                     [0, 0,0,1]])

def rotateAboutVector((cx,cy,cz), (x,y,z), radians):
    """ Rotate wireframe about given vector by 'radians' radians. """

    # Find angle and matrix needed to rotate vector about the z-axis such that its y-component is 0
    rotZ = np.arctan2(y, x)
    rotZ_matrix = rotateZMatrix(rotZ)

    # Find angle and matrix needed to rotate vector about the y-axis such that its x-component is 0
    (x, y, z, _) = np.dot(np.array([x,y,z,1]), rotZ_matrix)
    rotY = np.arctan2(x, z)

    matrix = translationMatrix(dx=-cx, dy=-cy, dz=-cz)
    matrix = np.dot(matrix, rotZ_matrix)
    matrix = np.dot(matrix, rotateYMatrix(rotY))
    matrix = np.dot(matrix, rotateZMatrix(radians))
    matrix = np.dot(matrix, rotateYMatrix(-rotY))
    matrix = np.dot(matrix, rotateZMatrix(-rotZ))
    matrix = np.dot(matrix, translationMatrix(dx=cx, dy=cy, dz=cz))

    return matrix

class Node(np.ndarray):
    def __new__(subtype,wf, l, dtype=float, buffer=None, offset=0,strides=None, order=None, shape=None):
        obj = super(Node, subtype).__new__(subtype, (3,), dtype,buffer, offset, strides,order)
        obj.id = wf.IDITERATOR
        wf.IDITERATOR += 1
        obj.name = ''
        obj[0] = l[0]
        obj[1] = l[1]
        obj[2] = l[2]
        obj.x = l[0]
        obj.y = l[1]
        obj.z = l[2]
        obj.colour = (255,255,255)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)
        self.z = getattr(obj, 'z', None)

def uniquePair(x,y):
    if x < y:
        return y*y+x
    else:
        return x*x+y

class Edge:
    def __init__(self,wf, n1, n2):
        self.id = wf.IDITERATOR
        self.internal_id = uniquePair(n1,n2)
        wf.IDITERATOR += 1
        self.nodes = (n1,n2)
        self.start = n1
        self.stop  = n2

class Face:
    def __init__(self,wf, node_list, colour):
        self.id = wf.IDITERATOR
        wf.IDITERATOR += 1
        self.nodes = node_list
        #v1 = (wf.nodes[self.nodes[1]] - wf.nodes[self.nodes[0]])
        #v2 = (wf.nodes[self.nodes[2]] - wf.nodes[self.nodes[0]])
        #self.normal = np.cross(v1, v2)
        self.colour = colour

class Wireframe:
    """ An array of vectors in R3 and list of edges connecting them. """

    def __init__(self, nodes=None):
        self.IDITERATOR = 0
        self.nodes = []
        self.edges = []
        self.faces = []
        self.center = np.array([0,0,0])
        self.os = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
        if nodes:
            self.addNodes(nodes)

    def addNodes(self, node_array):
        """ Append 1s to a list of 3-tuples and add to self.nodes. """
        for n in node_array:
            self.nodes.append(Node(self,n))
        #ones_added = np.hstack((node_array, np.ones((len(node_array),1))))
        #self.nodes = np.vstack((self.nodes, node_array))

    def addEdges(self, edge_list):
        """ Add edges as a list of 2-tuples. """

        # Is it better to use a for loop or generate a long list then add it?
        # Should raise exception if edge value > len(self.nodes)
        self.edges += [edge for edge in edge_list if not any(e.internal_id == edge.internal_id for e in self.edges)]

    def addFaces(self, face_list, face_colour=(255,255,255)):
        for node_list in face_list:
            #self.faces.append([self.nodes[node] for node in node_list])
            self.faces.append(Face(self,node_list, np.array(face_colour, np.uint8)))
            #self.addEdges([Edge(self,node_list[n-1], node_list[n]) for n in range(len(node_list))])
            for n in range(len(node_list)):
                #if not any( == e.internal_id for e in self.edges):
                self.edges.append(Edge(self,node_list[n-1], node_list[n]))


    def output(self):
        if len(self.nodes) > 1:
            self.outputNodes()
        if self.edges:
            self.outputEdges()
        if self.faces:
            self.outputFaces()

    def outputNodes(self):
        print "\n --- Nodes --- "
        for i, (x, y, z, _) in enumerate(self.nodes):
            print "   %d: (%d, %d, %d)" % (i, x, y, z)

    def outputEdges(self):
        print "\n --- Edges --- "
        for i, (node1, node2) in enumerate(self.edges):
            print "   %d: %d -> %d" % (i, node1, node2)

    def outputFaces(self):
        print "\n --- Faces --- "
        for i, nodes in enumerate(self.faces):
            print "   %d: (%s)" % (i, ", ".join(['%d' % n for n in nodes]))

    def transform(self, transformation_matrix):
        """ Apply a transformation defined by a transformation matrix. """
        for node in self.nodes:
            self.os = np.matmul(self.os,transformation_matrix)
            new_node = np.dot(np.array([node[0],node[1],node[2],1]),transformation_matrix)[:3]
            node[0] = new_node[0]
            node[1] = new_node[1]
            node[2] = new_node[2]

    def findCentre(self):
        """ Find the spatial centre by finding the range of the x, y and z coordinates. """

        min_values = self.nodes[:,:].min(axis=0)
        max_values = self.nodes[:,:].max(axis=0)
        return 0.5*(min_values + max_values)

    def sortedFaces(self):
        return sorted(self.faces, key=lambda face: min(self.nodes[f][2] for f in face.nodes))

    def update(self):
        """ Override this function to control wireframe behaviour. """
        pass

class WireframeGroup:
    """ A dictionary of wireframes and methods to manipulate them all together. """

    def __init__(self):
        self.wireframes = {}

    def addWireframe(self, name, wireframe):
        self.wireframes[name] = wireframe

    def output(self):
        for name, wireframe in self.wireframes.items():
            print name
            wireframe.output()

    def outputNodes(self):
        for name, wireframe in self.wireframes.items():
            print name
            wireframe.outputNodes()

    def outputEdges(self):
        for name, wireframe in self.wireframes.items():
            print name
            wireframe.outputEdges()

    def findCentre(self):
        """ Find the central point of all the wireframes. """

        # There may be a more efficient way to find the minimums for a group of wireframes
        min_values = np.array([np.array(wireframe.nodes[:]).min(axis=0) for wireframe in self.wireframes.values()]).min(axis=0)
        max_values = np.array([np.array(wireframe.nodes[:]).max(axis=0) for wireframe in self.wireframes.values()]).max(axis=0)
        return 0.5*(min_values + max_values)

    def transform(self, matrix):
        for wireframe in self.wireframes.values():
            wireframe.transform(matrix)

    def update(self):
        for wireframe in self.wireframes.values():
            wireframe.update()
