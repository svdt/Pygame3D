import wireframeDisplay as wd
import wireframe as wf
import basicShapes as shape
import time


if __name__ == '__main__':
    pv = wd.WireframeViewer(1200, 600)
    s = time.time()
    toroid = shape.Toroid((400,200,100), 50,100, 30)
    sphere = shape.Spheroid((100,100,100), (10,10,10),30)
    print time.time()-s
    #cube = wf.Wireframe()
    #cube.addNodes([(x,y,z) for x in (50,250) for y in (50,250) for z in (50,250)])
    #cube.addEdges([(n,n+4) for n in range(0,4)]+[(n,n+1) for n in range(0,8,2)]+[(n,n+2) for n in (0,1,4,5)])
    print len(toroid.nodes)
    pv.addWireframe('torus', toroid)
    pv.addWireframe('shpere', sphere)
    pv.run()
