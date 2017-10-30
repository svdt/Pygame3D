import pygame, math
import numpy as np
import wireframe as wf
import time

# Radian rotated by a key event
ROTATION_AMOUNT = np.pi/16
MOVEMENT_AMOUNT = 10

key_to_function = {
    pygame.K_LEFT:   (lambda x: x.transform(wf.translationMatrix(dx=-MOVEMENT_AMOUNT))),
    pygame.K_RIGHT:  (lambda x: x.transform(wf.translationMatrix(dx= MOVEMENT_AMOUNT))),
    pygame.K_UP:     (lambda x: x.transform(wf.translationMatrix(dy=-MOVEMENT_AMOUNT))),
    pygame.K_DOWN:   (lambda x: x.transform(wf.translationMatrix(dy= MOVEMENT_AMOUNT))),
    pygame.K_EQUALS: (lambda x: x.scale(1.25)),
    pygame.K_MINUS:  (lambda x: x.scale(0.8)),
    pygame.K_q:      (lambda x: x.rotate('x', ROTATION_AMOUNT)),
    pygame.K_w:      (lambda x: x.rotate('x',-ROTATION_AMOUNT)),
    pygame.K_a:      (lambda x: x.rotate('y', ROTATION_AMOUNT)),
    pygame.K_s:      (lambda x: x.rotate('y',-ROTATION_AMOUNT)),
    pygame.K_z:      (lambda x: x.rotate('z', ROTATION_AMOUNT)),
    pygame.K_x:      (lambda x: x.rotate('z',-ROTATION_AMOUNT))
}

light_movement = {
    pygame.K_q:      (lambda x: x.transform(wf.rotateXMatrix(-ROTATION_AMOUNT))),
    pygame.K_w:      (lambda x: x.transform(wf.rotateXMatrix( ROTATION_AMOUNT))),
    pygame.K_a:      (lambda x: x.transform(wf.rotateYMatrix(-ROTATION_AMOUNT))),
    pygame.K_s:      (lambda x: x.transform(wf.rotateYMatrix( ROTATION_AMOUNT))),
    pygame.K_z:      (lambda x: x.transform(wf.rotateZMatrix(-ROTATION_AMOUNT))),
    pygame.K_x:      (lambda x: x.transform(wf.rotateZMatrix( ROTATION_AMOUNT)))
}

class WireframeViewer(wf.WireframeGroup):
    """ A group of wireframes which can be displayed on a Pygame screen """

    def __init__(self, width, height, name="Wireframe Viewer"):
        self.width = width
        self.height = height
        self.center = np.array([0,0,0])
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(name)

        self.wireframes = {}
        self.wireframe_colours = {}
        self.object_to_update = []

        self.displayNodes = True
        self.displayEdges = True
        self.displayFaces = True

        self.perspective = False#300.
        self.eyeX = self.width/2
        self.eyeY = 100
        self.view_vector = np.array([0, 0, -1])

        self.light = wf.Wireframe()
        self.light.addNodes([[0, -1, 0]])

        self.min_light = 0.02
        self.max_light = 0.5
        self.light_range = self.max_light - self.min_light

        self.background = (10,10,50)
        self.nodeColour = (250,250,250)
        self.edgeColour = (237, 235, 137)
        self.nodeRadius = 2

        self.control = 0

    def addWireframe(self, name, wireframe):
        self.wireframes[name] = wireframe
        #   If colour is set to None, then wireframe is not displayed
        self.wireframe_colours[name] = (250,250,250)

    def addWireframeGroup(self, wireframe_group):
        # Potential danger of overwriting names
        for name, wireframe in wireframe_group.wireframes.items():
            self.addWireframe(name, wireframe)

    def scale(self, scale):
        """ Scale wireframes in all directions from the centre of the group. """

        scale_matrix = wf.scaleMatrix(scale, self.width/2, self.height/2, 0)
        self.transform(scale_matrix)

    def rotate(self, axis, amount):
        #(x, y, z) = self.findCentre()
        (x, y, z) = self.wireframes.values()[0].os[3,:-1]
        translation_matrix1 = wf.translationMatrix(-x, -y, -z)
        translation_matrix2 = wf.translationMatrix(x, y, z)

        if axis == 'x':
            rotation_matrix = wf.rotateXMatrix(amount)
        elif axis == 'y':
            rotation_matrix = wf.rotateYMatrix(amount)
        elif axis == 'z':
            rotation_matrix = wf.rotateZMatrix(amount)

        rotation_matrix = np.dot(np.dot(translation_matrix1, rotation_matrix), translation_matrix2)
        self.transform(rotation_matrix)

    def display(self):
        self.screen.fill(self.background)
        light = self.light.nodes[0][:3]
        spectral_highlight = light + self.view_vector
        spectral_highlight /= np.linalg.norm(spectral_highlight)

        for name, wireframe in self.wireframes.items():
            nodes = wireframe.nodes

            if self.displayFaces:
                for face in wireframe.sortedFaces():
                    colour = face.colour
                    face = face.nodes
                    v1 = (nodes[face[1]] - nodes[face[0]])
                    v2 = (nodes[face[2]] - nodes[face[0]])

                    normal = np.cross(v1, v2)
                    towards_us = np.dot(normal, self.view_vector)
                    #print normal
                    # Only draw faces that face us
                    if towards_us > 0:
                        normal /= np.linalg.norm(normal)
                        theta = np.dot(normal, light)
                        #catchlight_face = np.dot(normal, spectral_highlight) ** 25

                        c = 0
                        if theta < 0:
                            shade = self.min_light * colour
                        else:
                            shade = (theta * self.light_range + self.min_light) * colour

                        pygame.draw.polygon(self.screen, shade, [(nodes[node][0], nodes[node][1]) for node in face], 0)

                        #mean_x = sum(nodes[node][0] for node in face) / len(face)
                        #mean_y = sum(nodes[node][1] for node in face) / len(face)
                        #pygame.draw.aaline(self.screen, (255,255,255), (mean_x, mean_y), (mean_x+25*normal[0], mean_y+25*normal[1]), 1)

                if self.displayEdges:
                    for edge in wireframe.edges:
                        (n1, n2) = edge.nodes
                        if self.perspective:
                            if wireframe.nodes[n1][2] > -self.perspective and nodes[n2][2] > -self.perspective:
                                z1 = self.perspective/ (self.perspective + nodes[n1][2])
                                x1 = self.width/2  + z1*(nodes[n1][0] - self.width/2)
                                y1 = self.height/2 + z1*(nodes[n1][1] - self.height/2)

                                z2 = self.perspective/ (self.perspective + nodes[n2][2])
                                x2 = self.width/2  + z2*(nodes[n2][0] - self.width/2)
                                y2 = self.height/2 + z2*(nodes[n2][1] - self.height/2)

                                pygame.draw.aaline(self.screen, self.edgeColour, (x1, y1), (x2, y2), 1)
                        else:
                            pygame.draw.aaline(self.screen, self.edgeColour, (nodes[n1][0], nodes[n1][1]), (nodes[n2][0], nodes[n2][1]), 1)

            if self.displayNodes:
                for node in nodes:
                    pygame.draw.circle(self.screen, node.colour, (int(node[0]), int(node[1])), self.nodeRadius, 0)
                    myfont = pygame.font.Font(None, 15)
                    label = myfont.render(node.name, 1, (255,255,255))
                    self.screen.blit(label, (int(node[0]), int(node[1])))
        pygame.display.flip()

    def keyEvent(self, key):
        if key in key_to_function:
            key_to_function[key](self)
            #light_movement[key](self.light)

    def mouseRotate(self):
        rotDir = pygame.mouse.get_rel()
        amount = math.radians(math.sqrt(rotDir[0]*rotDir[0]+rotDir[1]*rotDir[1]))

        #rotation_matrix = wf.rotateAboutVector(self.wireframes.values()[0].os[3,:-1], (rotDir[1],-rotDir[0],0), amount)
        rotation_matrix = wf.rotateAboutVector(self.findCentre(), (rotDir[1],-rotDir[0],0), amount)
        self.transform(rotation_matrix)

    def mouseTranslate(self):
        rotDir = pygame.mouse.get_rel()
        amount = math.sqrt(rotDir[0]*rotDir[0]+rotDir[1]*rotDir[1])
        translation_matrix = wf.translateAlongVectorMatrix((rotDir[0], rotDir[1], 0))
        self.transform(translation_matrix)


    def mouseScale(self,amount):
        self.scale(amount)

    def run(self):
        """ Display wireframe on screen and respond to keydown events """

        running = True
        mousetrig = False
        key_down = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    key_down = event.key
                elif event.type == pygame.KEYUP:
                    key_down = None
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.mouseScale(0.9)
                    elif event.button == 5:
                        self.mouseScale(1.1)
                    else:
                        mousetrig = True
                        pygame.mouse.get_rel()
                elif event.type == pygame.MOUSEBUTTONUP:
                    mousetrig = False
                elif mousetrig and pygame.mouse.get_pressed()[0] and event.type == pygame.MOUSEMOTION:
                    self.mouseRotate()
                elif mousetrig and pygame.mouse.get_pressed()[2] and event.type == pygame.MOUSEMOTION:
                    self.mouseTranslate()


            if key_down:
                self.keyEvent(key_down)

            self.display()
            self.update()

        pygame.quit()
