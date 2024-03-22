#-*- coding:Latin-1 -*-
#objectif de la s�ance afficher des triangles pleins et fils de fer
from math import cos, sin
from operator import pos
import sys
from turtle import window_height, window_width
import numpy.matlib 
import numpy as np
from vispy import gloo, app
from vispy.app import MouseEvent,KeyEvent
from vispy.util import keys
from vispy.gloo import Program, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
sys.path.insert(1, 'Z:/BUT3/sae/PythonApplication1/pyKinectAzure-master')
import pykinect_azure as pykinect

import matplotlib.image as mpimage
#load textures
img1 = mpimage.imread('Z:/BUT3/sae/chat1.jpg')

#img2 = mpimage.imread('Z:/kinect/chat2.jfif')
#img3 = mpimage.imread('Z:/kinect/chat3.jpg')
#img4 = mpimage.imread('Z:/kinect/chat4.jfif')
#img5 = mpimage.imread('Z:/kinect/chat5.jpg')
#img6 = mpimage.imread('Z:/kinect/chat6.jfif')


#dessin de primitives
class Triangle:
    'cr�ation d''un triangle avec couleur'
    def __init__(self,x1,y1,z1,x2,y2,z2,x3,y3,z3,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)] # les vertex
        self.program['color'] = [(1,0,0,1),(0,1,0,1),(0,0,1,1)]; #la couleur de chaque vertex
        self.triangle = IndexBuffer([0,1,2]); # la topologie: ordre des vertex pour le dessin
    def draw(self):
        self.program.draw('triangles',self.triangle)

class textcolorCube:
    'cr�ation d''une cube avec couleur'
    def __init__(self,x,y,z,r,g,b,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [(-1.*x,1*y,1*z),(-1.*x,-1.*y,1*z),(1*x,-1.*y,1*z),(1*x,1*y,1*z),(-1.*x,1*y,-1.*z),(-1.*x,-1.*y,-1.*z),(1*x,-1.*y,-1.*z),(1*x,1*y,-1.*z)] # les vertex
        self.program['color'] = [(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1)]; #la couleur de chaque vertex
        self.triangle = IndexBuffer([[0,1,2,0,2,3,4,5,6,4,6,7,4,5,1,4,1,0,7,6,2,7,2,3,4,0,7,7,0,3,5,1,6,6,1,2]]); # la topologie: ordre des vertex pour le dessin
    def draw(self):
        self.program.draw('triangles',self.triangle)

class TriangleWireFrame:
    'cr�ation d''un triangle en fils de fer color�s'
    def __init__(self,x1,y1,z1,x2,y2,z2,x3,y3,z3,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)] # les vertex
        self.program['color'] = [(1,0,0,1),(0,1,0,1),(0,0,1,1)]; #la couleur de chaque vertex
        self.triangle = IndexBuffer([[0,1],[1,2],[2,0]]); # la topologie: ordre des vertex pour le dessin
    def draw(self):
        self.program.draw('lines',self.triangle)

class Cube_Filaire:
    'cr�ation d''une cube en fils de fer color�s'
    def __init__(self,x,y,z,r,g,b,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [(-1.*x,1*y,1*z),(-1.*x,-1.*y,1*z),(1*x,-1.*y,1*z),(1*x,1*y,1*z),(-1.*x,1*y,-1.*z),(-1.*x,-1.*y,-1.*z),(1*x,-1.*y,-1.*z),(1*x,1*y,-1.*z)]
        #self.program['position'] = [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4),(x5,y5,z5),(x6,y6,z6),(x7,y7,z7),(x8,y8,z8)] # les vertex
        self.program['color'] = [(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1),(r,g,b,1)]; #la couleur de chaque vertex
        self.triangle = IndexBuffer([[0,1],[1,2],[2,0],[0,2],[2,3],[3,0],[4,5],[5,6],[6,4],[4,6],[6,7],[7,4],[4,5],[5,1],[1,4],[4,1],[1,0],[0,4],[7,6],[6,2],[2,7],[7,2],[2,3],[3,7],[4,0],[0,3],[3,4],[4,3],[3,7],[7,4],[5,1],[1,2],[2,5],[5,2],[2,6],[6,5]]); # la topologie: ordre des vertex pour le dessin
    def draw(self):
        self.program.draw('lines',self.triangle)

    '''def __init__(self,x4,y4,z4,x5,y5,z5,x6,y6,z6,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [(x4,y4,z4),(x5,y5,z5),(x6,y6,z6)] # les vertex
        self.program['color'] = [(0,1,0,1),(0,1,0,1),(0,1,0,1)]; #la couleur de chaque vertex
        self.triangle = IndexBuffer([[0,2],[2,3],[3,0]]); # la topologie: ordre des vertex pour le dessin
    def draw(self):
        self.program.draw('lines',self.triangle)'''

class Line:
    'cr�ation d''une ligne en fils de fer color�s'
    def __init__(self,x1,y1,z1,x2,y2,z2,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [((x1,y1,z1),(x2,y2,z2))] # les vertex
        self.line = IndexBuffer([[0,1]]); # la topologie: ordre des vertex pour le dessin
    def draw(self):
        self.program.draw('lines',self.line)

class CubeTextureNew:
    'cr�ation d''une cube avec couleur'
    def __init__(self,x,y,z,texture,program):
        self.program = program #le programme que l'on va utiliser avec ces shaders
        self.position1= [(-1.*x,1*y,1*z),(-1.*x,-1.*y,1*z),(1*x,-1.*y,1*z),(1*x,1*y,1*z)]
        self.position2= [(1*x,1*y,-1.*z),(1*x,-1.*y,-1.*z),(-1.*x,-1.*y,-1.*z),(-1.*x,1*y,-1.*z)]
        self.position3= [(-1.*x,1*y,-1.*z),(-1.*x,-1.*y,-1.*z),(-1.*x,-1.*y,1*z),(-1.*x,1*y,1*z)]
        self.position4= [(1*x,1*y,1*z),(1*x,-1.*y,1*z),(1*x,-1.*y,-1.*z),(1*x,1*y,-1.*z)]
        self.position5= [(-1.*x,1*y,-1.*z),(-1.*x,1*y,1*z),(1*x,1*y,1*z),(1*x,1*y,-1.*z)]
        self.position6= [(-1.*x,-1.*y,1*z),(-1.*x,-1.*y,-1.*z),(1*x,-1.*y,-1.*z),(1*x,-1.*y,1*z)]
        self.program['texcoord']  = [(0,0),(0,1),(1,1),(1,0)]
        self.face = IndexBuffer([[0,1,2,0,2,3]])

    def draw(self):
        self.program['texture'] = img1
        self.program['position'] = self.position1
        self.program.draw('triangles',self.face)
        self.program['texture'] = img2
        self.program['position'] = self.position2
        self.program.draw('triangles',self.face)
        self.program['texture'] = img3
        self.program['position'] = self.position3
        self.program.draw('triangles',self.face)
        self.program['texture'] = img4
        self.program['position'] = self.position4
        self.program.draw('triangles',self.face)
        self.program['texture'] = img5
        self.program['position'] = self.position5
        self.program.draw('triangles',self.face)
        self.program['texture'] = img6
        self.program['position'] = self.position6
        self.program.draw('triangles',self.face) 

class Cylindre:
    'cr�ation dun cylindre'
    def __init__(self,program):
        self.program = program #le programme que l'on va utiliser avec ces shaders
   
        
    def draw(self,R,h, nb_pas):
        self.program['texture'] = img1
        
        dtheta = 2*np.pi/nb_pas
        theta=0.0

        for i in range(0,nb_pas,1):
         #for i in range(0,nb_pas,1):
         #self.program['texcoord']  = [(1,1),(1,0),(0,1),(0,0)]
            x12 = i / nb_pas
            x34 = (i + 1) / nb_pas
            
            self.program['texcoord']  = [(x12,1),(x12,0),(x34,1),(x34,0)]
            x1 = R * cos(theta)
            y1 = R * sin(theta)
            z1 = -h

            x2 = R * cos(theta)
            y2 = R * sin(theta)
            z2 = h

            x3 = R * cos(theta+dtheta)
            y3 = R * sin(theta+dtheta)
            z3 = -h
        
            x4 = R * cos(theta+dtheta)
            y4 = R * sin(theta+dtheta)
            z4 = h
            self.program['position'] = [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4)]
            self.face = IndexBuffer([[1,0,3,3,2,0]])
            self.program.draw('triangles',self.face)
            theta = theta + dtheta
            

class Sphere:
    'cr�ation dune sphere'
    def __init__(self,program):
        self.program = program #le programme que l'on va utiliser avec ces shaders

    def draw(self,R, nb_pas):
        self.program['texture'] = img1

        dtheta = 2*np.pi/nb_pas
        theta=0.0

        dphi = np.pi/nb_pas
        phi=0.0

        #equation pour la sphere

        for i in range(0,nb_pas,1):
            #y02 = i / nb_pas
           #y13 = (i + 1) / nb_pas
            y02 =(nb_pas-i)/nb_pas
            y13 = (nb_pas -(i + 1) )/nb_pas
            for j in range(0,nb_pas,1):
                x12 = j / nb_pas
                x34 = (j + 1) / nb_pas
                
                #self.program['texcoord']  = [(x12,y02),(x12,y13),(x34,y02),(x34,y13)]
                self.program['texcoord']  = [(x12,y02),(x12,y13),(x34,y02),(x34,y13)]
                x1 = R * sin(phi)*cos(theta)
                y1 = R * sin(phi)*sin(theta)
                z1 = R * cos(phi)

                x2 = R * sin(phi + dphi) * cos(theta)
                y2 = R * sin(phi + dphi) * sin(theta)
                z2 = R * cos(phi + dphi)

                x3 = R * sin(phi) * cos(theta+dtheta)
                y3 = R * sin(phi) * sin(theta+dtheta)
                z3 = R * cos(phi)
        
                x4 = R * sin(phi+dphi) * cos(theta + dtheta)
                y4 = R * sin(phi+dphi) * sin(theta + dtheta)
                z4 = R * cos(phi + dphi)

                self.program['position'] = [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4)]
                self.face = IndexBuffer([[0,1,2,2,3,1]])
                self.program.draw('triangles',self.face)
                theta = theta + dtheta               
            phi = phi + dphi
            
            

#theta entre 0 et 2pi, phi entre 0 et pi
    

#class CylindreTexture:
#    'cr�ation dun cylindre'
#    def __init__(self,x,y,z,texture,program):

#vertex shader------------------------------
vertexColor = """
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

attribute vec3 position;
attribute vec4 color;

varying vec4 v_color;
void main()
{
    gl_Position = projection * view * model * vec4(position,1.0);
    v_color = color;
}
"""
#fragment shader---------------------------------------------------------------
fragmentColor = """

varying vec4 v_color;
void main()
{
    gl_FragColor =v_color;
} """

##-------------------
##-------------------
vertexTexture = """
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform sampler2D texture;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;

varying vec2 v_texcoord;
void main()
{
    gl_Position = projection * view * model * vec4(position,1.0);
    v_texcoord = texcoord;
}
"""
#fragment shader---------------------------------------------------------------
fragmentTexture = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    } """
##------------------
##------------------

class Canvas(app.Canvas):
    def __init__(self):

        app.Canvas.__init__(self, size=(512, 512), title='World Frame',keys='interactive')
        # Build program & data
        self.programTexture= Program(vertexTexture, fragmentTexture) #les shaders que l'on va utiliser
        self.program = Program(vertexColor, fragmentColor) #les shaders que l'on va utiliser
        self.thetax = 0.0 #variable d'angle
        self.timer = app.Timer('auto', self.on_timer) #construction d'un timer
        self.timer.start() #lancement du timer
        

        # Build view, model, projection & normal
        view = translate((0, 0, -4)) #on recule la camera suivant z
        model = np.eye(4, dtype=np.float32) #matrice identit�e
        self.program['model'] = model #matrice de l'objet
        self.program['view'] = view #matrice de la camera
        self.programTexture['model'] = model #matrice de l'objet
        self.programTexture['view'] = view #matrice de la camera

        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True) #couleur du fond et test de profond
        self.activate_zoom() #generation de la matrice de projection
        self.show() #rendu


    def drawFrame(self):
        ####t1 = Triangle(0,0,0,0,1,0,0.5,0.5,0,self.program) #construction d'un objet triangle
        ###t1.draw() #affichage de l'objet

        #t2 = TriangleWireFrame(-1.,0,0,-1.,1,0,-0.5,0.5,0,self.program) #construction d'un objet triangle
        #t2.draw() #affichage de l'objet

        t3 = Line(0,0,0,1,0,0,self.program) #construction d'un objet ligne
        self.program['color'] = [(1,0,0,1),(1,0,0,1)]; #la couleur de chaque vertex
        t3.draw() #affichage de l'objet

        t4 = Line(0,0,0,0,1,0,self.program) #construction d'un objet ligne
        self.program['color'] = [(0,1,0,1),(0,1,0,1)]; #la couleur de chaque vertex
        t4.draw() #affichage de l'objet

        t5 = Line(0,0,0,0,0,1,self.program) #construction d'un objet ligne
        self.program['color'] = [(0,0,1,1),(0,0,1,1)]; #la couleur de chaque vertex
        t5.draw() #affichage de l'objet


        #cubeTexture = CubeTextureNew(1,1,1,img1,self.programTexture)
        #cubeTexture.draw()

        #kitty cat cilinder
        #cylindre = Cylindre(self.programTexture)
        #cylindre.draw(rayon,hauteur,nb_pas)

        #sphere
        #sphere = Sphere(self.programTexture)
        #sphere.draw(rayon, nb_pas)
                    
        #t6 = Cube_Filaire(x,y,z,1,0,0,self.program)
        #t6.draw()

        #t6 = CubeTexture(x,y,z,0,1,2,0,2,3,img1,self.programTexture) #front
        #t6.draw()
        #t7 = CubeTexture(x,y,z,4,5,6,4,6,7,img2,self.programTexture) #back
        #t7.draw()
        #t8 = CubeTexture(x,y,z,4,5,1,4,1,0,img3,self.programTexture) #left
        #t8.draw()
        #t9 = CubeTexture(x,y,z,7,6,2,7,2,3,img4,self.programTexture) #right
        #t9.draw()
        #t10 = CubeTexture(x,y,z,4,0,3,4,3,7,img5,self.programTexture) #top
        #self.programTexture['texcoord']  = []
        #t10.draw()
        #t11 = CubeTexture(x,y,z,5,1,6,6,1,2,img6,self.programTexture) #bottom
        #t11.draw()

        #t6 = CubeTexture(x,y,z,0,1,2,0,2,3,img1,self.programTexture) #front
        #t6.draw()
        #t7 = CubeTexture(x,y,z,4,5,6,4,6,7,img2,self.programTexture) #back
        #t7.draw()
        #t8 = CubeTexture(x,y,z,8,9,13,13,8,12,img3,self.programTexture) #left
        #t8.draw()
        #t9 = CubeTexture(x,y,z,11,10,14,14,11,15,img4,self.programTexture) #right
        #t9.draw()
        #t10 = CubeTexture(x,y,z,16,19,20,20,19,23,img5,self.programTexture) #top
        #t10.draw()
        #t11 = CubeTexture(x,y,z,17,21,18,21,18,22,img6,self.programTexture) #bottom
        #t11.draw()
        

        t6 = Line(self.tetex,self.tetey,self.tetez,self.nezx,self.nezy,self.nezz,self.program)
        self.program['color'] = [(0,0,1,1),(0,0,1,1)];
        t6.draw()

    def on_draw(self, event):
        gloo.set_clear_color('grey')
        gloo.clear(color=True)
        self.drawFrame()

        self.tetex = self.joints3D[26].position.x/scale
        self.tetey = self.joints3D[26].position.y/scale
        self.tetez = self.joints3D[26].position.z/scale

        self.nezx = self.joints3D[27].position.x/scale
        self.nezy = self.joints3D[27].position.y/scale
        self.nezz = self.joints3D[27].position.z/scale

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size) #l'�cran d'affichage
    
    def activate_zoom(self): #pour crer la matrice de projection
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(45.0, self.size[0] / float(self.size[1]),2.0, 10.0) #matrice de projection
        self.program['projection'] = projection
        self.programTexture['projection'] = projection

    #def on_mouse_move(self, event):
    #    x, y = event.pos #on r�cup�re la position 
    #    largeur, hauteur = self.size #on r�cup�re la taille de l'�cran
    #    print (x) 
    #    print (y) 
    #    print(largeur, hauteur)

    #    self.thetax = 360*x/hauteur -180 #calcul de l'angle x
    #    self.thetay = 360*y/largeur -180 #calcul de l'angle y

    #    rx = rotate(self.thetax,(0,1,0)) #rotation en x
    #    ry = rotate(self.thetay,(1,0,0)) #rotation en y

    #    r=np.matmul(rx,ry) #calcul des rotations
    #    self.program['model']=r #matrice de l'objet
    #    self.programTexture['model']=r #matrice de l'objet
    #    self.update()

    def on_timer(self, event):
        # Get capture
        capture = device.update()
        # Get body tracker frame
        body_frame = bodyTracker.update()
            
        self.numbersbody = body_frame.get_num_bodies()
        
        if self.numbersbody >0 :
            body = body_frame.get_body()
            self.joints3D = body.joints
            print(f"joint[0] = {self.joints3D[jointTete['tete']].position.x}")
                
        #t6 = Line(self.tetex,self.tetey,self.tetez,self.nezx,self.nezy,self.nezz,self.program)
        #self.program['color'] = [(0,0,1,1),(0,0,1,1)];
        #t6.draw()
            


if __name__ == "__main__":
    '''x=float(input("echelle sur x:"))
    y=float(input("echelle sur y:"))
    z=float(input("echelle sur x:"))
    print(x,y,z)'''
    scale =1000
    pykinect.initialize_libraries(track_body=True)

    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    
    print("start device")
    
    device = pykinect.start_device(config=device_config)
    bodyTracker = pykinect.start_body_tracker()

    jointTete = {"tete":26,"nez":27,"oeil gauche":28,"oreille gauche":29,"oeil droit":30,"oreille droite":31}
    jointBrasGauche = {"epaule gauche":5,"coude gauche":6,"poignee gauche":7,"main gauche":8}
    jointBrasDroit = {"epaule droite":12,"coude droit":13,"poignee droite":14,"main droite":15}
    jointCorps = {"cou":3,"pec":1, "nombril":2,"bassin":0} 
    jointJambeGauche = {"hanche gauche":18, "genou gauche":19}
    jointJambeDroite = {"hanche droite":22, "genou droit":23}
    jointPiedGauche = {"cheville gauche":20,"pied gauche":21}
    jointPiedDroit = {"cheville droit":24,"pied droit":25}

    x=1
    y=1
    z=1
    nb_pas = 20
    nb_pas_phi = 10
    nb_pas_theta = 5
    rayon = 0.5
    hauteur = 1
    c = Canvas() #construction d'un objet Canvas
    app.run()

