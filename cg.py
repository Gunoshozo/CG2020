import random
import string
import time

import numpy as np
from PIL import Image

def random_string(string_length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))

def func(el1,el2):
    return '%s|%s' % (el1,el2)


colors = {'blue':(0,0,255),
          'green':(0, 255, 0),
          'red': (255, 0, 0),
          'pink':(255, 192, 203),
          'black':(255,255,255),
          'yellow':(255,255,0),
          'cyan':(0,255,255),
          'purple':(255,0,255)}

class Model:

    def __init__(self, filename):
        self.points = np.empty((0, 3), dtype=np.float)
        self.faces = np.empty((0, 3), dtype=np.int)
        self.normals = np.empty((0, 3), dtype=np.float)
        self.poly_norms = np.empty((0, 3), dtype=np.float)
        self.vns = np.empty((0, 3), dtype=np.float)
        self.rot = np.eye(3,dtype=np.float)
        self.translation = np.zeros((3),dtype=np.float)
        self.parse_file(filename)

    def set_translation(self, arr):
        self.translation = np.array(arr, dtype=np.float)

    def parse_file(self, filename):
        file = open(filename, 'r')
        res_p = np.empty((0, 3), dtype=np.float64)
        res_vn = np.empty((0, 3), dtype=np.float64)
        res_f = np.empty((0, 3), dtype=np.int)
        res_n = np.empty((0, 3), dtype=np.int)
        while True:
            line = file.readline()
            if line == '':
                break
            line_array = line.split(' ')
            if line_array[0] == 'v':
                line_array = [np.float64(i) for i in line_array[1:]]
                res_p = np.append(res_p, [line_array], axis=0)
            elif line_array[0] == 'vn':
                line_array = [np.float64(i) for i in line_array[1:]]
                res_vn = np.append(res_vn, [line_array], axis=0)
            elif line_array[0] == 'f':
                line_array1 = [int(i) for i in [j.split('/')[0] for j in line_array[1:]]]
                line_array2 = [int(i) for i in [j.split('/')[2] for j in line_array[1:]]]
                res_f = np.append(res_f, [line_array1], axis=0)
                res_n = np.append(res_n, [line_array2], axis=0)

        self.points, self.faces, self.vns, self.poly_norms = res_p, res_f, res_vn, res_n
        self.points = np.dot(self.points, self.rot)
        self.calculate_normals()
        file.close()

    def calculate_normals(self):
        for i in self.faces:
            p1 = np.array([self.points[i[0] - 1]])
            p2 = np.array([self.points[i[1] - 1]])
            p3 = np.array([self.points[i[2] - 1]])
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n = n / np.linalg.norm(n)
            self.normals = np.append(self.normals, n, axis=0)

    def set_rotation_matrix(self, x, y, z):
        x, y, z = np.array([x, y, z]) * np.pi / 180
        x_r = np.array([[1, 0, 0],
                       [0, np.cos(x), -np.sin(x)],
                       [0, np.sin(x), np.cos(x)]])
        y_r = np.array([[np.cos(y), 0, np.sin(y)],
                       [0, 1, 0],
                       [-np.sin(y), 0, np.cos(y)]])
        z_r = np.array([[np.cos(z), -np.sin(z), 0],
                       [np.sin(z), np.cos(z), 0],
                       [0, 0, 1]])
        self.rot = np.dot(np.dot(x_r, y_r), z_r)

class Camera:
    def __init__(self, width, height, scale, dx, dy):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.depthMap = np.full((height,width), float('inf'), dtype=np.float64)
        self.width = width
        self.height = height
        self.scale = scale
        self.dx = dx
        self.dy = dy

        self.k = np.array([[scale, 0, width / 2],
                           [0, -scale, height / 2],
                           [0, 0, 1]], dtype=np.float)

    def show_image(self):
        im = Image.fromarray(self.canvas)
        im.show()

    def save_image(self, name):
        im = Image.fromarray(self.canvas)
        im.save("name")

class BasicShader:

    def __init__(self):
        self.tmp = np.zeros((3, 3), np.float)
        self.intensity = 0
        self.back = True

    def vertex_shader(self, model, poly_n, camera):
        vrtx = model.points[model.faces[poly_n] - 1]
        #camera_vrtx = np.column_stack((vrtx[:,0] * camera.scale + camera.dx, -vrtx[:,1] * camera.scale + camera.dy, vrtx[:,2]))

        new_vertex = np.dot(model.rot,vrtx) + [model.translation, model.translation, model.translation]
        new_vertex[:] = new_vertex / new_vertex[:, 2, None]

        camera_vrtx = np.zeros((3, 3), dtype=np.float)

        camera_vrtx[0, :] = camera.k @ new_vertex[0, :]
        camera_vrtx[1, :] = camera.k @ new_vertex[1, :]
        camera_vrtx[2, :] = camera.k @ new_vertex[2, :]
        camera_vrtx[:, 2] = vrtx[:, 2]

        self.tmp = np.copy(vrtx)


        d1 = self.tmp[1] - self.tmp[0]
        d2 = self.tmp[2] - self.tmp[0]

        n = np.cross(d1, d2)
        n = n / np.linalg.norm(n)
        self.back = (n[2] > 0)
        self.intensity = -n[2]
        return camera_vrtx

    def fragment_shader(self, barycentric, camera, model,base_color):
        return self.back, tuple([int(round(i*self.intensity)) for i in color])

class SmoothShader:

    def __init__(self):
        self.vertex_intensity = np.zeros(3, np.float)

    def vertex_shader(self, model, poly_n, camera):
        vrtx = model.points[model.faces[poly_n][:] - 1]

        new_vertex = np.dot(model.rot, vrtx) + [model.translation,model.translation,model.translation]
        new_vertex[:] = new_vertex/new_vertex[:,2,None]

        camera_vertex = np.zeros((3,3),dtype=np.float)

        camera_vertex[0,:] = camera.k @ new_vertex[0,:]
        camera_vertex[1,:] = camera.k @ new_vertex[1,:]
        camera_vertex[2,:] = camera.k @ new_vertex[2,:]
        camera_vertex[:,2] = vrtx[:,2]
        self.vertex_intensity = model.vns[model.poly_norms[poly_n][:]-1][:,2]

        return camera_vertex

    def fragment_shader(self, barycentric, camera, model,base_color):
        portion = np.dot(barycentric,self.vertex_intensity)
        color = tuple([int(round(i*portion)) for i in base_color])
        if any(x < 0 for x in color):
            return True, (0,0,0)
        else:
            return False, color


def barycentric(v0, v1, v2, x, y):
    x0, y0 = v0[0], v0[1]
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]

    l0 = ((y - y2) * (x1 - x2) - (x - x2) * (y1 - y2)) / ((y0 - y2) * (x1 - x2) - (x0 - x2) * (y1 - y2))
    l1 = ((y - y0) * (x2 - x0) - (x - x0) * (y2 - y0)) / ((y1 - y0) * (x2 - x0) - (x1 - x0) * (y2 - y0))
    l2 = ((y - y1) * (x0 - x1) - (x - x1) * (y0 - y1)) / ((y2 - y1) * (x0 - x1) - (x2 - x1) * (y0 - y1))
    return np.array([l0, l1, l2], dtype=np.float)


def draw(filename, width, height, scale, dx, dy, sh,base_color):
    Krol = Model(filename)
    Krol.set_rotation_matrix(0,180,0)
    Krol.set_translation([0,-0.05,0.1])

    cam = Camera(width, height, scale, dx, dy)

    shader = sh()

    cam_vertex = np.zeros((3, 3), dtype=np.float)
    start = time.time()
    for poly_n in range(Krol.faces.shape[0]):
        cam_vertex = shader.vertex_shader(Krol,poly_n,cam)

        x_min = max(int(np.floor(min(cam_vertex[0][0],cam_vertex[1][0],cam_vertex[2][0]))),0)
        y_min = max(int(np.floor(min(cam_vertex[0][1],cam_vertex[1][1],cam_vertex[2][1]))),0)

        x_max = min(int(np.ceil(max(cam_vertex[0][0],cam_vertex[1][0],cam_vertex[2][0]))),cam.width)
        y_max = min(int(np.ceil(max(cam_vertex[0][1],cam_vertex[1][1],cam_vertex[2][1]))),cam.height)


        for x in range(x_min,x_max):
            for y in range(y_min,y_max):
                l = barycentric(cam_vertex[0],cam_vertex[1],cam_vertex[2],x,y)
                z = l[0]*cam_vertex[0,2] + l[1]*cam_vertex[1,2] + l[2]*cam_vertex[2,2]
                if (l > 0).all() and cam.depthMap[y][x] > z:
                    discard, color = shader.fragment_shader(l,cam,Krol,base_color)
                    if not discard:
                        cam.canvas[y][x] = color
                        cam.depthMap[y][x] = z
    print(time.time()-start)
    cam.show_image()

if __name__ == '__main__':
    color = colors['black']
    draw("file.obj", 800, 600, 50, 400, 300, BasicShader,color)
