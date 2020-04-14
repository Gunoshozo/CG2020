import random
import string

import numpy as np
from PIL import Image

pink = (255, 192, 203)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


def random_string(string_length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


class Model:

    def __init__(self, filename):
        self.points = np.empty((0, 3), dtype=np.float)
        self.faces = np.empty((0, 3), dtype=np.int)
        self.norms = np.empty((0, 3), dtype=np.float)
        self.rot = np.eye(3, dtype=np.float)
        self.parse_file(filename)
        # self.depthMap = np.full(self.size, float('inf'), dtype=np.float64)
        # self.k = np.array([[scale, 0, 700],
        #                    [0, -scale, 1200],
        #                    [0, 0, 1]])
        # self.set_rotation_matrix(rot[0], rot[1], rot[2])

    def parse_file(self, filename):
        file = open(filename, 'r')
        res_p = np.empty((0, 3), dtype=np.float64)
        res_f = np.empty((0, 3), dtype=np.int)
        while True:
            line = file.readline()
            if line == '':
                break
            line_array = line.split(' ')
            if line_array[0] == 'v':
                line_array = [np.float64(i) for i in line_array[1:]]
                res_p = np.append(res_p, [line_array], axis=0)
            elif line_array[0] == 'f':
                line_array = [int(i) for i in [j.split('/')[0] for j in line_array[1:]]]
                res_f = np.append(res_f, [line_array], axis=0)
        self.points, self.faces = res_p, res_f
        self.points = np.dot(self.points, self.rot)
        self.calculate_norms()
        file.close()

    @staticmethod
    def draw_triangle(image, x0, y0, x1, y1, x2, y2, color):
        pixels = image.load()
        a0 = 1 / ((y0 - y2) * (x1 - x2) - (x0 - x2) * (y1 - y2))
        a1 = 1 / ((y1 - y0) * (x2 - x0) - (x1 - x0) * (y2 - y0))
        a2 = 1 / ((y2 - y1) * (x0 - x1) - (x2 - x1) * (y0 - y1))
        for x in range(int(max(0, (min([x0, x1, x2])))), int(min(image.size[0], (max([x0, x1, x2])))) + 1):
            for y in range(int(max(0, (min([y0, y1, y2])))), int(min(image.size[1], (max([y0, y1, y2])))) + 1):
                L0 = ((y - y2) * (x1 - x2) - (x - x2) * (y1 - y2)) * a0
                L1 = ((y - y0) * (x2 - x0) - (x - x0) * (y2 - y0)) * a1
                L2 = ((y - y1) * (x0 - x1) - (x - x1) * (y0 - y1)) * a2
                if L0 > 0 and L1 > 0 and L2 > 0:
                    pixels[x, y] = color

    def draw_polygons(self):
        image = Image.new(mode='RGB', size=self.size)
        tmp_point = self.points
        # tmp_point[:, 2] = 1
        tmp_point = np.dot(tmp_point, self.k.T)
        for i in enumerate(self.faces):
            if self.norms[i[0]][2] <= 0:
                val = int(np.round(-255 * self.norms[i[0]][2]))
                col = (val, val, val)
                self.draw_triangle(image, tmp_point[i[1][0] - 1][0], tmp_point[i[1][0] - 1][1],
                                   tmp_point[i[1][1] - 1][0], tmp_point[i[1][1] - 1][1],
                                   tmp_point[i[1][2] - 1][0], tmp_point[i[1][2] - 1][1], col)
        image.show()
        image.save("krol.PNG")

    def draw_polygons_with_z(self):
        image = Image.new(mode='RGB', size=self.size)
        tmp_point = np.copy(self.points)
        tmp_point[:, 2] = 1
        tmp_point = np.dot(tmp_point, self.k.T)
        tmp_point[:, 2] = self.points[:, 2]
        for i in enumerate(self.faces):
            if self.norms[i[0]][2] <= 0:
                val1 = int(np.round(-57 * self.norms[i[0]][2]))
                val2 = int(np.round(-255 * self.norms[i[0]][2]))
                val3 = int(np.round(-20 * self.norms[i[0]][2]))
                col = (val1, val2, val3)
                self.draw_triangle_with_z(image, tmp_point[i[1][0] - 1], tmp_point[i[1][1] - 1], tmp_point[i[1][2] - 1],
                                          col)
        image.show()
        image.save(random_string(4) + "_z.PNG")

    def draw_triangle_with_z(self, image, p0, p1, p2, col):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        pixel = image.load()
        a0 = 1 / ((y0 - y2) * (x1 - x2) - (x0 - x2) * (y1 - y2))
        a1 = 1 / ((y1 - y0) * (x2 - x0) - (x1 - x0) * (y2 - y0))
        a2 = 1 / ((y2 - y1) * (x0 - x1) - (x2 - x1) * (y0 - y1))
        for x in range(int(max(0, (min([x0, x1, x2])))), int(min(image.size[0], (max([x0, x1, x2])) + 1))):
            for y in range(int(max(0, (min([y0, y1, y2])))), int(min(image.size[1], (max([y0, y1, y2])) + 1))):
                l0 = ((y - y2) * (x1 - x2) - (x - x2) * (y1 - y2)) * a0
                l1 = ((y - y0) * (x2 - x0) - (x - x0) * (y2 - y0)) * a1
                l2 = ((y - y1) * (x0 - x1) - (x - x1) * (y0 - y1)) * a2
                z = z0 * l0 + z1 * l1 + z2 * l2
                if l0 > 0 and l1 > 0 and l2 > 0 and z < self.depthMap[x][y]:
                    self.depthMap[x][y] = z
                    pixel[x, y] = col

    def calculate_norms(self):
        for i in self.faces:
            p1 = np.array([self.points[i[0] - 1]])
            p2 = np.array([self.points[i[1] - 1]])
            p3 = np.array([self.points[i[2] - 1]])
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n = n / np.linalg.norm(n)
            self.norms = np.append(self.norms, n, axis=0)

    def set_rotation_matrix(self, x, y, z):
        x, y, z = np.array([x, y, z]) * np.pi / 180
        xR = np.array([[1, 0, 0],
                       [0, np.cos(x), -np.sin(x)],
                       [0, np.sin(x), np.cos(x)]])
        yR = np.array([[np.cos(y), 0, np.sin(y)],
                       [0, 1, 0],
                       [-np.sin(y), 0, np.cos(y)]])
        zR = np.array([[np.cos(z), -np.sin(z), 0],
                       [np.sin(z), np.cos(z), 0],
                       [0, 0, 1]])
        self.rot = np.dot(np.dot(xR, yR), zR)


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

    def vertex_shader(self, model, poly_n, vertex_n, camera):
        vrtx = model.points[model.faces[poly_n][:] - 1]
        camera_vrtx = np.column_stack((vrtx[:,0] * camera.scale + camera.dx, -vrtx[:,1] * camera.scale + camera.dy, vrtx[:,2]))

        self.tmp = np.copy(vrtx)


        d1 = self.tmp[1] - self.tmp[0]
        d2 = self.tmp[2] - self.tmp[0]

        n = np.cross(d1, d2)
        n = n / np.linalg.norm(n)
        self.back = (n[2] > 0)
        self.intensity = int(round(-255 * n[2]))
        return camera_vrtx

    def fragment_shader(self, barycentric, camera, model):
        return self.back, (self.intensity, self.intensity, self.intensity)




def barycentric(v0, v1, v2, x, y):
    x0, y0 = v0[0], v0[1]
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]

    l0 = ((y - y2) * (x1 - x2) - (x - x2) * (y1 - y2)) / ((y0 - y2) * (x1 - x2) - (x0 - x2) * (y1 - y2))
    l1 = ((y - y0) * (x2 - x0) - (x - x0) * (y2 - y0)) / ((y1 - y0) * (x2 - x0) - (x1 - x0) * (y2 - y0))
    l2 = ((y - y1) * (x0 - x1) - (x - x1) * (y0 - y1)) / ((y2 - y1) * (x0 - x1) - (x2 - x1) * (y0 - y1))
    return np.array([l0, l1, l2], dtype=np.float)


def draw(filename, width, height, scale, dx, dy, sh):
    Krol = Model(filename)

    cam = Camera(width, height, scale, dx, dy)

    shader = sh()

    cam_vertex = np.zeros((3, 3), dtype=np.float)

    for poly_n in range(Krol.faces.shape[0]):
        cam_vertex = shader.vertex_shader(Krol,poly_n,0,cam)

        x_min = max(int(np.floor(min(cam_vertex[0][0],cam_vertex[1][0],cam_vertex[2][0]))),0)
        y_min = max(int(np.floor(min(cam_vertex[0][1],cam_vertex[1][1],cam_vertex[2][1]))),0)

        x_max = min(int(np.ceil(max(cam_vertex[0][0],cam_vertex[1][0],cam_vertex[2][0]))),cam.width)
        y_max = min(int(np.ceil(max(cam_vertex[0][1],cam_vertex[1][1],cam_vertex[2][1]))),cam.height)

        for x in range(x_min,x_max):
            for y in range(y_min,y_max):
                l = barycentric(cam_vertex[0],cam_vertex[1],cam_vertex[2],x,y)
                z = l[0]*cam_vertex[0,2] + l[1]*cam_vertex[1,2] + l[2]*cam_vertex[2,2]
                if (l > 0).all() and cam.depthMap[y][x] > z:
                    discard, color = shader.fragment_shader(l,cam,Krol)
                    if not discard:
                        cam.canvas[y][x] = color
                        cam.depthMap[y][x] = z

    cam.show_image()

if __name__ == '__main__':
    draw("file.obj",1280,720,7000,790,800,BasicShader)
