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

    def __init__(self,scale=10000,rot=(0,0,0)):
        self.points = np.empty((0, 3), dtype=np.float64)
        self.faces = np.empty((0, 3), dtype=np.int)
        self.norms = np.empty((0, 3), dtype=np.float64)
        self.size = (1500, 1500)
        self.depthMap = np.full(self.size, float('inf'), dtype=np.float64)
        self.k = np.array([[scale, 0, 700],
                           [0, -scale, 1200],
                           [0, 0, 1]])
        self.rot = self.get_rotation_matrix(rot[0], rot[1], rot[2])

    def set_k(self, arr):
        self.k = arr

    def set_array(self, points):
        self.points = np.array(points)

    def get_point(self, index):
        return self.points[index]

    def get_face(self, index):
        return self.faces[index]

    def draw_line(self, image, x0, y0, x1, y1):
        pixels = image.load()
        steep = False
        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)
        if np.abs(x0 - x1) < np.abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        for x in range(x0, x1, 1):
            t = (x - x0) / (x1 - x0)
            y = int(y0 * (1. - t) + y1 * t)
            if 0 <= x < 2000 and y >= 0 and y < 2000:
                if steep:
                    pixels[x, y] = green
                else:
                    pixels[y, x] = green

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

    def draw_points(self):
        image = Image.new(mode='RGB', size=self.size)
        pixels = image.load()
        for i in self.points:
            pixels[4000 * i[0] + 500, -4000 * i[1] + 500] = (255, 255, 255)
        image.show()

    def draw_faces(self):
        image = Image.new(mode='RGB', size=self.size)
        pixels = image.load()
        tmp_point = self.points
        tmp_point[:, 2] = 1
        tmp_point = np.dot(tmp_point, self.k.T)
        for i in self.faces:
            Model.draw_line(image, tmp_point[i[0] - 1][0], tmp_point[i[0] - 1][1], tmp_point[i[1] - 1][0],
                            tmp_point[i[1] - 1][1])
            Model.draw_line(image, tmp_point[i[1] - 1][0], tmp_point[i[1] - 1][1], tmp_point[i[2] - 1][0],
                            tmp_point[i[2] - 1][1])
            Model.draw_line(image, tmp_point[i[0] - 1][0], tmp_point[i[0] - 1][1], tmp_point[i[2] - 1][0],
                            tmp_point[i[2] - 1][1])
        image.show()
        image.save("krol.PNG")

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
        image.save(random_string(4)+"_z.PNG")

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

    @staticmethod
    def get_rotation_matrix(x, y, z):
        x,y,z = np.array([x,y,z])*np.pi/180
        xR = np.array([[1, 0, 0],
                       [0, np.cos(x), -np.sin(x)],
                       [0, np.sin(x), np.cos(x)]])
        yR = np.array([[np.cos(y), 0, np.sin(y)],
                       [0, 1, 0],
                       [-np.sin(y), 0, np.cos(y)]])
        zR = np.array([[np.cos(z), -np.sin(z), 0],
                       [np.sin(z), np.cos(z), 0],
                       [0, 0, 1]])
        return np.dot(np.dot(xR, yR), zR)


if __name__ == '__main__':
    Krol = Model(scale=10000,rot=(0,70,0))
    Krol.parse_file('file.obj')
    Krol.draw_polygons_with_z()
