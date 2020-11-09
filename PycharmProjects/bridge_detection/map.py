import cv2
from math import cos, sin, atan2, sqrt, radians, degrees, asin, modf, log, pi, tan
import time
import math
import csv
from operator import itemgetter
import pyproj as proj
from shapely import geometry

import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()

ROOT.withdraw()
main_lst = []

######################## graphs ############################


class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

############################### path generation ###############################
paths = []


def find_all_paths(graph, start, end, path=[], newpaths=None):
    path = path + [start]
    if start == end:
        return path
    for node in graph[start].get_connections():
        if node.get_id() not in path:
            newpaths = find_all_paths(graph, node.get_id(), end, path)
    if newpaths not in paths and type(newpaths) == list:
        paths.append(newpaths)
    return paths


def find_path(graph, start, end,heading,path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    print(graph[start])
    for node in graph[start].get_connections():
        lat1 = float(main_lst[int(start)][3])
        lon1 = float(main_lst[int(start)][4])
        lat2 = float(main_lst[int(node.get_id())][3])
        lon2 = float(main_lst[int(node.get_id())][4])
        heading1 = get_azimuth(lat1,lon1,lat2,lon2)
        # if(abs(heading-heading1)>180):
        #     diff = abs(360-heading1+heading)
        #     if(diff > 360):
        #         diff = diff - 360
        # else:
        #     diff = abs(heading1-heading)
        diff = get_heading_diff(heading1,heading)
        if(diff > 120):
            continue
        if node.get_id() not in path or end not in path:
            newpath = find_path(graph, node.get_id(), end, heading1 ,path)
            if newpath: return newpath
    return None


########################## basic layout ###############################

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
original_img = cv2.imread("bigmap.png")
img = []
img = original_img.copy()

img_c = 913
img_r = 613

csv_otput = open("output.csv",'w')
spamwriter = csv.writer(csv_otput, delimiter=',')

spamwriter.writerow(["lat"] + ["lon"])

g = Graph()

lt1 = 18.540958
ln1 = 73.955254

lt2 = 18.542574
ln2 = 73.957739

recorded_lst = []

lon_factor = (ln2-ln1)/img_c
lat_factor = (lt2-lt1)/img_r

steps = 3
counter = 0

############################### latlon calculations ##############################


def get_heading_diff(a1,a2):
    angle = abs(a1 - a2)
    if (angle > 180):
        angle = 360 - angle
    return(angle)


def getDestinationLatLong(lat,lng,azimuth,distance):
    '''returns the lat an long of destination point
    given the start lat, long, aziuth, and distance'''
    R = 6378.1 #Radius of the Earth in km
    brng = radians(azimuth)
    d = distance/1000

    lat1 = radians(lat)
    lon1 = radians(lng)

    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1)* sin(d/R)* cos(brng))

    lon2 = lon1 + atan2(sin(brng) * sin(d/R)* cos(lat1),
           cos(d/R)- sin(lat1)* sin(lat2))

    #convert to degrees
    lat2 = degrees(lat2)
    lon2 = degrees(lon2)

    return[lat2, lon2]


def getPathLength(lat1, lng1, lat2, lng2):
    '''calculates the distance between two lat, long coordinate pairs'''
    R = 6371000
    lat1rads = radians(lat1)
    lat2rads = radians(lat2)
    deltaLat = radians((lat2-lat1))
    deltaLng = radians((lng2-lng1))
    a = sin(deltaLat/2) * sin(deltaLat/2) + cos(lat1rads) * cos(lat2rads) * sin(deltaLng/2) * sin(deltaLng/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c
    return d


def get_azimuth(lat1, lng1, lat2, lng2):
    global  img
    global  steps
    dLong = lng2 - lng1
    dPhi = log(tan(lat2 / 2.0 + pi / 4.0) / tan(lat1 / 2.0 + pi / 4.0))
    azimuth = (degrees(atan2(dLong, dPhi)) + 360.0) % 360.0
    print('azimuth value : ', azimuth)
    return(azimuth)


def main(interval, azimuth, lat1, lng1, lat2, lng2):
    global img
    '''
    returns every coordinate pair in between two coordinate pairs given with certain interval
    '''
    coords = []
    d = getPathLength(lat1,lng1,lat2,lng2)
    remainder, dist = modf((d / interval))
    counter = 1.0
    coords.append([lat1,lng1])
    spamwriter.writerow([lat1] + [lng1])
    for distance in range(1, int(d),interval):
        c = getDestinationLatLong(lat1, lng1, azimuth, counter)
        counter += interval
        x,y = c[0],c[1]
        spamwriter.writerow([x] + [y])
        y_pt = img_r - (x - lt1) / lon_factor
        x_pt = (y - ln1) / lat_factor
        img = cv2.circle(img, (int(x_pt), int(y_pt)), radius=2, color=(0, 255, 0), thickness=6)
        cv2.imshow("image", img)
        cv2.waitKey(10)
        coords.append(c)
    counter += 1
    coords.append([lat2, lng2])
    return coords

################################ nearest point to path #################################


def draw_circle(event, x, y, flags, param):
    global img
    global stable_img
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #cv2.imshow('image', img)
        # current_location = simpledialog.askstring(title="Input", prompt="Please enter current location of vehicle")
        # x = current_location.split(',')[0]
        # y = current_location.split(',')[1]
        heading = simpledialog.askstring(title="Input", prompt="Please enter current heading of vehicle")
        print(heading)
        lat, lon, strt = get_closest_point(x, y, int(heading))
        path_genration(float(lat), float(lon), strt, int(heading))
        cv2.waitKey(2000)
        cv2.imshow('image', original_img)
        img = original_img.copy()
        draw_point()
        cv2.waitKey(2)


def index_to_latlon(x,y):
    lon = (x)*lon_factor + ln1
    lat = (img_r-y)*lat_factor + lt1
    return lat,lon


def get_closest_point(x, y, heading):
    tmp_file = open("constants.csv", 'r')
    csv_read = csv.reader(tmp_file)
    lst = []
    sorted_lst = []
    for row in csv_read:
        x1, y1, lt, ln, id = row
        #setup projections
        global res
        points_dict = dict()
        crs_wgs = proj.Proj(init='epsg:4326')  #WGS84 geographic - which is a standard representation for lat lon
        crs_bng = proj.Proj(init='epsg:27700')  #appropriate CRS - co ordinate refrence system
        lat, lon = index_to_latlon(x, y)
        #create two points for comparison
        point_1 = geometry.Point(float(lon), float(lat))
        point_2 = geometry.Point(float(ln), float(lt))
        #create your circle buffer
        distance = 0.0002
        circle_buffer = point_1.buffer(distance)
        #check if point is inside
        if circle_buffer.contains(point_2):
            # print('circle contains point ', number)
            dist = getPathLength(float(x), float(y), float(lt), float(ln))
            points_dict['point'] = [lt, ln, id]
            points_dict['distance'] = dist
        else:
            pass
            # print('not in circle')
        if len(points_dict) > 0:
            lst.append(points_dict)
    # sort list based on distance
    final = sorted(lst, key=itemgetter('distance'))
    print(final)
    for i in final:
        sorted_lst.append(i['point'])
    head_lst = []
    min = 360
    index = 0
    for i, data in enumerate(sorted_lst):
        heading1 = get_azimuth(float(lat), float(lon), float(data[0]), float(data[1]))
        diff = get_heading_diff(heading, heading1)
        if(diff < min):
            min = diff
            index = i
    if(min > 120):
        adj_lst = g.vert_dict[sorted_lst[index][2]].get_connections()
        for point in adj_lst:
            heading1 = get_azimuth(float(x), float(y), float(main_lst[int(point.get_id())][3]), float(main_lst[int(point.get_id())][4]))
            if(get_heading_diff(heading1, heading) < 120):
                return(x, y, str(point.get_id()))
    tmp_file.close()
    return(lat, lon, sorted_lst[index][2])


def path_genration(lat_init, lon_init, strt, heading):
    endt = 33   # fixed for now
    lst = find_all_paths(g.vert_dict, str(strt), str(endt))
    # print(type(lst))
    print(lst)
    lst.sort(key=lambda x: len(x))
    # print(lst[0])
    print(lst[1])
    lst = lst[1]

    #plotting the points in path

    if(lst == []):
        print("No path Found")
        return None

    lt_lst = []
    ln_lst = []

    for i in range(0, len(lst)):
        lt_lst.append(recorded_lst[int(lst[i])][1])
        ln_lst.append(recorded_lst[int(lst[i])][2])

    lt_lst = list(map(float, lt_lst))
    ln_lst = list(map(float, ln_lst))

    tmlt = lat_init
    tmln = lon_init

    for (x, y) in zip(lt_lst, ln_lst):
        azimuth = get_azimuth(tmlt, tmln, x, y)
        main(steps, azimuth, tmlt, tmln, x, y)
        tmlt = x
        tmln = y

############################# graph ##############################


def graph_creation():
    global g
    for i in range(0,53):
        g.add_vertex(str(i))

    for i in range(0,16):
        g.add_edge(str(i),str(i+1),1)

    for i in range(17,42):
        g.add_edge(str(i),str(i+1),1)

    for i in range(43,53):
        g.add_edge(str(i),str(i+1),1)

    g.add_edge('0', '16', 1)
    g.add_edge('0', '17', 1)
    g.add_edge('43', '27', 1)
    g.add_edge('43', '26', 1)

########################### image #################################


def draw_point():
    csv_file = open("coord_cor.csv", 'r')
    csv_reader = csv.reader(csv_file)
    const_output = open("constants.csv", 'w')
    const_writer = csv.writer(const_output, delimiter=',', lineterminator='\n')
    global counter
    global img
    counter = 0
    for row in csv_reader:
        lon, lt, number = row
        recorded_lst.append([number, lt, lon])
        y_pt = round(img_r - (float(lt) - float(lt1)) / lon_factor)
        x_pt = round((float(lon) - float(ln1)) / lat_factor)
        const_writer.writerow([x_pt] + [y_pt] + [lt] + [lon] + [counter])
        main_lst.append([counter, x_pt, y_pt, lt, lon])
        img = cv2.circle(img, (int(x_pt), int(y_pt)), radius=1, color=(0, 255, 0), thickness=4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(counter), (int(x_pt) + 5, int(y_pt) + 5), font, .4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("image", img)
        cv2.waitKey(1)
        counter = counter + 1
    const_output.close()
    csv_file.close()


if __name__ == '__main__':
    graph_creation()
    cv2.setMouseCallback("image", draw_circle)
    draw_point()
    while(1):
        cv2.imshow('image',img)
        cv2.waitKey(300)
        
    csv_file.close()
    csv_otput.close()