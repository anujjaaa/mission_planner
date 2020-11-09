import cv2
from math import cos, sin, atan2, sqrt, radians, degrees, asin, modf, log, pi, tan
import time
import math
import csv
# from operator import itemgetter
# import pyproj as proj
# from shapely import geometry

import tkinter as tk
from tkinter import simpledialog
from collections import defaultdict
from operator import itemgetter, attrgetter

ROOT = tk.Tk()

ROOT.withdraw()
main_lst = []


class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


graph = Graph()

############################### path generation ###############################
paths = []

max_turning_deg = 85
path_list = []


def find_all_paths(graph, start, end, veh_head, path=[], true_start_heading=0):
    global path_list
    path = path + [start]
    if start == end:
        path_list.append(path)
        return path
    total_adjacent = len(graph.edges[start])
    for i, node in enumerate(graph.edges[start]):
        lat1 = float(main_lst[int(start)][3])
        lon1 = float(main_lst[int(start)][4])
        lat2 = float(main_lst[int(node)][3])
        lon2 = float(main_lst[int(node)][4])
        nxt_point_heading = get_azimuth(lat1, lon1, lat2, lon2)
        diff = get_heading_diff(nxt_point_heading, veh_head)
        if (diff > max_turning_deg):
            continue
        if start in path[:-1]:
            index = path[:-1].index(start)
            lat1 = float(main_lst[int(path[index])][3])
            lon1 = float(main_lst[int(path[index])][4])
            lat2 = float(main_lst[int(path[index + 1])][3])
            lon2 = float(main_lst[int(path[index + 1])][4])
            past_point_heading = get_azimuth(lat1, lon1, lat2, lon2)
            if (abs(past_point_heading - nxt_point_heading) > 10):
                newpaths = find_all_paths(graph, node, end, nxt_point_heading, path)
                if newpaths and total_adjacent == (i + 1):
                    return (newpaths)
            else:
                pass
        else:
            newpaths = find_all_paths(graph, node, end, nxt_point_heading, path)
            if newpaths and total_adjacent == (i + 1):
                return (newpaths)
    return (path)


########################## basic layout ###############################

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

original_img = cv2.imread("bigmap.png")
img = []
img = original_img.copy()

img_c = 913  # number of columns in map
img_r = 613  # number of rows in map

csv_otput = open("output.csv", 'w')
spamwriter = csv.writer(csv_otput, delimiter=',')

spamwriter.writerow(["lat"] + ["lon"])

###########   Left - down lat lon points of map
lt1 = 18.540958
ln1 = 73.955254

# test for big map
lt2 = 18.542574
ln2 = 73.957739

recorded_lst = []

# for big map   Calculation of x and y factor for lat lon to pixel conversion vice versa
lon_factor = (ln2 - ln1) / img_c
lat_factor = (lt2 - lt1) / img_r

steps = 3
counter = 0


############################### latlon calculations ##############################


def get_heading_diff(a1, a2):
    angle = abs(a1 - a2)
    if (angle > 180):
        angle = 360 - angle
    return (angle)


def getDestinationLatLong(lat, lng, azimuth, distance):
    '''returns the lat an long of destination point
    given the start lat, long, aziuth, and distance'''
    R = 6378.1  # Radius of the Earth in km
    brng = radians(azimuth)
    d = distance / 1000

    lat1 = radians(lat)
    lon1 = radians(lng)
    #
    lat2 = asin(sin(lat1) * cos(d / R) + cos(lat1) * sin(d / R) * cos(brng))

    lon2 = lon1 + atan2(sin(brng) * sin(d / R) * cos(lat1),
                        cos(d / R) - sin(lat1) * sin(lat2))

    # convert back to degrees
    lat2 = degrees(lat2)
    lon2 = degrees(lon2)

    return [lat2, lon2]


def getPathLength(lat1, lng1, lat2, lng2):
    '''calculates the distance between two lat, long coordinate pairs'''
    if (isinstance(lat1, str)):
        lat1 = float(lat1)
        lng1 = float(lng1)
        lat2 = float(lat2)
        lng2 = float(lng2)
    R = 6371000  # radius of earth in m
    lat1rads = radians(lat1)
    lat2rads = radians(lat2)
    deltaLat = radians((lat2 - lat1))
    deltaLng = radians((lng2 - lng1))
    a = sin(deltaLat / 2) * sin(deltaLat / 2) + cos(lat1rads) * cos(lat2rads) * sin(deltaLng / 2) * sin(deltaLng / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c
    return d


def get_azimuth(lat1, lng1, lat2, lng2):
    global img
    global steps
    dLong = lng2 - lng1
    dPhi = log(tan(lat2 / 2.0 + pi / 4.0) / tan(lat1 / 2.0 + pi / 4.0))
    # if abs(dLong) > pi:
    #     dLong = -(2.0 * pi - dLong)
    #     # print('greater', dLong)
    # else:
    #     dLong = 2.0 * pi + dLong
    #     # print('less', dLong)
    # #     dLong = (dLong > 0.0) ? -(2.0 * pi - dLong): (2.0 * pi + dLong)
    azimuth = (degrees(atan2(dLong, dPhi)) + 360.0) % 360.0
    # print('azimuth value : ', azimuth)
    return (azimuth)


def main(interval, azimuth, lat1, lng1, lat2, lng2):
    global img
    d = getPathLength(lat1, lng1, lat2, lng2)
    remainder, dist = modf((d / interval))
    counter = 1.0
    spamwriter.writerow([lat1] + [lng1])
    for distance in range(1, int(d), interval):
        c = getDestinationLatLong(lat1, lng1, azimuth, counter)
        counter += interval
        x, y = c[0], c[1]
        spamwriter.writerow([x] + [y])
        y_pt = img_r - (x - lt1) / lon_factor
        x_pt = (y - ln1) / lat_factor
        img = cv2.circle(img, (int(x_pt), int(y_pt)), radius=2, color=(0, 255, 0), thickness=6)
        cv2.imshow("image", img)
        cv2.waitKey(10)
    counter += 1


################################ nearest point to path ###################################


def draw_circle(event, x, y, flags, param):
    global img
    global stable_img
    if event == cv2.EVENT_LBUTTONDBLCLK:
        lat, lon = index_to_latlon(x, y)
        heading = simpledialog.askstring(title="Input", prompt="Please enter current heading of vehicle")
        print(heading)
        lat, lon, strt = get_closest_point(x, y, int(heading))
        path_genration(float(lat), float(lon), strt, int(heading))
        cv2.waitKey(2000)
        cv2.imshow('image', original_img)
        img = original_img.copy()
        draw_point()
        cv2.waitKey(2)


def index_to_latlon(x, y):
    lon = (x) * lon_factor + ln1
    lat = (img_r - y) * lat_factor + lt1
    return lat, lon


def get_closest_point(x, y, veh_head):
    tmp_file = open("constants.csv", 'r')
    csv_read = csv.reader(tmp_file)
    lst = []
    for row in csv_read:
        x1, y1, lt, ln, id = row
        if (math.sqrt((math.pow((x - int(x1)), 2)) + (math.pow((y - int(y1)), 2))) < 49):
            lst.append([x1, y1, lt, ln, id])
    lat, lon = index_to_latlon(x, y)
    min = 360
    index = 0
    for i, data in enumerate(lst):
        nxt_point_heading = get_azimuth(lat, lon, float(data[2]), float(data[3]))
        distance = getPathLength(lat, lon, float(data[2]), float(data[3]))
        lst[i].append(distance)
        diff = get_heading_diff(veh_head, nxt_point_heading)
        if (diff < min):
            min = diff
            index = i
    if (min > max_turning_deg):
        adj_lst = graph.edges[lst[index][4]]
        for point in adj_lst:
            nxt_point_heading = get_azimuth(lat, lon, float(main_lst[int(point)][3]), float(main_lst[int(point)][4]))
            if (get_heading_diff(nxt_point_heading, veh_head) < max_turning_deg):
                return (lat, lon, str(point))
    tmp_file.close()
    return (lat, lon, lst[index][4])


def path_genration(lat_init, lon_init, strt, heading):
    global path_list
    endt = 61  # fixed for now
    find_all_paths(graph, str(strt), str(endt), heading)
    # lst  = dijsktra(graph,str(strt),str(endt),heading)
    print(path_list)
    dst_lst = []
    temp = 0
    if (len(path_list) > 1):
        for lst in path_list:
            temp = 0
            for i in range(0, len(lst) - 1):
                temp = temp + graph.weights[(lst[i], lst[i + 1])]
            dst_lst.append(temp)
        print(dst_lst)
        lst = path_list[dst_lst.index(min(dst_lst))]
    else:
        lst = path_list[0]
    print(lst)
    path_list = []
    if (lst == None):
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
    # global g
    # for i in range(0,53):
    #     g.add_vertex(str(i))

    for i in range(0, 24):
        dist = getPathLength(main_lst[i][3], main_lst[i][4], main_lst[i + 1][3], main_lst[i + 1][4])
        graph.add_edge(str(i), str(i + 1), dist)

    for i in range(25, 38):
        dist = getPathLength(main_lst[i][3], main_lst[i][4], main_lst[i + 1][3], main_lst[i + 1][4])
        graph.add_edge(str(i), str(i + 1), dist)

    for i in range(39, 62):
        dist = getPathLength(main_lst[i][3], main_lst[i][4], main_lst[i + 1][3], main_lst[i + 1][4])
        graph.add_edge(str(i), str(i + 1), dist)

    dist = getPathLength(main_lst[0][3], main_lst[0][4], main_lst[25][3], main_lst[25][4])
    graph.add_edge('0', '25', dist)
    dist = getPathLength(main_lst[0][3], main_lst[0][4], main_lst[24][3], main_lst[24][4])
    graph.add_edge('0', '24', dist)
    dist = getPathLength(main_lst[45][3], main_lst[45][4], main_lst[38][3], main_lst[38][4])
    graph.add_edge('45', '38', dist)
    dist = getPathLength(main_lst[2][3], main_lst[2][4], main_lst[26][3], main_lst[26][4])
    graph.add_edge('2', '26', dist)
    dist = getPathLength(main_lst[43][3], main_lst[43][4], main_lst[37][3], main_lst[37][4])
    graph.add_edge('43', '37', dist)


########################### image #################################


def draw_point():
    csv_file = open("mapturning.csv", 'r')
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
    cv2.setMouseCallback("image", draw_circle)
    draw_point()
    graph_creation()
    while (1):
        cv2.imshow('image', img)
        cv2.waitKey(300)
    csv_file.close()
    csv_otput.close()
