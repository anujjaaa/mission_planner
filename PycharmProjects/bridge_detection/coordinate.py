from math import cos, sin, atan2, sqrt, radians, degrees, asin, modf, log, pi, tan
#from pykml import parser
#from os import path
g_list = []

def getPathLength(lat1, lng1, lat2, lng2):
    '''calculates the distance between two lat, long coordinate pairs'''
    R = 6371000 #radius of earth in m
    lat1rads = radians(lat1)
    lat2rads = radians(lat2)
    deltaLat = radians((lat2-lat1))
    deltaLng = radians((lng2-lng1))
    a = sin(deltaLat/2) * sin(deltaLat/2) + cos(lat1rads) * cos(lat2rads) * sin(deltaLng/2) * sin(deltaLng/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c
    return d


def getDestinationLatLong(lat,lng,azimuth,distance):
    '''returns the lat an long of destination point
    given the start lat, long, aziuth, and distance'''
    R = 6378.1 #Radius of the Earth in km
    brng = radians(azimuth) #Bearing is degrees converted to radians.
    d = distance/1000 #Distance m converted to km

    lat1 = radians(lat) #Current dd lat point converted to radians
    lon1 = radians(lng) #Current dd long point converted to radians
#
    lat2 = asin( sin(lat1) * cos(d/R) + cos(lat1)* sin(d/R)* cos(brng))

    lon2 = lon1 + atan2(sin(brng) * sin(d/R)* cos(lat1),
           cos(d/R)- sin(lat1)* sin(lat2))

    #convert back to degrees
    lat2 = degrees(lat2)
    lon2 = degrees(lon2)

    return[lat2, lon2]


def main(interval, azimuth, lat1, lng1, lat2, lng2):
    '''
    returns every coordinate pair in between two coordinate pairs given with certain interval
    '''
    coords = []
    d = getPathLength(lat1,lng1,lat2,lng2)
    remainder, dist = modf((d / interval))
    counter = 1.0
    coords.append([lat1,lng1])
    for distance in range(1, int(dist)):
        c = getDestinationLatLong(lat1, lng1, azimuth, counter)
        counter += 1.0
        coords.append(c)
    counter += 1
    coords.append([lat2, lng2])
    return coords

def get_azimuth(lat1, lng1, lat2, lng2):
    dLong = abs(lng2 - lng1)
    dPhi = log(tan(lat2 / 2.0 + (pi / 4.0)) / tan(lat1 / 2.0 + (pi / 4.0)))
    # if abs(dLong) > pi:
    #     dLong = -(2.0 * pi - dLong)
    #     # print('greater', dLong)
    # else:
    #     dLong = 2.0 * pi + dLong
    #     # print('less', dLong)
    # #     dLong = (dLong > 0.0) ? -(2.0 * pi - dLong): (2.0 * pi + dLong)
    azimuth = (degrees(atan2(dLong, dPhi)) + 360) % (360)
    print('azimuth value : ', azimuth)
    coords = main(interval, azimuth, lat1, lng1, lat2, lng2)
    # print(coords)
    for val in coords:
        g_list.append(",".join(str(round(float(v), 6)) for v in val))
    print(g_list)


if __name__ == "__main__":
    #point interval in meters
    interval = 1
    #direction of line in degrees
    # azimuth = 50#89.22#343.75#279.91
    #open kml file to get coordinates
    # coordinates = ['73.955936,18.5415196|73.9563052,18.5415137|73.9567464,18.5415138|73.9570642,18.5415112|73.9571313,18.541557|73.9571984,18.5416155|73.9572453,18.5416295|73.9573003,18.5416714|73.9572708,18.5417782|73.9572574,18.5418927|73.9571501,18.5418838|73.9569798,18.5418749|73.9568653,18.541868|73.9565023,18.5418596|73.9561215,18.5418583|73.9561107,18.5419957|73.9560343,18.5419995|73.9558264,18.5420046|73.9555917,18.5420058|73.9553915,18.5420015']
    # lon_lat = coordinates[0].split('|')
    # for i in range(len(lon_lat)-1):
        #start point

    lat1 = 18.542219#float((lon_lat[i].split(','))[1])#18.541545
    lng1 = 73.957323#float((lon_lat[i].split(','))[0])#73.956031
        # print(lat1, lng1)
        # #end point
   # 18.542132, 73.957373
    18.542254,
    lat2 = 18.542132#float((lon_lat[i+1].split(','))[1])#18.541528
    lng2 = 73.957441#float((lon_lat[i+1].split(','))[0])#73.956598
        # print(lat2, lng2)
    get_azimuth(lat1, lng1, lat2, lng2)
    # coords = main(interval, azimuth, lat1, lng1, lat2, lng2)
    # print(coords)


    # print(azimuth)