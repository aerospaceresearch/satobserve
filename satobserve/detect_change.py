import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from scipy import ndimage
import scipy as sp

import pickle

class satellite:
    def __init__(self, name='Unknown'):
        self.name = name

        self.by_light = []
        self.by_move = []
        self.detected = []
        self.detected_over_steps = []
        self.by_light_over_steps = []

        self.object_list = []

        self.countID = 0
        self.frame_no = 0


    def add_step_data(self):
        self.detected_over_steps.append([self.frame_no, self.detected])
        self.by_light_over_steps.append([self.frame_no, self.by_light])

    def restore_data(self, filename):
        print("test")
        with (open(filename, "rb")) as openfile:
            data = pickle.load(openfile)
            print(data)

            self.by_light_over_steps = data["by_light_over_steps"]
            self.detected_over_steps = data["detected_over_steps"]
            self.detected = data["detected"]
            self.countID = data["countID"]
            self.object_list = data["object_list"]
            self.frame_no = data["frame_no"] + 1


    def store_data(self, filename):
        data = {}
        data["by_light_over_steps"] = self.by_light_over_steps
        data["detected_over_steps"] = self.detected_over_steps
        data["detected"] = self.detected
        data["countID"] = self.countID
        data["object_list"] = self.object_list
        data["frame_no"] = self.frame_no

        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



    def add_by_light(self, sats):

        self.by_light = sats


    def detect(self, sats):

        if len(self.detected) == 0:

            for i in range(len(sats)):
                sats[i].append(1) # counter of detection
                sats[i].append(self.countID)

                self.detected.append(sats[i])
                self.countID += 1


        else:
            # adding the new sats into the detected
            for i in range(len(sats)):
                found = -1

                for j in range(len(self.detected)):
                    if sats[i][0] == self.detected[j][0] and sats[i][1] == self.detected[j][1]:
                        #self.detected[j][-2] = self.detected[j][-2] + 1 not always counting. put somewhere else
                        found = j


                if found == -1:
                    sats[i].append(self.countID)
                    self.detected.append(sats[i])

                    self.countID += 1

    def move_it(self, frame0):

        next = []

        for i in range(len(self.detected)):
            xn = self.detected[i][0] + self.detected[i][3]
            yn = self.detected[i][1] + self.detected[i][4]

            minc = xn - 5
            maxc = xn + 5
            minr = yn - 5
            maxr = yn + 5

            if minr > 0 and maxr < len(frame0) and minc > 0 and maxc < len(frame0[0]):

                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                # ax.plot(bx, by, '-g', linewidth=2.5)
                # ax.plot(xn, yn, '.g', markersize=10)

                starsize = 15
                pascal = pascal_triangle(starsize)
                section = frame0[int(minr):int(maxr), int(minc):int(maxc)]
                scale = 3
                section = upscale(section, scale)
                section = star(section, pascal)
                section_max = np.max(section)
                #section_mean_old = np.mean(section)

                result = np.where(section == section_max)
                x_moved = int(minc) + np.mean(result[1]) / scale
                y_moved = int(minr) + np.mean(result[0]) / scale
                x_dif = ((x_moved - xn) + self.detected[i][3]) / 2.0
                y_dif = ((y_moved - yn) + self.detected[i][4]) / 2.0
                # ax.plot(x_moved, y_moved, '.b', markersize=15)
                # ax.plot(now[next][0], now[next][1], '.r', markersize=15)

                ## todo: star is asumed in the middle, but it is not! change the code
                mini_box = starsize // 2
                section_mean = (np.sum(section) - \
                                np.sum(section[int(np.mean(result[1])) - (mini_box): int(np.mean(result[1])) + (mini_box),
                                       int(np.mean(result[0])) - (mini_box): int(np.mean(result[0])) + (mini_box)])) / \
                               (len(section) * len(section[0]) - (starsize ** 2))

                print("snrX", section_mean, len(section) * len(section[0]),
                      len(section) * len(section[0]) - (starsize ** 2), starsize ** 2)
                snr = section_max / section_mean

                #if self.detected[i][-2] < 3:

                window_decay = 0

                if snr >= 1.4 and (x_dif**2 + y_dif**2)**0.5 > min_distance / 2:

                    next.append([x_moved, y_moved, (snr + self.detected[i][2]) / 2.0, x_dif, y_dif, section_max, section_mean, window_decay, self.detected[i][-2], self.detected[i][-1]])

        self.detected = next


    def check_for_duplicates(self):

        counter = 0
        max_run = len(self.detected)
        run = 1
        if len(self.detected) > 1:
            while counter < max_run and run == 1:

                for i in range(counter+1, len(self.detected), 1):
                    if self.detected[counter][0] == self.detected[i][0] and self.detected[counter][1] == self.detected[i][1]:
                        print("double", counter, i, self.detected[counter], self.detected[i], len(self.detected))
                        del self.detected[i]
                        break

                counter += 1
                if counter >= len(self.detected):
                    run = 0



def pascal_triangle(level):

    triangle =[[1], [1,1]]


    for l in range(level-2):
        line = []
        line.append(1)
        for i in range(len(triangle[-1])-1):
            line.append(triangle[-1][i] + triangle[-1][i+1])
        line.append(1)

        triangle.append(line)

    return triangle[-1]


def star(x, pascal = [1, 6, 15, 20, 15, 6, 1]):
    #pascal = [1, 8, 28, 56, 70, 56, 28, 8, 1]
    #pascal = [1, 6, 15, 20, 15, 6, 1]
    #pascal = [1, 4, 6, 4, 1]
    kernel = np.zeros((len(pascal), len(pascal)))
    for i in range(len(pascal)):
        kernel[0][i] = pascal[i]
        kernel[-1][i] = pascal[i]
        kernel[i][0] = pascal[i]
        kernel[i][-1] = pascal[i]

    for i in range(1, len(pascal) - 1, 1):
        for j in range(1, len(pascal) - 1, 1):
            kernel[i][j] = pascal[i] * pascal[j]

    #print(kernel)

    weights3 = np.array(kernel, dtype=np.float)
    weights3 = weights3 / np.sum(weights3[:])

    y = sp.ndimage.filters.convolve(x, weights3, mode='constant')
    return y


def upscale(image, scale):

    image_upscape = []

    for r in range(len(image)):
        for i in range(scale):

            tmp =[]

            for c in range(len(image[r])):
                for j in range(scale):
                    tmp.append(image[r][c])

            image_upscape.append(tmp)



    return np.array(image_upscape)


def multi_dil(im, num, element = np.array([[0,1,0], [1,1,1], [0,1,0]])):
    for i in range(num):
        im = dilation(im, element)
    return im

def multi_ero(im, num, element = np.array([[0,1,0], [1,1,1], [0,1,0]])):
    for i in range(num):
        im = erosion(im, element)
    return im



satellite = satellite()

cap = cv2.VideoCapture("output9.MP4")
filename_out = "output9.mp4.pickle"
framenumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)

now = []
next_window = []
move_window = []

#satellite.restore_data(filename_out)

satellite.frame_no = int(4.51*60*25)

#object_list = []

for step in range(satellite.frame_no, int(framenumber)-1, 1):
    satellite.frame_no = step
    # cap.set(2, frame_no)


    cap.set(cv2.CAP_PROP_POS_FRAMES, satellite.frame_no)
    ret, frame = cap.read()
    frame0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 1.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, satellite.frame_no+1)
    ret, frame = cap.read()
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 1.0

    frame_dif = np.subtract(frame1, frame0)
    frame_dif = np.abs(frame_dif)

    number_mean = np.mean(frame_dif)
    number_max = np.max(frame_dif)
    number_std = np.std(frame_dif)
    print(number_mean, number_std, number_max)

    frame_dif[frame_dif < number_mean + number_std*6] = 0.0
    frame_dif[frame_dif > 0.0] = 255.0

    # filling the holes
    frame_dif = ndimage.binary_fill_holes(frame_dif)*1.0
    frame_dif1 = frame_dif

    frame_dif = multi_dil(frame_dif, 2)
    frame_dif = multi_ero(frame_dif, 2)

    #frame_dif = np.add(frame_dif, frame_dif1)

    #plt.imshow(frame_dif)
    #plt.show()

    label_img = label(frame_dif)
    regions = regionprops(label_img)
    print("regions", len(regions))

    #fig, ax = plt.subplots()
    #ax.imshow(frame_dif)

    test = np.zeros((1080, 1920))

    xl = []
    yl = []

    xll = []
    yll = []

    tmp1 = []

    for props in regions:
        y0, x0 = props.centroid

        box = 5
        if props.area > 1 \
                and x0 >= box and y0 >= box and x0 < len(frame1[0]) - box and y0 < len(frame1) - box:

            xll.append(x0)
            yll.append(y0)


            #ax.plot(x0, y0, '.g', markersize=10)
            tmp = frame1[int(y0)-box:int(y0)+box, int(x0)-box:int(x0)+box]

            starsize = 15
            pascal = pascal_triangle(starsize)
            section = tmp
            scale = 3
            section = upscale(section, scale)
            section = star(section, pascal)
            section_max = np.max(section)
            section_mean = np.mean(section)
            # section_mean_old = np.mean(section)

            result = np.where(section == section_max)
            x_moved = int(x0) + np.mean(result[1]) / scale - box
            y_moved = int(y0) + np.mean(result[0]) / scale - box

            snr = section_max / section_mean
            if snr > 1.7:
                xl.append(x_moved)
                yl.append(y_moved)

                tmp1.append(
                    {"area": props.area, "centroid": props.centroid, "bbox": props.bbox, "moved": [x_moved, y_moved],
                     "snr": snr})

            test[int(y0)-box:int(y0)+box, int(x0)-box:int(x0)+box] = tmp


            #fig, ax = plt.subplots()
            #ax.plot(np.mean(result[1]), np.mean(result[0]), '.g', markersize=10)
            #ax.imshow(section)
            #plt.show()

    #plt.show()

    print(step, len(xl), len(xll))
    '''
    fig, ax = plt.subplots()
    ax.imshow(test)

    for i in range(len(xl)):
        ax.plot(xl[i], yl[i], '.g', markersize=3)

    for i in range(len(xll)):
        ax.plot(xll[i], yll[i], '.r', markersize=2)
    plt.show()
    '''

    satellite.object_list.append(tmp1)


    img = np.array(frame1, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    min_distance = 0.3
    max_distance = 2.1

    for r in range(len(satellite.object_list) - 2):

        sat_confirmed_step = []
        sat_moved_step = []
        next_window = []
        now = []

        time1 = time.time()

        #fig, ax = plt.subplots()
        #ax.imshow(test)#, cmap=plt.cm.gray)

        satellite.move_it(frame1)
        satellite.check_for_duplicates()


        for p0 in range(len(satellite.object_list[r])):




            if 0 == 0:
                #print(object_list[r][p0]["moved"])
                x0 = satellite.object_list[r][p0]["moved"][0]
                y0 = satellite.object_list[r][p0]["moved"][1]


                box = satellite.object_list[r][p0]["bbox"]

                distance_total = -1
                distance_min1 = 0
                distance_min2 = 0

                x0_min = 0
                y0_min = 0

                x1_min = 0
                y1_min = 0

                x2_min = 0
                y2_min = 0

                for p1 in range(len(satellite.object_list[r + 1])):

                    #y1, x1 = object_list[r + 1][p1]["centroid"]
                    x1 = satellite.object_list[r+1][p1]["moved"][0]
                    y1 = satellite.object_list[r+1][p1]["moved"][1]

                    distance1 = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
                    if distance1 > min_distance and distance1 < max_distance:

                        for p2 in range(len(satellite.object_list[r + 2])):

                            #y2, x2 = object_list[r + 2][p2]["centroid"]
                            x2 = satellite.object_list[r + 2][p2]["moved"][0]
                            y2 = satellite.object_list[r + 2][p2]["moved"][1]

                            distance2 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

                            if distance2 > min_distance and distance2 < max_distance:

                                if distance_total == -1:
                                    distance_total = distance1 + distance2
                                    distance_min1 = distance1
                                    distance_min2 = distance2

                                    x0_min = x0
                                    y0_min = y0

                                    x1_min = x1
                                    y1_min = y1

                                    x2_min = x2
                                    y2_min = y2

                                    box = satellite.object_list[r + 2][p2]["bbox"]

                                else:
                                    if distance_total > distance1 + distance2:
                                        distance_total = distance1 + distance2
                                        distance_min1 = distance1
                                        distance_min2 = distance2

                                        x0_min = x0
                                        y0_min = y0

                                        x1_min = x1
                                        y1_min = y1

                                        x2_min = x2
                                        y2_min = y2

                                        box = satellite.object_list[r + 2][p2]["bbox"]

                #if distance_total > -1:
                #    print("ggg", r, p0, x0_min, y0_min, x1_min, y1_min, x2_min, y2_min, distance_total, distance_min1, distance_min2, satellite.object_list[r][p0]["area"], satellite.object_list[r + 1][p1]["area"], satellite.object_list[r + 2][p2]["area"])



                if distance_total > -1:
                    #frame0 = np.array(frame0)


                    angle = ((x1_min - x0_min) * (x1_min - x2_min) + (y1_min - y0_min) * (y1_min - y2_min)) / \
                            (((x1_min - x0_min) ** 2 + (y1_min - y0_min) ** 2) ** 0.5 *
                             ((x1_min - x2_min) ** 2 + (y1_min - y2_min) ** 2) ** 0.5)
                    #print("angle", angle)

                    angle = np.arccos(angle) * 180.0 / np.pi
                    if np.isnan(angle) == True:
                        #print("angle1", angle)
                        angle = 180.

                    #print("angle2", angle)

                    #ax.plot(x0_min, y0_min, '.b', markersize=3)
                    #ax.plot(x1_min, y1_min, '.b', markersize=6)
                    #ax.plot(x2_min, y2_min, '.b', markersize=9)

                    if angle > 180-30:

                        #ax.plot(x0_min, y0_min, '.r', markersize=4)
                        #ax.plot(x1_min, y1_min, '.r', markersize=7)
                        #ax.plot(x2_min, y2_min, '.r', markersize=10)

                        direction = np.array([x1_min, y1_min])
                        direction = np.add(direction, np.multiply(np.array([x2_min - x1_min, y2_min - y1_min]), 1))
                        next_window.append(direction)

                        direction = np.array([x1_min, y1_min])
                        x_dif = x2_min - x1_min
                        y_dif = y2_min - y1_min
                        direction = np.add(direction, np.multiply(np.array([x_dif, y_dif]), 1))

                        xn = direction[0]
                        yn = direction[1]

                        minc = xn - 6
                        maxc = xn + 6
                        minr = yn - 6
                        maxr = yn + 6

                        if minr > 0 and maxr < len(frame1) and minc > 0 and maxc < len(frame1[0]):
                            # drawing boxes next to the window limits makes the code crash.

                            pascal = pascal_triangle(15)
                            section = frame1[int(minr):int(maxr), int(minc):int(maxc)]
                            scale = 3
                            section = upscale(section, scale)
                            section = star(section, pascal)
                            section_max = np.max(section)
                            section_mean = np.mean(section)
                            snr = section_max / section_mean
                            print("snr", xn, yn, section_max / np.mean(section))
                            result = np.where(section == section_max)
                            window_decay = 0
                            now1 = [int(minc) + np.mean(result[1]) / scale,
                                    int(minr) + np.mean(result[0]) / scale,
                                    snr,
                                    x_dif, y_dif,
                                    section_max,
                                    section_mean,
                                    window_decay]

                            now.append(now1)

                            minr, minc, maxr, maxc = box

                            bx = (minc, maxc, maxc, minc, minc)
                            by = (minr, minr, maxr, maxr, minr)
                            # ax.plot(bx, by, '-r', linewidth=2.5)
                            # print(bx, by)

                            start_point = (minc, minr)
                            end_point = (maxc, maxr)

                            # Blue color in BGR
                            color = (0, 0, 255)
                            # Line thickness of 2 px
                            thickness = 1
                            # Using cv2.rectangle() method
                            # Draw a rectangle with blue line borders of thickness of 2 px
                            img = cv2.rectangle(img, start_point, end_point, color, thickness)

        # predicted windows
        print("sss", step, len(satellite.detected))
        for ii in range(len(satellite.detected)):

            satellite.detected[ii][-2] = satellite.detected[ii][-2] + 1

            xn = satellite.detected[ii][0]
            yn = satellite.detected[ii][1]
            minc = xn - 7
            maxc = xn + 7
            minr = yn - 7
            maxr = yn + 7

            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            #ax.plot(bx, by, '-g', linewidth=1.5)
            # print(bx, by)

            start_point = (int(minc), int(minr))
            end_point = (int(maxc), int(maxr))

            # Green color in BGR
            if satellite.detected[ii][-2] > 25:
                color = (0, 255, 0)
                print(step, ii, satellite.detected[ii][-2], "green")
            else:
                color = (0, 255, 255) #yellow
                print(step, ii, satellite.detected[ii][-2], "yellow")
            # Line thickness of 2 px
            thickness = 1
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            xbox = int(minc) + 19
            ybox = int(minr) + 10
            if xbox < 0:
                xbox = 0
            if ybox < 0:
                ybox = 0
            org = (xbox, ybox)
            lineType = 1
            img = cv2.putText(img, str(satellite.detected[ii][-1]), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                              1, lineType)



        #plt.show()

    path = "test2/test2_" + str(step) + ".jpg"
    cv2.imwrite(path, img)



    satellite.add_by_light(now)
    if len(satellite.by_light) > 0:
        satellite.detect(satellite.by_light)

    print("by light", satellite.countID)
    print(satellite.by_light)
    print(satellite.detected)

    if len(satellite.object_list) >= 3:
        del satellite.object_list[0]

    satellite.add_step_data()


    if step%5==0:
        satellite.store_data(filename_out)