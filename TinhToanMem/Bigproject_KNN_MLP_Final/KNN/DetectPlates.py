# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# biến cấp mô-đun ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


###################################################################################################
def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []  # đây sẽ là giá trị trả về

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.showSteps:  # show steps #######################################################
        cv2.imshow("0", imgOriginalScene)
    # end if # show steps #########################################################################

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(
        imgOriginalScene)  # tiền xử lý để có được hình ảnh ngưỡng và thang độ xám

    if Main.showSteps:  # show steps #######################################################
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    # end if # show steps #########################################################################

    # tìm tất cả các ký tự có thể có trong cảnh, chức năng này trước tiên tìm tất cả các đường bao, sau đó chỉ bao
    # gồm các đường bao có thể là các ký tự (chưa có so sánh với các ký tự khác)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if Main.showSteps:  # show steps #######################################################
        print("Step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))  # 131 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    # end if # show steps #########################################################################

    # đã đưa ra danh sách tất cả các ký tự có thể có, tìm các nhóm ký tự phù hợp
    # trong các bước tiếp theo, mỗi nhóm ký tự phù hợp sẽ cố gắng được nhận dạng là một bảng
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps:  # show steps #######################################################
        print("Step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  # 13 với hình ảnh MCLRNF1

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if
    # show steps #########################################################################

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:  # cho mỗi nhóm ký tự phù hợp
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)  # cố gắng giải nén

        if possiblePlate.imgPlate is not None:  # nếu đĩa được tìm thấy
            listOfPossiblePlates.append(possiblePlate)  # thêm vào danh sách các tấm có thể
        # end if
    # end for

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  # 13 với hình ảnh MCLRNF1

    if Main.showSteps:  # show steps #######################################################
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("Possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print("\n Plate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
    # end if # show steps #########################################################################

    return listOfPossiblePlates


# end function

###################################################################################################
def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []  # đây sẽ là giá trị trả về

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST,
                                                           cv2.CHAIN_APPROX_SIMPLE)  # tìm tất cả các đường viền

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):  # cho mỗi đường viền

        if Main.showSteps:  # show steps ###################################################
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
        # end if # show steps #####################################################################

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(
                possibleChar):  # nếu đường viền là một ký tự có thể, lưu ý rằng điều này không so sánh với các ký tự
            # khác (chưa). . .
            intCountOfPossibleChars = intCountOfPossibleChars + 1  # số gia tăng của các ký tự có thể
            listOfPossibleChars.append(possibleChar)  # và thêm vào danh sách các ký tự có thể
        # end if
    # end for

    if Main.showSteps:  # show steps #######################################################
        print("\n Step 2 - len(contours) = " + str(len(contours)))  # 2362 với hình ảnh MCLRNF1
        print("Step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))  # 131 với hình ảnh MCLRNF1
        cv2.imshow("2a", imgContours)
    # end if # show steps #########################################################################

    return listOfPossibleChars


# end function


###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()  # đây sẽ là giá trị trả về

    listOfMatchingChars.sort(
        key=lambda matchingChar: matchingChar.intCenterX)  # sắp xếp các ký tự từ trái sang phải dựa trên vị trí x

    # tính toán điểm trung tâm của tấm
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # tính toán chiều rộng và chiều cao tấm
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[
                             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # tính toán góc hiệu chỉnh của vùng tấm
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0],
                                                     listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # đóng gói điểm trung tâm khu vực tấm, chiều rộng và chiều cao và góc hiệu chỉnh vào biến thành viên trực thuộc
    # xoay của tấm
    possiblePlate.rrLocationOfPlateInScene = (
        tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)

    # các bước cuối cùng là thực hiện luân chuyển thực tế

    # lấy ma trận xoay cho góc hiệu chỉnh được tính toán của chúng tôi
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape  # giải nén chiều rộng và chiều cao của hình ảnh gốc

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))  # xoay toàn bộ hình ảnh

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped  # sao chép hình ảnh đĩa đã cắt vào biến thành viên áp dụng của
    # tấm có thể

    return possiblePlate
# end function
