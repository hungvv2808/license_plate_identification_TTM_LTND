# DetectChars.py

import cv2
import numpy as np
import math
import random
import os

import Main
import Preprocess
import PossibleChar

# biến cấp mô-đun ##########################################################################

kNearest = cv2.ml.KNearest_create()

# hằng số checkIfPossibleChar, điều này chỉ kiểm tra một ký tự khả thi (không so sánh với một ký tự khác)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

# hằng số để so sánh hai ký tự
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# các hằng số khác
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


###################################################################################################
def loadKNNDataAndTrainKNN():
    allContoursWithData = []  # khai báo danh sách trống,
    validContoursWithData = []  # chúng tôi sẽ điền vào những điều này trong thời gian ngắn

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)  # đọc trong phân loại đào tạo
    except:  # nếu tệp không thể mở được
        print("Error, unable to open classifications.txt, exiting program\n")  # hiển thị thông báo lỗi
        os.system("Pause")
        return False  # và trả về False
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # đọc trong hình ảnh đào tạo
    except:  # nếu tệp không thể mở được
        print("Error, unable to open flattened_images.txt, exiting program\n")  # hiển thị thông báo lỗi
        os.system("Pause")
        return False  # và trả về False
    # end try

    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))  # định hình lại mảng numpy thành 1d, cần chuyển để gọi đến huấn luyện

    kNearest.setDefaultK(1)  # đặt K mặc định thành 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)  # huấn luyện đối tượng KNN

    return True  # nếu chúng tôi đến đây đào tạo đã thành công vì vậy hãy trả về true


# end function

###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:  # nếu danh sách các tấm có thể trống
        return listOfPossiblePlates  # return
    # end if

    # tại thời điểm này, chúng tôi có thể chắc chắn rằng danh sách các tấm có thể có ít nhất một tấm

    for possiblePlate in listOfPossiblePlates:  # cho mỗi tấm có thể, đây là một vòng lặp for lớn chiếm hầu hết chức năng

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(
            possiblePlate.imgPlate)  # tiền xử lý để có được hình ảnh ngưỡng và thang độ xám

        if Main.showSteps == True:  # show steps ###################################################
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if # show steps #####################################################################

        # tăng kích thước của hình ảnh tấm để dễ dàng xem và phát hiện ký tự
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx=1.6, fy=1.6)

        # ngưỡng một lần nữa để loại bỏ mọi vùng xám
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0,
                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.showSteps == True:  # show steps ###################################################
            cv2.imshow("5d", possiblePlate.imgThresh)
        # end if # show steps #####################################################################

        # tìm tất cả các ký tự có thể có trong đĩa, hàm này đầu tiên tìm tất cả các đường bao, sau đó chỉ bao gồm các
        # đường bao có thể là các ký tự (mà chưa so sánh với các ký tự khác)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if Main.showSteps == True:  # show steps ###################################################
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]  # xóa danh sách đường viền

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("6", imgContours)
        # end if # show steps #####################################################################

        # đưa ra một danh sách tất cả các ký tự có thể có, tìm các nhóm ký tự phù hợp trong bảng
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if Main.showSteps == True:  # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if # show steps #####################################################################

        if (len(listOfListsOfMatchingCharsInPlate) == 0):  # nếu không có nhóm ký tự phù hợp nào được tìm thấy trong đĩa

            if Main.showSteps == True:  # show steps ###############################################
                print("Chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if # show steps #################################################################

            possiblePlate.strChars = ""
            continue  # quay lại đầu vòng lặp for
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):  # trong mỗi danh sách các ký tự phù hợp
            listOfListsOfMatchingCharsInPlate[i].sort(
                key=lambda matchingChar: matchingChar.intCenterX)  # sắp xếp các ký tự từ trái sang phải
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(
                listOfListsOfMatchingCharsInPlate[i])  # và xóa các ký tự chồng chéo bên trong
        # end for

        if Main.showSteps == True:  # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if # show steps #####################################################################

        # trong mỗi mảng có thể, giả sử danh sách dài nhất các ký tự phù hợp tiềm năng là danh sách ký tự thực tế
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        # lặp qua tất cả các vectơ của các ký tự phù hợp, lấy chỉ mục của một vectơ có nhiều ký tự nhất
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        # giả sử rằng danh sách ký tự phù hợp dài nhất trong bảng là danh sách ký tự thực tế
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main.showSteps == True:  # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if # show steps #####################################################################

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main.showSteps == True:  # show steps ###################################################
            print("Chars found in plate number " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if # show steps #####################################################################

    # cuối vòng lặp for lớn chiếm hầu hết chức năng

    if Main.showSteps == True:
        print("\n Char detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates


# end function

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []  # đây sẽ là giá trị trả về
    contours = []
    imgThreshCopy = imgThresh.copy()

    # tìm tất cả các đường viền trong tấm
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:  # cho mỗi đường viền
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(
                possibleChar):  # nếu đường viền là một ký tự có thể, lưu ý rằng điều này không so sánh với các ký tự
            # khác (chưa). . .
            listOfPossibleChars.append(possibleChar)  # thêm vào danh sách các ký tự có thể
        # end if
    # end if

    return listOfPossibleChars


# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
    # hàm này là 'lần vượt qua đầu tiên' kiểm tra sơ bộ một đường bao để xem nó có thể là một ký tự hay không,
    # lưu ý rằng chúng tôi chưa (chưa) so sánh ký tự với các ký tự khác để tìm kiếm nhóm
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if


# end function

###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):
    # với hàm này, chúng tôi bắt đầu với tất cả các ký tự có thể có trong một danh sách lớn
    # mục đích của hàm này là sắp xếp lại một danh sách lớn các ký tự thành một danh sách các ký tự phù hợp,
    # lưu ý rằng các ký tự không được tìm thấy trong một nhóm đối sánh không cần phải xem xét thêm
    listOfListsOfMatchingChars = []  # đây sẽ là giá trị trả về

    for possibleChar in listOfPossibleChars:  # cho mỗi ký tự có thể có trong một danh sách lớn các ký tự
        listOfMatchingChars = findListOfMatchingChars(possibleChar,
                                                      listOfPossibleChars)
        # tìm tất cả các ký tự trong danh sách lớn phù hợp với ký tự hiện tại

        listOfMatchingChars.append(possibleChar)
        # cũng thêm các ký tự hiện tại vào danh sách các ký tự phù hợp có thể có hiện tại

        if len(
                listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
            # nếu danh sách các ký tự phù hợp có thể có hiện tại không đủ dài để tạo thành một bảng có thể
            continue  # quay lại đầu vòng lặp for và thử lại với ký tự tiếp theo, lưu ý rằng không cần thiết
            # để lưu danh sách theo bất kỳ cách nào vì nó không có đủ ký tự để trở thành một bảng khả thi
        # end if

        # nếu chúng tôi đến đây, danh sách hiện tại đã vượt qua kiểm tra dưới dạng "nhóm" hoặc "cụm" các ký tự phù hợp
        listOfListsOfMatchingChars.append(
            listOfMatchingChars)  # vì vậy hãy thêm vào danh sách danh sách các ký tự phù hợp của chúng tôi

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        # xóa danh sách hiện tại gồm các ký tự phù hợp khỏi danh sách lớn để chúng tôi không sử dụng các ký tự đó hai
        # lần, hãy đảm bảo tạo một danh sách lớn mới cho việc này vì chúng tôi không muốn thay đổi danh sách lớn ban
        # đầu
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(
            listOfPossibleCharsWithCurrentMatchesRemoved)  # cuộc gọi đệ quy

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            # cho mỗi danh sách các ký tự phù hợp được tìm thấy bằng lệnh gọi đệ quy
            listOfListsOfMatchingChars.append(
                recursiveListOfMatchingChars)  # thêm vào danh sách danh sách các ký tự phù hợp ban đầu của chúng tôi
        # end for

        break  # exit for

    # end for

    return listOfListsOfMatchingChars


# end function

###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
    # mục đích của hàm này là, với một ký tự khả thi và một danh sách lớn các ký tự có thể, tìm tất cả các ký tự trong
    # danh sách lớn phù hợp với các ký tự duy nhất có thể và trả về các ký tự phù hợp đó dưới dạng danh sách
    listOfMatchingChars = []  # đây sẽ là giá trị trả về

    for possibleMatchingChar in listOfChars:  # cho mỗi char trong danh sách lớn
        if possibleMatchingChar == possibleChar:  # nếu biểu đồ chúng tôi cố gắng tìm các kết quả phù hợp hoàn toàn
            # giống nhau
            # char là char trong danh sách lớn mà chúng tôi hiện đang kiểm tra thì chúng tôi không nên đưa nó vào
            # danh sách khớp với b / c sẽ kết thúc gấp đôi bao gồm cả ký tự hiện tại
            continue  # so do not add to list of matches and jump back to top of for loop
        # end if
        # tính toán công cụ để xem liệu các ký tự có khớp không
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(
            abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(
            possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(
            abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(
            possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(
            abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(
            possibleChar.intBoundingRectHeight)

        # kiểm tra xem các ký tự có khớp không
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
                fltChangeInArea < MAX_CHANGE_IN_AREA and
                fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
                fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
            listOfMatchingChars.append(
                possibleMatchingChar)
            # nếu các ký tự trùng khớp, hãy thêm ký tự hiện tại vào danh sách các ký tự phù hợp
        # end if
    # end for

    return listOfMatchingChars  # return result


# end function

###################################################################################################
# sử dụng định lý Pitago để tính khoảng cách giữa hai ký tự
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))


# end function

###################################################################################################
# sử dụng lượng giác cơ bản (SOH CAH TOA) để tính góc giữa các ký tự
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:
        # kiểm tra để đảm bảo chúng ta không chia cho 0 nếu các vị trí tâm X bằng nhau, phép chia float cho 0
        # sẽ gây ra sự cố trong Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)  # nếu liền kề không bằng 0, hãy tính góc
    else:
        fltAngleInRad = 1.5708
        # nếu liền kề là 0, hãy sử dụng giá trị này làm góc, điều này phù hợp với phiên bản C ++ của chương trình này
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)  # tính toán góc theo độ

    return fltAngleInDeg


# end function

# ################################################################################################## nếu chúng ta có
# hai # ký tự chồng chéo hoặc gần nhau để có thể là các ký tự riêng biệt, hãy xóa ký tự bên trong (nhỏ hơn),
# điều này là để ngăn việc bao gồm cùng một ký tự hai lần nếu hai đường viền được tìm thấy cho cùng một ký tự,
# ví dụ đối với # ký tự 'O' cả vòng trong và vòng ngoài có thể được tìm thấy dưới dạng đường viền, nhưng chúng ta chỉ
# nên bao gồm các ký tự một lần
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)  # đây sẽ là giá trị trả về

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:  # nếu char hiện tại và char khác không cùng một char. . .
                # nếu char hiện tại và các char khác có điểm trung tâm ở gần như cùng một vị trí. . .
                if distanceBetweenChars(currentChar, otherChar) < (
                        currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # nếu chúng tôi vào đây, chúng tôi đã tìm thấy các ký tự trùng lặp, tiếp theo chúng tôi xác định
                    # ký tự nào nhỏ hơn, thì nếu char đó chưa được xóa trên thẻ trước đó, hãy xóa nó
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:  # nếu char hiện tại nhỏ hơn
                        # so với các ký tự khác
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:  # nếu char hiện tại chưa có
                            # Đã xóa trên thẻ trước. . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)  # sau đó loại bỏ ký tự hiện tại
                        # end if
                    else:  # khác nếu char khác nhỏ hơn char hiện tại
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:  # nếu char khác chưa có
                            # Đã xóa trên thẻ trước. . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)  # sau đó loại bỏ các ký tự khác
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved


# end function

###################################################################################################
# đây là nơi chúng tôi áp dụng nhận dạng char thực tế
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""  # đây sẽ là giá trị trả về, các ký tự trong bảng lic

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)  # sắp xếp các ký tự từ trái sang phải

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR,
                 imgThreshColor)  # tạo phiên bản màu của hình ảnh ngưỡng để chúng tôi có thể vẽ các đường viền bằng
    # màu trên đó

    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth),
               (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)  # vẽ hộp màu xanh lá cây xung quanh char

        # cắt biểu đồ ra khỏi ngưỡng hình ảnh
        imgROI = imgThresh[
                 currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                 currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (
            RESIZED_CHAR_IMAGE_WIDTH,
            RESIZED_CHAR_IMAGE_HEIGHT))  # thay đổi kích thước hình ảnh, điều này là cần thiết để nhận dạng ký tự

        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))  # làm phẳng hình ảnh thành mảng numpy 1d

        npaROIResized = np.float32(npaROIResized)  # chuyển đổi từ mảng ints 1d numpy thành mảng float 1d numpy

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                     k=1)  # cuối cùng chúng ta có thể gọi findNearest !!!

        strCurrentChar = str(chr(int(npaResults[0][0])))  # lấy nhân vật từ kết quả

        strChars = strChars + strCurrentChar  # nối ký tự hiện tại vào chuỗi đầy đủ

    # end for

    if Main.showSteps == True:  # show steps #######################################################
        cv2.imshow("10", imgThreshColor)
    # end if # show steps #########################################################################

    return strChars
# end function
