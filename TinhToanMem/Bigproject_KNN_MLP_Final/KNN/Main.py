# Main.py

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate

# biến cấp mô-đun ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


###################################################################################################
def main(filePath, nameFile):
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # cố gắng đào tạo KNN

    if not blnKNNTrainingSuccessful:  # nếu đào tạo KNN không thành công
        print("\n Error: KNN traning was not successful\n")  # hiển thị thông báo lỗi
        return  # và thoát khỏi chương trình
    # end if

    imgOriginalScene = cv2.imread(filePath)  # open image

    if imgOriginalScene is None:  # nếu hình ảnh không được đọc thành công
        print("\n Error: image not read from file \n\n")  # in thông báo lỗi tới stdout
        os.system("Pause")  # tạm dừng để người dùng có thể thấy thông báo lỗi
        return  # và thoát khỏi chương trình
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # phát hiện tấm

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # phát hiện ký tự trong đĩa

    # cv2.imshow("imgOriginalScene", imgOriginalScene)  # hiển thị hình ảnh cảnh

    if len(listOfPossiblePlates) == 0:  # nếu không có tấm nào được tìm thấy
        print("\n No license plates were detected\n")  # thông báo cho người dùng không tìm thấy tấm
    else:  # else
        # nếu chúng ta vào đây danh sách các hành tinh có thể có ít nhất một đĩa
        # sắp xếp danh sách các bảng có thể có theo thứ tự MONG MUỐN (số ký tự nhiều nhất đến số ký tự ít nhất)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # giả sử đĩa có các ký tự được nhận dạng nhiều nhất (đĩa đầu tiên được sắp xếp theo độ dài chuỗi sắp xếp
        # giảm dần) là tấm thực tế
        licPlate = listOfPossiblePlates[0]

        # cv2.imshow("imgPlate", licPlate.imgPlate)  # hiển thị cắt tấm và ngưỡng của tấm
        # cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:  # nếu không có ký tự nào được tìm thấy trong đĩa
            print("\n No characters were detected\n\n")  # show message
            return  # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # vẽ hình chữ nhật màu đỏ xung quanh đĩa

        print("\nName image: " + nameFile)
        print("\nLicense plate read from image = " + licPlate.strChars + "\n")  # viết văn bản biển số xe lên stdout
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)  # viết văn bản biển số xe trên hình ảnh

        # cv2.imshow(nameFile, imgOriginalScene)  # hiển thị lại hình ảnh cảnh

        cv2.imwrite("./output_img/IMG_Result_" + nameFile, imgOriginalScene)  # ghi hình ảnh ra tệp

    # end if else

    # cv2.waitKey(0)  # hold windows open until user presses a key

    return


# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # lấy 4 đỉnh của trực thuộc đã quay

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)  # vẽ 4 đường màu đỏ
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0  # đây sẽ là trung tâm của khu vực mà văn bản sẽ được viết
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0  # đây sẽ là phần dưới cùng bên trái của khu vực mà văn bản sẽ được ghi vào
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # chọn một phông chữ jane đơn giản
    fltFontScale = float(plateHeight) / 30.0  # tỷ lệ phông chữ cơ sở trên chiều cao của vùng tấm
    intFontThickness = int(round(fltFontScale * 1.5))  # độ dày phông chữ cơ bản trên thang phông chữ

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  # gọi getTextSize

    # giải nén hình chữ nhật đã xoay thành trung tâm, chiều rộng và chiều cao và góc
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)  # đảm bảo tâm là một số nguyên
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)  # vị trí ngang của vùng văn bản giống như tấm

    if intPlateCenterY < (sceneHeight * 0.75):  # nếu biển số xe nằm ở 3/4 phía trên của hình ảnh
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # viết các ký tự vào bên dưới tấm
    else:  # khác nếu biển số xe nằm ở 1/4 dưới của hình ảnh
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # viết các ký tự phía trên đĩa
    # end if

    textSizeWidth, textSizeHeight = textSize  # giải nén chiều rộng và kích thước văn bản

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))  # tính toán gốc dưới bên trái của vùng văn bản
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))  # dựa trên trung tâm vùng văn bản, chiều rộng và chiều cao

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


# end function

###################################################################################################
if __name__ == "__main__":
    file_paths = os.listdir("./dataset")
    for i in range(len(file_paths)):
        print(f"index: {i+1}")
        main("./dataset/" + file_paths[i], file_paths[i])
