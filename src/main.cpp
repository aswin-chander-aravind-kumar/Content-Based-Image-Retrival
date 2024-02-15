#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "functions.h"

using namespace cv;
using namespace std;

// Define target image paths for each task
#define TARGET_IMAGE_PATH_TASK_1 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.1016.jpg"
#define TARGET_IMAGE_PATH_TASK_2 "/Users/aswinchanderaravindkumar/Desktop/IMG_3736.tif" 
#define TARGET_IMAGE_PATH_TASK_21 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0164.jpg" 
#define TARGET_IMAGE_PATH_TASK_3 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0274.jpg" 
#define TARGET_IMAGE_PATH_TASK_31 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0274.jpg" 
#define TARGET_IMAGE_PATH_TASK_32 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0274.jpg" 
#define TARGET_IMAGE_PATH_TASK_4 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0535.jpg"
#define TARGET_IMAGE_PATH_TASK_41 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0628.jpg"
#define TARGET_IMAGE_PATH_TASK_42 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0746.jpg"
#define TARGET_IMAGE_PATH_TASK_5 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0893.jpg"
#define TARGET_IMAGE_PATH_TASK_51 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0164.jpg"
#define TARGET_IMAGE_PATH_TASK_6 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.1062.jpg"
#define TARGET_IMAGE_PATH_TASK_7 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0746.jpg"
#define TARGET_IMAGE_PATH_TASK_8 "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/olympus/pic.0164.jpg"
#define RESULTS_DIRECTORY "/Users/aswinchanderaravindkumar/Desktop/CV_Project_2/src/results"

// Tasks Description:

// Task 51: Deep Netwprk Embeddings with image 0164 using cosine distance
// Task 5:  Deep Netwprk Embeddings with image 0893 using cosine distance
// Task 42 : Extension - Features of one or more cooccurence matrices : energy ,entropy contract,homogenity and max probability
// Task 41 : Extension - Features extracted from a Fourier transform of the image , such as resizing the power spectrum to a 16x16 image
// Task 4 : Used whole image histogram and whole image texture histogram 
// Task 32: Applying one histogram on top and the eother at bottom
// Task 31: Using centered region histogram and hue saturation histogram
// Task 3: whole image and cetered image histogram


int main(int argc, char* argv[]) {
    if (argc < 5 ) {
        cout << "Usage: " << argv[0] << " <directory path> <top N matches> <task number> <embeddings file path>" << endl;
        return -1;
    }


    
    string directoryPath = argv[1];
    int topN = stoi(argv[2]); // Convert the command-line argument to an integer
    int taskNumber = stoi(argv[3]);
    string embeddingsFilePath = argv[4];
    vector<string> imageFiles;
    readFiles(directoryPath, imageFiles);

    // Select the target image based on the task number
    string targetImagePath;
    if (taskNumber == 1) {
        targetImagePath = TARGET_IMAGE_PATH_TASK_1;
    } else if (taskNumber == 2) {
        targetImagePath = TARGET_IMAGE_PATH_TASK_2;
    } else if (taskNumber == 21) {
        targetImagePath = TARGET_IMAGE_PATH_TASK_21;
    }else if (taskNumber == 3){
        targetImagePath = TARGET_IMAGE_PATH_TASK_3;
    } else if (taskNumber == 31){
        targetImagePath = TARGET_IMAGE_PATH_TASK_31;
    }else if (taskNumber == 32){
        targetImagePath = TARGET_IMAGE_PATH_TASK_32;
    }else if (taskNumber == 4){
        targetImagePath = TARGET_IMAGE_PATH_TASK_4;
    }else if (taskNumber == 41){
        targetImagePath = TARGET_IMAGE_PATH_TASK_41;
    }else if (taskNumber == 42){
        targetImagePath = TARGET_IMAGE_PATH_TASK_42;
    }else if (taskNumber == 5){
        targetImagePath = TARGET_IMAGE_PATH_TASK_5;
    }else if (taskNumber == 51){
        targetImagePath = TARGET_IMAGE_PATH_TASK_51;
    }else if (taskNumber == 6){
        targetImagePath = TARGET_IMAGE_PATH_TASK_6;
    }else if (taskNumber == 7){
        targetImagePath = TARGET_IMAGE_PATH_TASK_7;
    }else if (taskNumber == 8){
        targetImagePath = TARGET_IMAGE_PATH_TASK_8;
    }else {
        cerr << "Invalid task number." << endl;
        return -1;
    }

    vector<pair<string, float>> ssdScores;
    pipeline(targetImagePath, imageFiles, ssdScores, topN, taskNumber,embeddingsFilePath);

    // Display the top N matches
    cout << "Top " << topN << " matches for task " << taskNumber << ":" << endl;
    for (int i = 0; i < topN && i < ssdScores.size(); ++i) {
        cout << ssdScores[i].first << " - Score: " << ssdScores[i].second << endl;
    }

    // Saving the top N matches
    saveMatchedImages(cv::imread(targetImagePath, cv::IMREAD_COLOR), ssdScores, RESULTS_DIRECTORY, topN, taskNumber);

    return 0;
}