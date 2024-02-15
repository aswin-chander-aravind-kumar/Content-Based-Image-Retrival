#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>



//function for calculating sum-of-squared-diffrence
float SSD(const cv::Mat &src, const cv::Mat &dst);



// Function to read image files from a directory and store their paths in a vector
int readFiles(const std::string& dirname, std::vector<std::string>& files);



//Saves the target image and top N matched images based on SSD (Sum of Squared Differences) scores to a specified results directory.
// The function assumes that the 'ssdScores' vector is already sorted in ascending order,so the image with the smallest SSD score is considered the best match.
void saveMatchedImages(const cv::Mat &targetImage, const std::vector<std::pair<std::string, float>> &ssdScores, const std::string &resultsDirectory, int topN, int taskNumber);


int pipeline(const std::string &src_dir, const std::vector<std::string> &files, std::vector<std::pair<std::string, float>> &scores, int topN, int taskNumber, const std::string& embeddingsFilePath);

cv::Mat generateHistogram(const cv::Mat& src, int histsize = 16);

float histogramIntersection(const cv::Mat& histA, const cv::Mat& histB);

cv::Mat generateRGBHistogram(const cv::Mat& src, int bins = 8);

float histogramIntersection3D(const cv::Mat& histA, const cv::Mat& histB);

cv::Mat generateHueSaturationHistogram(const cv::Mat& src, int hBins, int sBins, bool useHalf);

cv::Mat generateCenteredRegionHistogram(const cv::Mat& src, int bins, float regionFraction);

cv::Mat generateHistogramTop(const cv::Mat& src, int bins, double top_percentage);

cv::Mat generateHistogramBot(const cv::Mat& src, int bins, double bot_percentage);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);

int sobelY3x3(cv::Mat &src, cv::Mat &dst);

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

cv::Mat calculateMagnitudeHistogram(cv::Mat& src, int bins);

cv::Mat computeFourierFeatures(const cv::Mat& src, const cv::Size& featureSize);

cv::Mat calculateCoOccurrenceMatrix(const cv::Mat& gray, int dx, int dy) ;

cv::Mat normalizeCoOccurrenceMatrix(const cv::Mat& coMat);

void calculateTextureFeatures(const cv::Mat& coMat, float& energy, float& entropy, float& contrast, float& homogeneity);

std::vector<cv::Mat> createLawsFilters();

std::vector<cv::Mat> applyLawsFiltersAndGetEnergy(const cv::Mat& src, const std::vector<cv::Mat>& filters) ;

std::vector<cv::Mat> createEnergyHistograms(const std::vector<cv::Mat>& energies, int bins );

void readEmbeddings(const std::string& filePath, std::unordered_map<std::string, std::vector<float>>& embeddings);

float cosineDistance(const std::vector<float>& v1, const std::vector<float>& v2);

std::string extractFilename(const std::string& filepath);

cv::Mat computeGaborHistogram(const cv::Mat& image, int bins);

std::vector<float> extractHOGFeatures(const cv::Mat& image);

float matchHOGFeatures(const std::vector<float>& descriptors1, const std::vector<float>& descriptors2);


#endif // FUNCTIONS_H
