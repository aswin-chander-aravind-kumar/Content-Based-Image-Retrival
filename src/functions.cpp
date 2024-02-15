#include <filesystem>
#include <iostream>
#include "functions.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include<vector>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <chrono>

namespace fs = std::filesystem;

// Inside your pipeline function, before starting task 1 or 2:
auto start = std::chrono::high_resolution_clock::now();


// SSD Calculation
float SSD(const cv::Mat& src, const cv::Mat& dst) {
    int center_row = src.rows / 2;
    int center_col = src.cols / 2;
    float sum = 0.0f;

    for (int i = center_row - 3; i <= center_row + 3; i++) {
        for (int j = center_col - 3; j <= center_col + 3; j++) {
            cv::Vec3b srcPixel = src.at<cv::Vec3b>(i, j);
            cv::Vec3b dstPixel = dst.at<cv::Vec3b>(i, j);
            for (int k = 0; k < 3; k++) {
                float diff = static_cast<float>(srcPixel[k] - dstPixel[k]);
                sum += diff * diff;
            }
        }
    }
    return sum;
}

// Reading Image Files from Directory
int readFiles(const std::string& dirname, std::vector<std::string>& files) {
    try {
        for (const auto& entry : fs::directory_iterator(dirname)) {
            if (entry.is_regular_file()) {
                auto path = entry.path();
                if (path.extension() == ".jpg" || path.extension() == ".png" || 
                    path.extension() == ".ppm" || path.extension() == ".tif") {
                    files.push_back(path.string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "General error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

// Saving Matched Images with Task Number in Filenames
void saveMatchedImages(const cv::Mat &targetImage, const std::vector<std::pair<std::string, float>> &ssdScores, const std::string &resultsDirectory, int topN, int taskNumber) {
    fs::create_directory(resultsDirectory);

    std::string targetPath = resultsDirectory + "/target_task_" + std::to_string(taskNumber) + ".jpg";
    cv::imwrite(targetPath, targetImage);

    for (int i = 0; i < topN && i < ssdScores.size(); ++i) {
        cv::Mat image = cv::imread(ssdScores[i].first, cv::IMREAD_COLOR);
        std::string matchedImagePath = resultsDirectory + "/match_task_" + std::to_string(taskNumber) + "_" + std::to_string(i + 1) + ".jpg";
        cv::imwrite(matchedImagePath, image);
    }
}

// Generate Histogram Function
cv::Mat generateHistogram(const cv::Mat& src, int histsize) {
    cv::Mat hist = cv::Mat::zeros(histsize, histsize, CV_32F);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            float R = pixel[2];
            float G = pixel[1];
            float B = pixel[0];
            float sum = R + G + B;
            sum = sum > 0 ? sum : 1.0f; // Prevent division by zero
            float r = (R / sum) * (histsize - 1);
            float g = (G / sum) * (histsize - 1);
            int idxR = static_cast<int>(r);
            int idxG = static_cast<int>(g);
            hist.at<float>(idxR, idxG) += 1.0f;
        }
    }

    // Normalize the histogram
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

// Histogram Intersection Function
float histogramIntersection(const cv::Mat& histA, const cv::Mat& histB) {
    float intersection = 0.0f;
    for (int i = 0; i < histA.rows; i++) {
        for (int j = 0; j < histA.cols; j++) {
            intersection += std::min(histA.at<float>(i, j), histB.at<float>(i, j));
        }
    }
    return intersection;
}

cv::Mat generateRGBHistogram(const cv::Mat& src, int bins) {
    // Histogram ranges for each channel (B, G, R)
    float range[] = {0, 256};
    const float* histRange[] = {range, range, range};

    // Set up bins and histogram
    int histSize[] = {bins, bins, bins};
    cv::Mat hist;
    int channels[] = {0, 1, 2}; // Indexes of channels

    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 3, histSize, histRange, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX); // Normalize the histogram

    return hist;
}   


float histogramIntersection3D(const cv::Mat& histA, const cv::Mat& histB) {
    CV_Assert(histA.size == histB.size && histA.type() == histB.type());

    float intersection = 0.0f;
    for (int r = 0; r < histA.size[0]; r++) {
        for (int g = 0; g < histA.size[1]; g++) {
            for (int b = 0; b < histA.size[2]; b++) {
                float binValA = histA.at<float>(r, g, b);
                float binValB = histB.at<float>(r, g, b);
                intersection += std::min(binValA, binValB);
            }
        }
    }
    return intersection;
}
cv::Mat generateCenteredRegionHistogram(const cv::Mat& src, int bins, float regionFraction) {
    // Determine the region of interest (centered region)
    int width = src.cols;
    int height = src.rows;
    int centerX = width / 2;
    int centerY = height / 2;
    int regionWidth = static_cast<int>(width * regionFraction);
    int regionHeight = static_cast<int>(height * regionFraction);
    int xStart = centerX - (regionWidth / 2);
    int yStart = centerY - (regionHeight / 2);

    // Create ROI based on calculated dimensions
    cv::Rect roi(xStart, yStart, regionWidth, regionHeight);
    cv::Mat srcRegion = src(roi);

    // Now generate the histogram for this region
    return generateRGBHistogram(srcRegion, bins); // Reuse the RGB histogram function for the region
}

// HueSaturation histogram
cv::Mat generateHueSaturationHistogram(const cv::Mat& src, int hBins, int sBins, bool useHalf) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    int fromRow = useHalf ? src.rows / 2 : 0;
    cv::Mat halfSrc = hsv(cv::Range(fromRow, src.rows), cv::Range::all());

    int histSize[] = { hBins, sBins };
    float hRanges[] = { 0, 180 };
    float sRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges, sRanges };
    int channels[] = { 0, 1 };

    cv::Mat hist;
    cv::calcHist(&halfSrc, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

cv::Mat generateHistogramTop(const cv::Mat& src, int bins, double top_percentage) {
    int top_rows = static_cast<int>(src.rows * top_percentage);
    cv::Rect roi_rect_1(0, 0, src.cols, top_rows); // ROI covering the top part of the image
    cv::Mat src_top = src(roi_rect_1); // Extract the top part of the image

    // Histogram ranges for each channel (B, G, R)
    float range[] = {0, 256};
    const float* histRange[] = {range, range, range};

    // Set up bins and histogram
    int histSize[] = {bins, bins, bins};
    cv::Mat hist_top, hist_bot; // Separate histograms for top and bottom parts
    int channels[] = {0, 1, 2}; // Indexes of channels

    // Compute histograms for top and bottom parts separately
    cv::calcHist(&src_top, 1, channels, cv::Mat(), hist_top, 3, histSize, histRange, true, false);

    // Normalize the histograms
    cv::normalize(hist_top, hist_top, 0, 1, cv::NORM_MINMAX);

    return hist_top;
}  

cv::Mat generateHistogramBot(const cv::Mat& src, int bins, double bot_percentage) {
    int bot_rows = static_cast<int>(src.rows * bot_percentage);
    cv::Rect roi_rect_2(0, 0, src.cols, bot_rows); // ROI covering the bottom part of the image
    cv::Mat src_bot = src(roi_rect_2); // Extract the bottom part of the image

    // Histogram ranges for each channel (B, G, R)
    float range[] = {0, 256};
    const float* histRange[] = {range, range, range};

    // Set up bins and histogram
    int histSize[] = {bins, bins, bins};
    cv::Mat hist_top, hist_bot; // Separate histograms for top and bottom parts
    int channels[] = {0, 1, 2}; // Indexes of channels

    // Compute histograms for top and bottom parts separately
    cv::calcHist(&src_bot, 1, channels, cv::Mat(), hist_bot, 3, histSize, histRange, true, false);

    // Normalize the histograms
    cv::normalize(hist_bot, hist_bot, 0, 1, cv::NORM_MINMAX);

    return hist_bot;
}  
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int i = 1; i < src.rows - 1; i++) {
        // src pointer
        cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i + 1);

        // destination pointer
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);

        // for each column
        for (int j = 1; j < src.cols - 1; j++) {
            // for each color channel
            for (int c = 0; c < 3; c++) {
                dptr[j][c] = (-1 * rptrm1[j - 1][c] + 1 * rptrm1[j + 1][c] +
                              -2 * rptr[j - 1][c] + 2 * rptr[j + 1][c] +
                              -1 * rptrp1[j - 1][c] + 1 * rptrp1[j + 1][c]) ;
            }
        }
    }

    return 0;
}

// Sobel Y 3x3
// Applies the Sobel operator to detect vertical edges in the image.
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int i = 1; i < src.rows - 1; i++) {
        // src pointer
        cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i + 1);

        // destination pointer
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);

        // for each column
        for (int j = 1; j < src.cols - 1; j++) {
            // for each color channel
            for (int c = 0; c < 3; c++) {
                dptr[j][c] = -rptrm1[j - 1][c] - 2 * rptrm1[j][c] - rptrm1[j + 1][c] +
                              rptrp1[j - 1][c] + 2 * rptrp1[j][c] + rptrp1[j + 1][c];
            }
        }
    }

    return 0;
}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ) {
    // allocate destination space
    //dst = cv::Mat::zeros(sx.size(), CV_16SC3);
    dst = sx.clone();

    // loop through src and apply vertical filter
    // go through rows
    for (int i = 0; i < sx.rows; i++) {
        
        // source row pointer
        cv::Vec3s *rowptrsx = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *rowptrsy = sy.ptr<cv::Vec3s>(i);
    
        // destination pointer
        cv::Vec3s *dstptr = dst.ptr<cv::Vec3s>(i);

        // go through columes
        for (int j = 0; j < sx.cols; j++) {
            
            // go though each color channels
            for (int k = 0; k < 3; k++) {
                // calculate maginitude
                dstptr[j][k] = static_cast<short>(sqrt(static_cast<double>(rowptrsx[j][k] * rowptrsx[j][k]) + static_cast<double>(rowptrsy[j][k] * rowptrsy[j][k])));

            }
        }
    }
    return 0;
}
cv::Mat calculateMagnitudeHistogram(cv::Mat& src, int bins) {
    cv::Mat histogram; // Declare histogram as cv::Mat

    cv::Mat sobelX, sobelY, mag;
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);
    magnitude(sobelX, sobelY, mag);

    
    float range[] = {0, 256};
    const float* histRange[] = {range, range, range};

    // Set up bins and histogram
    int histSize[] = {bins, bins, bins};
    
    int channels[] = {0, 1, 2}; // Indexes of channels
    cv::calcHist(&src, 1, channels, cv::Mat(), histogram, 3, histSize, histRange, true, false);
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX); // Normalize the histogram

    return histogram; // Return the calculated histogram
}



cv::Mat computeFourierFeatures(const cv::Mat& src, const cv::Size& featureSize = cv::Size(16, 16)) {
    // Ensure the image is in grayscale
    cv::Mat grayImage;
    if (src.channels() > 1) {
        cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = src.clone();
    }

    // Convert image to float
    cv::Mat floatSrc;
    grayImage.convertTo(floatSrc, CV_32F); // Ensure it's CV_32F for cv::dft

    // Expand the image to an optimal size for DFT
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(floatSrc.rows);
    int n = cv::getOptimalDFTSize(floatSrc.cols);
    cv::copyMakeBorder(floatSrc, padded, 0, m - floatSrc.rows, 0, n - floatSrc.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Prepare for DFT
    cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI); // Merge into a two-channel image for DFT

    // Perform DFT
    cv::dft(complexI, complexI);

    // Compute the magnitude
    cv::split(complexI, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::Mat magI = planes[0];

    // Switch to logarithmic scale to visualize DFT
    magI += cv::Scalar::all(1);
    log(magI, magI);

    // Crop the spectrum
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // Normalize to [0, 1]
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

    // Resize the magnitude image to feature size
    cv::Mat featureImg;
    cv::resize(magI, featureImg, featureSize);

    return featureImg;
}

cv::Mat calculateCoOccurrenceMatrix(const cv::Mat& gray, int dx, int dy) {
    cv::Mat coMat = cv::Mat::zeros(256, 256, CV_32F); // Assuming 8-bit grayscale images

    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            int pixelValue = gray.at<uchar>(y, x);
            int neighborX = x + dx;
            int neighborY = y + dy;

            if (neighborX >= 0 && neighborX < gray.cols && neighborY >= 0 && neighborY < gray.rows) {
                int neighborValue = gray.at<uchar>(neighborY, neighborX);
                coMat.at<float>(pixelValue, neighborValue) += 1.0f;
            }
        }
    }
    return coMat;
}

cv::Mat normalizeCoOccurrenceMatrix(const cv::Mat& coMat) {
    cv::Mat normalizedCoMat;
    cv::normalize(coMat, normalizedCoMat, 0, 1, cv::NORM_MINMAX);
    return normalizedCoMat;
}

void calculateTextureFeatures(const cv::Mat& coMat, float& energy, float& entropy, float& contrast, float& homogeneity) {
    energy = 0.0f;
    entropy = 0.0f;
    contrast = 0.0f;
    homogeneity = 0.0f;

    for (int i = 0; i < coMat.rows; i++) {
        for (int j = 0; j < coMat.cols; j++) {
            float value = coMat.at<float>(i, j);
            energy += value * value;
            if (value > 0)
                entropy -= value * log2(value);
            contrast += (i - j) * (i - j) * value;
            homogeneity += value / (1 + abs(i - j));
        }
    }
}

// Define basic vectors (Gaussian and its derivatives)
cv::Mat L5 = (cv::Mat_<float>(1, 5) << 1, 4, 6, 4, 1);
cv::Mat E5 = (cv::Mat_<float>(1, 5) << -1, -2, 0, 2, 1);
cv::Mat S5 = (cv::Mat_<float>(1, 5) << -1, 0, 2, 0, -1);
cv::Mat W5 = (cv::Mat_<float>(1, 5) << -1, 2, 0, -2, 1);
cv::Mat R5 = (cv::Mat_<float>(1, 5) << 1, -4, 6, -4, 1);

// Function to create Laws' filters by convolving vectors
std::vector<cv::Mat> createLawsFilters() {
    std::vector<cv::Mat> filters;

    
    filters.push_back(L5.t() * L5);
    filters.push_back(L5.t() * E5);
    filters.push_back(L5.t() * S5);
    filters.push_back(E5.t() * L5);
    filters.push_back(E5.t() * E5);
    

    return filters;
}

// Function to apply Laws' filters and calculate texture energy
std::vector<cv::Mat> applyLawsFiltersAndGetEnergy(const cv::Mat& src, const std::vector<cv::Mat>& filters) {
    std::vector<cv::Mat> energies;
    cv::Mat srcGray;
    cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY); // Convert to grayscale

    for (const auto& filter : filters) {
        cv::Mat response, energy;
        cv::filter2D(srcGray, response, CV_32F, filter);
        cv::convertScaleAbs(response, response); // Convert to absolute values
        cv::blur(response, energy, cv::Size(5, 5)); // Sum over local 5x5 area
        energies.push_back(energy);
    }

    return energies;
}

// Function to create histograms from energies
std::vector<cv::Mat> createEnergyHistograms(const std::vector<cv::Mat>& energies, int bins = 256) {
    std::vector<cv::Mat> histograms;

    for (const auto& energy : energies) {
        cv::Mat hist;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::calcHist(&energy, 1, 0, cv::Mat(), hist, 1, &bins, &histRange, true, false);
        histograms.push_back(hist);
    }

    return histograms;
}

// Function to read embeddings from a CSV file
void readEmbeddings(const std::string& filePath, std::unordered_map<std::string, std::vector<float>>& embeddings) {
    std::ifstream file(filePath);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string filename;
        std::getline(iss, filename, ','); // Assuming the filename is the first value before the first comma

        std::vector<float> embedding;
        std::string value;
        while (std::getline(iss, value, ',')) {
            embedding.push_back(std::stof(value));
        }

        embeddings[filename] = embedding;
    }
}

float cosineDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dotProduct = 0.0, denom_v1 = 0.0, denom_v2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        denom_v1 += v1[i] * v1[i];
        denom_v2 += v2[i] * v2[i];
    }
    denom_v1 = sqrt(denom_v1);
    denom_v2 = sqrt(denom_v2);
    float cosine = dotProduct / (denom_v1 * denom_v2);
    return 1.0 - cosine; // Convert similarity to distance
}

std::string extractFilename(const std::string& filepath) {
    fs::path pathObj(filepath);
    return pathObj.filename().string();
}

cv::Mat computeGaborHistogram(const cv::Mat& image, int bins) {
    // Define Gabor filter parameters
    int ksize = 21;
    double sigma = 4, theta = CV_PI/4, lambda = 5, gamma = 0.5, psi = CV_PI/2;

    cv::Mat srcGray;
    if (image.channels() > 1) {
        cv::cvtColor(image, srcGray, cv::COLOR_BGR2GRAY);
    } else {
        srcGray = image.clone();
    }

    // Apply Gabor filter
    cv::Mat gaborKernel = cv::getGaborKernel(cv::Size(ksize, ksize), sigma, theta, lambda, gamma, psi, CV_32F);
    cv::Mat filteredImage;
    cv::filter2D(srcGray, filteredImage, CV_32F, gaborKernel);

    // Prepare for histogram computation
    // Convert to proper type if necessary
    filteredImage.convertTo(filteredImage, CV_8U);

    // Compute histogram from the Gabor-filtered image
    int channels[] = {0};
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&filteredImage, 1, channels, cv::Mat(), hist, 1, &bins, &histRange, true, false);

    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX); // Normalize the histogram

    return hist;
}

std::vector<float> extractHOGFeatures(const cv::Mat& image) {
    cv::HOGDescriptor hog;
    std::vector<float> descriptors;
    hog.compute(image, descriptors);
    return descriptors;
}

float matchHOGFeatures(const std::vector<float>& descriptors1, const std::vector<float>& descriptors2) {
    if (descriptors1.empty() || descriptors2.empty()) return std::numeric_limits<float>::max();

    // Convert descriptors to cv::Mat for FLANN
    cv::Mat mat1(1, descriptors1.size(), CV_32F, (void*)&descriptors1[0]);
    cv::Mat mat2(1, descriptors2.size(), CV_32F, (void*)&descriptors2[0]);

    // Create and use FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(mat1, mat2, matches);

    // Calculate average distance among all matches as the score
    float averageDistance = 0;
    for (auto& match : matches) {
        averageDistance += match.distance;
    }
    return averageDistance / matches.size();
}


// Pipeline for Task Selection and Execution
int pipeline(const std::string &src_dir, const std::vector<std::string> &files, std::vector<std::pair<std::string, float>> &scores, int topN, int taskNumber, const std::string& embeddingsFilePath) {
    cv::Mat targetImage = cv::imread(src_dir, cv::IMREAD_COLOR);
    if (targetImage.empty()) {
        std::cerr << "Could not read the target image from " << src_dir << std::endl;
        return -1;
    }

    if (taskNumber == 1) { // SSD Matching
        for (const auto& imagePath : files) {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (!image.empty()) {
                float ssdScore = SSD(targetImage, image);
                scores.emplace_back(imagePath, ssdScore);
            }
        }

        // Sort based on SSD score in ascending order
        std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second < b.second;
        });
    } else if (taskNumber == 2) { // Histogram Matching
        cv::Mat targetHist = generateHistogram(targetImage, 16);

        for (const auto& imagePath : files) {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (!image.empty()) {
                cv::Mat hist = generateHistogram(image, 16);
                cv::imshow("2D Histogram", hist);
                cv::waitKey(0);
                float distance = histogramIntersection(targetHist, hist);
                scores.emplace_back(imagePath, distance); // Use distance directly for sorting
            }
        }

        // Sort scores for histogram intersection in descending order
        std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second > b.second; // Higher intersection values indicate closer matches
        });
    } else if (taskNumber == 21) { // RGB Histogram Matching
        cv::Mat targetHist = generateRGBHistogram(targetImage, 8); // Use 8 bins for each channel

        for (const auto& imagePath : files) {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (!image.empty()) {
                cv::Mat hist = generateRGBHistogram(image, 8); // Generate histogram for each image
                float distance = histogramIntersection3D(targetHist, hist); // Assuming you have a suitable function for 3D histogram intersection
                scores.emplace_back(imagePath, distance); // Store the distance (intersection value)
                
            }
        }

        // Sort based on histogram intersection score in descending order
        std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second > b.second; // Higher scores indicate better matches
        });
    } else if (taskNumber == 3) { // Task for combined whole image and centered region histogram matching
        cv::Mat targetHistWhole = generateRGBHistogram(targetImage, 8);
        cv::Mat targetHistCentered = generateCenteredRegionHistogram(targetImage, 8, 0.5);

        for (const auto& imagePath : files) {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (!image.empty()) {
                cv::Mat histWhole = generateRGBHistogram(image, 8);
                cv::Mat histCentered = generateCenteredRegionHistogram(image, 8, 0.5);

                // Combine distances from both histograms (simple averaging or weighted averaging)
                float distanceWhole = histogramIntersection3D(targetHistWhole, histWhole);
                float distanceCentered = histogramIntersection3D(targetHistCentered, histCentered);
                float combinedDistance = (distanceWhole + distanceCentered) / 2; // Simple averaging

                scores.emplace_back(imagePath, combinedDistance); // Use combined distance for sorting
            }
        }

        // Sort based on combined distance metric in descending order (assuming higher values mean more similarity)
        std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
         return a.second > b.second;
        });
    }else if (taskNumber == 31) {
    // Generate histograms for the target image
        cv::Mat targetHistRGB = generateCenteredRegionHistogram(targetImage, 8, 0.5); // Centered region RGB histogram
        cv::Mat targetImageHS = generateHueSaturationHistogram(targetImage, 8, 8, false); // Use entire image for Hue-Saturation

        for (const auto& imagePath : files) {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (!image.empty()) {
                // Generate histograms for the candidate image
                cv::Mat histRGB = generateCenteredRegionHistogram(image, 8, 0.5); // Centered region RGB histogram
                //cv::Mat histHS = generateHueSaturationHistogram(image, 8, false); // Lower half RG hue saturation histogram

                int hBins = 8; // Hue bins
                int sBins = 8; // Saturation bins
                bool useHalf = true; // Assuming you want to use half the image, adjust based on your need
                cv::Mat histHS = generateHueSaturationHistogram(image, hBins, sBins, useHalf);
            
                // Calculate distances and combine them
                float distanceRGB = histogramIntersection3D(targetHistRGB, histRGB);
                float distanceHS = histogramIntersection(targetImageHS, histHS);
                float combinedDistance = (distanceRGB + distanceHS) / 2; // Average of the two distances

                scores.emplace_back(imagePath, combinedDistance);
            }
        }

        // Sort based on combined distance in descending order
        std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second > b.second;
       });
    } else if (taskNumber == 32) { // RGB Histogram Matching
        cv::Mat targetHist_Top = generateHistogramTop(targetImage, 8, 0.5); // Use 8 bins for each channel
        cv::Mat targetHist_Bot = generateHistogramBot(targetImage, 8, 0.5);
        for (const auto& imagePath : files) {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (!image.empty()) {
                cv::Mat hist_top = generateHistogramTop(image, 8, 0.5); // Generate histogram for each image
                cv::Mat hist_bot = generateHistogramBot(image, 8, 0.5);
                float distance_1 = histogramIntersection3D(targetHist_Top, hist_top); // Assuming you have a suitable function for 3D histogram intersection
                float distance_2 = histogramIntersection3D(targetHist_Bot, hist_bot);
                float combinedDistance = (distance_1 + distance_2) / 2;
                scores.emplace_back(imagePath, combinedDistance); // Store the distance (intersection value)
            }
        }

        // Sort based on histogram intersection score in descending order
        std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second > b.second; // Higher scores indicate better matches
        });
    } else if (taskNumber == 4) { // RGB Histogram Matching
        cv::Mat targetHist_whole = generateRGBHistogram(targetImage, 8);
        cv::Mat targetHist_magnitude = calculateMagnitudeHistogram(targetImage, 8); // Use 8 bins for each channel
        for (const auto& imagePath : files) {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (!image.empty()) {
                cv::Mat hist_whole= generateRGBHistogram(image, 8); // Generate histogram for each image
                cv::Mat hist_magnitude = calculateMagnitudeHistogram(image, 8);
                float distance_1 = histogramIntersection3D(targetHist_whole, hist_whole); // Assuming you have a suitable function for 3D histogram intersection
                float distance_2 = histogramIntersection3D(targetHist_magnitude, hist_magnitude);
                float combinedDistance = (distance_1 + distance_2) / 2;
                scores.emplace_back(imagePath, combinedDistance); // Store the distance (intersection value)
            }
        }

        // Sort based on histogram intersection score in descending order
        std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second > b.second; // Higher scores indicate better matches
        });
    } else if (taskNumber == 41) {
        cv::Mat targetFeatures = computeFourierFeatures(targetImage);

        for (const auto& imagePath : files) {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE); // Ensure images are read in grayscale
            if (!image.empty()) {
                cv::Mat features = computeFourierFeatures(image);

                // Compute similarity between feature vectors (e.g., using Euclidean distance)
                float score = cv::norm(targetFeatures, features, cv::NORM_L2);
                scores.emplace_back(imagePath, score);
            }
        }

        // Sort scores in ascending order since lower distance indicates higher similarity
        std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
        return a.second < b.second;
        });
    } else if (taskNumber == 42) {
        // Convert target image to grayscale
        cv::Mat grayImage;
        cv::cvtColor(targetImage, grayImage, cv::COLOR_BGR2GRAY);

         // Calculate co-occurrence matrix for a specific direction (e.g., horizontally right)
        cv::Mat coMat = calculateCoOccurrenceMatrix(grayImage, 1, 0);
        cv::Mat normalizedCoMat = normalizeCoOccurrenceMatrix(coMat);

        // Calculate texture features
        float energy, entropy, contrast, homogeneity;
        calculateTextureFeatures(normalizedCoMat, energy, entropy, contrast, homogeneity);

        std::cout << "Texture Features for Target Image:" << std::endl;
        std::cout << "Energy: " << energy << std::endl;
        std::cout << "Entropy: " << entropy << std::endl;
        std::cout << "Contrast: " << contrast << std::endl;
        std::cout << "Homogeneity: " << homogeneity << std::endl;

    }else if (taskNumber == 5) {
        // Load embeddings for task 5 - Image 0893
        std::unordered_map<std::string, std::vector<float>> embeddings;
        readEmbeddings(embeddingsFilePath, embeddings);     

        
        std::string targetFilename = extractFilename(src_dir); 
        if (embeddings.find(targetFilename) == embeddings.end()) {
            std::cerr << "Embedding not found for " << targetFilename << std::endl;
            return -1;
        }

        std::vector<float> targetEmbedding = embeddings[targetFilename];
        for (const auto& file : files) {
            std::string filename = extractFilename(file); 
            if (embeddings.find(filename) != embeddings.end()) {
                float distance = cosineDistance(targetEmbedding, embeddings[filename]); 
                scores.emplace_back(file, distance);
            }
        }

        // Sort the scores
        std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
            return a.second < b.second; // Ascending order for distance
        });
    }else if (taskNumber == 51) {
        // Load embeddings for task 51 - Image 0164
        std::unordered_map<std::string, std::vector<float>> embeddings;
        readEmbeddings(embeddingsFilePath, embeddings); 

        
        std::string targetFilename = extractFilename(src_dir); 
        if (embeddings.find(targetFilename) == embeddings.end()) {
            std::cerr << "Embedding not found for " << targetFilename << std::endl;
            return -1;
        }

        std::vector<float> targetEmbedding = embeddings[targetFilename];
        for (const auto& file : files) {
            std::string filename = extractFilename(file); 
            if (embeddings.find(filename) != embeddings.end()) {
                float distance = cosineDistance(targetEmbedding, embeddings[filename]); 
                scores.emplace_back(file, distance);
            }
        }

        // Sort the scores
        std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
            return a.second < b.second; // Ascending order for distance
        });
    }else if(taskNumber==6){
        cv::Mat targetImage = cv::imread(src_dir, cv::IMREAD_COLOR);
        if (targetImage.empty()) {
            std::cerr << "Could not read target image: " << src_dir << std::endl;
            return -1;
        }

        // Prepare for comparisons
        std::vector<std::pair<std::string, float>> embeddingScores, histogramScores;
        std::unordered_map<std::string, std::vector<float>> embeddings;

        // Load embeddings from file
        readEmbeddings(embeddingsFilePath, embeddings);
        std::string targetFilename = fs::path(src_dir).filename().string();
    
        if (embeddings.find(targetFilename) == embeddings.end()) {
            std::cerr << "Embedding not found for target image" << std::endl;
            return -1;
        }

        std::vector<float> targetEmbedding = embeddings[targetFilename];

        // RGB Histogram of target image
        cv::Mat targetHist = generateRGBHistogram(targetImage, 8); // Adjust bin size if necessary

        for (const auto& imagePath : files) {
            std::string filename = fs::path(imagePath).filename().string();
        
            // DNN Embeddings comparison
            if (embeddings.find(filename) != embeddings.end()) {
                float dist = cosineDistance(targetEmbedding, embeddings[filename]);
                embeddingScores.push_back({imagePath, dist});
            }

            // RGB Histogram comparison
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            cv::Mat hist = generateRGBHistogram(image, 8); // Adjust bin size if necessary
            float histScore = histogramIntersection3D(targetHist, hist);
            histogramScores.push_back({imagePath, histScore});
        }

        // Sort scores
        auto sortRule = [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) -> bool {
            return a.second < b.second;
        };

        std::sort(embeddingScores.begin(), embeddingScores.end(), sortRule);
        std::sort(histogramScores.begin(), histogramScores.end(), sortRule);

        // Display top N matches
        std::cout << "Top " << topN << " matches (Embedding Score vs. Histogram Score):" << std::endl;
        for (int i = 0; i < topN; ++i) {
            std::cout << i+1 << ". " << embeddingScores[i].first << " (E: " << embeddingScores[i].second << " vs. H: " << histogramScores[i].second << ")" << std::endl;
        }    
    } else if (taskNumber == 7){
        cv::Mat targetHist_gabor = computeGaborHistogram(targetImage, 256);
        std::unordered_map<std::string, std::vector<float>> embeddings;
readEmbeddings(embeddingsFilePath, embeddings);     

std::string targetFilename = extractFilename(src_dir); 
if (embeddings.find(targetFilename) == embeddings.end()) {
    std::cerr << "Embedding not found for " << targetFilename << std::endl;
    return -1;
}

std::vector<float> targetEmbedding = embeddings[targetFilename];

for (const auto& imagePath : files) {
    //std::cout << "Image path: " << imagePath << std::endl;

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << imagePath << std::endl;
        continue;
    }

    // Calculate histogram intersection between targetHist_Gabor and hist_Gabor
    cv::Mat hist_gabor = computeGaborHistogram(image, 256);
    float histIntersection = cosineDistance(targetHist_gabor, hist_gabor);

    // Calculate cosine distance between targetEmbedding and embeddings of the current image
    std::string filename = extractFilename(imagePath); 
    if (embeddings.find(filename) != embeddings.end()) {
        float cosDist = cosineDistance(targetEmbedding, embeddings[filename]); 
        float combinedScore = (histIntersection + cosDist) / 2.0f; // Combine the scores
        scores.emplace_back(imagePath, combinedScore);
    }
}

// Sort the scores
std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
    return a.second < b.second; // Ascending order for combined score
});
    }else if (taskNumber == 8) {

    // Display the top N matches
    for (int i = 0; i < std::min(topN, static_cast<int>(scores.size())); ++i) {
        std::cout << "Match " << i + 1 << ": " << scores[i].first << " with score " << scores[i].second << std::endl;
    }
    }

    // After completing the task:
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Execution time for task " << taskNumber << ": " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
    
    