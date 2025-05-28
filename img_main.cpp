#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat original, modified; // Declare oriignal and modified as cv::Mat(rices)

// Default params for adjustment later.
int brightness = 50;

int colorX = 50;  // left to right
int colorY = 50;  // bottom to top

int colorShift = 0;
int blurAmount = 1;
bool showEdges = false;
int saveCount = 0;
int edgeThreshold = 50;

// Resize image so it fits window
void showImagePreservingAspect(const std::string& winName, const cv::Mat& img,
                                int maxWidth = 1000, int maxHeight = 800,
                                int minWidth = 300, int minHeight = 300) {
    double imgAspect = static_cast<double>(img.cols) / img.rows;
    double winAspect = static_cast<double>(maxWidth) / maxHeight;

    int displayWidth, displayHeight;
    if (imgAspect > winAspect) {
        displayWidth = maxWidth;
        displayHeight = static_cast<int>(maxWidth / imgAspect);
        if (displayHeight < minHeight) {
            displayHeight = minHeight;
            displayWidth = static_cast<int>(minHeight * imgAspect);
        }
    } else {
        displayHeight = maxHeight;
        displayWidth = static_cast<int>(maxHeight * imgAspect);
        if (displayWidth < minWidth) {
            displayWidth = minWidth;
            displayHeight = static_cast<int>(minWidth / imgAspect);
        }
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(displayWidth, displayHeight));

    cv::Mat canvas(maxHeight, maxWidth, resized.type(), cv::Scalar(20, 20, 20));
    int xOffset = std::max((maxWidth - displayWidth) / 2, 0);
    int yOffset = std::max((maxHeight - displayHeight) / 2, 0);
    resized.copyTo(canvas(cv::Rect(xOffset, yOffset, displayWidth, displayHeight)));

    cv::imshow(winName, canvas);
}

void overlayEdges(cv::Mat &img) {
    cv::Mat gray, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, edges, CV_8U, 1, 1);

    // Create a mask where edges are strong
    cv::Mat mask;
    cv::threshold(edges, mask, edgeThreshold, 255, cv::THRESH_BINARY);  // adjust threshold for sharpness

    // Make a yellow overlay image (BGR = [0,255,255])
    cv::Mat yellow(img.size(), img.type(), cv::Scalar(0, 255, 255));

    // Copy yellow onto img only where mask is non-zero
    yellow.copyTo(img, mask);
}

void showHistogram(const cv::Mat& image) {
    std::vector<cv::Mat> bgr_planes;
    cv::split(image, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX);

    for (int i = 1; i < histSize; i++) {
        cv::line(histImage,
                 cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                 cv::Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
                 cv::Scalar(255, 0, 0), 2);
        cv::line(histImage,
                 cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                 cv::Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
                 cv::Scalar(0, 255, 0), 2);
        cv::line(histImage,
                 cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                 cv::Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
                 cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("Histogram", histImage);
}

void drawValueDisplay() {
    cv::Mat valueDisplay(220, 300, CV_8UC3, cv::Scalar(20, 20, 20));

    int y = 30;
    int dy = 30;

    auto drawLine = [&](const std::string &text, cv::Scalar color = cv::Scalar(255, 255, 255)) {
        cv::putText(valueDisplay, text, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1);
        y += dy;
    };

    drawLine("Brightness: " + std::to_string(brightness));
    drawLine("Tint (G↔M): " + std::to_string(colorX));
    drawLine("Temp (B↔Y): " + std::to_string(colorY));  
    drawLine("Color Shift: " + std::to_string(colorShift));
    drawLine("Blur: " + std::to_string(blurAmount));
    drawLine("Edge Threshold: " + std::to_string(edgeThreshold));

    y += 10; // extra spacing before toggle section
    drawLine("Press E to toggle edges", cv::Scalar(200, 200, 200));
    drawLine("Edges: " + std::string(showEdges ? "On" : "Off"),
             showEdges ? cv::Scalar(0, 255, 255) : cv::Scalar(100, 100, 100));

    cv::imshow("Values", valueDisplay);
}

void drawColorMap() {
    const int width = 300, height = 300;
    cv::Mat colorMap(height, width, CV_8UC3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float xf = static_cast<float>(x) / width;   // 0.0 → 1.0
            float yf = static_cast<float>(y) / height;  // 0.0 → 1.0

            // Define axes: X = green ↔ magenta, Y = blue ↔ yellow
            float r = std::min(1.0f, xf + yf);     // red increases toward right and bottom
            float g = std::min(1.0f, (1.0f - xf) + yf); // green increases toward left and bottom
            float b = std::min(1.0f, (1.0f - yf));  // blue increases toward top

            cv::Vec3b color(
                static_cast<uchar>(b * 255),
                static_cast<uchar>(g * 255),
                static_cast<uchar>(r * 255)
            );
            colorMap.at<cv::Vec3b>(y, x) = color;
        }
    }

    // Draw black dot at selected X/Y
    int cx = static_cast<int>((colorX / 100.0) * (width - 1));
    int cy = static_cast<int>((1.0 - colorY / 100.0) * (height - 1)); // Flip Y (top = high)
    cv::circle(colorMap, cv::Point(cx, cy), 5, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);

    cv::imshow("Color Spectrum", colorMap);
}



void updateImage(int, void*) {
    // Manually fetch values before updating image
    brightness     = cv::getTrackbarPos("Brightness", "Image");
    colorX         = cv::getTrackbarPos("Tint (G↔M)", "Image");
    colorY         = cv::getTrackbarPos("Temp (B↔Y)", "Image");
    colorShift     = cv::getTrackbarPos("Color Shift", "Image");
    blurAmount     = cv::getTrackbarPos("Blur", "Image");
    edgeThreshold  = cv::getTrackbarPos("Edge Threshold", "Image");

    colorX = cv::getTrackbarPos("Tint (G↔M)", "Image");
    colorY = cv::getTrackbarPos("Temp (B↔Y)", "Image");
    colorShift = cv::getTrackbarPos("Color Shift", "Image");
    blurAmount = cv::getTrackbarPos("Blur", "Image");
    edgeThreshold = cv::getTrackbarPos("Edge Threshold", "Image");

    modified = original.clone();

    double alpha = 1.0;
    int beta = brightness - 50;
    original.convertTo(modified, -1, alpha, beta);

    // Split into hue, sat and brightness channels, modify hue, recombine.
    std::vector<cv::Mat> channels(3);
    cv::split(modified, channels);

    // Determine target color direction from X/Y
    int dx = colorX - 50;
    int dy = colorY - 50;

    double angle = std::atan2(dy, dx) * 180.0 / CV_PI;
    if (angle < 0) angle += 360;

    // Map angle to RGB channel weights (simplified)
    double r_weight = std::max(0.0, std::cos((angle - 0) * CV_PI / 180));
    double g_weight = std::max(0.0, std::cos((angle - 120) * CV_PI / 180));
    double b_weight = std::max(0.0, std::cos((angle - 240) * CV_PI / 180));

    // Normalize
    double sum = r_weight + g_weight + b_weight + 1e-6;
    r_weight /= sum;
    g_weight /= sum;
    b_weight /= sum;

    // Scale intensity of shift
    int shift = 20;
    channels[2] = cv::min(channels[2] + r_weight * shift, 255); // Red
    channels[1] = cv::min(channels[1] + g_weight * shift, 255); // Green
    channels[0] = cv::min(channels[0] + b_weight * shift, 255); // Blue

    cv::merge(channels, modified);

    
    cv::Mat hsv;
    cv::cvtColor(modified, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    hsvChannels[0].forEach<uchar>([&](uchar &pixel, const int*) {
        pixel = (pixel + colorShift) % 180;
    });
    cv::merge(hsvChannels, hsv);
    cv::cvtColor(hsv, modified, cv::COLOR_HSV2BGR);

    int blurVal = blurAmount * 2 + 1;
    cv::GaussianBlur(modified, modified, cv::Size(blurVal, blurVal), 0);

    if (showEdges) {
        overlayEdges(modified);
    }

    
    showImagePreservingAspect("Image", modified);

    // Show histogram after modifying the image
    showHistogram(modified);

    // Lil display with info on it.
    drawValueDisplay();

    // Show color value map
    drawColorMap();
}

int main() {

    // Write a short message to the console.
    std::cout << "=== Image Adjuster v1.0 ===\n";
    std::cout << "Created by J\n";
    std::cout << "Use sliders to adjust brightness, color, blur, and edge detection.\n";
    std::cout << "Press [E] to toggle edge overlay, [S] to save, [ESC] to exit.\n\n";

    original = cv::imread("Windmill.jpg");
    if (original.empty()) {
        std::cout << "Failed to load image\n";
        return 1;
    }

    // Null pointer just fetches slider values instead of global values.
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Values", cv::WINDOW_NORMAL);
    cv::createTrackbar("Brightness", "Image", nullptr, 100, updateImage);
    
    cv::createTrackbar("Tint (G↔M)", "Image",  nullptr, 100, updateImage);   // green ↔ magenta
    cv::createTrackbar("Temp (B↔Y)", "Image",  nullptr, 100, updateImage);   // blue ↔ yellow

    cv::namedWindow("Color Spectrum", cv::WINDOW_NORMAL);
    cv::createTrackbar("Color Shift", "Image",  nullptr, 100, updateImage);
    cv::createTrackbar("Blur", "Image",  nullptr, 10, updateImage);
    cv::createTrackbar("Edge Threshold", "Image",  nullptr, 255, updateImage);
    

    updateImage(0, nullptr);

    // Keep the window alive
    while (true) {
        int key = cv::waitKey(30);
        if (key == 27) break; // ESC
        else if (key == 'e') {
            showEdges = !showEdges;
            updateImage(0, nullptr);
        
        // Save image
        } else if (key == 's') {
            std::string filename = "output_" + std::to_string(saveCount++) + ".jpg";
            cv::imwrite(filename, modified);
            std::cout << "Saved to " << filename << "\n";
        }
    }

    return 0;
}

