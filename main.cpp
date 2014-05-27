/*

Eye detection and tracking
 
Author: Prateek Joshi

This program detects and tracks the user's eye with webcam.

*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eye_cascade;

// Function to detect human face and the eyes from an image.
// im:      The source image
// tpl:     If successfully detected, it will contain the eye template
// rect:    Bounding box of the eye
// Return:  0 = failed, 1 = success
int detectEye(cv::Mat& im, cv::Mat& tpl, cv::Rect& rect)
{
	std::vector<cv::Rect> faces, eyes;
	face_cascade.detectMultiScale(im, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));
    
	for (int i = 0; i < faces.size(); i++)
	{
		cv::Mat face = im(faces[i]);
		eye_cascade.detectMultiScale(face, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20,20));
        
		if (eyes.size())
		{
			rect = eyes[0] + cv::Point(faces[i].x, faces[i].y);
			tpl  = im(rect);
		}
	}
    
	return eyes.size();
}

// Perform template matching to search the user's eye in the given image.
// im:      The source image
// tpl:     The eye template
// rect:    The eye bounding box, will be updated with the new location of the eye
void trackEye(cv::Mat& im, cv::Mat& tpl, cv::Rect& rect)
{
	cv::Size size(rect.width * 2, rect.height * 2);
	cv::Rect window(rect + size - cv::Point(size.width/2, size.height/2));
    
	window &= cv::Rect(0, 0, im.cols, im.rows);
    
	cv::Mat dst(window.width - tpl.rows + 1, window.height - tpl.cols + 1, CV_32FC1);
	cv::matchTemplate(im(window), tpl, dst, CV_TM_SQDIFF_NORMED);
    
	double minval, maxval;
	cv::Point minloc, maxloc;
	cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);
    
	if (minval <= 0.2)
	{
		rect.x = window.x + minloc.x;
		rect.y = window.y + minloc.y;
	}
	else
		rect.x = rect.y = rect.width = rect.height = 0;
}

int main(int argc, char** argv)
{
	// Load the cascade classifiers
	// Make sure you point the XML files to the right path, or
	// just copy the files from [OPENCV_DIR]/data/haarcascades directory
	face_cascade.load("haarcascade_frontalface_alt2.xml");
	eye_cascade.load("haarcascade_eye.xml");
    
	// Open webcam
	cv::VideoCapture cap(0);
    
	// Check if everything is ok
	if (face_cascade.empty() || eye_cascade.empty() || !cap.isOpened())
		return 1;
    
	// Set video to 320x240
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    
	cv::Mat frame, eye_tpl;
	cv::Rect eye_bb;
    
	while (cv::waitKey(15) != 'q')
	{
		cap >> frame;
		if (frame.empty())
			break;
        
		// Flip the frame horizontally, Windows users might need this
		cv::flip(frame, frame, 1);
        
		// Convert to grayscale and
		// adjust the image contrast using histogram equalization
		cv::Mat gray;
		cv::cvtColor(frame, gray, CV_BGR2GRAY);
        
		if (eye_bb.width == 0 && eye_bb.height == 0)
		{
			// Detection stage
			// Try to detect the face and the eye of the user
			detectEye(gray, eye_tpl, eye_bb);
		}
		else
		{
			// Tracking stage with template matching
			trackEye(gray, eye_tpl, eye_bb);
            
			// Draw bounding rectangle for the eye
			cv::rectangle(frame, eye_bb, CV_RGB(0,255,0));
		}
        
		// Display video
		cv::imshow("video", frame);
	}
    
	return 0;
}
