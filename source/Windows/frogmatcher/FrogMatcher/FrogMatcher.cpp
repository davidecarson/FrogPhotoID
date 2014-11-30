// FrogMatcher.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FrogMatcher.h"
#include <iostream>
#include <vector>
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>
#include <string>
#include <ctime>
// #include "opencv2/stitching/detail/util.hpp"		// for LOGLN


using namespace std;
using namespace cv;
using namespace cv::gpu;

// --------------------------------------------------------------------------------------
// Tuning parameters
// --------------------------------------------------------------------------------------
float ratio = 0.70f;
double confidence = 0.99;
double dist = 1.0;
int minHessian = 500;
bool GPUMODE = false;


// ##########################################################################################
// Readme instructions for command-line 
// ##########################################################################################
static void readme()
{
    cout << "\nFrogMatcher.exe\n"
			"This enhanced image matching application was developed to match frogs from\n"
            "a frog identification database.  \n"
            "Place the query image list and database image list in their respective folders.\n"
            "List file paths will be used to locate image files.\n"
			"Default output is: matchReport.txt.\n"
			"Options:  \n"
			"  --debug "
			"		Enable debug console and logging\n"
			"  --gpu\n"
			"		Try to use GPU. All default values are for CPU mode.\n"
			"  --output [filename]\n"
			"		Report filename.  Default is 'matchReport.html'.\n"
            "Usage:\n"
            "./FrogMatcher.exe <file with query images> <file with database images> <options>\n" << endl;
}


// ##########################################################################################
// Read image filenames from input file. 
// Used to parse both query and train image lists.
// ##########################################################################################
static void readSetFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == String::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}


// ##########################################################################################
// main() routine entry 
// ##########################################################################################
int main( int argc, char** argv )
{
	// --------------------------------------------------------------------------------------
	// Variables
	// --------------------------------------------------------------------------------------
    int64 app_start_time = getTickCount();
	int64 t;
    string fileWithQueryImages, queryDirName, trainDirName;
    string fileWithTrainImages;
    string dirToSaveResImages;
	string reportFile = "matchReport.html";
	bool DEBUG;
	int matchCount = 0;
	double rank;


	// --------------------------------------------------------------------------------------
	// Argument check 
	// Application takes one argument of path to directory containing images.
	// --------------------------------------------------------------------------------------
    if(( argc == 1) || ( argc == 2))
	{
		readme();
		return -1;
	}
	else if ( argc == 3 )
    {
        fileWithQueryImages = argv[1]; fileWithTrainImages = argv[2];
    }
	else if ( argc >= 4 )
	{
        fileWithQueryImages = argv[1]; fileWithTrainImages = argv[2];

		for (int i = 3; i < argc; ++i)
		{
			if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
			{
				readme();
				return -1;
			}
			else if (string(argv[i]) == "--debug")
			{
				DEBUG = true;
			}
			else if (string(argv[i]) == "--gpu")
			{
				GPUMODE = true;
			}
			else if (string(argv[i]) == "--output")
			{
				reportFile = string(argv[i+1]);
			}
	   }
	}

	// --------------------------------------------------------------------------------------
	// Prepare output report file
	// --------------------------------------------------------------------------------------
	//string outputfilename = "matchreport-" + dt + ".html";
	ofstream fout(reportFile);
	//fout << std::string(dt) << endl;
	std::time_t result = std::time(NULL);
	/* old report format
	
	fout << "Frog Image Match Report" << endl;		// Log file output
	if (DEBUG == true){
		fout << "Ratio, conf, dist, minH: " << ratio << "\t" << confidence << "\t" << dist << "\t" << minHessian << endl;
	}
	fout << "Start " << std::asctime(std::localtime(&result));		// Log file output
	*/

	// HTML report begin
	fout << "<html xmlns='http://www.w3.org/1999/xhtml' xml:lang='en' lang='en'>" << endl;;
	fout << "<head> \n <meta http-equiv='content-type' content='text/html; charset=iso-8859-1' />" << endl;
	fout << "<title>Wood Frog Match Report</title>" << endl;
	fout << "<style type='text/css' media='screen'>@import 'includes/layout.css';</style>";
	fout << "\n</head>" << endl;
	fout << "\n<body>\n<div id='Header'>Frog Matching System</div>";
	fout << "\n<div id='Content'>";
	fout << "\n<h1>Frog Match Report</h1>";
	fout << "\n<p><table width='643' border='0' cellspacing='5' cellpadding='5'>";
	fout << "\n<tr>\n <td><B>Query Image</td>\n <td>Database Image</td>\n <td>Score</td>\n <td>Rank</B></td>\n </tr>";
	fout << endl;

	// --------------------------------------------------------------------------------------
	// Begin image match sequence
	// --------------------------------------------------------------------------------------
	vector<string> queryImagesNames;
	readSetFilenames(fileWithQueryImages, queryDirName, queryImagesNames);

	// Iterate through query images OUTER LOOP
	for ( size_t i = 0; i < queryImagesNames.size(); i++)
	{
       string queryfilename = queryDirName + queryImagesNames[i];

		int innerCount=0;
		t = getTickCount();
		Mat image1 = imread( queryfilename, CV_LOAD_IMAGE_GRAYSCALE );

		if( image1.empty() )
		{
			cout << "\n Error, couldn't read image filename " << queryfilename << endl;
			return -1;
		}
        
		// --------------------------------------------------------------------------------------
		// Loop through database images to test files against the outer loop one by one
		// --------------------------------------------------------------------------------------
		vector<string> trainImagesNames;
		readSetFilenames(fileWithTrainImages, trainDirName, trainImagesNames);

		// Iterate through train (database) images  INNER LOOP
		for ( size_t j = 0; j < trainImagesNames.size(); j++)
		{
	       string trainfilename = trainDirName + trainImagesNames[j];

			Mat image2 = imread( trainfilename, CV_LOAD_IMAGE_GRAYSCALE );
			if (!image2.data)
					return 0; 
			
			// ---------------------------------------------------------------------------------
			// Prepare the matcher
			// ---------------------------------------------------------------------------------
			FrogMatcher fmatcher;
			fmatcher.setConfidenceLevel(confidence);
			fmatcher.setMinDistanceToEpipolar(dist);
			fmatcher.setRatio(ratio);
			fmatcher.setMinHess(minHessian);
			if (GPUMODE == true)
			{
				//fout << "GPU mode ENABLED" << endl;
				fmatcher.setGpu();
			}

			//fmatcher.setFeatureDetector(pfd);
			//fmatcher.setDescriptorExtractor(extractor);

			// ---------------------------------------------------------------------------------
			// Match the two images
			// ---------------------------------------------------------------------------------
            std::cout << queryfilename << " : " << trainfilename << std::endl;
			std::vector<cv::DMatch> matches;
			std::vector<cv::KeyPoint> keypoints1, keypoints2;
			//cv::Mat fundamental= fmatcher.match(image1,image2,matches, keypoints1, keypoints2);
			Mat fundamental= fmatcher.match(image1,image2,matches, keypoints1, keypoints2);

			if (matches.size()>0) {

				// We have a match;  draw the matches
				cv::Mat imageMatches;
				//Mat image1, image2;
				//image1 = Mat(imageGpu1);
				//image2 = Mat(imageGpu2);

				/* -----------------------------------------------------------------------------
				cv::drawMatches(image1,keypoints1,  // 1st image and its keypoints
									image2,keypoints2,  // 2nd image and its keypoints
												matches,                        // the matches
												imageMatches,           // the image produced
												cv::Scalar(255,255,255)); // color of the lines
				------------------------------------------------------------------------------- */

				// Convert keypoints into Point2f       
				std::vector<cv::Point2f> points1, points2;
        
				for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
								 it!= matches.end(); ++it) {

								 // Get the position of left keypoints
								 float x= keypoints1[it->queryIdx].pt.x;
								 float y= keypoints1[it->queryIdx].pt.y;
								 points1.push_back(cv::Point2f(x,y));
								 cv::circle(image1,cv::Point(x,y),3,cv::Scalar(255,255,255),3);
								 // Get the position of right keypoints
								 x= keypoints2[it->trainIdx].pt.x;
								 y= keypoints2[it->trainIdx].pt.y;
								 cv::circle(image2,cv::Point(x,y),3,cv::Scalar(255,255,255),3);
								 points2.push_back(cv::Point2f(x,y));
				}
        
				// ---------------------------------------------------------------------------------
				// Draw the epipolar lines
				// ---------------------------------------------------------------------------------
				std::vector<cv::Vec3f> lines1; 
				cv::computeCorrespondEpilines(cv::Mat(points1),1,fundamental,lines1);
                
				for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
								 it!=lines1.end(); ++it) {

								 cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
													 cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
																 cv::Scalar(255,255,255));
				}

				std::vector<cv::Vec3f> lines2; 
				cv::computeCorrespondEpilines(cv::Mat(points2),2,fundamental,lines2);
        
				for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
							 it!=lines2.end(); ++it) {

								 cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
													 cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
																 cv::Scalar(255,255,255));
				}

				Mat combinedImage = cvCreateMat(MAX(image1.rows,image2.rows), image1.cols + image2.cols, image1.type());
				cv::Mat tmp = combinedImage(cv::Rect(0, 0, image1.cols, image1.rows));
				image1.copyTo(tmp);
				tmp = combinedImage(cv::Rect(image1.cols, 0, image2.cols, image2.rows));
				image2.copyTo(tmp);

				// ---------------------------------------------------------------------------------
				// Output the combined image (epipolar lines matches)
				// ---------------------------------------------------------------------------------
				string outputFile = ".\\Output\\" + queryImagesNames[i] + ".matches." + trainImagesNames[j];
				//string outimFile = ".\\Output\\" + myFileName + ".im." + myFileName2;
				cv::imwrite(outputFile, combinedImage);
				//cv::imwrite(outimFile, imageMatches);
				
				// Write out filenames and 1 to indicate a match
				double score1 = fmatcher.getScore();
				double score2 = fmatcher.getScore2();
				rank = score1/score2;
				/* old output
				fout << queryImagesNames[i] + "\t " + trainImagesNames[j] + "\t " << score1 << "\t " << rank;
				if (DEBUG == true){
					fout << "\t " << fmatcher.getFerr() << "\t " << fmatcher.getScore2();
				}
				fout << endl;
				*/
				fout << "\n<tr>\n <td>" + queryImagesNames[i] + "</td>\n <td>" + trainImagesNames[j] + "</td>\n <td>" << score1;
				fout << "</td>\n <td>" << rank << "</td>\n </tr>" << endl;
			}
			else if (DEBUG == true){
				// Write out filenames and 0 to indicate not a match
				fout << queryImagesNames[i] + "\t " + trainImagesNames[j] + "\t 0" << "\t " << fmatcher.getScore() << endl;
			}
			matchCount += 1;
			innerCount += 1;
			}	// end INNER LOOP
		// Next iteration (same left image, next right image)
		
			//Output Run time per iteration
			//fout << queryfilename + "\t " + " run time (secs): " + "\t " + "\t " + "\t " + "\t " + "\t "<< ((getTickCount() - t) / getTickFrequency()) << "\n";

	}	// end OUTER LOOP


	// Exit application
	result = std::time(NULL);
	/* Old report totals
	fout << "Total Matches\t" << matchCount << endl;
	fout << "Run time: " << ((getTickCount() - app_start_time) / getTickFrequency());		// Log file output
	*/

	// HTML report totals
	fout << "\n</tr>\n</table>\n</p>";
	fout << "\n<p>Image Pairs Analyzed: " << matchCount << endl;
	fout << "</p>\n<p>Run time: " << ((getTickCount() - app_start_time) / getTickFrequency());
	fout << "</p>\n</body></html>" << endl;
	fout.close();
	return 0;
}
