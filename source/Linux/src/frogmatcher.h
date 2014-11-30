/*
 * frogmatcher.h
 *
 *  Created on: Apr 19, 2014
 *      Author: root
 */

#ifndef FROGMATCHER_H_
#define FROGMATCHER_H_

#pragma once
#if !defined MATCHER
#define MATCHER

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>		//needed for SurfFeatureDetector
#include <opencv2/calib3d/calib3d.hpp>			//needed for  RANSAC CV_FM_8POINT
#include "opencv2/gpu/gpu.hpp"
//#include "opencv2/nonfree/gpu.hpp"

class FrogMatcher {

  private:

          // pointer to the feature point detector object
          cv::Ptr<cv::FeatureDetector> detector;
          // pointer to the feature descriptor extractor object
          cv::Ptr<cv::DescriptorExtractor> extractor;
		  cv::Ptr<cv::DescriptorMatcher> matcher;
		  bool GPUMODE;
          float ratio; // max ratio between 1st and 2nd NN
		  int minHessian; // minimum Hessian threshold for detecting keypoints
          bool refineF; // if true will refine the Fundamental matrix
          double distance; // min distance to epipolar
          double confidence; // confidence level (probability)
		  int matchedPoints1; // number of matched points 1->2
		  int matchedPoints2; // number of matched points 2->1
		  int ratioTest1; // number of matched points following ratio test 1->2
		  int ratioTest2; // number of matched points following ratio test 2->1
		  int symTest; // number of matched points following symmetry test
		  int ransacResults; // number of matched points following RANSAC
		  double thresholdDist; // max distance between paired points
		  double thresholdDist2; // for second image when image size differs
		  double xDist, yDist, x2Dist, y2Dist;
		  double xSize, ySize, x2Size, y2Size;
		  double f_err, f_err2; // Average error of fundamental matrix
		  double score; /*	computed score for match
							Lower is better.  0 is identical match */
		  double score2, score3;
		  //double surfTime, knnMatchTime, ratioTime, symTime, ransacTime;

  public:

          FrogMatcher() : ratio(0.85f), refineF(true), confidence(0.99), distance(3.0) {

                  // SURF is the default feature detector
                  detector= new cv::SurfFeatureDetector();
                  extractor= new cv::SurfDescriptorExtractor();
                  //detector= new cv::OrbFeatureDetector();
                  //extractor= new cv::OrbDescriptorExtractor();
				  // FLANN is the default descriptor matcher
				  //matcher = new cv::FlannBasedMatcher;
                  //cv::BFMatcher matcher;
		          //matcher= new cv::BFMatcher;

          }

		// ------------------------------------------------------------------------------
		// Get properties
		// ------------------------------------------------------------------------------
		int getMatchedPoints1() {
			return matchedPoints1;
		}

		int getMatchedPoints2() {
			return matchedPoints2;
		}

		int getRatioTest1() {
			return ratioTest1;
		}

		int getRatioTest2() {
			return ratioTest2;
		}

		int getSymTest() {
			return symTest;
		}

		int getRansac() {
			return ransacResults;
		}

		double getFerr() {
			return f_err;
		}

		int getScore() {
			return score;
		}

		int getScore2() {
			return score2;
		}

		int getScore3() {
			return score3;
		}

		int getFerr2() {
			return f_err2;
		}

		  // Set the feature detector
          void setFeatureDetector(cv::Ptr<cv::FeatureDetector>& detect) {

                  detector= detect;
          }

          // Set descriptor extractor
          void setDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>& desc) {

                  extractor= desc;
          }

		  // Set the matcher
		  //void setDescriptorMatcher(cv::Ptr<cv::DescriptorMatcher>& match) {

			//  matcher= match;

		  //}

          // Set the minimum distance to epipolar in RANSAC
          void setMinDistanceToEpipolar(double d) {

                  distance= d;
          }

          // Set confidence level in RANSAC
          void setConfidenceLevel(double c) {

                  confidence= c;
          }

          // Set the NN ratio
          void setRatio(float r) {

                  ratio= r;
          }

           // Set the minimum Hessian
          void setMinHess(int m) {

                  minHessian= m;
          }

           // Set GPU mode
          void setGpu(void) {

                  GPUMODE= true;
          }

         // if you want the F matrix to be recalculated
          void refineFundamental(bool flag) {

                  refineF= flag;
          }


		// ##########################################################################################
		// ratioTest Function
        // Clear matches for which NN ratio is > than threshold
		// return the number of removed points
 		// ##########################################################################################
          int ratioTest(std::vector<std::vector<cv::DMatch> >& matches) {

                int removed=0;

        // for all matches
                for (std::vector<std::vector<cv::DMatch> >::iterator matchIterator= matches.begin();
                         matchIterator!= matches.end(); ++matchIterator) {

                                 // if 2 NN has been identified
                                 if (matchIterator->size() > 1) {

                                         // check distance ratio
                                         if ((*matchIterator)[0].distance/(*matchIterator)[1].distance > ratio) {

                                                 matchIterator->clear(); // remove match
                                                 removed++;
                                         }

                                 } else { // does not have 2 neighbours

                                         matchIterator->clear(); // remove match
                                         removed++;
                                 }
                }

                return removed;
          }

		// ##########################################################################################
		// symmetryTest Function
        // Insert symmetrical matches in symMatches vector
 		// ##########################################################################################
          void symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1,
                                const std::vector<std::vector<cv::DMatch> >& matches2,
                                            std::vector<cv::DMatch>& symMatches) {

                // for all matches image 1 -> image 2
                for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator1= matches1.begin();
                         matchIterator1!= matches1.end(); ++matchIterator1) {

                        if (matchIterator1->size() < 2) // ignore deleted matches
                                continue;

                        // for all matches image 2 -> image 1
                        for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator2= matches2.begin();
                                matchIterator2!= matches2.end(); ++matchIterator2) {

                                if (matchIterator2->size() < 2) // ignore deleted matches
                                        continue;

                                // Match symmetry test
                                if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx  &&
                                        (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {

                                                // add symmetrical match
                                                symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
                                                                                                            (*matchIterator1)[0].trainIdx,
                                                                                                            (*matchIterator1)[0].distance));
                                                break; // next match in image 1 -> image 2
                                }
                        }
                }
          }

		// ##########################################################################################
		// RANSACTEST Function
        // Identify good matches using RANSAC
        // Return fundemental matrix
		// ##########################################################################################
          cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
                                 const std::vector<cv::KeyPoint>& keypoints1,
                                                 const std::vector<cv::KeyPoint>& keypoints2,
                                             std::vector<cv::DMatch>& outMatches) {

				cv::Mat f_mask;
				//cv::gpu::GpuMat fundamentalGPU;
                // Convert keypoints into Point2f
                std::vector<cv::Point2f> points1, points2;
                for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
                         it!= matches.end(); ++it) {

                         // Get the position of left keypoints
                         float x= keypoints1[it->queryIdx].pt.x;
                         float y= keypoints1[it->queryIdx].pt.y;
                         points1.push_back(cv::Point2f(x,y));
                         // Get the position of right keypoints
                         x= keypoints2[it->trainIdx].pt.x;
                         y= keypoints2[it->trainIdx].pt.y;
                         points2.push_back(cv::Point2f(x,y));
            }

                // Compute F matrix using RANSAC
                std::vector<uchar> inliers(points1.size(),0);
                cv::Mat fundamental= cv::findFundamentalMat(
                        cv::Mat(points1),cv::Mat(points2), // matching points
                    inliers,      // match status (inlier ou outlier)
                    CV_FM_RANSAC, // RANSAC method
                    distance,     // distance to epipolar line
                    confidence);  // confidence probability

                // extract the surviving (inliers) matches
                std::vector<uchar>::const_iterator itIn= inliers.begin();
                std::vector<cv::DMatch>::const_iterator itM= matches.begin();
                // for all matches
                for ( ;itIn!= inliers.end(); ++itIn, ++itM) {

                        if (*itIn) { // it is a valid match

                                outMatches.push_back(*itM);
                        }
                }

                std::cout << "Number of matched points (after RANSAC): " << outMatches.size() << std::endl;

				if (outMatches.size()>0) {

					if (refineF) {
					// The F matrix will be recomputed with all accepted matches

							// Convert keypoints into Point2f for final F computation
							points1.clear();
							points2.clear();

							for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();
									 it!= outMatches.end(); ++it) {

									 // Get the position of left keypoints
									 float x= keypoints1[it->queryIdx].pt.x;
									 float y= keypoints1[it->queryIdx].pt.y;
									 points1.push_back(cv::Point2f(x,y));
									 // Get the position of right keypoints
									 x= keypoints2[it->trainIdx].pt.x;
									 y= keypoints2[it->trainIdx].pt.y;
									 points2.push_back(cv::Point2f(x,y));
							}

							// Compute 8-point F from all accepted matches
							fundamental = cv::findFundamentalMat(
									cv::Mat(points1),cv::Mat(points2), // matching points
									CV_FM_8POINT); // 8-point method
									//3.0, 0.99, f_mask);
					}


					// Refine matches by relative image position
					for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();
							it!= outMatches.end(); ++it) {
								// Get the position of left keypoints
								float x= keypoints1[it->queryIdx].pt.x;
								float y= keypoints1[it->queryIdx].pt.y;
								// Get the position of right keypoints
								float x2= keypoints2[it->trainIdx].pt.x;
								float y2= keypoints2[it->trainIdx].pt.y;

								double dist = sqrtl(((x-x2)*(x-x2) + (y-y2)*(y-y2)));

								if (dist < thresholdDist2)
								{
									f_err += 1;
								}
								else if (dist < thresholdDist)
								{
									f_err += 2;
								}
								else
								{
									f_err += 3;
								}

							}

					//fundamentalGPU.upload(fundamental);
					return fundamental;
				}
				else
	                std::cout << "Frogs are not a match!!" << std::endl << std::endl;

							// Compute 8-point F from all accepted matches
							//cv::Mat(points2) = cv::Mat(points1);
							//fundemental= cv::findFundamentalMat(
							//		cv::Mat(points1),cv::Mat(points2), // matching points
							//		CV_FM_8POINT); // 8-point method
					//fundamentalGPU.upload(fundamental);
					score = 1000000.0;
					return fundamental;
          }

		// ##########################################################################################
		// match Function
        // Match feature points using symmetry test and RANSAC
        // returns fundemental matrix
		// ##########################################################################################
          cv::Mat match(cv::Mat& image1, cv::Mat& image2, // input images
                  std::vector<cv::DMatch>& matches, // output matches and keypoints
                  std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2) {

				//int64 t = cv::getTickCount();
				score = 0;
				int nOctaves = 4; // number of gaussian pyramid octaves for SURF detector
				int nLayers = 2; // number of octave layers for SURF detector
				bool extDescriptors = false; // extended descriptors
				bool upright = true; // upright orientation for descriptors
				std::vector<std::vector<cv::DMatch> > matches1;
				std::vector<std::vector<cv::DMatch> > matches2;

				// 1. Detection of the keypoint features, compute descriptors
				if (GPUMODE && cv::gpu::getCudaEnabledDeviceCount() > 0)
				{
					std::cout << "GPU mode enabled " << std::endl;
					cv::gpu::GpuMat imageGpu1, imageGpu2;
					cv::gpu::GpuMat keypoints1GPU, keypoints2GPU;
					cv::gpu::GpuMat descriptors1GPU, descriptors2GPU;

					imageGpu1.upload(image1);
					imageGpu2.upload(image2);

					// --------------------------------------------------------------------------------------
					// Construction of the SURF feature detector
					// --------------------------------------------------------------------------------------
					//double minHessian = 500;  // minimum Hessian threshold

					cv::gpu::SURF_GPU surf( minHessian, // minimum Hessian threshold
											  nOctaves, /*	The number of gaussian pyramid octaves that the detector uses.
															It is set to 4 by default.
															For very large features, use a larger value.
															For small features, decrease it. */
											   nLayers, //	The number of octave layers.
										extDescriptors, /*	Extended descriptors bool
															False indicates basic descriptors (64 elements each)
															True set extended descriptors (128 elements ea.) shall be computed */
											   upright); /*	Upright orientation of features
															True - upright only (do not compute orientation)
															False - compute orientation  */

					surf(imageGpu1, cv::gpu::GpuMat(), keypoints1GPU, descriptors1GPU);
					surf(imageGpu2, cv::gpu::GpuMat(), keypoints2GPU, descriptors2GPU);

					cv::gpu::BFMatcher_GPU matcher(4);

					// from image 1 to image 2
					// based on k nearest neighbours (with k=2)
					matcher.knnMatch(descriptors1GPU,descriptors2GPU,
							matches1, // vector of matches (up to 2 per entry)
							2);               // return 2 nearest neighbours

					// from image 2 to image 1
					// based on k nearest neighbours (with k=2)
					matcher.knnMatch(descriptors2GPU,descriptors1GPU,
							matches2, // vector of matches (up to 2 per entry)
							2);               // return 2 nearest neighbours
					//std::cout << "knnMatch 2->1" << std::endl;

					// 2b. Download results
					surf.downloadKeypoints(keypoints1GPU, keypoints1);
					surf.downloadKeypoints(keypoints2GPU, keypoints2);
				}
				else
				{
					std::cout << "Running in CPU mode... " << std::endl;
					cv::Mat descriptors1, descriptors2;

					cv::SURF surf( minHessian, // minimum Hessian threshold
											  nOctaves, /*	The number of gaussian pyramid octaves that the detector uses.
															It is set to 4 by default.
															For very large features, use a larger value.
															For small features, decrease it. */
											   nLayers, //	The number of octave layers.
										extDescriptors, /*	Extended descriptors bool
															False indicates basic descriptors (64 elements each)
															True set extended descriptors (128 elements ea.) shall be computed */
											   upright); /*	Upright orientation of features
															True - upright only (do not compute orientation)
															False - compute orientation  */

					surf(image1, cv::Mat(), keypoints1, descriptors1);
					surf(image2, cv::Mat(), keypoints2, descriptors2);
					cv::BFMatcher matcher(4);

					// from image 1 to image 2
					// based on k nearest neighbours (with k=2)
					matcher.knnMatch(descriptors1,descriptors2,
							matches1, // vector of matches (up to 2 per entry)
							2);               // return 2 nearest neighbours
					//std::cout << "knnMatch 1->2" << std::endl;

					// from image 2 to image 1
					// based on k nearest neighbours (with k=2)
					matcher.knnMatch(descriptors2,descriptors1,
							matches2, // vector of matches (up to 2 per entry)
							2);               // return 2 nearest neighbours
					//std::cout << "knnMatch 2->1" << std::endl;
					}

                std::cout << "Number of SURF points (1): " << keypoints1.size() << std::endl;
                std::cout << "Number of SURF points (2): " << keypoints2.size() << std::endl;
                //std::cout << "SURF time: " << surfTime << std::endl;
                //std::cout << "knnMatch time: " << knnMatchTime << std::endl;
				//std::cout << "Number of matched points 1->2: " << matches1.size() << std::endl;
                //std::cout << "Number of matched points 2->1: " << matches2.size() << std::endl;

                // 3. Remove matches for which NN ratio is > than threshold
				//t = cv::getTickCount();
                // clean image 1 -> image 2 matches
                int removed= ratioTest(matches1);
                std::cout << "Number of matched points 1->2 (ratio test) : " << matches1.size()-removed << std::endl;
                // clean image 2 -> image 1 matches
                removed= ratioTest(matches2);
                std::cout << "Number of matched points 2->1 (ratio test) : " << matches2.size()-removed << std::endl;
				//ratioTime = (cv::getTickCount() - t) / cv::getTickFrequency();
                //std::cout << "Ratio test time: " << ratioTime << std::endl;

                // 4. Remove non-symmetrical matches
				std::vector<cv::DMatch> symMatches;
                symmetryTest(matches1,matches2,symMatches);

                std::cout << "Number of matched points (symmetry test): " << symMatches.size() << std::endl;

				// If there are no symmetrically matched points, skip the RANSAC test (which will fail)
				if (symMatches.size()==0)
					{
						score = 1000000.0;
						cv::Mat fundamental;
						return fundamental;
				}

				// 5. Validate matches using RANSAC
				f_err = 0;
				thresholdDist = 0.20 * sqrtl((image1.size().width*image1.size().width + image1.size().height*image1.size().height));
				thresholdDist2 = 0.3 * sqrtl((image2.size().width*image2.size().width + image2.size().height*image2.size().height));
				cv::Mat fundamental= ransacTest(symMatches, keypoints1, keypoints2, matches);

				if(matches.size() == 0)
				{
					score = 1000000.0;
				}
				else
				{
					for(unsigned int i=0; i<matches.size(); i++)
						score += (double) matches[i].distance;
					score = 10000.0 * score / (matches.size() * matches.size());
					score2 = score * f_err / matches.size();
                std::cout << "Match count : Score  " << matches.size() << " : " << score << " , " << score2 << std::endl << std::endl;
				}

                // return the found fundemental matrix
                return fundamental;
        }
};

#endif



#endif /* FROGMATCHER_H_ */
