// CMakeProject1.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <Eigen>
struct exportableDatum {
	Eigen::Vector3f scale;
	Eigen::Vector3f rotationAngles;
	Eigen::Vector3f translations;
};

// TODO: Reference additional headers your program requires here.