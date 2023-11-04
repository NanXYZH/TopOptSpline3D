//////////////////////////////////////////////////////////////////////////
// implicit function format:
// 1. polynomial implicit function
// 2. blobs implicit function
// 3. tv spline implicit function
//////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>
#include <iostream>
#include"MarchingCube.h"
#include "TriMesh3D.h"
using namespace std;

class BlobsFunction
{
public:
	BlobsFunction();
	~BlobsFunction();
	double calculate_value(double x, double y, double z);

public:
	std::vector<std::vector<double>>      VecAngle_alpha;
	std::vector<std::vector<double>>      VecAngle_beta;
	std::vector<std::vector<double>>      VecAngle_gama;
	std::vector<std::vector<double>>      r_1;
	std::vector<std::vector<double>>      r_2;
	std::vector<std::vector<double>>      r_3;
	//std::vector<std::vector<double>>      VecAngle;
	std::vector<std::vector<Point3>>     cell_centers;
	std::vector<std::vector<double>>  cell_radius;
	double value_;
};

class TVsplineFunction
{
public:
	TVsplineFunction();
	~TVsplineFunction();
	void InitKnot(double min_x_,double min_y_,double min_z_, double max_x_, double max_y_, double max_z_);
	void InitCoffi(string filename_);
	double calculate_cublic_tensor_product(double x, double y ,double z);
	int IsNull(){ return mKnotVector_X.size();}
	double calculate_cubic_spline_value(int k1, double x, const std::vector<double> Knot_t);

private:
	std::vector<double>		 mKnotVector_X;
	std::vector<double>       mKnotVector_Y;
	std::vector<double>       mKnotVector_Z;
	std::vector<std::vector<double>>       coffi;
	
	int		degree[3];
	int		partition[3];	
};