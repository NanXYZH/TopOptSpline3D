#include "ImplicitFunction.h"
#include<fstream>
#define PI_DIV_180 (0.017453292519943296)  //π/180  
#define DegToRad(x) ((x)*PI_DIV_180)

using namespace std;

//blobs function
BlobsFunction::BlobsFunction(){}

BlobsFunction::~BlobsFunction(){}

double BlobsFunction::calculate_value(double x, double y, double z)
{
	double value_1 = 0;
	for (int i = 0; i < cell_centers.size(); i++)
	{
		for (int j = 0; j < cell_centers[i].size(); j++)
		{
			//circle
			double d = ((x - cell_centers[i][j].data()[0])*(x - cell_centers[i][j].data()[0]) + (y - cell_centers[i][j].data()[1])*(y - cell_centers[i][j].data()[1]) + (z - cell_centers[i][j].data()[2])*(z - cell_centers[i][j].data()[2]));
			value_1 = value_1 + cell_radius[i][j] * cell_radius[i][j] / d;

			//ellipse
			//method_1
			/*double alpha = DegToRad(VecAngle_alpha[i][j]);
			double beta = DegToRad(VecAngle_beta[i][j]);
			double gama = DegToRad(VecAngle_gama[i][j]);
			double u_x = x*cos(alpha)*cos(beta) + y * sin(alpha)*cos(beta) - z*sin(beta);
			double u_y = x*(cos(alpha)*sin(beta)*sin(gama) - sin(alpha)*cos(gama))
				+ y *( sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama))
				+z*(cos(beta)*sin(gama));
			double u_z = x*(cos(alpha)*cos(beta)*cos(gama)+sin(alpha)*sin(gama) )
				+ y * (sin(alpha)*sin(beta)*cos(gama)-cos(alpha)*sin(gama))
				+z*(cos(beta)*cos(gama));
			double d = ((u_x - cell_centers[i][j].data()[0])*(u_x - cell_centers[i][j].data()[0]) / ( r_1[i][j] * r_1[i][j])
				+ (u_y - cell_centers[i][j].data()[1])*(u_y - cell_centers[i][j].data()[1]) / ( r_2[i][j]*r_2[i][j])
				+ (u_z - cell_centers[i][j].data()[2])*(u_z - cell_centers[i][j].data()[2]) / ( r_3[i][j]*r_3[i][j]));
			value_1 = value_1 + cell_radius[i][j] * cell_radius[i][j] / d;*/

			//method_2
			//double d = ((x - cell_centers[i][j].data()[0])*(x - cell_centers[i][j].data()[0]) / (r_1[i][j] * r_1[i][j])
			//	+ (y - cell_centers[i][j].data()[1])*(y - cell_centers[i][j].data()[1]) / (r_2[i][j] * r_2[i][j])
			//	+ (z - cell_centers[i][j].data()[2])*(z - cell_centers[i][j].data()[2])) / (r_3[i][j] * r_3[i][j]);
			//value_1 = value_1 + cell_radius[i][j] * cell_radius[i][j] / d;

			/*if (d >= cell_radius[i][j])
			{
				value_1 = value_1 + 0;
			}
			else if (d >= cell_radius[i][j] / 3 && d <= cell_radius[i][j])
			{
				value_1 = value_1 + 1.5*(1 - d / cell_radius[i][j])*(1 - d / cell_radius[i][j]);
			}
			else if (d >= 0 && d <= cell_radius[i][j] / 3)
			{
				value_1 = value_1 + (1 - 3* d *d/ cell_radius[i][j]);
			}*/
		}
	}
	//cout << "value_1" << value_1 << endl;
	return value_1;
}

//TVspline Function
TVsplineFunction::TVsplineFunction(){}

TVsplineFunction::~TVsplineFunction(){}

void TVsplineFunction::InitKnot(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z)
{
	degree[0] = 3;
	degree[1] = 3;
	degree[2] = 3;
	partition[0] = 16;
	partition[1] = 8;
	partition[2] = 24;

	mKnotVector_X.clear();
	mKnotVector_Y.clear();
	mKnotVector_Z.clear();
	mKnotVector_X.push_back(min_x);
	mKnotVector_X.push_back(min_x);
	mKnotVector_X.push_back(min_x);
	mKnotVector_X.push_back(min_x);
	double delta_x = (max_x-min_x) / (partition[0]);
	double add_x = min_x;
	for (int i = 0; i < partition[0]-1; i++)
	{
		add_x = add_x + delta_x;
		mKnotVector_X.push_back(add_x);
	}
	mKnotVector_X.push_back(max_x);
	mKnotVector_X.push_back(max_x);
	mKnotVector_X.push_back(max_x);
	mKnotVector_X.push_back(max_x);

	double delta_y = (max_y - min_y) / (partition[1]);
	double add_y = min_y;
	mKnotVector_Y.push_back(min_y);
	mKnotVector_Y.push_back(min_y);
	mKnotVector_Y.push_back(min_y);
	mKnotVector_Y.push_back(min_y);
	for (int i = 0; i < partition[1]-1; i++)
	{
		add_y = add_y + delta_y;
		mKnotVector_Y.push_back(add_y);
	}
	mKnotVector_Y.push_back(max_y);
	mKnotVector_Y.push_back(max_y);
	mKnotVector_Y.push_back(max_y);
	mKnotVector_Y.push_back(max_y);

	double delta_z = (max_z - min_z) / (partition[2]);
	double add_z = min_z;
	mKnotVector_Z.push_back(min_z);
	mKnotVector_Z.push_back(min_z);
	mKnotVector_Z.push_back(min_z);
	mKnotVector_Z.push_back(min_z);
	for (int i = 0; i < partition[2]-1; i++)
	{
		add_z = add_z + delta_z;
		mKnotVector_Z.push_back(add_z);
	}
	mKnotVector_Z.push_back(max_z);
	mKnotVector_Z.push_back(max_z);
	mKnotVector_Z.push_back(max_z);
	mKnotVector_Z.push_back(max_z);
}

void TVsplineFunction::InitCoffi(string filename_)
{
	ifstream outFile("./output/coef_" + filename_ + ".txt");
	char line[1024] = { 0 };
	coffi.clear();

	while (outFile.getline(line, sizeof(line)))
	{
		istringstream is(line);
		std::string ss = is.str();
		ss.erase(0, ss.find_first_not_of(' '));
		ss.erase(ss.find_last_not_of(' ') + 1);

		const char* loc = ss.c_str();
		int res = 0;
		atoi(ss.c_str());
		loc = strstr(ss.c_str(), " ");
		while (loc != NULL)
		{
			atoi(loc + 1);
			res++;
			loc = strstr(loc + 1, " ");
		}
		std::vector<double> v_index;
		v_index.resize(res + 1);
		for (int i = 0; i < res + 1; i++)
		{
			is >> v_index[i];
		}
		coffi.push_back(v_index);
	}
}

double TVsplineFunction::calculate_cublic_tensor_product(double x, double y, double z)
{
	double value_ = 0;
	std::vector<double>  x_spline, y_spline, z_spline;
	for (int k = 0; k < mKnotVector_Z.size() - 4;k++)
	{
		double basis_z = calculate_cubic_spline_value(k, z, mKnotVector_Z);
		z_spline.push_back(basis_z);
	}
	for (int j = 0; j < mKnotVector_Y.size() - 4;j++)
	{
		double basis_y = calculate_cubic_spline_value(j, y, mKnotVector_Y);
		y_spline.push_back(basis_y);
	}
	for (int i = 0; i < mKnotVector_X.size() - 4;i++)
	{
		double basis_x = calculate_cubic_spline_value(i, x, mKnotVector_X);
		x_spline.push_back(basis_x);
	}

	for (int k = 0; k < mKnotVector_Z.size()-4; k++)
	{
		if (z_spline[k] != 0)
		{
			for (int j = 0; j < mKnotVector_Y.size() - 4; j++)
			{
				if (y_spline[j] != 0)
				{
					for (int i = 0; i < mKnotVector_X.size() - 4; i++)
					{
						if (x_spline[i]!= 0)
						{
							value_ = value_ + coffi[k*(partition[1] + degree[1]) + j][i] * x_spline[i] * y_spline[j] * z_spline[k];
						}
					}
				}
			}
		}
	}
	return value_;
}

double TVsplineFunction::calculate_cubic_spline_value(int k1, double x, std::vector<double>  Knot_t)
{
	double value = 0;
	if (x>= Knot_t[k1] && x<= Knot_t[k1 + 1] && (Knot_t[k1 + 3] - Knot_t[k1]) && (Knot_t[k1 + 2] - Knot_t[k1]) && (Knot_t[k1 + 1] - Knot_t[k1]))
	{
		value = (x- Knot_t[k1]) *(x- Knot_t[k1]) *(x- Knot_t[k1]) / ((Knot_t[k1 + 3] - Knot_t[k1])*(Knot_t[k1 + 2] - Knot_t[k1])*(Knot_t[k1 + 1] - Knot_t[k1]));
		return value;
	}
	else if (x>= Knot_t[k1 + 1] && x<= Knot_t[k1 + 2] && (Knot_t[k1 + 3] - Knot_t[k1]) && (Knot_t[k1 + 2] - Knot_t[k1]) && (Knot_t[k1 + 2] - Knot_t[k1 + 1]) && (Knot_t[k1 + 3] - Knot_t[k1 + 1]) && (Knot_t[k1 + 4] - Knot_t[k1 + 1]))
	{
		value = (x- Knot_t[k1])*(x- Knot_t[k1])*(Knot_t[k1 + 2] - x) / ((Knot_t[k1 + 3] - Knot_t[k1])*(Knot_t[k1 + 2] - Knot_t[k1])*(Knot_t[k1 + 2] - Knot_t[k1 + 1]))
			+ (x- Knot_t[k1])*(Knot_t[k1 + 3] - x)*(x- Knot_t[k1 + 1]) / ((Knot_t[k1 + 3] - Knot_t[k1])*(Knot_t[k1 + 3] - Knot_t[k1 + 1])*(Knot_t[k1 + 2] - Knot_t[k1 + 1]))
			+ (Knot_t[k1 + 4] - x)*(x- Knot_t[k1 + 1])*(x- Knot_t[k1 + 1]) / ((Knot_t[k1 + 4] - Knot_t[k1 + 1])*(Knot_t[k1 + 3] - Knot_t[k1 + 1])*(Knot_t[k1 + 2] - Knot_t[k1 + 1]));
		return value;
	}
	else if (x>= Knot_t[k1 + 2] && x<= Knot_t[k1 + 3] && (Knot_t[k1 + 3] - Knot_t[k1]) && (Knot_t[k1 + 3] - Knot_t[k1 + 1]) && (Knot_t[k1 + 3] - Knot_t[k1 + 2]) && (Knot_t[k1 + 4] - Knot_t[k1 + 1]) && (Knot_t[k1 + 4] - Knot_t[k1 + 2]))
	{
		value = (x- Knot_t[k1])*(Knot_t[k1 + 3] - x)*(Knot_t[k1 + 3] - x) / ((Knot_t[k1 + 3] - Knot_t[k1])*(Knot_t[k1 + 3] - Knot_t[k1 + 1])*(Knot_t[k1 + 3] - Knot_t[k1 + 2]))
			+ (Knot_t[k1 + 4] - x)*(x- Knot_t[k1 + 1])*(Knot_t[k1 + 3] - x) / ((Knot_t[k1 + 4] - Knot_t[k1 + 1])*(Knot_t[k1 + 3] - Knot_t[k1 + 1])*(Knot_t[k1 + 3] - Knot_t[k1 + 2]))
			+ (Knot_t[k1 + 4] - x)*(Knot_t[k1 + 4] - x)*(x- Knot_t[k1 + 2]) / ((Knot_t[k1 + 4] - Knot_t[k1 + 1])*(Knot_t[k1 + 4] - Knot_t[k1 + 2])*(Knot_t[k1 + 3] - Knot_t[k1 + 2]));
		return value;
	}
	else if (x>= Knot_t[k1 + 3] && x<= Knot_t[k1 + 4] && (Knot_t[k1 + 4] - Knot_t[k1 + 1]) && (Knot_t[k1 + 4] - Knot_t[k1 + 2]) && (Knot_t[k1 + 4] - Knot_t[k1 + 3]))
	{
		value = (Knot_t[k1 + 4] - x)*(Knot_t[k1 + 4] - x)*(Knot_t[k1 + 4] - x) / ((Knot_t[k1 + 4] - Knot_t[k1 + 1])*(Knot_t[k1 + 4] - Knot_t[k1 + 2])*(Knot_t[k1 + 4] - Knot_t[k1 + 3]));
		return value;
	}
	return value;
}


