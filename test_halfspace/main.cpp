#include<iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;

// ���� ANSI escape codes
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

//��άdoubleʸ��
struct Vec3d
{
	double x, y, z;

	Vec3d()
	{
		x = 0.0;
		y = 0.0;
		z = 0.0;
	}
	Vec3d(double dx, double dy, double dz)
	{
		x = dx;
		y = dy;
		z = dz;
	}
	void Set(double dx, double dy, double dz)
	{
		x = dx;
		y = dy;
		z = dz;
	}
};

// �����������ķ�����
// p1(x1,y1,z1),p2(x2,y2,z2),p3(x3,y3,z3)
// p1p2(x2-x1,y2-y1,z2-z1),p1p3(x3-x1,y3-y1,z3-z1)
// a=(a1,a2,a3) b=(b1,b2,b3)
// a��b=(a2b3-a3b2��a3b1-a1b3��a1b2-a2b1)
void Cal_Normal_3D(const Vec3d& v1, const Vec3d& v2, const Vec3d& v3, Vec3d& vn)
{
	//v1(n1,n2,n3);
	//ƽ�淽��: na * (x �C n1) + nb * (y �C n2) + nc * (z �C n3) = 0 ;
	double na = (v2.y - v1.y) * (v3.z - v1.z) - (v2.z - v1.z) * (v3.y - v1.y);
	double nb = (v2.z - v1.z) * (v3.x - v1.x) - (v2.x - v1.x) * (v3.z - v1.z);
	double nc = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);

	//ƽ�淨����
	vn.Set(na, nb, nc);
}

// �����������ɰ��з�ת�ľ���
std::vector<std::vector<int>> flipMatrix(const std::vector<std::vector<int>>& matrix) {
	std::vector<std::vector<int>> flippedMatrix;

	// ��ȡ�����������
	int rows = matrix.size();
	int cols = matrix[0].size();

	// ��ʼ����ת��ľ���
	flippedMatrix.resize(cols, std::vector<int>(rows, 0));

	// ִ���з�ת����
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			flippedMatrix[j][i] = matrix[i][j];
		}
	}

	return flippedMatrix;
}

// �����������ɰ��жԳƵľ���
std::vector<std::vector<int>> ReverseMatrix(const std::vector<std::vector<int>>& matrix) {
	std::vector<std::vector<int>> ReversedMatrix;

	// ��ȡ�����������
	int rows = matrix.size();
	int cols = matrix[0].size();

	// ��ʼ�����ҵߵ��ľ���
	ReversedMatrix.resize(rows, std::vector<int>(cols, 0));

	// ִ���з�ת����
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			ReversedMatrix[i][j] = matrix[i][cols - j - 1];
		}
	}

	return ReversedMatrix;
}

// �����������ɰ��жԳƵľ���
std::vector<std::vector<int>> SymmetryMatrix2(const std::vector<int> matrix[3]) {
	std::vector<std::vector<int>> SymmetricMatrix;

	// ��ȡ�����������
	int rows = 3;
	int cols = matrix[0].size();

	// ��ʼ�����ҵߵ��ľ���
	SymmetricMatrix.resize(rows, std::vector<int>(cols, 0));

	// ִ���з�ת����
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			SymmetricMatrix[i][j] = matrix[i][cols - j - 1];
		}
	}

	return SymmetricMatrix;
}

// ��������ƴ����������
std::vector<std::vector<int>> concatenateMatrices(const std::vector<std::vector<int>>& matrix1, const std::vector<std::vector<int>>& matrix2) {
	std::vector<std::vector<int>> concatenatedMatrix;

	// ��ȡ����1��������
	int rows1 = matrix1.size();
	int cols1 = matrix1[0].size();

	// ��ȡ����2��������
	int rows2 = matrix2.size();
	int cols2 = matrix2[0].size();

	// �������Ƿ������ͬ������
	if (rows1 != rows2) {
		std::cerr << "Error: Matrices have different number of rows." << std::endl;
		return concatenatedMatrix;
	}

	// ��ʼ��ƴ�Ӻ�ľ���
	concatenatedMatrix.resize(rows1, std::vector<int>(cols1 + cols2, 0));

	// ���ƾ���1������
	for (int i = 0; i < rows1; ++i) {
		for (int j = 0; j < cols1; ++j) {
			concatenatedMatrix[i][j] = matrix1[i][j];
		}
	}

	// ���ƾ���2������
	for (int i = 0; i < rows2; ++i) {
		for (int j = 0; j < cols2; ++j) {
			concatenatedMatrix[i][j + cols1] = matrix2[i][j];
		}
	}

	return concatenatedMatrix;
}

std::vector<std::vector<int>> SymmetryMatrix3(const std::vector<int> matrix[3], std::vector<std::vector<int>> bdbox, int plane)
{
	std::vector<std::vector<int>> SyMatrx;
	// x(yoz plane) 01 
	// y(xoz plane) 23 
	// z(xoy plane) 45 
	// 0 2 4 left
	// 1 3 5 right

	SyMatrx.resize(3);
	for (int i = 0; i < 3; i++)
	{
		SyMatrx[i].resize(matrix[0].size());
	}

	for (int i = 0; i  < matrix[0].size(); ++i)
	{
		int tmp[3];		
		
		for (int j = 0; j < 3; j ++)
		{
			tmp[j] = matrix[j][i];
			if (plane % 2 == 0)
			{
				tmp[j] = tmp[j] - bdbox[0][j];
			}
			else
			{
				tmp[j] = tmp[j] - bdbox[1][j];
			}			
		}

		if (plane == 0 || plane == 1)
		{
			tmp[0] = -tmp[0];
		}
		else if (plane == 2 || plane == 3)
		{
			tmp[1] = -tmp[1];
		}
		else if (plane == 4 || plane == 5)
		{
			tmp[2] = -tmp[2];
		}

		for (int j = 0; j < 3; j++)
		{
			if (plane % 2 == 0)
			{
				tmp[j] = tmp[j] + bdbox[0][j];
			}
			else
			{
				tmp[j] = tmp[j] + bdbox[1][j];
			}

			SyMatrx[j][i] = tmp[j];
		}
	}

	return SyMatrx;
}

std::vector<std::vector<int>> findVdbBoundingbox(std::vector<int> pos[3])
{
	std::vector<std::vector<int>> boundingind;
	boundingind.resize(2);
	boundingind[0].resize(3);
	boundingind[1].resize(3);

	int minX, minY, minZ, maxX, maxY, maxZ;
	minX = std::numeric_limits<int>::max();
	minY = std::numeric_limits<int>::max();
	minZ = std::numeric_limits<int>::max();

	maxX = std::numeric_limits<int>::lowest();
	maxY = std::numeric_limits<int>::lowest();
	maxZ = std::numeric_limits<int>::lowest();

	for (size_t i = 0; i < pos[0].size(); ++i) {
		minX = std::min(minX, pos[0][i]);
		minY = std::min(minY, pos[1][i]);
		minZ = std::min(minZ, pos[2][i]);

		maxX = std::max(maxX, pos[0][i]);
		maxY = std::max(maxY, pos[1][i]);
		maxZ = std::max(maxZ, pos[2][i]);
	}

	boundingind[0][0] = minX;
	boundingind[0][1] = minY;
	boundingind[0][2] = minZ;

	boundingind[1][0] = maxX;
	boundingind[1][1] = maxY;
	boundingind[1][2] = maxZ;

	return boundingind;
}

int main()
{

	// �������ɫ���ı�
	printf("%sRed Text%s\n", RED, RESET);
	printf("%sGreen Text%s\n", GREEN, RESET);
	printf("%sYellow Text%s\n", YELLOW, RESET);
	printf("%sBlue Text%s\n", BLUE, RESET);

	int boundingind[2][3] = { std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest() };

	// ʾ������
	std::vector<std::vector<int>> originalMatrix = {
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9}
	};

	std::vector<int> epos[3];
	int rows = 3;
	int cols = 10;
	//epos.resize(rows);
	for (int i = 0; i < 3; i++) epos[i].resize(cols);
	int t = 1;

	for (int j = 0; j < epos[0].size(); ++j)
	{
		for (int i = 0; i < 3; ++i)
		{
			epos[i][j] = t;
			t = t + 1;
		}
	}

	std::vector<std::vector<int>> bound_ = findVdbBoundingbox(epos);
	for (const auto& row : bound_) {
		for (int value : row) {
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}

	std::vector<std::vector<int>> epos2;
	epos2.resize(rows);
	for (int i = 0; i < 3; i++) epos2[i] = epos[i];

	// ��ӡ
	std::cout << "Origin Matrix:" << std::endl;
	for (const auto& row : epos2) {
		for (int value : row) {
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}

	std::vector<std::vector<int>> SymMatrix;
	for (int i = 0; i < 6; ++i)
	{
		SymMatrix = SymmetryMatrix3(epos, bound_, i);

		// ��ӡ��ת��ľ���
		std::cout << "Symmetry Matrix: " << i <<  std::endl;
		for (const auto& row : SymMatrix) {
			for (int value : row) {
				std::cout << value << " ";
			}
			std::cout << std::endl;
		}
	}

	// ƴ����������
	std::vector<std::vector<int>> concatenatedMatrix = concatenateMatrices(epos2, SymMatrix);
	std::vector<int> concatenatedMatrix2[3];
	for (int i = 0; i < 3; i++) concatenatedMatrix2[i] = concatenatedMatrix[i];

	// ��ӡƴ�Ӻ�ľ���
	std::cout << "Concatenated Matrix:" << std::endl;
	for (const auto& row : concatenatedMatrix2) {
		for (int value : row) {
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}

	float angle_epsilon = 170 / 180;
	float test1 = 10., test2 = 20.;

	std::cout << angle_epsilon << std::endl;
	std::cout << test1 / test2 << std::endl;

	int x = 10;
	int y = 20;

	x = 11;
	y = 21;
	// ��ֵ����x��y
	auto lambda = [=]() {
		std::cout << "x: " << x << ", y: " << y << std::endl;
	};

	lambda(); // ��� x: 10, y: 20

	x = 100;
	y = 200;

	lambda();

	int num = 42;
	int width = 10;
	printf("Integer number: %*d, %*d\n", width, num, width, num + 1);

	unsigned int flag = 0; // ��ʼ��Ϊ0������λ����0

	// �����ض�λ
	int positionToSet = 15;
	flag |= (1u << positionToSet);

	// ����ض�λ
	int positionToClear = 2;
	flag &= ~(1u << positionToClear);

	// ����ض�λ
	int positionToCheck = 3;
	bool isSet = (flag & (1u << positionToCheck)) != 0;

	std::cout << "Size of unsigned int: " << sizeof(unsigned int) * 8 << " bits" << std::endl;
	std::cout << "Flag: " << flag << std::endl;
	std::cout << "Is bit " << positionToCheck << " set? " << isSet << std::endl;


	Vec3d v1(-14.411007249513272, 6.6273702236878158, 0.0);
	Vec3d v2(-14.411007249513272, 6.6273702236878158, 1.0);
	Vec3d v3(-14.411007249513272, 7.6273702236878158, 0.0);
	Vec3d vn;
	Cal_Normal_3D(v1, v2, v3, vn);
	cout << "������Ϊ��" << vn.x << '\t' << vn.y << '\t' << vn.z << '\n';

	return 0;
}