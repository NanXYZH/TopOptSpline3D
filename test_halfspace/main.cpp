#include<iostream>

using namespace std;

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

int main()
{

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