// D:\[2]topopt_spline\result\cube24x24x12\ordsplinetopmma19

#include <iostream>
#include <fstream>
#include <vector>
#include "binaryIO.h"

int main() {
	// 文件路径
	std::string filename = "../../[2]topopt_spline/result/cube24x24x12/ordsplinetopmma19/c";

	std::vector<double> cRecord, volRecord;

	bio::read_vector(filename, volRecord);

	std::cout << volRecord.size() << std::endl;

	for (int i = 0; i < volRecord.size(); i++)
	{
		printf("--   v = %9.6lf \n", volRecord[i]);
	}

	std::vector<double> fhost;
	bool suc = bio::read_vector(filename, fhost);
	if (!suc) {
		printf("\033[31mFailed to open file %s \n\033[0m", filename.c_str());
		throw std::runtime_error("error open file");
	}
	//if (fhost.size() != n_gsvertices * 3) {
	//	printf("\033[31mForce Size does not match\033[0m\n");
	//	printf("\033[31mForce Size Load: %i, Force size: %i\033[0m\n", fhost.size() / 3, n_gsvertices);
	//	throw std::runtime_error("invalid size");
	//}

	// 打开文件
	std::ifstream file(filename, std::ios::binary);

	// 检查文件是否成功打开
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return 1;
	}

	// 读取文件内容到 vector<char>
	std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	
	// 关闭文件
	file.close();


	// 处理读取的二进制数据（这里只是简单地打印）
	for (char byte : data) {
		std::cout << static_cast<int>(byte) << " "; // 打印每个字节的整数值
	}

	return 0;
}
