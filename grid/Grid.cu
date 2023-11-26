#include "Grid.h"
//#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "templateMatrix.h"
#include "lib.cuh"
#include "projection.h"
#include "tictoc.h"
//#define GLM_FORCE_CUDA
//// #define GLM_FORCE_PURE (not needed anymore with recent GLM versions)
//#include <glm/glm.hpp>
#include "matlab_utils.h"

#define DIRICHLET_DIAGONAL_WEIGHT 1e6f
//#define DIRICHLET_DIAGONAL_WEIGHT 1

using namespace grid;

__constant__ double gTemplateMatrix[24][24];
__constant__ int* gV2E[8];
__constant__ int* gV2Vfine[27];
__constant__ int* gV2Vcoarse[8];
__constant__ int* gV2V[27];
__constant__ int* gVfine2Vfine[27];
__constant__ int* gV2VfineC[64];// vertex to fine grid element center 
__constant__ int* gVfine2Efine[8];
__constant__ int* gVfine2Effine[8];
__constant__ float power_penalty[1];
__constant__ double* gU[3];
__constant__ double* gF[3];
__constant__ double* gR[3];
__constant__ double* gUworst[3];
__constant__ double* gFworst[3];
__constant__ double* gRfine[3];
__constant__ double* gUcoarse[3];
__constant__ int gGS_num[8];
__constant__ int gmode[1];
__constant__ int* gVflag[1];
__constant__ int* gEflag[1];
__constant__ int gLayerid[1];
__constant__ int gDEBUG[1];
__constant__ int gorder[1];
__constant__ int gnpartition[3];
__constant__ int gnbasis[3];
__constant__ int gnknotspan[3];
__constant__ float gnstep[3];
__constant__ float gnBoundMin[3];
__constant__ float gnBoundMax[3];

__constant__ float* gpu_KnotSer[3];
__constant__ float* gpu_cijk;
__constant__ float* gpu_SurfacePoints[3];
__constant__ float* gpu_surface_normal[3];
__constant__ float* gpu_surface_normal_direction[1];

__constant__ float* gpu_surface_normal_dc[3];         // derivative to coeffs
__constant__ float* gpu_surface_normal_norm_dc[1];

__constant__ int gssmode[1];
__constant__ int gdripmode[1];
__constant__ float gdefaultPrintAngle[1];
__constant__ float goptPrintAngle[1];

extern __constant__ double* gLoadtangent[2][3];
extern __constant__ double* gLoadnormal[3];

extern gBitSAT<unsigned int> vid2loadid;

__device__ float Heaviside(float s, float beta) {
#if 0
	return s;
#elif 1
	float eta = 0.5;
	return (tanhf(beta * eta) + tanhf(beta * (s - eta))) / (tanhf(beta * eta) + tanhf(beta * (1 - eta)));
#elif 0
	float eta = 0.5;
	return (tanhf(beta * eta) + tanhf(beta * (s - eta))) / (tanhf(beta * eta) + tanhf(beta * (1 - eta)));
#else 
	float T = 0.5f;
	float eta = 0.1f;
	float alpha = 0.001f;
	if (s > (T + eta)) {
		return 1;
	}
	else if (s < (T - eta)) {
		return alpha;
	}
	else {
		return 3 * (1 - alpha) / 4 * ((s - T) / eta - (s - T) * (s - T) * (s - T) / (eta * eta * eta * 3)) + (1 + alpha) / 2;
	}
#endif
}

__device__ float Dirac(float s, float beta) {
#if 0
	return 1;
#elif 1
	float eta = 0.5;
	return beta * (1 - tanhf(beta * (s - eta)) * tanhf(beta * (s - eta))) / (tanhf(beta * eta) + tanhf(beta * (1 - eta)));
#elif 0
	float eta = 0.5;
	return beta * (1 - tanhf(beta * (s - eta)) * tanhf(beta * (s - eta))) / (tanhf(beta * eta) + tanhf(beta * (1 - eta)));
#else
	float T = 0.5f;
	float eta = 0.1f;
	float alpha = 0.001f;
	if (s > (T + eta) || s < (T - eta)) {
		return 0;
	}
	else {
		return 3 * (1 - alpha) / (4 * eta) - 3 * (1 - alpha) * (s - T) * (s - T) / (4 * eta * eta * eta);
	}
#endif
}

__device__ float h(float x, float para_func) {
#if 1
	return x >= 1 ? para_func * (x - 1) * (x - 1) : 0; // hfunction_c
#else 
	// 4 --> tanh func (这个不适用于这里，具体的看函数图像就知道了,因为有部分是负的，算出来约束不对)
	 //return 2 / (1 + expf(-2 * x / tanh_g)) - 1;
	// 5 --> sigmoid function
	return 1 / (1 + expf(-(x - 1) / para_func)); // sigmoid_c
	//
#endif
}

__device__ float dh(float x, float para_func) {
#if 1
	return x >= 1 ? 2 * para_func * (x - 1) : 0; // hfunction_c
#else
	// 4 --> tanh func (这个不适用于这里，具体的看函数图像就知道了,因为有部分是负的，算出来约束不对)
	 //return 4 * expf(-2 * x / tanh_g) / (1 + expf(-2 * x / tanh_g)) / (1 + exp(-2 * x / tanh_g)) / tanh_g;
	// 5 --> Sigmoid function
	return 1 / (1 + expf(-(x - 1) / para_func)) / (1 + expf(-(x - 1) / para_func)) / para_func * expf(-(x - 1) / para_func); // sigmoid_c
#endif
}

__device__ float oh(float x, float para_func) {
#if 0
	return x >= 1 ? para_func * (x - 1) * (x - 1) + 1 : 1;  // hfunction_c
#else 
	// 1 --> tanh func (这个不适用于这里，具体的看函数图像就知道了,因为有部分是负的，算出来约束不对)
	 //return 2 / (1 + expf(-2 * x / tanh_g)) - 1;
	// 2 --> sigmoid function (false)
	//return 1 / (1 + expf(-(x - 1) / sigmoid_c));
	// 3 --> sigmoid function (good)
	return 1 / (1 + expf(-(x - 1.5) / para_func)) + 1;  // sigmoid_c
	// 4 --> polynomial function
	//return -(x - 0.7) * (x - 0.7) + 1;
#endif
}

__device__ float doh(float x, float para_func) {
#if 0
	return x >= 1 ? 2 * para_func * (x - 1) : 0; // hfunction_c
#else
	// 1 --> tanh func (这个不适用于这里，具体的看函数图像就知道了,因为有部分是负的，算出来约束不对)
	 //return 4 * expf(-2 * x / tanh_g) / (1 + expf(-2 * x / tanh_g)) / (1 + exp(-2 * x / tanh_g)) / tanh_g;
	// 2 --> Sigmoid function (false)
	//return 1 / (1 + expf(-(x - 1) / sigmoid_c)) / (1 + expf(-(x - 1) / sigmoid_c)) / sigmoid_c * expf(-(x - 1) / sigmoid_c);
	// 3 --> Sigmoid function (good)
	return 1 / (1 + expf(-(x - 1.5) / para_func)) / (1 + expf(-(x - 1.5) / para_func)) / para_func * expf(-(x - 1.5) / para_func); // sigmoid_c
	// 4 --> polynomial function
	//return -2 * (x - 0.7);
#endif
}


void Grid::use_grid(void)
{
	cudaMemcpyToSymbol(gV2V, _gbuf.v2v, sizeof(gV2V));
	cudaMemcpyToSymbol(gV2Vfine, _gbuf.v2vfine, sizeof(gV2Vfine));
	cudaMemcpyToSymbol(gV2Vcoarse, _gbuf.v2vcoarse, sizeof(gV2Vcoarse));
	cudaMemcpyToSymbol(gV2E, _gbuf.v2e, sizeof(gV2E));
	cudaMemcpyToSymbol(gV2VfineC, _gbuf.v2vfinecenter, sizeof(gV2VfineC));
	cudaMemcpyToSymbol(gU, _gbuf.U, sizeof(gU));
	cudaMemcpyToSymbol(gF, _gbuf.F, sizeof(gF));
	cudaMemcpyToSymbol(gR, _gbuf.R, sizeof(gR));
	cudaMemcpyToSymbol(gUworst, _gbuf.Uworst, sizeof(gUworst));
	cudaMemcpyToSymbol(gFworst, _gbuf.Fworst, sizeof(gFworst));
	cudaMemcpyToSymbol(gGS_num, gs_num, sizeof(gGS_num));
	cudaMemcpyToSymbol(gVflag, &_gbuf.vBitflag, sizeof(gVflag));
	cudaMemcpyToSymbol(gEflag, &_gbuf.eBitflag, sizeof(gEflag));
	cudaMemcpyToSymbol(gLayerid, &_layer, sizeof(gLayerid));

	if (fineGrid != nullptr) {
		cudaMemcpyToSymbol(gVfine2Vfine, fineGrid->_gbuf.v2v, sizeof(gVfine2Vfine));
		cudaMemcpyToSymbol(gVfine2Efine, fineGrid->_gbuf.v2e, sizeof(gVfine2Efine));
		cudaMemcpyToSymbol(gRfine, fineGrid->_gbuf.R, sizeof(gRfine));
	}
	if (coarseGrid != nullptr) {
		cudaMemcpyToSymbol(gUcoarse, coarseGrid->_gbuf.U, sizeof(gUcoarse));
	}
	//cudaDeviceSynchronize();
	cuda_error_check;
}

void grid::Grid::uploadSymbol2device(void)
{
	cudaMemcpyToSymbol(gpu_surface_normal, _gbuf.surface_normal, sizeof(gpu_surface_normal));
	cuda_error_check;

	cudaMemcpyToSymbol(gpu_surface_normal_direction, &_gbuf.surface_normal_direction, sizeof(gpu_surface_normal_direction));
	cuda_error_check;

	cudaMemcpyToSymbol(gpu_surface_normal_dc, _gbuf.surface_normal_dc, sizeof(gpu_surface_normal_dc));
	cuda_error_check;

	cudaMemcpyToSymbol(gpu_surface_normal_norm_dc, &_gbuf.surface_normal_norm_dc, sizeof(gpu_surface_normal_norm_dc));
	cuda_error_check;
}

void grid::Grid::scaleVector(float* p_data, size_t len, float scale)
{
	array_t<float> vec_map(p_data, len);
	vec_map *= scale;
	cuda_error_check;
}

__device__ bool isValidNode(int vid) {
	return gV2V[13][vid] != -1;
}

__device__ void loadTemplateMatrix(volatile double KE[24][24]) {
	int i = threadIdx.x / 24;
	int j = threadIdx.x % 24;
	if (i < 24) {
		KE[i][j] = gTemplateMatrix[i][j];
	}
	int nfill = blockDim.x;
	while (nfill < 24 * 24) {
		int kid = nfill + threadIdx.x;
		i = kid / 24;
		j = kid % 24;
		if (i < 24) {
			KE[i][j] = gTemplateMatrix[i][j];
		}
		nfill += blockDim.x;
	}
	__syncthreads();
}

__device__ void loadNeighborNodesAndFlags(int vid, int v2v[27], bool vfix[27], bool vload[27]) {
	int* pflag = gVflag[0];
	for (int i = 0; i < 27; i++) {
		v2v[i] = gV2V[i][vid];
		if (v2v[i] != -1) {
			int flag = pflag[v2v[i]];
			vfix[i] = flag & grid::Grid::Bitmask::mask_supportnodes;
			vload[i] = flag & grid::Grid::Bitmask::mask_loadnodes;
		}
	}
}

__device__ void loadNeighborNodes(int vid, int v2v[27]) {
	for (int i = 0; i < 27; i++) { v2v[i] = gV2V[i][vid]; }
}

// Spline
__device__ void SplineBasisX(float x, float* pNX)
{
	//float* left = new float[m_iM];
	//float* right = new float[m_iM];
	float left[m_iM], right[m_iM];
	pNX[0] = 1.0;

	int l = (int)((x - gnBoundMin[0]) / gnstep[0]) + m_iM - 1;
	for (int j = 1; j < m_iM; j++)
	{
		//left[j] = x - gpu_ptrfKnotSerX[l + 1 - j];
		//right[j] = gpu_ptrfKnotSerX[l + j] - x;
		left[j] = x - gpu_KnotSer[0][l + 1 - j];
		right[j] = gpu_KnotSer[0][l + j] - x;

		float saved = 0.0;
		for (int r = 0; r < j; r++)
		{
			float temp = pNX[r] / (right[r + 1] + left[j - r]);
			pNX[r] = saved + right[r + 1] * temp;
			saved = left[j - r] * temp;
		}

		pNX[j] = saved;
	}
	pNX[m_iM] = 0.0;
}

__device__ void SplineBasisY(float y, float* pNY)
{
	//float* left = new float[m_iM];
	//float* right = new float[m_iM];
	float left[m_iM], right[m_iM];
	pNY[0] = 1.0;

	int l = (int)((y - gnBoundMin[1]) / gnstep[1]) + m_iM - 1;
	for (int j = 1; j < m_iM; j++)
	{
		//left[j] = y - gpu_ptrfKnotSerY[l + 1 - j];
		//right[j] = gpu_ptrfKnotSerY[l + j] - y;
		left[j] = y - gpu_KnotSer[1][l + 1 - j];
		right[j] = gpu_KnotSer[1][l + j] - y;

		float saved = 0.0;
		for (int r = 0; r < j; r++)
		{
			float temp = pNY[r] / (right[r + 1] + left[j - r]);
			pNY[r] = saved + right[r + 1] * temp;
			saved = left[j - r] * temp;
		}

		pNY[j] = saved;
	}
	pNY[m_iM] = 0.0;
}

__device__ void SplineBasisZ(float z, float* pNZ)
{
	//float* left = new float[m_iM];
	//float* right = new float[m_iM];
	float left[m_iM], right[m_iM];
	pNZ[0] = 1.0;

	int l = (int)((z - gnBoundMin[2]) / gnstep[2]) + m_iM - 1;
	for (int j = 1; j < m_iM; j++)
	{
		//left[j] = z - gpu_ptrfKnotSerZ[l + 1 - j];
		//right[j] = gpu_ptrfKnotSerZ[l + j] - z;
		left[j] = z - gpu_KnotSer[2][l + 1 - j];
		right[j] = gpu_KnotSer[2][l + j] - z;

		float saved = 0.0;
		for (int r = 0; r < j; r++)
		{
			float temp = pNZ[r] / (right[r + 1] + left[j - r]);
			pNZ[r] = saved + right[r + 1] * temp;
			saved = left[j - r] * temp;
		}

		pNZ[j] = saved;
	}
	pNZ[m_iM] = 0.0;
}

/////////////////////////////////////////////////////////////////////////////
// SplineBasisDeriX:
//		calculate the derivative of spline basis of x direction
__device__ void SplineBasisDeriX(float x, const int n, float* value)
{
	int l = (int)((x - gnBoundMin[0]) / gnstep[0]) + m_iM - 1;

	// allocate the array
	int i, j, k, r;
	float ders[10][m_iM] = { {0.f} };
	float test[m_iM] = { 0.0f };
	float ndu[m_iM][m_iM] = { {0.f} };
	float a[2][m_iM] = { {0.f} };

	float left[m_iM], right[m_iM];

	// store functions and knot differences
	ndu[0][0] = 1.0f;
	for (j = 1; j < m_iM; j++)
	{
		//left[j] = x - gpu_ptrfKnotSerX[l + 1 - j];
		//right[j] = gpu_ptrfKnotSerX[l + j] - x;
		left[j] = x - gpu_KnotSer[0][l + 1 - j];
		right[j] = gpu_KnotSer[0][l + j] - x;

		float saved = 0.0f;
		for (r = 0; r < j; r++)
		{
			ndu[j][r] = right[r + 1] + left[j - r];
			float temp = ndu[r][j - 1] / ndu[j][r];
			ndu[r][j] = saved + right[r + 1] * temp;
			saved = left[j - r] * temp;
		}
		ndu[j][j] = saved;
	}

	// load the basis functions
	for (j = 0; j < m_iM; j++)
		ders[0][j] = ndu[j][m_iM - 1];

	// compute the derivatives
	for (r = 0; r < m_iM; r++)
	{
		int s1 = 0, s2 = 1;
		a[0][0] = 1.0f;

		for (k = 1; k < n; k++)
		{
			int j1, j2;
			float d = 0.0f;
			int rk = r - k, pk = m_iM - 1 - k;

			if (r >= k)
			{
				a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
				d = a[s2][0] * ndu[rk][pk];
			}

			if (rk >= -1)
				j1 = 1;
			else
				j1 = -rk;

			if (r - 1 <= pk)
				j2 = k - 1;
			else
				j2 = m_iM - 1 - r;

			for (j = j1; j <= j2; j++)
			{
				a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
				d += a[s2][j] * ndu[rk + j][pk];
			}

			if (r <= pk)
			{
				a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
				d += a[s2][k] * ndu[r][pk];
			}

			ders[k][r] = d;
			j = s1; s1 = s2; s2 = j;
		}
	}

	r = m_iM - 1;
	for (k = 1; k < n; k++)
	{
		for (j = 0; j < m_iM; j++)
			ders[k][j] *= r;
		r *= (m_iM - 1 - k);
	}

	for (i = 0; i < m_iM; i++)
	{
		value[i] = ders[n - 1][i];
	}

}

/////////////////////////////////////////////////////////////////////////////
// SplineBasisDeriY:
//		calculate the derivative of spline basis of y direction
__device__ void SplineBasisDeriY(float y, int n, float* value)
{
	int l = (int)((y - gnBoundMin[1]) / gnstep[1]) + m_iM - 1;

	// allocate the array
	int i, j, k, r;
	float ders[10][m_iM] = { {0.f} };
	float test[m_iM] = { 0.0f };
	float ndu[m_iM][m_iM] = { {0.f} };
	float a[2][m_iM] = { {0.f} };

	float left[m_iM], right[m_iM];

	//float** ndu = new float* [m_iM];
	//float** a = new float* [2];
	//for (i = 0; i < m_iM; i++)
	//{
	//	ndu[i] = new float[m_iM];
	//	if (i < 2)
	//		a[i] = new float[m_iM];
	//}

	//float* left = new float[m_iM];
	//float* right = new float[m_iM];

	// store functions and knot differences
	ndu[0][0] = 1.0f;
	for (j = 1; j < m_iM; j++)
	{
		//left[j] = y - gpu_ptrfKnotSerY[l + 1 - j];
		//right[j] = gpu_ptrfKnotSerY[l + j] - y;
		left[j] = y - gpu_KnotSer[1][l + 1 - j];
		right[j] = gpu_KnotSer[1][l + j] - y;

		float saved = 0.0f;
		for (r = 0; r < j; r++)
		{
			ndu[j][r] = right[r + 1] + left[j - r];
			float temp = ndu[r][j - 1] / ndu[j][r];
			ndu[r][j] = saved + right[r + 1] * temp;
			saved = left[j - r] * temp;
		}
		ndu[j][j] = saved;
	}

	// load the basis functions
	for (j = 0; j < m_iM; j++)
		ders[0][j] = ndu[j][m_iM - 1];

	// compute the derivatives
	for (r = 0; r < m_iM; r++)
	{
		int s1 = 0, s2 = 1;
		a[0][0] = 1.0f;

		for (k = 1; k < n; k++)
		{
			int j1, j2;
			float d = 0.0f;
			int rk = r - k, pk = m_iM - 1 - k;

			if (r >= k)
			{
				a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
				d = a[s2][0] * ndu[rk][pk];
			}

			if (rk >= -1)
				j1 = 1;
			else
				j1 = -rk;

			if (r - 1 <= pk)
				j2 = k - 1;
			else
				j2 = m_iM - 1 - r;

			for (j = j1; j <= j2; j++)
			{
				a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
				d += a[s2][j] * ndu[rk + j][pk];
			}

			if (r <= pk)
			{
				a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
				d += a[s2][k] * ndu[r][pk];
			}

			ders[k][r] = d;
			j = s1; s1 = s2; s2 = j;
		}
	}

	r = m_iM - 1;
	for (k = 1; k < n; k++)
	{
		for (j = 0; j < m_iM; j++)
			ders[k][j] *= r;
		r *= (m_iM - 1 - k);
	}

	for (i = 0; i < m_iM; i++)
	{
		value[i] = ders[n - 1][i];
	}
	//// free the array
	//for (i = 0; i < m_iM; i++)
	//{
	//	delete[] ndu[i];
	//	if (i < 2)
	//		delete[] a[i];
	//}
	//delete[] ndu;
	//delete[] a;

	//delete[] left;
	//delete[] right;
}

/////////////////////////////////////////////////////////////////////////////
// SplineBasisDeriZ:
//		calculate the derivative of spline basis of z direction
__device__ void SplineBasisDeriZ(float z, int n, float* value)
{
	int l = (int)((z - gnBoundMin[2]) / gnstep[2]) + gorder[0] - 1;

	// allocate the array
	int i, j, k, r;
	float ders[10][m_iM] = { {0.f} };
	float test[m_iM] = { 0.0f };
	float ndu[m_iM][m_iM] = { {0.f} };
	float a[2][m_iM] = { {0.f} };

	float left[m_iM], right[m_iM];

	//float** ndu = new float* [m_iM];
	//float** a = new float* [2];
	//for (i = 0; i < m_iM; i++)
	//{
	//	ndu[i] = new float[m_iM];
	//	if (i < 2)
	//		a[i] = new float[m_iM];
	//}

	//float* left = new float[m_iM];
	//float* right = new float[m_iM];

	// store functions and knot differences
	ndu[0][0] = 1.0f;
	for (j = 1; j < m_iM; j++)
	{
		//left[j] = z - gpu_ptrfKnotSerZ[l + 1 - j];
		//right[j] = gpu_ptrfKnotSerZ[l + j] - z;
		left[j] = z - gpu_KnotSer[2][l + 1 - j];
		right[j] = gpu_KnotSer[2][l + j] - z;

		float saved = 0.0f;
		for (r = 0; r < j; r++)
		{
			ndu[j][r] = right[r + 1] + left[j - r];
			float temp = ndu[r][j - 1] / ndu[j][r];
			ndu[r][j] = saved + right[r + 1] * temp;
			saved = left[j - r] * temp;
		}
		ndu[j][j] = saved;
	}

	// load the basis functions
	for (j = 0; j < m_iM; j++)
		ders[0][j] = ndu[j][m_iM - 1];

	// compute the derivatives
	for (r = 0; r < m_iM; r++)
	{
		int s1 = 0, s2 = 1;
		a[0][0] = 1.0f;

		for (k = 1; k < n; k++)
		{
			int j1, j2;
			float d = 0.0f;
			int rk = r - k, pk = m_iM - 1 - k;

			if (r >= k)
			{
				a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
				d = a[s2][0] * ndu[rk][pk];
			}

			if (rk >= -1)
				j1 = 1;
			else
				j1 = -rk;

			if (r - 1 <= pk)
				j2 = k - 1;
			else
				j2 = m_iM - 1 - r;

			for (j = j1; j <= j2; j++)
			{
				a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
				d += a[s2][j] * ndu[rk + j][pk];
			}

			if (r <= pk)
			{
				a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
				d += a[s2][k] * ndu[r][pk];
			}

			ders[k][r] = d;
			j = s1; s1 = s2; s2 = j;
		}
	}

	r = m_iM - 1;
	for (k = 1; k < n; k++)
	{
		for (j = 0; j < m_iM; j++)
			ders[k][j] *= r;
		r *= (m_iM - 1 - k);
	}

	for (i = 0; i < m_iM; i++)
	{
		value[i] = ders[n - 1][i];
	}
	//// free the array
	//for (i = 0; i < m_iM; i++)
	//{
	//	delete[] ndu[i];
	//	if (i < 2)
	//		delete[] a[i];
	//}
	//delete[] ndu;
	//delete[] a;

	//delete[] left;
	//delete[] right;
}


__device__ float norm(float v[3]) {
#ifndef USE_CUDA_FAST_MATH
	return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
#else
	return __fsqrt_rn(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
#endif
}

__device__ float norm(float x, float y, float z) {
#ifndef USE_CUDA_FAST_MATH
	return sqrtf(x * x + y * y + z * z);
#else
	return __fsqrt_rn(x * x + y * y + z * z);
#endif
}

__device__ float dot(float v1[3], float v2[3]) {
	return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
}

/*
	//rxcoarse[32(27)][9][nv]
	rxcoarse[27][9][nv]
*/
template<int BlockSize = 32 * 9>
__global__ void restrict_stencil_dyadic_kernel(int nv_coarse, double* rxcoarse_, int nv_fine, double* rxfine_) {
	size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	int ke_id = tid / nv_coarse;
	int vid = tid % nv_coarse;

	if (ke_id >= 9) return;

	GraftArray<double, 27, 9> rxCoarse(rxcoarse_, nv_coarse);
	GraftArray<double, 27, 9> rxFine(rxfine_, nv_fine);

	//__shared__ double coarseStencil[27][BlockSize / 32][32];
	//initSharedMem(&coarseStencil[0][0][0], sizeof(coarseStencil) / sizeof(double));
	double coarseStencil[27] = { 0. };

	int warpid = threadIdx.x / 32;
	int warptid = threadIdx.x % 32;

	double w[4] = { 1.0,1.0 / 2,1.0 / 4,1.0 / 8 };
	for (int i = 0; i < 27; i++) {
		int neipos[3] = { i % 3 + 1 ,i % 9 / 3 + 1 ,i / 9 + 1 };

		int wneighpos[3] = { abs(neipos[0] - 2),abs(neipos[1] - 2),abs(neipos[2] - 2) };

		if (wneighpos[0] >= 2 || wneighpos[1] >= 2 || wneighpos[2] >= 2) continue;

		double weight = w[wneighpos[0] + wneighpos[1] + wneighpos[2]];

		int vn = gV2Vfine[i][vid];

		if (vn == -1) continue;

		// traverse fine stencil component (each neighbor vertex has a component)
		for (int j = 0; j < 27; j++) {

			double kij = rxFine[j][ke_id][vn] * weight;

			// DEBUG
			if (gVfine2Vfine[j][vn] == -1) { if (kij != 0) { printf("-- error on stencil 1\n"); } continue; }

			int vjpos[3] = { neipos[0] + j % 3 - 1 ,neipos[1] + j % 9 / 3 - 1 ,neipos[2] + j / 9 - 1 };

			// traverse coarse vertices to scatter the stencil component to them
			for (int vsplit = 0; vsplit < 27; vsplit++) {
				int vsplitpos[3] = { vsplit % 3 * 2, vsplit % 9 / 3 * 2, vsplit / 9 * 2 };
				int wsplitpos[3] = { abs(vsplitpos[0] - vjpos[0]), abs(vsplitpos[1] - vjpos[1]), abs(vsplitpos[2] - vjpos[2]) };
				if (wsplitpos[0] >= 2 || wsplitpos[1] >= 2 || wsplitpos[2] >= 2) continue;
				double wsplit = w[wsplitpos[0] + wsplitpos[1] + wsplitpos[2]];
				coarseStencil[vsplit] += wsplit * kij;
			}
		}
	}

	for (int i = 0; i < 27; i++) {
		//rxCoarse[i][ke_id][vid] = coarseStencil[i][warpid][warpid];
		rxCoarse[i][ke_id][vid] = coarseStencil[i];
	}
}

// on the fly assembly
template<int BlockSize = 32 * 9>
__global__ void restrict_stencil_dyadic_OTFA_kernel(int nv_coarse, double* rxcoarse_, int nv_fine, float* rhofine) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//__shared__ int restrict_elements[64];
	__shared__ double KE[24][24];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	int warpid = threadIdx.x / 32;
	int warptid = threadIdx.x % 32;

	int ke_id = tid / nv_coarse;
	int vid = tid % nv_coarse;

	if (ke_id >= 9) return;

	GraftArray<double, 27, 9> rxCoarse(rxcoarse_, nv_coarse);

	//__shared__ double coarseStencil[27][BlockSize / 32][32];
	//initSharedMem(&coarseStencil[0][0][0], sizeof(coarseStencil) / sizeof(double));

	double coarseStencil[27] = { 0. };

	
	//for (int i = 0; i < 27; i++) {
	//	coarseStencil[i][warpid][warptid] = 0;
	//}
	

	// reorder K3 in row major order 
	int k3row = ke_id / 3;
	int k3col = ke_id % 3;

	double w[4] = { 1.0,1.0 / 2,1.0 / 4,1.0 / 8 };
	double kc[27] = { 0. };

	int ebit[2] = { 0 };

	float power = power_penalty[0];

	// traverse neighbor nodes on fine grid
	for (int i = 0; i < 27; i++) {
		int neipos[3] = { i % 3 + 1 ,i % 9 / 3 + 1 ,i / 9 + 1 };

		int vn = gV2Vfine[i][vid];

		if (vn == -1) continue;

		// traverse the neighbor element of each neighbor nodes
		for (int j = 0; j < 8; j++) {
			int epos[3] = { neipos[0] + j % 2 - 1,neipos[1] + j % 4 / 2 - 1,neipos[2] + j / 4 - 1 };
			int eposid = epos[0] + epos[1] * 4 + epos[2] * 16;
			if (read_gbit(ebit, eposid)) continue;
			set_gbit(ebit, eposid);
			float rho_p = 0;
			int eid = gVfine2Efine[j][vn];
			if (eid == -1) continue;
			rho_p = powf(rhofine[eid], power);
			// traverse vertex of neighbor elements (rows of element matrix)
			for (int vi = 0; vi < 8; vi++) {
				int vipos[3] = { epos[0] + vi % 2,epos[1] + vi % 4 / 2,epos[2] + vi / 4 };
				int wipos[3] = { abs(vipos[0] - 2) , abs(vipos[1] - 2) , abs(vipos[2] - 2) };
				if (wipos[0] >= 2 || wipos[1] >= 2 || wipos[2] >= 2) continue;
				int wiid = wipos[0] + wipos[1] + wipos[2];
				if (wiid >= 4) continue;
				double wi_p = w[wiid] * rho_p;

				// traverse another vertex of neighbor element (cols of element matrix), compute Ke 3x3
				for (int vj = 0; vj < 8; vj++) {
					int vjpos[3] = { epos[0] + vj % 2,epos[1] + vj % 4 / 2,epos[2] + vj / 4 };
					double ke = 0;
					double wk = wi_p * KE[vi * 3 + k3row][vj * 3 + k3col];

					// scatter 3x3 Ke to coarse nodes, traverse coarse nodes
					for (int vsplit = 0; vsplit < 27; vsplit++) {
						int vsplitpos[3] = { vsplit % 3 * 2, vsplit % 9 / 3 * 2, vsplit / 9 * 2 };
						int wspos[3] = { abs(vsplitpos[0] - vjpos[0]), abs(vsplitpos[1] - vjpos[1]), abs(vsplitpos[2] - vjpos[2]) };
						if (wspos[0] >= 2 || wspos[1] >= 2 || wspos[2] >= 2) continue;
						int wsid = wspos[0] + wspos[1] + wspos[2];
						double wkw = wk * w[wsid];
						coarseStencil[vsplit] += wkw;
					}
				}
			}
		}

	}

	for (int i = 0; i < 27; i++) {
		//rxCoarse[i][ke_id][vid] = coarseStencil[i][warpid][warptid];
		rxCoarse[i][ke_id][vid] = coarseStencil[i];
	}
}

void HierarchyGrid::restrict_stencil_dyadic(Grid& dstcoarse, Grid& srcfine)
{
	dstcoarse.use_grid();
	size_t grid_size, block_size;
	constexpr int BlockSize = 32 * 6;
	if (dstcoarse._layer == 0 && srcfine._layer == 1) {
		make_kernel_param(&grid_size, &block_size, dstcoarse.n_gsvertices * 9, BlockSize);
		restrict_stencil_dyadic_OTFA_kernel<BlockSize> << <grid_size, block_size >> > (dstcoarse.n_gsvertices, dstcoarse._gbuf.rxStencil, srcfine.n_gsvertices, dstcoarse._gbuf.rho_e);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
		make_kernel_param(&grid_size, &block_size, dstcoarse.n_gsvertices * 9, BlockSize);
		restrict_stencil_dyadic_kernel<BlockSize> << <grid_size, block_size >> > (dstcoarse.n_gsvertices, dstcoarse._gbuf.rxStencil, srcfine.n_gsvertices, srcfine._gbuf.rxStencil);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

// on the fly assembly
template<int BlockSize = 32 * 9>
__global__ void restrict_stencil_nondyadic_OTFA_NS_kernel(int nv_coarse, double* rxcoarse_, int nv_fine, float* rhofine, int* vfineflag) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int warpid = threadIdx.x / 32;
	int warptid = threadIdx.x % 32;


	GraftArray<double, 27, 9> rxCoarse(rxcoarse_, nv_coarse);

	__shared__ double KE[24][24];
	__shared__ double W[4][4][4];
	//__shared__ double coarseStencil[27][BlockSize / 32][32];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	// compute weight
	if (threadIdx.x < 64) {
		int i = threadIdx.x % 4;
		int j = threadIdx.x % 16 / 4;
		int k = threadIdx.x / 16;
		W[k][j][i] = (4 - i)*(4 - j)*(4 - k) / 64.f;
	}
	__syncthreads();
	
	// init coarseStencil
	//initSharedMem(&coarseStencil[0][0][0], sizeof(coarseStencil) / sizeof(double));
	double coarseStencil[27] = { 0. };

	int ke_id = tid / nv_coarse;

	int vid = tid % nv_coarse;

	if (ke_id >= 9) return;

	//int flagword = vcoarseflag[vid];

	//if (flagword & Grid::Bitmask::mask_invalid) return;

	// reorder K3 in row major order
	int k3row = ke_id / 3;
	int k3col = ke_id % 3;

	float power = power_penalty[0];

	// traverse neighbor nodes of fine element center (which is the vertex on fine fine grid)
	for (int i = 0; i < 64; i++) {
		int i2[3] = { (i % 4) * 2 + 1 ,(i % 16 / 4) * 2 + 1 ,(i / 16) * 2 + 1 };
		//int m2 = i2[0] + i2[1] + i2[2] - 3;

		// get fine element center vertex
		int vn = gV2VfineC[i][vid];

		if (vn == -1) continue;

		// should traverse 7x7x7 neigbor nodes, and sum their weighted stencil, to reduce bandwidth, we traverse 8x8x8 elements 
		// traverse the neighbor fine fine element of this vertex and assembly the element matrices
		for (int j = 0; j < 8; j++) {
			int efineid = gVfine2Efine[j][vn];

			if (efineid == -1) continue;

			float rho_p = powf(rhofine[efineid], power);

			int epos[3] = { i2[0] + j % 2 - 1,i2[1] + j % 4 / 2 - 1,i2[2] + j / 4 - 1 };

			// traverse the vertex of neighbor element (rows of element matrix), compute the weight on this vertex 
			for (int ki = 0; ki < 8; ki++) {
				int vipos[3] = { epos[0] + ki % 2,epos[1] + ki % 4 / 2,epos[2] + ki / 4 };
				int wipos[3] = { abs(vipos[0] - 4),abs(vipos[1] - 4),abs(vipos[2] - 4) };
				if (wipos[0] >= 4 || wipos[1] >= 4 || wipos[2] >= 4) continue;
				double w_ki = W[wipos[0]][wipos[1]][wipos[2]] * rho_p;

				// traverse another vertex of neighbor element (cols of element matrix), get the 3x3 Ke and multiply the row weights
				for (int kj = 0; kj < 8; kj++) {
					int kjpos[3] = { epos[0] + kj % 2 , epos[1] + kj % 4 / 2 , epos[2] + kj / 4 };
					double wk = w_ki * KE[ki * 3 + k3row][kj * 3 + k3col];
					//  the weighted element matrix should split to coarse vertex, traverse the coarse vertices and split 3x3 Ke to coarse vertex by splitting weights
					for (int vsplit = 0; vsplit < 27; vsplit++) {
						int vsplitpos[3] = { vsplit % 3 * 4, vsplit % 9 / 3 * 4,vsplit / 9 * 4 };
						int wjpos[3] = { abs(vsplitpos[0] - kjpos[0]), abs(vsplitpos[1] - kjpos[1]), abs(vsplitpos[2] - kjpos[2]) };
						if (wjpos[0] >= 4 || wjpos[1] >= 4 || wjpos[2] >= 4) continue;
						double wkw = wk * W[wjpos[0]][wjpos[1]][wjpos[2]];
						coarseStencil[vsplit]/*[warpid][warptid]*/ += wkw;
					}
				}
			}
		}
	}

	for (int i = 0; i < 27; i++) {
		rxCoarse[i][ke_id][vid] = coarseStencil[i]/*[warpid][warptid]*/;
	}
}

// on the fly assembly
template<int BlockSize = 32 * 9>
__global__ void restrict_stencil_nondyadic_OTFA_WS_kernel(int nv_coarse, double* rxcoarse_, int nv_fine, float* rhofine, int* vfineflag) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int warpid = threadIdx.x / 32;
	int warptid = threadIdx.x % 32;


	GraftArray<double, 27, 9> rxCoarse(rxcoarse_, nv_coarse);

	__shared__ double KE[24][24];
	__shared__ double W[4][4][4];
	//__shared__ double coarseStencil[27][BlockSize / 32][32];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	// compute weight
	if (threadIdx.x < 64) {
		int i = threadIdx.x % 4;
		int j = threadIdx.x % 16 / 4;
		int k = threadIdx.x / 16;
		W[k][j][i] = (4 - i)*(4 - j)*(4 - k) / 64.f;
	}
	__syncthreads();
	
	// init coarseStencil
	//initSharedMem(&coarseStencil[0][0][0], sizeof(coarseStencil) / sizeof(double));
	double coarseStencil[27] = { 0. };

	int ke_id = tid / nv_coarse;

	int vid = tid % nv_coarse;

	if (ke_id >= 9) return;

	//int flagword = vcoarseflag[vid];

	//if (flagword & Grid::Bitmask::mask_invalid) return;

	// reorder K3 in row major order
	int k3row = ke_id / 3;
	int k3col = ke_id % 3;

	float power = power_penalty[0];

	// traverse neighbor nodes of fine element center (which is the vertex on fine fine grid)
	for (int i = 0; i < 64; i++) {
		int i2[3] = { (i % 4) * 2 + 1 ,(i % 16 / 4) * 2 + 1 ,(i / 16) * 2 + 1 };
		//int m2 = i2[0] + i2[1] + i2[2] - 3;

		// get fine element center vertex
		int vn = gV2VfineC[i][vid];

		if (vn == -1) continue;

		// should traverse 7x7x7 neigbor nodes, and sum their weighted stencil, to reduce bandwidth, we traverse 8x8x8 elements 
		// traverse the neighbor fine fine element of this vertex and assembly the element matrices
		for (int j = 0; j < 8; j++) {
			int efineid = gVfine2Efine[j][vn];

			if (efineid == -1) continue;

			float rho_p = powf(rhofine[efineid], power);

			int epos[3] = { i2[0] + j % 2 - 1,i2[1] + j % 4 / 2 - 1,i2[2] + j / 4 - 1 };

			// prefecth the flag of eight vertex
			bool vfix[8];
			for (int k = 0; k < 8; k++) {
				int vklid = j % 2 + k % 2 +
					(j / 2 % 2 + k / 2 % 2) * 3 +
					(j / 4 + k / 4) * 9;
				int vkvid = gVfine2Vfine[vklid][vn];
				if (vkvid == -1)printf("-- error in stencil restriction\n");
				int vkflag = vfineflag[vkvid];
				vfix[k] = vkflag & Grid::Bitmask::mask_supportnodes;
			}

			// traverse the vertex of neighbor element (rows of element matrix), compute the weight on this vertex 
			for (int ki = 0; ki < 8; ki++) {
				int vipos[3] = { epos[0] + ki % 2,epos[1] + ki % 4 / 2,epos[2] + ki / 4 };
				int wipos[3] = { abs(vipos[0] - 4),abs(vipos[1] - 4),abs(vipos[2] - 4) };
				if (wipos[0] >= 4 || wipos[1] >= 4 || wipos[2] >= 4) continue;
				double wi = W[wipos[0]][wipos[1]][wipos[2]];
				double w_ki = wi * rho_p;

				// traverse another vertex of neighbor element (cols of element matrix), get the 3x3 Ke and multiply the row weights
				for (int kj = 0; kj < 8; kj++) {
					int kjpos[3] = { epos[0] + kj % 2 , epos[1] + kj % 4 / 2 , epos[2] + kj / 4 };
					double wk = w_ki * KE[ki * 3 + k3row][kj * 3 + k3col];
									
					if (vfix[kj] || vfix[ki]) {
						wk = 0;
						if (ki == kj && k3row == k3col) {
							wk = wi * DIRICHLET_DIAGONAL_WEIGHT;
						}
					}

					//  the weighted element matrix should split to coarse vertex, traverse the coarse vertices and split 3x3 Ke to coarse vertex by splitting weights
					for (int vsplit = 0; vsplit < 27; vsplit++) {
						int vsplitpos[3] = { vsplit % 3 * 4, vsplit % 9 / 3 * 4,vsplit / 9 * 4 };
						int wjpos[3] = { abs(vsplitpos[0] - kjpos[0]), abs(vsplitpos[1] - kjpos[1]), abs(vsplitpos[2] - kjpos[2]) };
						if (wjpos[0] >= 4 || wjpos[1] >= 4 || wjpos[2] >= 4) continue;
						double wkw = wk * W[wjpos[0]][wjpos[1]][wjpos[2]];
						coarseStencil[vsplit]/*[warpid][warptid]*/ += wkw;
					}
				}
			}
		}
	}

	for (int i = 0; i < 27; i++) {
		rxCoarse[i][ke_id][vid] = coarseStencil[i]/*[warpid][warptid]*/;
	}
}

void HierarchyGrid::restrict_stencil_nondyadic(Grid& dstcoarse, Grid& srcfine)
{
	if (dstcoarse._layer != 2 || srcfine._layer != 0) {
		std::cout << "\033[31m" << "Non dyadic restriction is only applied on finest grid" << "\033[0m" << std::endl;
	}

	dstcoarse.use_grid();

	constexpr int BlockSize = 32 * 4;
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, dstcoarse.n_gsvertices * 9, BlockSize);
	if (_mode == no_support_constrain_force_direction || _mode == no_support_free_force) {
		restrict_stencil_nondyadic_OTFA_NS_kernel<BlockSize> << <grid_size, block_size >> > (dstcoarse.n_gsvertices, dstcoarse._gbuf.rxStencil, srcfine.n_gsvertices, srcfine._gbuf.rho_e, srcfine._gbuf.vBitflag);
	}
	else if (_mode == with_support_constrain_force_direction || _mode == with_support_free_force) {
		restrict_stencil_nondyadic_OTFA_WS_kernel<BlockSize> << <grid_size, block_size >> > (dstcoarse.n_gsvertices, dstcoarse._gbuf.rxStencil, srcfine.n_gsvertices, srcfine._gbuf.rho_e, srcfine._gbuf.vBitflag);
	}
	cudaDeviceSynchronize();
	cuda_error_check;
}

void HierarchyGrid::restrict_stencil(Grid& dstcoarse, Grid& srcfine)
{
	if (dstcoarse.is_dummy()) return;
	if (dstcoarse._layer == 0) return;

	init_array(dstcoarse._gbuf.rxStencil, double{ 0 }, 27 * 9 * dstcoarse.n_gsvertices);

	if (_setting.skiplayer1 && dstcoarse._layer == 2 && srcfine._layer == 0) {
		restrict_stencil_nondyadic(dstcoarse, srcfine);
	}
	else {
		if (dstcoarse._layer - srcfine._layer != 1) {
			printf("\033[31mOnly Support stencil restriction between neighbor layers!\033[0m\n");
			throw std::runtime_error("");
		}
		restrict_stencil_dyadic(dstcoarse, srcfine);
	}
}

void Grid::compute_gscolor(gpu_manager_t& gm, BitSAT<unsigned int>& vbit, BitSAT<unsigned int>& ebit, int vreso, int* vbitflaghost, int* ebitflaghost)
{
	int nv = vbit.total();
	int ne = ebit.total();
	int* vbitflagdevice = nullptr;
	int* ebitflagdevice = nullptr;
	int nvword = vbit._bitArray.size();
	int neword = ebit._bitArray.size();

	// build device SAT 
	gBitSAT<unsigned int> gvsat(vbit._bitArray, vbit._chunkSat);
	gBitSAT<unsigned int> gesat(ebit._bitArray, ebit._chunkSat);

	// copy bit flag to device 
	cudaMalloc(&vbitflagdevice, nv * sizeof(int));
	cudaMalloc(&ebitflagdevice, ne * sizeof(int));
	cudaMemcpy(vbitflagdevice, vbitflaghost, sizeof(int) * nv, cudaMemcpyHostToDevice);
	cudaMemcpy(ebitflagdevice, ebitflaghost, sizeof(int) * ne, cudaMemcpyHostToDevice);

	auto vkernel = [=] __device__(int tid) {
		// set vertex gs color  
		unsigned int word = gvsat._bitarray[tid];
		int vid = gvsat._chunksat[tid];
		int vreso2 = vreso * vreso;
		int nvbit = vreso2 * vreso;
		if (word != 0) {
			for (int ji = 0; ji < sizeof(unsigned int) * 8; ji++) {
				if (!read_gbit(word, ji)) continue;
				int vbitid = tid * BitCount<unsigned int>::value + ji;
				if (vbitid >= nvbit) break;
				int pos[3] = { vbitid % vreso, (vbitid % vreso2) / vreso, vbitid / vreso2 };
				int m2 = pos[0] % 2 + pos[1] % 2 * 2 + pos[2] % 2 * 4;
				// set vertex gs color id
				int bitword = vbitflagdevice[vid];
				bitword &= ~(int)Bitmask::mask_gscolor;
				bitword |= (m2 << Bitmask::offset_gscolor);
				vbitflagdevice[vid] = bitword;
				vid++;
			}
		}
		
		// set element gs color
		if (tid >= neword) return;
		word = gesat._bitarray[tid];
		if (word == 0) return;
		int eid = gesat._chunksat[tid];
		int ereso = vreso - 1;
		int ereso2 = ereso * ereso;
		int nebit = ereso * ereso2;
		for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
			if (!read_gbit(word, ji)) continue;
			int ebitid = tid * BitCount<unsigned int>::value + ji;
			if (ebitid >= nebit) break;
			int pos[3] = { ebitid % ereso, (ebitid % ereso2) / ereso, ebitid / ereso2 };
			int m2 = pos[0] % 2 + pos[1] % 2 * 2 + pos[2] % 2 * 4;
			int bitword = ebitflagdevice[eid];
			bitword &= ~(int)Bitmask::mask_gscolor;
			bitword |= (m2 << Bitmask::offset_gscolor);
			ebitflagdevice[eid] = bitword;
			eid++;
		}
	};
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, vbit._bitArray.size(), 512);
	traverse_noret << <grid_size, block_size >> > (vbit._bitArray.size(), vkernel);
	cudaDeviceSynchronize();
	cuda_error_check;

	cudaMemcpy(vbitflaghost, vbitflagdevice, sizeof(int) * nv, cudaMemcpyDeviceToHost);
	cudaMemcpy(ebitflaghost, ebitflagdevice, sizeof(int) * ne, cudaMemcpyDeviceToHost);

	gvsat.destroy();
	gesat.destroy();
	cudaFree(vbitflagdevice);
	cudaFree(ebitflagdevice);
}

void* Grid::getTempBuf(size_t requre)
{
	size_t req_size = snippet::Round<512>(requre);
	if (_tmp_buf == nullptr) {
		cudaMalloc(&_tmp_buf, req_size);
		cuda_error_check;
		_tmp_buf_size = req_size;
	}
	if (_tmp_buf_size < req_size) {
		cudaFree(_tmp_buf);
		cuda_error_check;
		_tmp_buf_size = snippet::Round<512>(req_size);
		cudaMalloc(&_tmp_buf, req_size);
		cuda_error_check;
	}
	return _tmp_buf;
}

void Grid::clearBuf(void)
{
	if (_tmp_buf != nullptr)
	{
		cudaFree(_tmp_buf);
		cuda_error_check;
		_tmp_buf = nullptr;
	}
}

void* Grid::getTempBuf1(size_t requre)
{
	size_t req_size = snippet::Round<512>(requre);
	if (_tmp_buf1 == nullptr) {
		cudaMalloc(&_tmp_buf1, req_size);
		_tmp_buf1_size = req_size;
	}
	if (_tmp_buf1_size < req_size) {
		cudaFree(_tmp_buf1);
		_tmp_buf1_size = snippet::Round<512>(req_size);
		cudaMalloc(&_tmp_buf1, req_size);
	}
	return _tmp_buf1;
}

void Grid::clearBuf1(void)
{
	if (_tmp_buf1 != nullptr)
	{
		cudaFree(_tmp_buf1);
		cuda_error_check;
		_tmp_buf1 = nullptr;
	}
}

void* Grid::getTempBuf2(size_t requre)
{
	size_t req_size = snippet::Round<512>(requre);
	if (_tmp_buf2 == nullptr) {
		cudaMalloc(&_tmp_buf2, req_size);
		_tmp_buf2_size = req_size;
	}
	if (_tmp_buf2_size < req_size) {
		cudaFree(_tmp_buf2);
		_tmp_buf2_size = snippet::Round<512>(req_size);
		cudaMalloc(&_tmp_buf2, req_size);
	}
	return _tmp_buf2;
}

void Grid::clearBuf2(void)
{
	if (_tmp_buf2 != nullptr)
	{
		cudaFree(_tmp_buf2);
		cuda_error_check;
		_tmp_buf2 = nullptr;
	}
}

void Grid::lexico2gsorder_g(int* idmap, int n_id, int* ids, int n_mapid, int* mapped_ids, int* valuemap /*= nullptr*/)
{
	int* pid = ids;
	int* old_ptr;
	if (ids == mapped_ids) {
		old_ptr = (int*)getTempBuf(sizeof(int)*n_id);
		cudaMemcpy(old_ptr, ids, sizeof(int) * n_id, cudaMemcpyDeviceToDevice);
		pid = old_ptr;
	}
	init_array(mapped_ids, -1, n_mapid);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_id, 512);
	auto permut = [=] __device__(int tid) {
		int newvalue = pid[tid];
		if (valuemap != nullptr) {
			if (newvalue != -1) {
				newvalue = valuemap[newvalue];
			}
		}
		if (idmap != nullptr) {
			mapped_ids[idmap[tid]] = newvalue;
		}
		else {
			mapped_ids[tid] = newvalue;
		}
	};
	traverse_noret << <grid_size, block_size >> > (n_id, permut);
	cudaDeviceSynchronize();
	cuda_error_check;
	
}

template<int BlockSize = 32 * 13>
__global__ void gs_relax_kernel(int n_vgstotal, int nv_gsset, double* rxstencil, int gs_offset) {
	GraftArray<double, 27, 9> stencil(rxstencil, n_vgstotal);
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//int mode = gmode[0];

	__shared__ double sumAu[3][13][32];
	int warpId = threadIdx.x / 32;
	int node_id_in_block = threadIdx.x % 32;
	int workId = node_id_in_block;
	int gs_vertex_id = blockIdx.x * 32 + node_id_in_block;

	int offset = gs_offset;

	double Au[3] = { 0.f,0.f,0.f };

	int node_id;

	int flag;

	bool invalid_node = true;

	if (gs_vertex_id < nv_gsset) {
		node_id = offset + gs_vertex_id;
		flag = gVflag[0][node_id];
		invalid_node = flag & Grid::Bitmask::mask_invalid;
		if (invalid_node) goto _blockSum;
		for (auto i : { 0,14 }) {
			double displacement[3];
			int neigh_th = warpId + i;
			int neigh = gV2V[neigh_th][node_id];	
			if (neigh == -1) continue;

			for (int j = 0; j < 3; j++) displacement[j] = gU[j][neigh];

			// K3 is ordered in row major 
			// traverse rows 
			for (int j = 0; j < 3; j++) {
				int jrows = j * 3;
				// traverse columns, dot u 
				for (int k = 0; k < 3; k++) {
					Au[j] += stencil[neigh_th][jrows + k][node_id] * displacement[k];
				}
			}

		}
	}

_blockSum:

	for (int i = 0; i < 3; i++) {
		sumAu[i][warpId][node_id_in_block] = Au[i];
	}
	__syncthreads();

	// gather all part
	if (warpId < 7) {
		for (int i = 0; i < 3; i++) {
			int addId = warpId + 7;
			if (addId < 13) {
				sumAu[i][warpId][node_id_in_block] += sumAu[i][addId][node_id_in_block];
			}
		}
	}
	__syncthreads();
	if (warpId < 4) {
		for (int i = 0; i < 3; i++) {
			int addId = warpId + 4;
			if (addId < 7) {
				sumAu[i][warpId][node_id_in_block] += sumAu[i][addId][node_id_in_block];
			}
		}
	}
	__syncthreads();
	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			int addId = warpId + 2;
			sumAu[i][warpId][node_id_in_block] += sumAu[i][addId][node_id_in_block];
		}
	}
	__syncthreads();
	if (warpId < 1) {
		for (int i = 0; i < 3; i++) {
			int addId = warpId + 1;
			Au[i] = sumAu[i][warpId][node_id_in_block] + sumAu[i][addId][node_id_in_block];
		}
	}
	//__syncthreads();

	if (gs_vertex_id < nv_gsset && !invalid_node) {
		double node_sum = 0;

		double displacement[3] = { 0. }; int	rowOffset = 0;

		if (warpId == 0) {
			for (int i = 0; i < 3; i++) displacement[i] = gU[i][node_id];
			node_sum = stencil[13][rowOffset + 1][node_id] * displacement[1] + stencil[13][rowOffset + 2][node_id] * displacement[2];
			displacement[0] = (gF[0][node_id] - Au[0] - node_sum) / stencil[13][0][node_id];
			gU[0][node_id] = displacement[0];

			rowOffset += 3;
			node_sum = stencil[13][rowOffset + 0][node_id] * displacement[0] + stencil[13][rowOffset + 2][node_id] * displacement[2];
			displacement[1] = (gF[1][node_id] - Au[1] - node_sum) / stencil[13][rowOffset + 1][node_id];
			gU[1][node_id] = displacement[1];

			rowOffset += 3;
			node_sum = stencil[13][rowOffset + 0][node_id] * displacement[0] + stencil[13][rowOffset + 1][node_id] * displacement[1];
			displacement[2] = (gF[2][node_id] - Au[2] - node_sum) / stencil[13][rowOffset + 2][node_id];
			gU[2][node_id] = displacement[2];
		}
	}
}

// map 32 vertices to 8 warp, each warp use specific neighbor element (density rho_i)
template<int BlockSize = 32 * 8>
__global__ void gs_relax_OTFA_NS_kernel(int nv_gs, int gs_offset, float* rholist) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//int mode = gmode[0];

	__shared__ double KE[24][24];

	__shared__ double sumKeU[3][4][32];

	__shared__ double sumS[9][4][32];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	int warpId = threadIdx.x / 32;
	int warpTid = threadIdx.x % 32;

	double KeU[3] = { 0. };
	double S[9] = { 0. };
	double* pU[3] = { gU[0],gU[1],gU[2] };

	bool invalid_node = false;
	// the id in a gs subset
	int vid = blockIdx.x * 32 + warpTid;

	// the id in total node set
	vid += gs_offset;

	int vi = 7 - warpId;
	double penalty = 0;
	int eid;

	int flag = gVflag[0][vid];
	invalid_node |= flag & Grid::Bitmask::mask_invalid;

	if (invalid_node) goto _blocksum;

	eid = gV2E[warpId][vid];

	if (eid != -1)
		penalty = powf(rholist[eid], power_penalty[0]);
	else
		goto _blocksum;

	if (gV2V[13][vid] == -1) {
		invalid_node = true;
		goto _blocksum;
	}

	// compute KU and S 
	for (int vj = 0; vj < 8; vj++) {
		// vjpos = epos + vjoffset
		int vjpos[3] = {
			vj % 2 + warpId % 2,
			vj % 4 / 2 + warpId % 4 / 2,
			vj / 4 + warpId / 4
		};
		int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
		int vj_vid = gV2V[vj_lid][vid];
		if (vj_vid == -1) continue;
		double U[3] = { pU[0][vj_vid],pU[1][vj_vid],pU[2][vj_vid] };
		if (vj_lid != 13) {
			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < 3; j++) {
					KeU[k] += penalty * KE[k + vi * 3][j + vj * 3] * U[j];
				}
			}
		}
		if (vj_lid == 13) {
			for (int i = 0; i < 9; i++) {
				S[i] = penalty * KE[vi * 3 + i / 3][vi * 3 + i % 3];
			}
		}
	}

_blocksum:

	if (warpId >= 4) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId - 4][warpTid] = KeU[i];
		}
		for (int i = 0; i < 9; i++) {
			sumS[i][warpId - 4][warpTid] = S[i];
		}
	}
	__syncthreads();

	if (warpId < 4) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId][warpTid] += KeU[i];
		}
		for (int i = 0; i < 9; i++) {
			sumS[i][warpId][warpTid] += S[i];
		}
	}
	__syncthreads();

	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId][warpTid] += sumKeU[i][warpId + 2][warpTid];
		}
		for (int i = 0; i < 9; i++) {
			sumS[i][warpId][warpTid] += sumS[i][warpId + 2][warpTid];
		}
	}
	__syncthreads();

	if (warpId < 1 && !invalid_node) {
		for (int i = 0; i < 3; i++) {
			KeU[i] = sumKeU[i][0][warpTid] + sumKeU[i][1][warpTid];
		}
		for (int i = 0; i < 9; i++) {
			S[i] = sumS[i][0][warpTid] + sumS[i][1][warpTid];
		}

		double newU[3] = { pU[0][vid],pU[1][vid],pU[2][vid] };
		double(*s)[3] = reinterpret_cast<double(*)[3]>(S);
		// s[][] is row major 
		newU[0] = (gF[0][vid] - s[0][1] * newU[1] - s[0][2] * newU[2] - KeU[0]) / s[0][0];
		newU[1] = (gF[1][vid] - s[1][0] * newU[0] - s[1][2] * newU[2] - KeU[1]) / s[1][1];
		newU[2] = (gF[2][vid] - s[2][0] * newU[0] - s[2][1] * newU[1] - KeU[2]) / s[2][2];
		pU[0][vid] = newU[0]; pU[1][vid] = newU[1]; pU[2][vid] = newU[2];

	}


}

// map 32 vertices to 8 warp, each warp use specific neighbor element (density rho_i)
template<int BlockSize = 32 * 8>
__global__ void gs_relax_OTFA_WS_kernel(int nv_gs, int gs_offset, float* rholist) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ double KE[24][24];

	__shared__ double sumKeU[3][4][32];

	__shared__ double sumS[9][4][32];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	int warpId = threadIdx.x / 32;
	int warpTid = threadIdx.x % 32;

	double KeU[3] = { 0. };
	double S[9] = { 0. };
	double* pU[3] = { gU[0],gU[1],gU[2] };

	bool invalid_node = false;
	// the id in a gs subset
	int vid = blockIdx.x * 32 + warpTid;

	// the id in total node set
	vid += gs_offset;

	int flag = gVflag[0][vid];
	int eid;
	double penalty = 0;
	int vi = 7 - warpId;
	bool viisfix;
	int* pflags;

	invalid_node |= flag & Grid::Bitmask::mask_invalid;
	if (invalid_node) goto _blocksum;

	eid = gV2E[warpId][vid];

	if (eid != -1)
		penalty = powf(rholist[eid], power_penalty[0]);
	else
		goto _blocksum;

	if (gV2V[13][vid] == -1) {
		invalid_node = true;
		goto _blocksum;
	}

	viisfix = flag & grid::Grid::Bitmask::mask_supportnodes;

	pflags = gVflag[0];

	// compute KU and S 
	for (int vj = 0; vj < 8; vj++) {
		// vjpos = epos + vjoffset
		int vjpos[3] = {
			vj % 2 + warpId % 2,
			vj % 4 / 2 + warpId % 4 / 2,
			vj / 4 + warpId / 4
		};
		int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
		int vj_vid = gV2V[vj_lid][vid];
		if (vj_vid == -1) continue;
		double U[3] = { pU[0][vj_vid],pU[1][vj_vid],pU[2][vj_vid] };

		// deal with fixed boundary
		int vjflag = pflags[vj_vid];
		bool vjisfix = vjflag & grid::Grid::Bitmask::mask_supportnodes;

		if (vj_lid != 13 && !vjisfix) {
			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < 3; j++) {
					KeU[k] += penalty * KE[k + vi * 3][j + vj * 3] * U[j];
				}
			}
		}
		if (vj_lid == 13) {
			if (!vjisfix) {
				for (int i = 0; i < 9; i++) {
					S[i] = penalty * KE[vi * 3 + i / 3][vi * 3 + i % 3];
				}
			}
			else {
				S[0] = 1; S[4] = 1; S[8] = 1;
			}
		}
	}

	if (viisfix) {
		KeU[0] = 0; KeU[1] = 0; KeU[2] = 0;
	}

_blocksum:

	if (warpId >= 4) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId - 4][warpTid] = KeU[i];
		}
		for (int i = 0; i < 9; i++) {
			sumS[i][warpId - 4][warpTid] = S[i];
		}
	}
	__syncthreads();

	if (warpId < 4) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId][warpTid] += KeU[i];
		}
		for (int i = 0; i < 9; i++) {
			sumS[i][warpId][warpTid] += S[i];
		}
	}
	__syncthreads();

	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId][warpTid] += sumKeU[i][warpId + 2][warpTid];
		}
		for (int i = 0; i < 9; i++) {
			sumS[i][warpId][warpTid] += sumS[i][warpId + 2][warpTid];
		}
	}
	__syncthreads();

	if (warpId < 1 && !invalid_node) {
		for (int i = 0; i < 3; i++) {
			KeU[i] = sumKeU[i][0][warpTid] + sumKeU[i][1][warpTid];
		}
		for (int i = 0; i < 9; i++) {
			S[i] = sumS[i][0][warpTid] + sumS[i][1][warpTid];
		}

		double newU[3] = { pU[0][vid],pU[1][vid],pU[2][vid] };
		double(*s)[3] = reinterpret_cast<double(*)[3]>(S);
		// s[][] is row major 
		newU[0] = (gF[0][vid] - s[0][1] * newU[1] - s[0][2] * newU[2] - KeU[0]) / s[0][0];
		newU[1] = (gF[1][vid] - s[1][0] * newU[0] - s[1][2] * newU[2] - KeU[1]) / s[1][1];
		newU[2] = (gF[2][vid] - s[2][0] * newU[0] - s[2][1] * newU[1] - KeU[2]) / s[2][2];
		pU[0][vid] = newU[0]; pU[1][vid] = newU[1]; pU[2][vid] = newU[2];

	}

}

void Grid::gs_relax(int n_times)
{
	if (is_dummy()) return;
	use_grid();
	cuda_error_check;
	if (_layer == 0) {
		for (int n = 0; n < n_times; n++) {
			int gs_offset = 0;
			for (int i = 0; i < 8; i++) {
				constexpr int BlockSize = 32 * 8;
				size_t grid_size, block_size;
				make_kernel_param(&grid_size, &block_size, gs_num[i] * 8, BlockSize);
				if (_mode == no_support_constrain_force_direction || _mode == no_support_free_force) {
					gs_relax_OTFA_NS_kernel<BlockSize> << <grid_size, block_size >> > (gs_num[i], gs_offset, _gbuf.rho_e);
				}
				else if (_mode == with_support_constrain_force_direction || _mode == with_support_free_force) {
					gs_relax_OTFA_WS_kernel<BlockSize> << <grid_size, block_size >> > (gs_num[i], gs_offset, _gbuf.rho_e);
				}
				//cudaDeviceSynchronize();
				//cuda_error_check;
				gs_offset += gs_num[i];
			}
			cudaDeviceSynchronize();
			cuda_error_check;
		}
	}
	else {
		check_array_len(_gbuf.rxStencil, 27 * 9 * n_gsvertices);
		for (int n = 0; n < n_times; n++) {
			int gs_offset = 0;
			for (int i = 0; i < 8; i++) {
				size_t grid_size, block_size;
				constexpr int BlockSize = 32 * 13;
				make_kernel_param(&grid_size, &block_size, gs_num[i] * 13, BlockSize);
				gs_relax_kernel<BlockSize> << <grid_size, block_size >> > (n_gsvertices, gs_num[i], _gbuf.rxStencil, gs_offset);
				//cudaDeviceSynchronize();
				//cuda_error_check;
				gs_offset += gs_num[i];
			}
			cudaDeviceSynchronize();
			cuda_error_check;
		}
	}
}

// map 1 vertices to 32 threads(1 warp), 4 vertices in 1 block
template<int BlockSize = 32 * 4 >
__global__ void restrict_adjoint_stencil_nondyadic_OTFA_constrain_force_direction_kernel_2(
	int nv_coarse, double* rxcoarse_, int nv_fine, float* rhofine, int* vfineflag, gBitSAT<unsigned int> vloadsat
) {
	size_t tid = size_t(blockDim.x) * blockIdx.x + threadIdx.x;
	int warpid = threadIdx.x / 32;
	int warptid = threadIdx.x % 32;

	GraftArray<double, 27, 9> rxCoarse(rxcoarse_, nv_coarse);

	__shared__ double KE[24][24];
	__shared__ double W[4][4][4];

	__shared__ double sumCoarseStencil[BlockSize / 32][27][32];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	// compute weight
	if (threadIdx.x < 64) {
		int i = threadIdx.x % 4;
		int j = threadIdx.x % 16 / 4;
		int k = threadIdx.x / 16;
		W[k][j][i] = (4 - i)*(4 - j)*(4 - k) / 64.f;
	}
	__syncthreads();
	
	// init coarseStencil
	//initSharedMem(&coarseStencil[0][0][0], sizeof(coarseStencil) / sizeof(double));
	double coarseStencil[27] = { 0. };

	bool validthread = true;

	int ke_id = (blockIdx.x * (BlockSize / 32) + warpid) / nv_coarse;

	// reorder K3 in row major order
	int k3row = ke_id / 3;
	int k3col = ke_id % 3;

	int vid = (blockIdx.x * (BlockSize / 32) + warpid) % nv_coarse;

	int flagword;
	float power;
	double* gvtan[2][3];

	if (ke_id >= 9) {
		validthread = false;
		goto __blocksum;
	}

	flagword = gVflag[0][vid];

	if (flagword & Grid::Bitmask::mask_invalid) {
		validthread = false;
		goto __blocksum;
	}

	power = power_penalty[0];

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++) gvtan[i][j] = gLoadtangent[i][j];


	// traverse neighbor nodes of fine element center (which is the vertex on fine fine grid)
	for (int ibase : {0, 32}) {
		int i = ibase + warptid;
		int i2[3] = { (i % 4) * 2 + 1 ,(i % 16 / 4) * 2 + 1 ,(i / 16) * 2 + 1 };
		//int m2 = i2[0] + i2[1] + i2[2] - 3;

		// get fine element center vertex
		int vn = gV2VfineC[i][vid];

		if (vn == -1) continue;

		// should traverse 7x7x7 neigbor nodes, and sum their weighted stencil, to reduce bandwidth, we traverse 8x8x8 elements 
		// traverse the neighbor fine fine element of this vertex and assembly the element matrices
		for (int j = 0; j < 8; j++) {
			int efineid = gVfine2Efine[j][vn];

			if (efineid == -1) continue;

			float rho_p = powf(rhofine[efineid], power);

			int epos[3] = { i2[0] + j % 2 - 1,i2[1] + j % 4 / 2 - 1,i2[2] + j / 4 - 1 };

			// prefecth the flag of eight vertex
			int vload[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
			for (int k = 0; k < 8; k++) {
				int vklid = j % 2 + k % 2 +
					(j / 2 % 2 + k / 2 % 2) * 3 +
					(j / 4 + k / 4) * 9;
				int vkvid = gVfine2Vfine[vklid][vn];
				int vkflag = vfineflag[vkvid];
				if (vkflag & Grid::Bitmask::mask_loadnodes) {
					vload[k] = vloadsat(vkvid);
				}
			}

			// traverse the vertex of neighbor element (rows of element matrix), compute the weight on this vertex 
			for (int ki = 0; ki < 8; ki++) {
				int vipos[3] = { epos[0] + ki % 2,epos[1] + ki % 4 / 2,epos[2] + ki / 4 };
				int wipos[3] = { abs(vipos[0] - 4),abs(vipos[1] - 4),abs(vipos[2] - 4) };
				if (wipos[0] >= 4 || wipos[1] >= 4 || wipos[2] >= 4) continue;
				double w_ki = W[wipos[0]][wipos[1]][wipos[2]] * rho_p;
				//double w_ki = (4 - wipos[0]) * (4 - wipos[1]) * (4 - wipos[2]) / 64.0;

				// fetch vi tangent vector if vi is load node
				double n1[3] = { 0. };
				n1[k3row] = 1;
				if (vload[ki] != -1) {
					n1[k3row] = 0;
					if (k3row < 2) {
						for (int m = 0; m < 3; m++) n1[m] = gvtan[k3row][m][vload[ki]];
					}
				}

				// traverse another vertex of neighbor element (cols of element matrix), get the 3x3 Ke and multiply the row weights
				for (int kj = 0; kj < 8; kj++) {
					int kjpos[3] = { epos[0] + kj % 2 , epos[1] + kj % 4 / 2 , epos[2] + kj / 4 };
					double wk = 0;

					double n2[3] = { 0. };
					n2[k3col] = 1;
					// check whether vi and vj are load nodes, multiply N if they are 
					if (vload[kj] != -1) {
						n2[k3col] = 0;
						if (k3col < 2) {
							for (int m = 0; m < 3; m++) n2[m] = gvtan[k3col][m][vload[kj]];
						}
					}

					// compute N * K * N^T
					for (int m = 0; m < 3; m++) {
						for (int n = 0; n < 3; n++) {
							wk += n1[m] * KE[ki * 3 + m][kj * 3 + n] * n2[n];
						}
					}

					// set degenerated diagonal as 1 
					if (ki == kj && vload[ki] != -1 && k3row == 2 && k3col == 2) {
						wk = 1;
					}

					// multiply weight on ki
					wk *= w_ki;

				_splitwk:
					//  the weighted element matrix should split to coarse vertex, traverse the coarse vertices and split 3x3 Ke to coarse vertex by splitting weights
					for (int vsplit = 0; vsplit < 27; vsplit++) {
						int vsplitpos[3] = { vsplit % 3 * 4, vsplit % 9 / 3 * 4,vsplit / 9 * 4 };
						int wjpos[3] = { abs(vsplitpos[0] - kjpos[0]), abs(vsplitpos[1] - kjpos[1]), abs(vsplitpos[2] - kjpos[2]) };
						if (wjpos[0] >= 4 || wjpos[1] >= 4 || wjpos[2] >= 4) continue;
						double wkw = wk * W[wjpos[0]][wjpos[1]][wjpos[2]];
						//double wkw = wk * (4 - wjpos[0]) * (4 - wjpos[1]) * (4 - wjpos[2]) / 64.;
						coarseStencil[vsplit]/*[warpid][warptid]*/ += wkw;
					}
				}
			}
		}
	}

__blocksum:

	for (int i = 0; i < 27; i++) {
		sumCoarseStencil[warpid][i][warptid] = coarseStencil[i];
	}

	__syncthreads();

#if 1
	// warp reduce sum on sumCoarseStencil[][][*]
	if (warptid < 16) {
		for (int i = 0; i < 27; i++) {
			sumCoarseStencil[warpid][i][warptid] += sumCoarseStencil[warpid][i][warptid + 16];
		}
	}
	if (warptid < 8) {
		for (int i = 0; i < 27; i++) {
			sumCoarseStencil[warpid][i][warptid] += sumCoarseStencil[warpid][i][warptid + 8];
		}
	}
	if (warptid < 4) {
		for (int i = 0; i < 27; i++) {
			sumCoarseStencil[warpid][i][warptid] += sumCoarseStencil[warpid][i][warptid + 4];
		}
	}
	if (warptid < 2) {
		for (int i = 0; i < 27; i++) {
			sumCoarseStencil[warpid][i][warptid] += sumCoarseStencil[warpid][i][warptid + 2];
		}
	}
	if (warptid < 1 && validthread) {
		for (int i = 0; i < 27; i++) {
			rxCoarse[i][ke_id][vid] = sumCoarseStencil[warpid][i][0] + sumCoarseStencil[warpid][i][1];
		}
	}
#else

	if (warptid == 0 && validthread) {
		for (int i = 0; i < 27; i++) {
			double sumst = 0;
			for (int j = 0; j < 32; j++) {
				sumst += sumCoarseStencil[warpid][i][j];
			}
			rxCoarse[i][ke_id][vid] = sumst;
		}
	}

#endif
}

__global__ void update_residual_kernel(int nv, double* rxstencil) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	int vid = tid;

	GraftArray<double, 27, 9> stencil(rxstencil, nv);
	//double f[3] = { gF[0][vid],gF[1][vid],gF[2][vid] };
	double KU[3] = { 0. };
	for (int i = 0; i < 27; i++) {
		int vj = gV2V[i][vid];
		if (vj == -1) continue;
		double u[3] = { gU[0][vj],gU[1][vj],gU[2][vj] };
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 3; col++) {
				KU[row] += stencil[i][row * 3 + col][vid] * u[col];
			}
		}
	}

	for (int i = 0; i < 3; i++) {
		gR[i][vid] = gF[i][vid] - KU[i];
	}
}

// map 32 vertices to 13 warp
template<int SetBlockSize = 32 * 13>
__global__ void update_residual_kernel_1(int nv, double* rxstencil) {
	GraftArray<double, 27, 9> stencil(rxstencil, nv);
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//int mode = gmode[0];

	__shared__ double sumAu[3][13][32];
	int warpId = threadIdx.x / 32;
	int warpTid = threadIdx.x % 32;
	int vid = blockIdx.x * 32 + warpTid;

	double Au[3] = { 0.f,0.f,0.f };

	int flag;

	bool invalid_node = true;

	if (vid < nv) {
		flag = gVflag[0][vid];
		invalid_node = flag & Grid::Bitmask::mask_invalid;
		if (invalid_node) goto _blockSum;
		for (auto i : { 0,14 }) {
			double displacement[3];
			int neigh_th = warpId + i;
			int neigh = gV2V[neigh_th][vid];
			if (neigh == -1) continue;

			for (int j = 0; j < 3; j++) displacement[j] = gU[j][neigh];

			// K3 is ordered in row major 
			// traverse rows 
			for (int j = 0; j < 3; j++) {
				int jrows = j * 3;
				// traverse columns, dot u 
				for (int k = 0; k < 3; k++) {
					Au[j] += stencil[neigh_th][jrows + k][vid] * displacement[k];
				}
			}

		}
	}

_blockSum:

	for (int i = 0; i < 3; i++) {
		sumAu[i][warpId][warpTid] = Au[i];
	}
	__syncthreads();

	// gather all part
	if (warpId < 7) {
		for (int i = 0; i < 3; i++) {
			int addId = warpId + 7;
			if (addId < 13) {
				sumAu[i][warpId][warpTid] += sumAu[i][addId][warpTid];
			}
		}
	}
	__syncthreads();
	if (warpId < 4) {
		for (int i = 0; i < 3; i++) {
			int addId = warpId + 4;
			if (addId < 7) {
				sumAu[i][warpId][warpTid] += sumAu[i][addId][warpTid];
			}
		}
	}
	__syncthreads();
	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			int addId = warpId + 2;
			sumAu[i][warpId][warpTid] += sumAu[i][addId][warpTid];
		}
	}
	__syncthreads();
	if (warpId < 1) {
		for (int i = 0; i < 3; i++) {
			int addId = warpId + 1;
			Au[i] = sumAu[i][warpId][warpTid] + sumAu[i][addId][warpTid];
		}
	}
	//__syncthreads();

	if (vid < nv) {
		double displacement[3] = { 0. };
		if (warpId == 0) {
			for (int i = 0; i < 3; i++) displacement[i] = gU[i][vid];
			for (int i = 0; i < 3; i++) {
				Au[0] += stencil[13][i][vid] * displacement[i];
				Au[1] += stencil[13][3 + i][vid] * displacement[i];
				Au[2] += stencil[13][6 + i][vid] * displacement[i];
			}
			for (int i = 0; i < 3; i++) {
				gR[i][vid] = gF[i][vid] - Au[i];
			}
		}
	}
}

__global__ void update_residual_OTFA_NS_kernel(int nv, float* rholist) {

	__shared__ double KE[24][24];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	loadTemplateMatrix(KE);

	if (tid >= nv) return;

	int vid = tid;

	//int mode = gmode[0];
	int v2v[27];
	loadNeighborNodes(vid, v2v);

	double KU[3] = { 0.,0.,0. };
	float power = power_penalty[0];
	for (int i = 0; i < 8; i++) {
		int eid = gV2E[i][vid];
		if (eid == -1) continue;
		double penalty = powf(rholist[eid], power);
		int vi = 7 - i;
		for (int vj = 0; vj < 8; vj++) {
			int vjpos[3] = {
				vj % 2 + i % 2,
				vj % 4 / 2 + i % 4 / 2,
				vj / 4 + i / 4
			};
			int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
			int vj_vid = v2v[vj_lid];
			if (vj_vid == -1) {
				// DEBUG
				printf("-- error in update residual otfa\n");
				continue;
			}
			double u[3] = { gU[0][vj_vid],gU[1][vj_vid],gU[2][vj_vid] };
			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 3; col++) {
					KU[row] += penalty * KE[row + vi * 3][col + vj * 3] * u[col];
				}
			}
		}

	}

	for (int i = 0; i < 3; i++) {
		gR[i][vid] = gF[i][vid] - KU[i];
	}
}

__global__ void update_residual_OTFA_WS_kernel(int nv, float* rholist) {

	__shared__ double KE[24][24];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	loadTemplateMatrix(KE);

	if (tid >= nv) return;

	int vid = tid;

	// add fixed flag check
	bool vfix[27], vload[27];
	int v2v[27];
	loadNeighborNodesAndFlags(vid, v2v, vfix, vload);

	double KU[3] = { 0.,0.,0. };
	float power = power_penalty[0];
	for (int i = 0; i < 8; i++) {
		int eid = gV2E[i][vid];
		if (eid == -1) continue;
		double penalty = powf(rholist[eid], power);
		int vi = 7 - i;
		for (int vj = 0; vj < 8; vj++) {
			int vjpos[3] = {
				vj % 2 + i % 2,
				vj % 4 / 2 + i % 4 / 2,
				vj / 4 + i / 4
			};
			int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
			int vj_vid = v2v[vj_lid];
			if (vj_vid == -1) {
				// DEBUG
				printf("-- error in update residual otfa\n");
				continue;
			}
			double u[3] = { gU[0][vj_vid],gU[1][vj_vid],gU[2][vj_vid] };
			if (vfix[vj_lid]) {
				u[0] = 0; u[1] = 0; u[2] = 0;
			}
			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 3; col++) {
					KU[row] += penalty * KE[row + vi * 3][col + vj * 3] * u[col];
				}
			}
		}
	}

	if (vfix[13]) {
		KU[0] = 0; KU[1] = 0; KU[2] = 0;
	}

	for (int i = 0; i < 3; i++) {
		gR[i][vid] = gF[i][vid] - KU[i];
	}
}

template<int SetBlockSize = 32 * 8>
__global__ void update_residual_OTFA_WS_kernel_1(int nv, float* rholist) {

	__shared__ double KE[24][24];
	__shared__ double sumKeU[3][4][32];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	loadTemplateMatrix(KE);

	int warpId = threadIdx.x / 32;
	int warpTid = threadIdx.x % 32;

	double KeU[3] = { 0.,0.,0. };

	float power = power_penalty[0];

	int vid = blockIdx.x * 32 + warpTid;

	// add fixed flag check
	bool vfix[27], vload[27];
	int v2v[27];

	if (vid >= nv) goto __blocksum;

	loadNeighborNodesAndFlags(vid, v2v, vfix, vload);

	// sum a element
	{
		int i = warpId;
		int eid = gV2E[i][vid];
		double penalty;
		int vi = 7 - i;
		if (eid == -1) goto __blocksum;
		penalty = powf(rholist[eid], power);
		for (int vj = 0; vj < 8; vj++) {
			int vjpos[3] = {
				vj % 2 + i % 2,
				vj % 4 / 2 + i % 4 / 2,
				vj / 4 + i / 4
			};
			int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
			int vj_vid = v2v[vj_lid];
			if (vj_vid == -1) {
				// DEBUG
				printf("-- error in update residual otfa\n");
				continue;
			}
			double u[3] = { gU[0][vj_vid],gU[1][vj_vid],gU[2][vj_vid] };
			if (vfix[vj_lid]) {
				u[0] = 0; u[1] = 0; u[2] = 0;
			}
			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 3; col++) {
					KeU[row] += penalty * KE[row + vi * 3][col + vj * 3] * u[col];
				}
			}
		}
	}

__blocksum:
	if (warpId >= 4) {
		for (int i = 0; i < 3; i++) { sumKeU[i][warpId - 4][warpTid] = KeU[i]; }
	}
	__syncthreads();

	if (warpId < 4) {
		for (int i = 0; i < 3; i++) { sumKeU[i][warpId][warpTid] += KeU[i]; }
	}
	__syncthreads();

	if (warpId < 2) {
		for (int i = 0; i < 3; i++) { sumKeU[i][warpId][warpTid] += sumKeU[i][warpId + 2][warpTid]; }
	}
	__syncthreads();

	if (warpId < 1 && v2v[13] != -1) {
		for (int i = 0; i < 3; i++) { KeU[i] = sumKeU[i][0][warpTid] + sumKeU[i][1][warpTid]; }

		if (vfix[13]) { KeU[0] = 0; KeU[1] = 0; KeU[2] = 0; }
		
		for (int i = 0; i < 3; i++) { gR[i][vid] = gF[i][vid] - KeU[i]; }
	}
}

void Grid::update_residual(void)
{
	if (is_dummy()) return;
	use_grid();
	size_t grid_size, block_size;
	if (_layer == 0) {
		if (_mode == no_support_constrain_force_direction || _mode == no_support_free_force) {
			make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
			update_residual_OTFA_NS_kernel << <grid_size, block_size >> > (n_gsvertices, _gbuf.rho_e);
		}
		else if (_mode == with_support_constrain_force_direction || _mode == with_support_free_force) {
#if 1
			make_kernel_param(&grid_size, &block_size, n_gsvertices, 256);
			update_residual_OTFA_WS_kernel << <grid_size, block_size >> > (n_gsvertices, _gbuf.rho_e);
#else
			make_kernel_param(&grid_size, &block_size, n_gsvertices * 8, 32 * 8);
			update_residual_OTFA_WS_kernel_1 << <grid_size, block_size >> > (n_gsvertices, _gbuf.rho_e);
#endif
		}
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
#if 0
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		update_residual_kernel << <grid_size, block_size >> > (n_gsvertices, _gbuf.rxStencil);
#else
		make_kernel_param(&grid_size, &block_size, n_gsvertices * 13, 32 * 13);
		update_residual_kernel_1 << <grid_size, block_size >> > (n_gsvertices, _gbuf.rxStencil);

#endif
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

__global__ void restrict_residual_kernel(int nv) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	
	//int mode = gmode[0];

	double res[3] = { 0.f };

	// volume center
	{
		int neigh = gV2Vfine[13][tid];
		if (neigh != -1) {
			for (int i = 0; i < 3; i++) {
				res[i] += gRfine[i][neigh];
			}
		}
	}

	// volume vertex
	for (int j : {0, 2, 6, 8, 18, 20, 24, 26}) {
		int neigh = gV2Vfine[j][tid];
		if (neigh != -1) {
			for (int i = 0; i < 3; i++) {
				res[i] += gRfine[i][neigh] * (1.0f / 8);
			}
		}
	}
	// face center
	for (int j : {4, 10, 12, 14, 16, 22}) {
		int neigh = gV2Vfine[j][tid];
		if (neigh != -1) {
			for (int i = 0; i < 3; i++) {
				res[i] += gRfine[i][neigh] * (1.0f / 2);
			}
		}
	}
	// edge center
	for (int j : {1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25}) {
		int neigh = gV2Vfine[j][tid];
		if (neigh != -1) {
			for (int i = 0; i < 3; i++) {
				res[i] += gRfine[i][neigh] * (1.0f / 4);
			}
		}
	}

__writeResidual:

	for (int i = 0; i < 3; i++) {
		gF[i][tid] = res[i] /*/ 8*/;
	}
}

__global__ void restrict_residual_nondyadic_kernel(int nv) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	//if (tid >= nv) return;

	//int mode = gmode[0];
	__shared__ double W[4][4][4];
	__shared__ int* vfine2vfine[27];

	if (threadIdx.x < 64) {
		int k = threadIdx.x % 4;
		int j = threadIdx.x / 4 % 4;
		int i = threadIdx.x / 16;
		W[i][j][k] = ((4 - i)*(4 - j)*(4 - k)) / 64.0;
		if (threadIdx.x < 27) {
			vfine2vfine[threadIdx.x] = gVfine2Vfine[threadIdx.x];
		}
	}
	__syncthreads();

	if (tid >= nv) return;

	int vid = tid;

	int aFlag[(7 * 7 * 7) / (sizeof(int) * 8) + 1] = { 0 };

	double sumR[3] = { 0. };

	// DEBUG
	//if (sumR[0] != 0 || sumR[1] != 0 || sumR[2] != 0) { printf("\033[31m-- kernel error, nonzero init at file %s, line %d\033[0m\n", __FILE__, __LINE__); }

	double* rfine[3] = { gRfine[0], gRfine[1], gRfine[2] };

	for (int i = 0; i < 64; i++) {
		int vff = gV2VfineC[i][vid];
		if (vff == -1) continue;
		int basepos[3] = { i % 4 * 2 - 3,i % 16 / 4 * 2 - 3,i / 16 * 2 - 3 };
		for (int dx = -1; dx <= 1; dx++) {
			int xj = basepos[0] + dx;
			if (xj <= -4 || xj >= 4) continue;
			for (int dy = -1; dy <= 1; dy++) {
				int yj = basepos[1] + dy;
				if (yj <= -4 || yj >= 4) continue;
				for (int dz = -1; dz <= 1; dz++) {
					int zj = basepos[2] + dz;
					if (zj <= -4 || zj >= 4) continue;
					int jid = xj + 3 + (yj + 3) * 7 + (zj + 3) * 49;
					if (read_gbit(aFlag, jid)) continue;
					set_gbit(aFlag, jid);
					int djid = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;
					int vj_vid = vfine2vfine[djid][vff];
					if (vj_vid == -1) continue;
					double r[3] = { rfine[0][vj_vid], rfine[1][vj_vid], rfine[2][vj_vid] };
					//double weight = (4 - abs(xj))*(4 - abs(yj))*(4 - abs(zj)) / 64.0;
					double weight = W[abs(xj)][abs(yj)][abs(zj)];
					for (int k = 0; k < 3; k++) sumR[k] += weight * r[k];
				}
			}
		}
	}

	for (int k = 0; k < 3; k++) { gF[k][vid] = sumR[k]; }
}

void Grid::restrict_residual(void)
{
	use_grid();

	size_t grid_size, block_size;
	if (_layer == 2 && is_skip()) {
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 256);
		restrict_residual_nondyadic_kernel << <grid_size, block_size >> > (n_gsvertices);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else if (_layer == 0) {
		msg() << "\033[31mCannot restrict residual to finest layer" << "\033[0m" << std::endl;
	}
	else {
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		restrict_residual_kernel << <grid_size, block_size >> > (n_gsvertices);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

__global__ void prolongate_correction_kernel(int nv) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//int mode = gmode[0];

	if (tid >= nv) return;

	int vid = tid;


	double c[3] = { 0. };

	double* pU[3] = { gUcoarse[0],gUcoarse[1],gUcoarse[2] };

	int flag = gVflag[0][vid];
	if (flag& Grid::Bitmask::mask_invalid) return;

	int posInE[3] = {
		((flag & Grid::Bitmask::mask_xmod7) >> Grid::Bitmask::offset_xmod7) % 2,
		((flag & Grid::Bitmask::mask_ymod7) >> Grid::Bitmask::offset_ymod7) % 2,
		((flag & Grid::Bitmask::mask_zmod7) >> Grid::Bitmask::offset_zmod7) % 2
	};

	for (int i = 0; i < 8; i++) {
		int vcoarsepos[3] = { i % 2 * 2, i % 4 / 2 * 2, i / 4 * 2 };
		int wpos[3] = { abs(vcoarsepos[0] - posInE[0]), abs(vcoarsepos[1] - posInE[1]), abs(vcoarsepos[2] - posInE[2]) };
		if (wpos[0] >= 2 || wpos[1] >= 2 || wpos[2] >= 2) continue;
		double weight = (2 - wpos[0]) * (2 - wpos[1]) * (2 - wpos[2]) / 8.;
		int vcoarseid = gV2Vcoarse[i][vid];
		if (vcoarseid == -1) continue;
		for (int j = 0; j < 3; j++) {
			c[j] += weight * pU[j][vcoarseid];
		}
	}

	for (int i = 0; i < 3; i++) {
		gU[i][vid] += c[i];
	}
}


__global__ void prolongate_correction_nondyadic_kernel(int nv, int* vbitflag) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= nv) return;

	int vid = tid;

	//int mode = gmode[0];

	double c[3] = { 0. };

	int flagword = vbitflag[vid];

	if (flagword & Grid::Bitmask::mask_invalid) return;

	int posInE[3] = {
		(flagword & Grid::Bitmask::mask_xmod7) >> Grid::Bitmask::offset_xmod7,
		(flagword & Grid::Bitmask::mask_ymod7) >> Grid::Bitmask::offset_ymod7,
		(flagword & Grid::Bitmask::mask_zmod7) >> Grid::Bitmask::offset_zmod7
	};
	for (int i = 0; i < 3; i++) posInE[i] %= 4;

	int nei_counter = 0;
	// traverse vertex of coarse element which contains this fine vertex
	for (int i = 0; i < 8; i++) {
		int finepos[3] = { i % 2 * 4, i % 4 / 2 * 4, i / 4 * 4 };
		int wpos[3] = { abs(finepos[0] - posInE[0]), abs(finepos[1] - posInE[1]), abs(finepos[2] - posInE[2]) };
		if (wpos[0] >= 4 || wpos[1] >= 4 || wpos[2] >= 4) continue;
		double weight = (4 - wpos[0]) * (4 - wpos[1]) * (4 - wpos[2]) / 64.0;
		int coarseid = gV2Vcoarse[i][vid];
		if (coarseid == -1) continue;
		for (int j = 0; j < 3; j++) c[j] += weight * gUcoarse[j][coarseid];
	}

	for (int i = 0; i < 3; i++) {
		gU[i][vid] += c[i];
	}
}


void Grid::prolongate_correction(void)
{
	if (is_dummy()) return;
	use_grid();
	size_t grid_size, block_size;
	if (_layer == 0 && is_skip()) {
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		prolongate_correction_nondyadic_kernel << <grid_size, block_size >> > (n_gsvertices, _gbuf.vBitflag);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		prolongate_correction_kernel << <grid_size, block_size >> > (n_gsvertices);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

void Grid::reset_displacement(void)
{
	for (int i = 0; i < 3; i++) {
		init_array(_gbuf.U[i], 0., n_gsvertices);
	}
}

void Grid::reset_force(void)
{
	cuda_error_check;
	for (int i = 0; i < 3; i++) {
		init_array(_gbuf.F[i], 0., n_gsvertices);
	}
}

void Grid::reset_residual(void)
{
	for (int i = 0; i < 3; i++) {
		init_array(_gbuf.R[i], 0., n_gsvertices);
	}
}

double Grid::v3norm(double* v[3])
{
	double* tmp = (double*)getTempBuf(n_nodes() * sizeof(double) / 100);
	double s = norm(v[0], v[1], v[2], tmp, n_nodes());
	return s;
}

double Grid::relative_residual(void)
{
	double r = v3norm(_gbuf.R);
	double f = v3norm(_gbuf.F);
	return r / f;
}

double Grid::residual(void)
{
	return v3norm(_gbuf.R);
}


//__global__ void mark_surface_nodes_kernel(int nv, int* vflag, int* eflag) {
//	int tid = threadIdx.x + blockDim.x*blockIdx.x;
//	if (tid >= nv) return;
//
//	int vid = tid;
//
//	bool surf = false;
//
//	bool axisHasNeighbor[3] = { false, false, false };
//
//	bool solid_flag[2][2][2];
//	for (int i = 0; i < 8; i++) {
//		int eid = gV2E[i][vid];
//		solid_flag[i % 2][i % 4 / 2][i / 4] = (eid != -1);
//	}
//
//	for (int i = 0; i < 2; i++) {
//		for (int j = 0; j < 2; j++) {
//			axisHasNeighbor[0] |= solid_flag[0][i][j] && solid_flag[1][i][j];
//			axisHasNeighbor[1] |= solid_flag[i][0][j] && solid_flag[i][1][j];
//			axisHasNeighbor[2] |= solid_flag[i][j][0] && solid_flag[i][j][1];
//		}
//	}
//
//	surf = (!axisHasNeighbor[0]) || (!axisHasNeighbor[1]) || (!axisHasNeighbor[2]);
//
//	int word = vflag[vid];
//	if (surf) {
//		word |= Grid::Bitmask::mask_surfacenodes;
//	}
//	else {
//		word &= ~(int)Grid::Bitmask::mask_surfacenodes;
//	}
//	vflag[vid] = word;
//}

//void Grid::mark_surface_nodes_g(void)
//{
//	use_grid();
//	size_t grid_size, block_size;
//	make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
//	mark_surface_nodes_kernel << <grid_size, block_size >> > (n_gsvertices, _gbuf.vBitflag, _gbuf.eBitflag);
//	cudaDeviceSynchronize();
//	cuda_error_check;
//}

__global__ void mark_surface_nodes_kernel(int nv, devArray_t<int*, 8> v2elist, int* vflag) {
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	if (tid >= nv) return;

	int vid = tid;

	bool surf = false;

	bool axisHasNeighbor[3] = { false, false, false };

	bool solid_flag[2][2][2];
	for (int i = 0; i < 8; i++) {
		int eid = v2elist[i][vid];
		solid_flag[i % 2][i % 4 / 2][i / 4] = (eid != -1);
	}

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			axisHasNeighbor[0] |= solid_flag[0][i][j] && solid_flag[1][i][j];
			axisHasNeighbor[1] |= solid_flag[i][0][j] && solid_flag[i][1][j];
			axisHasNeighbor[2] |= solid_flag[i][j][0] && solid_flag[i][j][1];
		}
	}

	surf = (!axisHasNeighbor[0]) || (!axisHasNeighbor[1]) || (!axisHasNeighbor[2]);

	int word = vflag[vid];
	if (surf) {
		word |= Grid::Bitmask::mask_surfacenodes;
	}
	else {
		word &= ~(int)Grid::Bitmask::mask_surfacenodes;
	}
	vflag[vid] = word;
}

void Grid::mark_surface_nodes_g(int nv, int* v2e[8], int* vflag)
{
	devArray_t<int*, 8> v2elist;
	for (int i = 0; i < 8; i++) v2elist[i] = v2e[i];

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv, 512);
	mark_surface_nodes_kernel << <grid_size, block_size >> > (nv, v2elist, _gbuf.vBitflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}

__global__ void mark_surface_elements_kernel(int nv, devArray_t<int*, 8> v2elist, int* vflag, int* eflag) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;

	int vfw = vflag[tid];

	if (vfw & Grid::Bitmask::mask_surfacenodes) {
		for (int i = 0; i < 8; i++) {
			int eid = v2elist[i][tid];
			if (eid == -1) continue;
			atomic_set_gbit(eflag, sizeof(int) * 8 * eid + Grid::Bitmask::offset_surfaceelements);
		}
	}

}

void grid::Grid::mark_surface_elements_g(int nv, int ne, int* v2e[8], int* vflag, int* eflag)
{
	devArray_t<int*, 8> v2elist;
	for (int i = 0; i < 8; i++) v2elist[i] = v2e[i];

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv, 512);
	mark_surface_elements_kernel << <grid_size, block_size >> > (nv, v2elist, vflag, eflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}



std::vector<int> Grid::getVflags(void)
{
	std::vector<int> hostflag(n_gsvertices);
	cudaMemcpy(hostflag.data(), _gbuf.vBitflag, sizeof(int)*n_gsvertices, cudaMemcpyDeviceToHost);
	cuda_error_check;
	return hostflag;
}

std::vector<int> Grid::getEflags(void)
{
	std::vector<int> hostflag(n_gselements);
	cudaMemcpy(hostflag.data(), _gbuf.eBitflag, sizeof(int)*n_gselements, cudaMemcpyDeviceToHost);
	cuda_error_check;
	return hostflag;
}

void Grid::getVflags(int nv, int* dst)
{
	cudaMemcpy(dst, _gbuf.vBitflag, sizeof(int)* nv, cudaMemcpyDeviceToHost);
}

void Grid::setVflags(int nv, int *src)
{
	cudaMemcpy(_gbuf.vBitflag, src, sizeof(int)* nv, cudaMemcpyHostToDevice);
}

void Grid::getEflags(int nv, int* dst)
{
	cudaMemcpy(dst, _gbuf.eBitflag, sizeof(int) * nv, cudaMemcpyDeviceToHost);
}


void Grid::v3_init(double* v[3], double val[3])
{
	for (int i = 0; i < 3; i++) {
		init_array(v[i], val[i], n_gsvertices);
	}
}


void Grid::v3_minus(double* a[3], double alpha, double* b[3])
{
	double* ax = a[0], *ay = a[1], *az = a[2];
	double* bx = b[0], *by = b[1], *bz = b[2];
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n_gsvertices, 512);
	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { ax[tid] -= alpha * bx[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { ay[tid] -= alpha * by[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { az[tid] -= alpha * bz[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;
}

void Grid::v3_minus(double* dst[3], double* a[3], double alpha, double* b[3])
{
	double* ax = a[0], *ay = a[1], *az = a[2];
	double* bx = b[0], *by = b[1], *bz = b[2];
	double* dstx = dst[0], *dsty = dst[1], *dstz = dst[2];
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n_gsvertices, 512);
	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { dstx[tid] = ax[tid] - alpha * bx[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { dsty[tid] = ay[tid] - alpha * by[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { dstz[tid] = az[tid] - alpha * bz[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;
}


void Grid::v3_add(double* a[3], double alpha, double* b[3])
{
	double* ax = a[0], *ay = a[1], *az = a[2];
	double* bx = b[0], *by = b[1], *bz = b[2];
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n_gsvertices, 512);
	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { ax[tid] += alpha * bx[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { ay[tid] += alpha * by[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { az[tid] += alpha * bz[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;
}

void Grid::v3_add(double alpha, double* a[3], double beta, double* b[3])
{
	double* ax = a[0], *ay = a[1], *az = a[2];
	double* bx = b[0], *by = b[1], *bz = b[2];
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n_gsvertices, 512);
	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { ax[tid] = alpha * ax[tid] + beta * bx[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { ay[tid] = alpha * ay[tid] + beta * by[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { az[tid] = alpha * az[tid] + beta * bz[tid]; });
	cudaDeviceSynchronize();
	cuda_error_check;
}

double Grid::v3_dot(double* v[3], double* u[3])
{
	double* tmp = (double*)getTempBuf(n_gsvertices / 100 * sizeof(double));
	double s = dot(v[0], v[1], v[2], u[0], u[1], u[2], tmp, n_gsvertices);
	return s;
}

double Grid::v3_diffdot(double* v1[3], double* v2[3], double* v3[3], double* v4[3])
{
	double sum = parallel_diffdot(n_nodes(), v1, v2, v3, v4, (double*)getTempBuf(n_gsvertices / 100 * sizeof(double)));
	cuda_error_check;
	return sum;
}

double Grid::v3_norm(double* v[3])
{
	double* tmp = (double*)getTempBuf(n_gsvertices / 100 * sizeof(double));
	double s = norm(v[0], v[1], v[2], tmp, n_nodes());
	return s;
}

double grid::Grid::v3_normalize(double* v[3])
{
	double nr = v3_norm(v);
	v3_scale(v, 1.0 / nr);
	return nr;
}

void grid::Grid::v3_destroy(double* dstv[3])
{
	for (int i = 0; i < 3; i++) {
		cudaFree(dstv[i]);
	}
	cuda_error_check;
}

void Grid::v3_rand(double* v[3], double low, double upp)
{
	randArray(v, 3, n_gsvertices, low, upp);
}

void Grid::randForce(void)
{
	v3_rand(_gbuf.F, -1, 1);
}

double Grid::unitizeForce(void)
{
	double fnorm = v3_norm(_gbuf.F);
	//printf("-- untize f norm = %lf\n", fnorm);
	v3_scale(_gbuf.F, 1.0 / fnorm);
	return fnorm;
}

double Grid::supportForceCh(void)
{
	double * fs[4];
	getTempBufArray(fs, 4, n_loadnodes());

	getForceSupport(_gbuf.F, fs);

	double sum = parallel_diffdot(n_loadnodes(), _gbuf.Fsupport, fs, _gbuf.Fsupport, fs, fs[3]);
	cuda_error_check;

	return sqrt(sum);
}

double grid::Grid::supportForceCh(double* newf[3])
{
	double* newfs[4];
	getTempBufArray(newfs, 4, n_loadnodes());

	getForceSupport(newf, newfs);

	double sum = parallel_diffdot(n_loadnodes(), _gbuf.Fsupport, newfs, _gbuf.Fsupport, newfs, newfs[3]);
	cuda_error_check;

	return sqrt(sum);
}

double Grid::supportForceNorm(void)
{
	double sum = norm(_gbuf.Fsupport[0], _gbuf.Fsupport[1], _gbuf.Fsupport[2], (double*)getTempBuf(sizeof(double) * n_loadnodes() / 100), n_loadnodes());
	cuda_error_check;
	return sum;
}

void Grid::v3_scale(double* v[3], double ampl)
{
	double *vx = v[0], *vy = v[1], *vz = v[2];
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n_nodes(), 512);
	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { vx[tid] *= ampl; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { vy[tid] *= ampl; });
	cudaDeviceSynchronize();
	cuda_error_check;

	map << <grid_dim, block_dim >> > (n_nodes(), [=]__device__(int tid) { vz[tid] *= ampl; });
	cudaDeviceSynchronize();
	cuda_error_check;
}


void Grid::v3_copy(double* vsrc[3], double* vdst[3])
{
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(vdst[i], vsrc[i], sizeof(double)*n_nodes(), cudaMemcpyDeviceToDevice);
		cuda_error_check;
	}
}

void HierarchyGrid::setMode(Mode mode)
{
	int modeid = mode;
	std::cout << "--[TEST] mode id: " << modeid << std::endl;
	_mode = mode;
	Grid::_mode = mode;
	cudaMemcpyToSymbol(gmode, &modeid, sizeof(int));
}

void HierarchyGrid::setSSMode(GlobalSSMode mode)
{
	int modeid = mode;
	std::cout << "--[TEST] ssmode id: " << modeid << std::endl;
	_ssmode = mode;
	Grid::_ssmode = mode;
	cudaMemcpyToSymbol(gssmode, &modeid, sizeof(int));
}

void HierarchyGrid::setDripMode(GlobalDripMode mode)
{
	int modeid = mode;
	std::cout << "--[TEST] dripmode id: " << modeid << std::endl;
	_dripmode = mode;
	Grid::_dripmode = mode;
	cudaMemcpyToSymbol(gdripmode, &modeid, sizeof(int));
}

void HierarchyGrid::setPrintAngle(float default_angle_ratio, float opt_angle_ratio)
{
	float sdefault = default_angle_ratio * M_PI;
	std::cout << "--[TEST] angle default: " << sdefault << std::endl;
	_setting.default_print_angle = sdefault;
	Grid::_default_print_angle = sdefault;
	cudaMemcpyToSymbol(gdefaultPrintAngle, &sdefault, sizeof(float));
	cuda_error_check;

	float sopt = opt_angle_ratio * M_PI;
	std::cout << "--[TEST] angle opt: " << sopt << std::endl;
	_setting.opt_print_angle = sopt;
	Grid::_opt_print_angle = sopt;
	cudaMemcpyToSymbol(goptPrintAngle, &sopt, sizeof(float));
}

void HierarchyGrid::set_spline_partition(int spartx, int sparty, int spartz, int sorder)
{
	int sporder = sorder;
	grid::Grid::n_order = sorder;
	cudaMemcpyToSymbol(gorder, &sporder, sizeof(int));
	cuda_error_check;

	_setting.n_partitionx = spartx;
	_setting.n_partitiony = sparty;
	_setting.n_partitionz = spartz;
	grid::Grid::n_partitionx = spartx;
	grid::Grid::n_partitiony = sparty;
	grid::Grid::n_partitionz = spartz;
	grid::Grid::sppartition[0] = spartx;
	grid::Grid::sppartition[1] = sparty;
	grid::Grid::sppartition[2] = spartz;
	cudaMemcpyToSymbol(gnpartition, grid::Grid::sppartition, sizeof(gnpartition));
	cuda_error_check;

	_setting.n_im = sorder + spartx;
	_setting.n_in = sorder + sparty;
	_setting.n_il = sorder + spartz;
	Grid::n_im = sorder + spartx;
	Grid::n_in = sorder + sparty;
	Grid::n_il = sorder + spartz;
	//int spbasis[3] = { sorder + spartx, sorder + sparty, sorder + spartz };
	Grid::spbasis[0] = sorder + spartx;
	Grid::spbasis[1] = sorder + sparty;
	Grid::spbasis[2] = sorder + spartz;
	cudaMemcpyToSymbol(gnbasis, grid::Grid::spbasis, sizeof(gnbasis));
	cuda_error_check;

	_setting.n_knotspanx = 2 * sorder + spartx;
	_setting.n_knotspany = 2 * sorder + sparty;
	_setting.n_knotspanz = 2 * sorder + spartz;
	Grid::n_knotspanx = 2 * sorder + spartx;
	Grid::n_knotspany = 2 * sorder + sparty;
	Grid::n_knotspanz = 2 * sorder + spartz;
	//int spknotspan[3] = { 2 * sorder + spartx, 2 * sorder + sparty, 2 * sorder + spartz };
	Grid::spknotspan[0] = 2 * sorder + spartx;
	Grid::spknotspan[1] = 2 * sorder + sparty;
	Grid::spknotspan[2] = 2 * sorder + spartz;
	cudaMemcpyToSymbol(gnknotspan, grid::Grid::spknotspan, sizeof(int) * 3);
	cuda_error_check;

	//std::cout << "-------------- spline info --------------------------- " << std::endl;
	//int orderhost;
	//int* orderhost2 = new int[1];
	//int* partitionhost = new int[3];
	//int* basishost = new int[3];
	//int* knotspanhost = new int[8];
	//cudaMemcpyFromSymbol(&orderhost, gorder, sizeof(int));
	//cudaMemcpyFromSymbol(orderhost2, gorder, sizeof(int));
	//cudaMemcpyFromSymbol(partitionhost, gnpartition, sizeof(int) * 3);
	//cudaMemcpyFromSymbol(basishost, gnbasis, sizeof(int) * 3);
	//cudaMemcpyFromSymbol(knotspanhost, gnknotspan, sizeof(int) * 3);
	//std::cout << "gnorder: " << orderhost << std::endl;
	//std::cout << "gnorder: " << orderhost2[0] << std::endl;
	//std::cout << "gnpartition: " << partitionhost[0] << ", " << partitionhost[1] << ", " << partitionhost[2] << std::endl;
	//std::cout << "gnpartition: " << grid::Grid::sppartition[0] << ", " << grid::Grid::sppartition[1] << ", " << grid::Grid::sppartition[2] << std::endl;
	//std::cout << "gnbasis: " << basishost[0] << ", " << basishost[1] << ", " << basishost[2] << std::endl;
	//std::cout << "gnbasis: " << grid::Grid::spbasis[0] << ", " << grid::Grid::spbasis[1] << ", " << grid::Grid::spbasis[2] << std::endl;
	//std::cout << "gnknotspan: " << knotspanhost[0] << ", " << knotspanhost[1] << ", " << knotspanhost[2] << std::endl;
	//std::cout << "gnknotspan: " << grid::Grid::spknotspan[0] << ", " << grid::Grid::spknotspan[1] << ", " << grid::Grid::spknotspan[2] << std::endl;
	//delete[] orderhost2;
	//delete[] partitionhost;
	//delete[] basishost;
	//delete[] knotspanhost;
	//std::cout << "------------------------------------------------------ " << std::endl;
}

void grid::Grid::set_spline_knot_infoSymbol(void)
{
	cudaMemcpyToSymbol(gpu_cijk, &_gbuf.coeffs, sizeof(gpu_cijk));
	cuda_error_check;

	cudaMemcpyToSymbol(gpu_KnotSer, _gbuf.KnotSer, sizeof(gpu_KnotSer));
	cuda_error_check;

	cudaMemcpyToSymbol(gnstep, m_sStep, sizeof(gnstep));
	cudaMemcpyToSymbol(gnBoundMin, m_3sBoundMin, sizeof(gnBoundMin));
	cudaMemcpyToSymbol(gnBoundMax, m_3sBoundMax, sizeof(gnBoundMax));
	cuda_error_check;

#if 0
	 //update to download (OK)
	//float penalhost[1];
	float stephost[3];
	float* Boundminhost = new float[3];
	float* Boundmaxhost = new float[3];
	//cudaMemcpyFromSymbol(penalhost, power_penalty, sizeof(float));
	cudaMemcpyFromSymbol(stephost, gnstep, sizeof(float) * 3);
	cudaMemcpyFromSymbol(Boundminhost, gnBoundMin, sizeof(float) * 3);
	cudaMemcpyFromSymbol(Boundmaxhost, gnBoundMax, sizeof(float) * 3);

	std::cout << "-------------- spline info --------------------------- " << std::endl;
	//std::cout << "penal: " << penalhost[0] << std::endl;
	std::cout << "gnStep: " << stephost[0] << ", " << stephost[1] << ", " << stephost[2] << std::endl;
	std::cout << "gnBoundmin: " << Boundminhost[0] << ", " << Boundminhost[1] << ", " << Boundminhost[2] << std::endl;
	std::cout << "gnBoundmax: " << Boundmaxhost[0] << ", " << Boundmaxhost[1] << ", " << Boundmaxhost[2] << std::endl;
	delete[] Boundminhost;
	delete[] Boundmaxhost;

	int orderhost;
	int* orderhost2 = new int[1];
	int* partitionhost = new int[3];
	int* basishost = new int[3];
	int* knotspanhost = new int[3];
	cudaMemcpyFromSymbol(&orderhost, gorder, sizeof(int));
	cudaMemcpyFromSymbol(orderhost2, gorder, sizeof(int));
	cudaMemcpyFromSymbol(partitionhost, gnpartition, sizeof(int) * 3);
	cudaMemcpyFromSymbol(basishost, gnbasis, sizeof(int) * 3);
	cudaMemcpyFromSymbol(knotspanhost, gnknotspan, sizeof(int) * 3);
	std::cout << "gnorder: " << orderhost << std::endl;
	std::cout << "gnorder: " << orderhost2[0] << std::endl;
	std::cout << "gnpartition: " << partitionhost[0] << ", " << partitionhost[1] << ", " << partitionhost[2] << std::endl;
	std::cout << "gnbasis: " << basishost[0] << ", " << basishost[1] << ", " << basishost[2] << std::endl;
	std::cout << "gnknotspan: " << knotspanhost[0] << ", " << knotspanhost[1] << ", " << knotspanhost[2] << std::endl;
	delete[] orderhost2;
	delete[] partitionhost;
	delete[] basishost;
	delete[] knotspanhost;
	std::cout << "------------------------------------------------------ " << std::endl;

	//float* cijkhost = new float[n_cijk()];
	//cudaMemcpyFromSymbol(cijkhost, gpu_cijk, sizeof(float) * n_cijk());
	//cuda_error_check;
	//gpu_manager_t::pass_buf_to_matlab("gncijk", cijkhost, n_cijk());

	float* KnotSerhost[3];
	for (int i = 0; i < 3; i++)
	{
		int n_basis = spknotspan[i];
		//std::cout << n_basis << std::endl;
		KnotSerhost[i] = new float[n_basis];
		cudaMemcpy(KnotSerhost[i], _gbuf.KnotSer[i], sizeof(float) * n_basis, cudaMemcpyDeviceToHost);
		cuda_error_check;
	}

	gpu_manager_t::pass_buf_to_matlab("gnknotserx", KnotSerhost[0], spknotspan[0]);
	gpu_manager_t::pass_buf_to_matlab("gnknotsery", KnotSerhost[1], spknotspan[1]);
	gpu_manager_t::pass_buf_to_matlab("gnknotserz", KnotSerhost[2], spknotspan[2]);

	for (int i = 0; i < 3; i++)
	{
		delete[] KnotSerhost[i];
		KnotSerhost[i] = nullptr;
	}
//	delete[] KnotSerhost;
#endif
}

void grid::Grid::uploadCoeffsSymbol(void)
{
	cudaMemcpyToSymbol(gpu_cijk, &_gbuf.coeffs, sizeof(gpu_cijk));
	cuda_error_check;

	//float* cijkhost = new float[n_cijk()];
	//cudaMemcpyFromSymbol(cijkhost, gpu_cijk, sizeof(float) * n_cijk());
	//cuda_error_check;
	//gpu_manager_t::pass_buf_to_matlab("gncijk", cijkhost, n_cijk());
}

void grid::Grid::uploadSurfacePointsSymbol(void)
{
	cudaMemcpyToSymbol(gpu_SurfacePoints, _gbuf.surface_points, sizeof(gpu_SurfacePoints));
	cuda_error_check;

#ifdef ENABLE_MATLAB
	float* surfpoints_host[3];
	for (int i = 0; i < 3; i++)
	{
		surfpoints_host[i] = new float[_num_surface_points];
		cudaMemcpy(surfpoints_host[i], _gbuf.surface_points[i], sizeof(float) * _num_surface_points, cudaMemcpyDeviceToHost);
		cuda_error_check;
	}

	gpu_manager_t::pass_buf_to_matlab("gpu_surf1_x", surfpoints_host[0], _num_surface_points);
	gpu_manager_t::pass_buf_to_matlab("gpu_surf1_y", surfpoints_host[1], _num_surface_points);
	gpu_manager_t::pass_buf_to_matlab("gpu_surf1_z", surfpoints_host[2], _num_surface_points);

	for (int i = 0; i < 3; i++)
	{
		delete[] surfpoints_host[i];
		surfpoints_host[i] = nullptr;
	}
	memset(surfpoints_host, 0, sizeof(surfpoints_host));
#endif
}


template<typename coeffdensity>
__global__ void coeff2density_kernel(int nebitword, float mindensity, gBitSAT<unsigned int> esat, int ereso, float* g_dst, coeffdensity calc_node, const int* eidmap, const int* eflag) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nebitword) return;

	const unsigned int* ebit = esat._bitarray;
	const int* sat = esat._chunksat;

	unsigned int eword = ebit[tid];

	if (eword == 0) return;

	int eidoffset = sat[tid];
	int ewordoffset = 0;
	for (int j = 0; j < BitCount<unsigned int>::value; j++) {
		if (read_gbit(eword, j)) {
			int bid = tid * BitCount<unsigned int>::value + j;
			int bpos[3] = { bid % ereso, bid % (ereso * ereso) / ereso, bid / (ereso * ereso) };
			int eid = eidoffset + ewordoffset;
			float node_value = calc_node(bid);
			if (eidmap != nullptr) eid = eidmap[eid];

			// check the shellelement of rho_e
			node_value = clamp(node_value, mindensity, 1.f);
			if (eflag[eid] & grid::Grid::mask_shellelement)
			{
				node_value = 1;
			}
	     	// rhoknew = clamp(rhoknew, rhomin, 1.f);
	     	// if (gEflag[0][eid] & grid::Grid::Bitmask::mask_shellelement) rhonew = 1;
			g_dst[eid] = node_value;
			ewordoffset++;
		}
	}
}

void Grid::coeff2density(void)
{
	if (_layer != 0) return;
	// computation
	float* cijk_value = _gbuf.coeffs;
	float* knotx_ = _gbuf.KnotSer[0];
	float* knoty_ = _gbuf.KnotSer[1];
	float* knotz_ = _gbuf.KnotSer[2];
	float* rholist = _gbuf.rho_e;
	
	int ereso = _ereso;
	float eh = elementLength();
	float boxOrigin[3] = { _box[0][0], _box[0][1], _box[0][2] };
	float min_Density = _min_density;
		
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, _gbuf.nword_ebits, 512);

	auto calc_node = [=] __device__(int id) {
		int xCoordi = id % ereso;
		int yCoordi = (id % (ereso * ereso)) / ereso;
		int zCoordi = id / (ereso * ereso);
		float pos[3] = { boxOrigin[0] + xCoordi * eh + 0.5 * eh,boxOrigin[1] + yCoordi * eh + 0.5 * eh, boxOrigin[2] + zCoordi * eh + 0.5 * eh };

		float val;
		int i, j, k, ir, it, is, index;

		float pNX[m_iM + 1];
		float pNY[m_iM + 1];
		float pNZ[m_iM + 1];

		// the first knot index (order ... order + partion + 1) in knotspan(1 ... 2*order + partion)
		i = (int)((pos[0] - gnBoundMin[0]) / gnstep[0]) + gorder[0];
		j = (int)((pos[1] - gnBoundMin[1]) / gnstep[1]) + gorder[0];
		k = (int)((pos[2] - gnBoundMin[2]) / gnstep[2]) + gorder[0];

		if ((i < gorder[0]) || (i > gnbasis[0]) || (j < gorder[0]) || (j > gnbasis[1]) || (k < gorder[0]) || (k > gnbasis[2]))
		{
			val = -0.2f;
		}
		else
		{
			SplineBasisX(pos[0], pNX);
			SplineBasisY(pos[1], pNY);
			SplineBasisZ(pos[2], pNZ);

			val = 0.0f;
			//index = i + j * m_im + k * m_im * m_in;
			for (ir = i - gorder[0]; ir < i; ir++)
			{
				for (is = j - gorder[0]; is < j; is++)
				{
					for (it = k - gorder[0]; it < k; it++)
					{
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];
						val += cijk_value[index] * pNX[ir - i + gorder[0]] * pNY[is - j + gorder[0]] * pNZ[it - k + gorder[0]];
					}
				}
			}
		}
		// MARK[TODO] : restrict the density bound
		// rhoknew = clamp(rhoknew, rhomin, 1.f);
     	// if (gEflag[0][eid] & grid::Grid::Bitmask::mask_shellelement) rhonew = 1;
		return val;
		//return Heaviside(val);
	};

	gBitSAT<unsigned int> esat(_gbuf.eActiveBits, _gbuf.eActiveChunkSum);

	init_array(_gbuf.rho_e, float{ 0 }, n_gselements);

	coeff2density_kernel << <grid_size, block_size >> > (_gbuf.nword_ebits, min_Density, esat, _ereso, _gbuf.rho_e, calc_node, _gbuf.eidmap, _gbuf.eBitflag);
	cudaDeviceSynchronize();
	cuda_error_check;

	int coeffsize = n_cijk();
	float* coeffhost = new float[n_cijk()];
	gpu_manager_t::download_buf(coeffhost, _gbuf.coeffs, n_cijk());
	gpu_manager_t::pass_buf_to_matlab("coeff1", coeffhost, n_cijk());
	cuda_error_check;

	float* rhohost = new float[n_gselements];
	//cudaMemcpy(rhohost, _gbuf.rho_e, n_gselements, cudaMemcpyDeviceToHost);
	gpu_manager_t::download_buf(rhohost, _gbuf.rho_e, sizeof(float)* n_gselements);
	gpu_manager_t::pass_buf_to_matlab("rhoe1", rhohost, n_gselements);
	cuda_error_check;

	delete[] coeffhost;
	coeffhost = nullptr;
	delete[] rhohost;
	rhohost = nullptr;
}

template<typename Func>
__global__ void ddensity2dcoeff_kernel(int nebitword, gBitSAT<unsigned int> esat, int ereso, Func func, const int* eidmap) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nebitword) return;

	const unsigned int* ebit = esat._bitarray;
	const int* sat = esat._chunksat;

	unsigned int eword = ebit[tid];

	if (eword == 0) return;

	int eidoffset = sat[tid];
	int ewordoffset = 0;
	for (int j = 0; j < BitCount<unsigned int>::value; j++) {
		if (read_gbit(eword, j)) {
			int bid = tid * BitCount<unsigned int>::value + j;
			int bpos[3] = { bid % ereso, bid % (ereso * ereso) / ereso, bid / (ereso * ereso) };
			int eid = eidoffset + ewordoffset;
			if (eidmap != nullptr) eid = eidmap[eid];
			func(bid, eid);
			
			//g_sens[eid] = node_value;
			ewordoffset++;
		}
	}
}

void Grid::ddensity2dcoeff(void)
{
	if (_layer != 0) return;
	int order3 = m_iM * m_iM * m_iM;

	//int* coeffindex = (int*)getTempBuf1(sizeof(int) * n_gselements * order3);
	//init_array(coeffindex, int{ 0 }, n_gselements * order3);
	//cuda_error_check;

	//float* part2c_value = (float*)getTempBuf2(sizeof(float) * n_gselements * order3);
	//init_array(part2c_value, float{ 0 }, n_gselements * order3);
	//cuda_error_check;

	int* coeffindex;
	cudaMalloc(&coeffindex, sizeof(int) * n_gselements * order3);
	init_array(coeffindex, std::numeric_limits<int>::quiet_NaN(), n_gselements * order3);
	cuda_error_check;

	float* part2c_value;
	cudaMalloc(&part2c_value, sizeof(float) * n_gselements * order3);
	init_array(part2c_value, float{ 0 }, n_gselements * order3);
	cuda_error_check;

	float* cijk_value = _gbuf.coeffs;
	float* knotx_ = _gbuf.KnotSer[0];
	float* knoty_ = _gbuf.KnotSer[1];
	float* knotz_ = _gbuf.KnotSer[2];
	float* rholist = _gbuf.rho_e;
	float* rho_diff = _gbuf.g_sens;
	float* vol_diff = _gbuf.vol_sens;
	int* eidmap = _gbuf.eidmap;
	int* eflag = _gbuf.eBitflag;

	int ereso = _ereso;
	float eh = elementLength();
	float boxOrigin[3] = { _box[0][0], _box[0][1], _box[0][2] };

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, _gbuf.nword_ebits, 512);
	
	auto node_diff = [ = ] __device__(int id, int eid) {
		int xCoordi = id % ereso;
		int yCoordi = (id % (ereso * ereso)) / ereso;
		int zCoordi = id / (ereso * ereso);
		float pos[3] = { boxOrigin[0] + xCoordi * eh + 0.5 * eh,boxOrigin[1] + yCoordi * eh + 0.5 * eh, boxOrigin[2] + zCoordi * eh + 0.5 * eh };

		float val;
		int i, j, k, ir, it, is, index;

		float pNX[m_iM + 1];
		float pNY[m_iM + 1];
		float pNZ[m_iM + 1];

		// the first knot index (order ... order + partion + 1) in knotspan(1 ... 2*order + partion)
		i = (int)((pos[0] - gnBoundMin[0]) / gnstep[0]) + gorder[0];
		j = (int)((pos[1] - gnBoundMin[1]) / gnstep[1]) + gorder[0];
		k = (int)((pos[2] - gnBoundMin[2]) / gnstep[2]) + gorder[0];

		if ((i < gorder[0]) || (i > gnbasis[0]) || (j < gorder[0]) || (j > gnbasis[1]) || (k < gorder[0]) || (k > gnbasis[2]))
		{
			val = -0.2f;
		}
		else
		{
			SplineBasisX(pos[0], pNX);
			SplineBasisY(pos[1], pNY);
			SplineBasisZ(pos[2], pNZ);

			val = 0.0f;
			int count4coeff = 0;
			for (ir = i - gorder[0]; ir < i; ir++)
			{
				for (is = j - gorder[0]; is < j; is++)
				{
					for (it = k - gorder[0]; it < k; it++)
					{
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];
						coeffindex[order3 * eid + count4coeff] = index;
						if (eflag[eid] & grid::Grid::mask_shellelement)
						{
							part2c_value[order3 * eid + count4coeff] = 0;
						}
						else
						{
							val = /*Dirac(rholist[eid]) * */rho_diff[eid] * pNX[ir - i + gorder[0]] * pNY[is - j + gorder[0]] * pNZ[it - k + gorder[0]];
							part2c_value[order3 * eid + count4coeff] = val;

						}
						count4coeff++;
						//dc_tmp[index] = /* rho_diff[cur_element] */  pNX[ir - i + gorder[0]] * pNY[is - j + gorder[0]] * pNZ[it - k + gorder[0]];
					}
				}
			}
		}
	};

	gBitSAT<unsigned int> esat(_gbuf.eActiveBits, _gbuf.eActiveChunkSum);

	ddensity2dcoeff_kernel << <grid_size, block_size >> > (_gbuf.nword_ebits, esat, _ereso, node_diff, _gbuf.eidmap);
	cudaDeviceSynchronize();
	cuda_error_check;

	// Memcpy to cpu 
	int* coeffindexhost = new int[n_gselements * order3];
	float* part2c_valuehost = new float[n_gselements * order3];
	cudaMemcpy(coeffindexhost, coeffindex, sizeof(int)* n_gselements * order3, cudaMemcpyDeviceToHost);
	cudaMemcpy(part2c_valuehost, part2c_value, sizeof(float)* n_gselements * order3, cudaMemcpyDeviceToHost);
	cuda_error_check;

	int col_tmp, indexlist_tmp;
	float value_tmp;
	float* c_sens = new float[n_cijk()];
	std::fill(c_sens, c_sens + n_cijk(), 0.0f);
	// compute in cpu
	for (int i = 0; i < n_gselements; i++)
	{
		for (int j = 0; j < order3; j++)
		{
			indexlist_tmp = i * order3 + j;
			col_tmp = coeffindexhost[indexlist_tmp];
			value_tmp = part2c_valuehost[indexlist_tmp];
			
			if (col_tmp >= n_cijk())
			{
				std::cout << "\033[31m Invalid cijk index !!! \033[0m" << std::endl;
				break;
			}
			c_sens[col_tmp] += value_tmp;
		}
	}

	// upload to _gbuf.c_sens
	init_array(_gbuf.c_sens, float{ 0 }, n_cijk());
	cudaMemcpy(_gbuf.c_sens, c_sens, n_cijk() * sizeof(float), cudaMemcpyHostToDevice);
	cuda_error_check;
//
//#ifdef ENABLE_MATLAB
//	gpu_manager_t::pass_buf_to_matlab("coeffindex", coeffindexhost, n_gselements* order3);
//	gpu_manager_t::pass_buf_to_matlab("part2c", part2c_valuehost, n_gselements* order3);
//	gpu_manager_t::pass_buf_to_matlab("csens1", c_sens, n_cijk());
//#endif
	
	cudaFree(coeffindex);
	cudaFree(part2c_value);
	coeffindex = nullptr;
	part2c_value = nullptr;

	//clearBuf1();
	//clearBuf2();
	delete[] coeffindexhost;
	delete[] part2c_valuehost;
	delete[] c_sens;
	coeffindexhost = nullptr;
	part2c_valuehost = nullptr;
	c_sens = nullptr;
}

void Grid::ddensity2dcoeff_update(void)
{
	if (_layer != 0) return;
	// computation
	float* dc_tmp;
	cudaMalloc(&dc_tmp, sizeof(float) * n_cijk());
	init_array(dc_tmp, float{ 0 }, n_cijk());
	cuda_error_check;

	int order3 = m_iM * m_iM * m_iM;

	float* cijk_value = _gbuf.coeffs;
	float* knotx_ = _gbuf.KnotSer[0];
	float* knoty_ = _gbuf.KnotSer[1];
	float* knotz_ = _gbuf.KnotSer[2];
	float* rholist = _gbuf.rho_e;
	float* rho_diff = _gbuf.g_sens;
	float* vol_diff = _gbuf.vol_sens;
	int* eidmap = _gbuf.eidmap;
	int* eflag = _gbuf.eBitflag;

	int ereso = _ereso;
	float eh = elementLength();
	float boxOrigin[3] = { _box[0][0], _box[0][1], _box[0][2] };

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, _gbuf.nword_ebits, 512);

	auto node_diff = [=] __device__(int id, int eid) {
		int xCoordi = id % ereso;
		int yCoordi = (id % (ereso * ereso)) / ereso;
		int zCoordi = id / (ereso * ereso);
		float pos[3] = { boxOrigin[0] + xCoordi * eh + 0.5 * eh,boxOrigin[1] + yCoordi * eh + 0.5 * eh, boxOrigin[2] + zCoordi * eh + 0.5 * eh };

		float val;
		int i, j, k, ir, it, is, index;

		float pNX[m_iM + 1];
		float pNY[m_iM + 1];
		float pNZ[m_iM + 1];

		// the first knot index (order ... order + partion + 1) in knotspan(1 ... 2*order + partion)
		i = (int)((pos[0] - gnBoundMin[0]) / gnstep[0]) + gorder[0];
		j = (int)((pos[1] - gnBoundMin[1]) / gnstep[1]) + gorder[0];
		k = (int)((pos[2] - gnBoundMin[2]) / gnstep[2]) + gorder[0];

		if ((i < gorder[0]) || (i > gnbasis[0]) || (j < gorder[0]) || (j > gnbasis[1]) || (k < gorder[0]) || (k > gnbasis[2]))
		{
			val = -0.2f;
		}
		else
		{
			SplineBasisX(pos[0], pNX);
			SplineBasisY(pos[1], pNY);
			SplineBasisZ(pos[2], pNZ);

			val = 0.0f;
			int count4coeff = 0;
			//index = i + j * m_im + k * m_im * m_in;
			for (ir = i - gorder[0]; ir < i; ir++)
			{
				for (is = j - gorder[0]; is < j; is++)
				{
					for (it = k - gorder[0]; it < k; it++)
					{
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];
						if (eflag[eid] & grid::Grid::mask_shellelement)
						{
							dc_tmp[index] += 0;
						}
						else
						{
							val = /*Dirac(rholist[eid]) * */rho_diff[eid] * pNX[ir - i + gorder[0]] * pNY[is - j + gorder[0]] * pNZ[it - k + gorder[0]];
							dc_tmp[index] += val;
						}
						count4coeff++;
					}
				}
			}
		}
	};

	gBitSAT<unsigned int> esat(_gbuf.eActiveBits, _gbuf.eActiveChunkSum);

	ddensity2dcoeff_kernel << <grid_size, block_size >> > (_gbuf.nword_ebits, esat, _ereso, node_diff, _gbuf.eidmap);
	cudaDeviceSynchronize();
	cuda_error_check;

	float* dc_host = new float[n_cijk()];
	cudaMemcpy(dc_host, dc_tmp, sizeof(float)* n_cijk(), cudaMemcpyDeviceToHost);
	cuda_error_check;
	// upload to _gbuf.c_sens
	init_array(_gbuf.c_sens, float{ 0 }, n_cijk());
	cudaMemcpy(_gbuf.c_sens, dc_host, n_cijk() * sizeof(float), cudaMemcpyHostToDevice);
	cuda_error_check;

#ifdef ENABLE_MATLAB
	gpu_manager_t::pass_buf_to_matlab("csens2", dc_host, n_cijk());
#endif

	cudaFree(dc_tmp);
	dc_tmp = nullptr;
	delete[] dc_host;
	dc_host = nullptr;
}

void Grid::dvol2dcoeff(void)
{
	if (_layer != 0) return;
	// computation
	float* dc_tmp;
	cudaMalloc(&dc_tmp, sizeof(float) * n_cijk());
	init_array(dc_tmp, float{ 0 }, n_cijk());
	cuda_error_check;

	int order3 = m_iM * m_iM * m_iM;

	float* cijk_value = _gbuf.coeffs;
	float* knotx_ = _gbuf.KnotSer[0];
	float* knoty_ = _gbuf.KnotSer[1];
	float* knotz_ = _gbuf.KnotSer[2];
	float* rholist = _gbuf.rho_e;
	float* rho_diff = _gbuf.g_sens;
	float* vol_diff = _gbuf.vol_sens;
	int* eidmap = _gbuf.eidmap;
	int* eflag = _gbuf.eBitflag;

	int ereso = _ereso;
	float eh = elementLength();
	float boxOrigin[3] = { _box[0][0], _box[0][1], _box[0][2] };

	double dvol2drho = 1.0 / n_gselements;
	//std::cout << "-- [TEST] " << dvol2drho << std::endl;
	//std::cout << "-- [TEST] " << n_gselements << std::endl;

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, _gbuf.nword_ebits, 512);

	auto node1_diff = [=] __device__(int id, int eid) {
		int xCoordi = id % ereso;
		int yCoordi = (id % (ereso * ereso)) / ereso;
		int zCoordi = id / (ereso * ereso);
		float pos[3] = { boxOrigin[0] + xCoordi * eh + 0.5 * eh,boxOrigin[1] + yCoordi * eh + 0.5 * eh, boxOrigin[2] + zCoordi * eh + 0.5 * eh };

		float val;
		int i, j, k, ir, it, is, index;

		float pNX[m_iM + 1];
		float pNY[m_iM + 1];
		float pNZ[m_iM + 1];

		// the first knot index (order ... order + partion + 1) in knotspan(1 ... 2*order + partion)
		i = (int)((pos[0] - gnBoundMin[0]) / gnstep[0]) + gorder[0];
		j = (int)((pos[1] - gnBoundMin[1]) / gnstep[1]) + gorder[0];
		k = (int)((pos[2] - gnBoundMin[2]) / gnstep[2]) + gorder[0];

		if ((i < gorder[0]) || (i > gnbasis[0]) || (j < gorder[0]) || (j > gnbasis[1]) || (k < gorder[0]) || (k > gnbasis[2]))
		{
			val = -0.2f;
		}
		else
		{
			SplineBasisX(pos[0], pNX);
			SplineBasisY(pos[1], pNY);
			SplineBasisZ(pos[2], pNZ);

			val = 0.0f;
			int count4coeff = 0;
			//index = i + j * m_im + k * m_im * m_in;
			for (ir = i - gorder[0]; ir < i; ir++)
			{
				for (is = j - gorder[0]; is < j; is++)
				{
					for (it = k - gorder[0]; it < k; it++)
					{
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];
						if (eflag[eid] & grid::Grid::mask_shellelement)
						{
							dc_tmp[index] += 0;
						}
						else
						{
							val = vol_diff[eid] * pNX[ir - i + gorder[0]] * pNY[is - j + gorder[0]] * pNZ[it - k + gorder[0]];
							dc_tmp[index] += val;
						}
						count4coeff++;
					}
				}
			}
		}
	};

	gBitSAT<unsigned int> esat(_gbuf.eActiveBits, _gbuf.eActiveChunkSum);

	ddensity2dcoeff_kernel << <grid_size, block_size >> > (_gbuf.nword_ebits, esat, _ereso, node1_diff, _gbuf.eidmap);
	cudaDeviceSynchronize();
	cuda_error_check;

	float* dc_host = new float[n_cijk()];
	cudaMemcpy(dc_host, dc_tmp, sizeof(float) * n_cijk(), cudaMemcpyDeviceToHost);
	cuda_error_check;
	// upload to _gbuf.c_sens
	init_array(_gbuf.volc_sens, float{ 1 }, n_cijk());
	cudaMemcpy(_gbuf.volc_sens, dc_host, n_cijk() * sizeof(float), cudaMemcpyHostToDevice);
	cuda_error_check;

#ifdef ENABLE_MATLAB
	gpu_manager_t::pass_buf_to_matlab("volcsens2", dc_host, n_cijk());
#endif

	cudaFree(dc_tmp);
	dc_tmp = nullptr;
	delete[] dc_host;
	dc_host = nullptr;
}

template <class T>
struct CudaAllocator {
	using value_type = T;

	CudaAllocator() noexcept = default;

	template <class U>
	CudaAllocator(const CudaAllocator<U>&) noexcept {}

	T* allocate(std::size_t n) {
		T* ptr = nullptr;
		cudaMallocManaged(&ptr, n * sizeof(T));
		return ptr;
	}

	void deallocate(T* ptr, std::size_t) noexcept {
		cudaFree(ptr);
	}
};

template <class Func>
__global__ void parallel_for(int n, Func func) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = idx; i < n; i += stride) {
		func(i);
	}
}

__global__ void sinfa(float* a, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = idx; i < n; i += stride) {
		a[i] = sinf(i);
	}
}

void HierarchyGrid::lambdatest(void)
{
	int n = 10;
	//std::vector<float, CudaAllocator<float>> arr(n);
	//std::vector<float, CudaAllocator<float>> brr(n);
	std::vector<float> cpu(n);
	int nByte = sizeof(float) * n;
	float* a_h = (float*)malloc(nByte);
	float* res_h = (float*)malloc(nByte);
	float* a_d;
	cudaMallocHost((float**)&a_d, nByte);
	cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice);

	float* crr;
	cudaMalloc(&crr, sizeof(float) * n);
	init_array(crr, std::numeric_limits<float>::quiet_NaN(), n);
	float* drr;
	cudaMalloc(&drr, sizeof(float) * n);
	init_array(drr, std::numeric_limits<float>::quiet_NaN(), n);

	// add1 need memcpy to device
	const float constants[] = { 0.1f, 2.0f, 3.0f, 4.0f, 5.0f };
	float* add1 = new float[5]; 
	for (int i = 0; i < 5; ++i) {
		add1[i] = constants[i];
	}

	float add11 = 0.1f;
	
	auto calc_node = [ = ] __device__(int i) {
		//arr[i] = sinf(i);
		//brr[i] = cosf(i);
		crr[i] = sinf(i) + add11;
		drr[i] = sinf(i);

	};
	parallel_for << <32, 128 >> > (n, calc_node);

	//parallel_for << <32, 128 >> > (n, [arr = arr.data(), brr = brr.data()] __device__(int i) {
	//	arr[i] = sinf(i);
	//	brr[i] = cosf(i);
	//	//crr[i] = sinf(i);
	//});
	cudaDeviceSynchronize();
	cuda_error_check;
	std::cout << "cudalloc time " << std::endl;
	
	//float* arr_data = new float[nByte];
	//float* brr_data = new float[nByte];
	float* crr_data = new float[nByte];
	float* drr_data = new float[nByte];
	//cudaMemcpy(arr_data, arr.data(), nByte, cudaMemcpyDeviceToHost);
	//cudaMemcpy(brr_data, brr.data(), nByte, cudaMemcpyDeviceToHost);
	cudaMemcpy(crr_data, crr, nByte, cudaMemcpyDeviceToHost);
	cudaMemcpy(drr_data, drr, nByte, cudaMemcpyDeviceToHost);
	//gpu_manager_t::pass_buf_to_matlab("arr", arr_data, n);
	//gpu_manager_t::pass_buf_to_matlab("brr", brr_data, n);
	gpu_manager_t::pass_buf_to_matlab("crr", crr_data, n);
	gpu_manager_t::pass_buf_to_matlab("drr", drr_data, n);
	for (int i = 0; i < 10; ++i)
	{
		//std::cout << arr_data[i] << " , " << brr_data[i] << std::endl;
		std::cout << crr_data[i] << " , " << drr_data[i] << std::endl;
	}


	sinfa << <32, 128 >> > (a_d, n);
	cudaDeviceSynchronize();
	cuda_error_check;
	std::cout << "cpu time " << std::endl;
	cudaMemcpy(res_h, a_d, nByte, cudaMemcpyDeviceToHost);
	cuda_error_check;
	cudaFreeHost(a_d);
	cuda_error_check;
	cudaFree(crr);
	cuda_error_check;
	cudaFree(drr);
	cuda_error_check;
	//cudaFree(a_h);
	//cudaFree(res_h);
	delete[] res_h;
	delete[] a_h;
	//delete[] arr_data;
	//delete[] brr_data;
	delete[] crr_data;
	delete[] drr_data;
}

void Grid::compute_background_mcPoints_value(std::vector<float>& bgnode_x, std::vector<float>& bgnode_y, std::vector<float>& bgnode_z, std::vector<float>& spline_value, int mc_ereso, float beta)
{
	if (_layer != 0) return;

	// computation
	float* cijk_value = _gbuf.coeffs;
	float* knotx_ = _gbuf.KnotSer[0];
	float* knoty_ = _gbuf.KnotSer[1];
	float* knotz_ = _gbuf.KnotSer[2];
	float* rholist = _gbuf.rho_e;

	int ereso = mc_ereso + 2;
	int vreso = ereso + 1;

	float ehx = (_mbox[1][0] - _mbox[0][0]) / mc_ereso;
	float ehy = (_mbox[1][1] - _mbox[0][1]) / mc_ereso;
	float ehz = (_mbox[1][2] - _mbox[0][2]) / mc_ereso;
	float boxOrigin[3] = { _mbox[0][0], _mbox[0][1], _mbox[0][2]};
	float boxEnd[3] = { _mbox[1][0], _mbox[1][1], _mbox[1][2]};
	float boxOrispan[3] = { _mbox[0][0] - ehx, _mbox[0][1] - ehy, _mbox[0][2] - ehz};
	float min_Density = _min_density;
	float isosurface_value = _isosurface_value; // [MARK] may update to node_value
	
	float* nodex;
	float* nodey;
	float* nodez;
	float* node_value;
	int ereso3 = ereso * ereso * ereso;
	int vreso3 = vreso * vreso * vreso;
	float t = 0.25;
	cudaMalloc(&node_value, sizeof(float) * ereso3);
	init_array(node_value, float{ 0 }, ereso3);
	cudaMalloc(&nodex, sizeof(float) * ereso3);
	init_array(nodex, float{ 0 }, ereso3);
	cudaMalloc(&nodey, sizeof(float) * ereso3);
	init_array(nodey, float{ 0 }, ereso3);
	cudaMalloc(&nodez, sizeof(float) * ereso3);
	init_array(nodez, float{ 0 }, ereso3);
	cuda_error_check;

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, ereso3, 512);

	auto calc_node = [=] __device__(int id) {
		int xCoordi = id % ereso;
		int yCoordi = (id % (ereso * ereso)) / ereso;
		int zCoordi = id / (ereso * ereso);
		float pos[3] = { boxOrispan[0] + xCoordi * ehx + 0.5 * ehx, boxOrispan[1] + yCoordi * ehy + 0.5 * ehy, boxOrispan[2] + zCoordi * ehz + 0.5 * ehz };

		float val;
		int i, j, k, ir, it, is, index;

		float pNX[m_iM + 1];
		float pNY[m_iM + 1];
		float pNZ[m_iM + 1];

		// the first knot index (order ... order + partion + 1) in knotspan(1 ... 2*order + partion)
		i = (int)((pos[0] - gnBoundMin[0]) / gnstep[0]) + gorder[0];
		j = (int)((pos[1] - gnBoundMin[1]) / gnstep[1]) + gorder[0];
		k = (int)((pos[2] - gnBoundMin[2]) / gnstep[2]) + gorder[0];

		if ((i < gorder[0]) || (i > gnbasis[0]) || (j < gorder[0]) || (j > gnbasis[1]) || (k < gorder[0]) || (k > gnbasis[2]))
		{
			val = -0.2f;
		}
		else
		{
			SplineBasisX(pos[0], pNX);
			SplineBasisY(pos[1], pNY);
			SplineBasisZ(pos[2], pNZ);

			val = 0.0f;
			for (ir = i - gorder[0]; ir < i; ir++)
			{
				for (is = j - gorder[0]; is < j; is++)
				{
					for (it = k - gorder[0]; it < k; it++)
					{
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];
						val += cijk_value[index] * pNX[ir - i + gorder[0]] * pNY[is - j + gorder[0]] * pNZ[it - k + gorder[0]];
					}
				}
			}
		}

		if (pos[0] < boxOrigin[0] || pos[0] > boxEnd[0] || pos[1] < boxOrigin[1] || pos[1] > boxEnd[1] || pos[2] < boxOrigin[2] || pos[2] > boxEnd[2])
		{
			val = 0;
		}

		nodex[id] = pos[0];
		nodey[id] = pos[1];
		nodez[id] = pos[2];
		node_value[id] = val - 0.5;
		//node_value[id] = Heaviside(val, beta) - 0.5;
		//node_value[id] = cosf(2 * M_PI * t * pos[0]) + cosf(2 * M_PI * t * pos[1]) + cosf(2 * M_PI * t * pos[2]) + 0.1; // to verify the Marching cube
	};

	traverse_noret << < grid_size, block_size >> > (ereso3, calc_node);
	cudaDeviceSynchronize();
	cuda_error_check;

	//gBitSAT<unsigned int> esat(_gbuf.eActiveBits, _gbuf.eActiveChunkSum);

	//init_array(_gbuf.rho_e, float{ 0 }, n_gselements);

	//MCdensity_kernel << <grid_size, block_size >> > (_gbuf.nword_ebits, min_Density, esat, _ereso, _gbuf.rho_e, calc_node, _gbuf.eidmap, _gbuf.eBitflag);
	//cudaDeviceSynchronize();
	//cuda_error_check;
	spline_value.resize(ereso3);
	cudaMemcpy(spline_value.data(), node_value, ereso3 * sizeof(float), cudaMemcpyDeviceToHost);
	cuda_error_check;
	bgnode_x.resize(ereso3);
	cudaMemcpy(bgnode_x.data(), nodex, ereso3 * sizeof(float), cudaMemcpyDeviceToHost);
	cuda_error_check;
	bgnode_y.resize(ereso3);
	cudaMemcpy(bgnode_y.data(), nodey, ereso3 * sizeof(float), cudaMemcpyDeviceToHost);
	cuda_error_check;
	bgnode_z.resize(ereso3);
	cudaMemcpy(bgnode_z.data(), nodez, ereso3 * sizeof(float), cudaMemcpyDeviceToHost);
	cuda_error_check;

#ifdef ENABLE_MATLAB
	float* rhohost = new float[ereso3];
	cudaMemcpy(rhohost, node_value, ereso3 * sizeof(float), cudaMemcpyDeviceToHost);
	cuda_error_check;
	gpu_manager_t::pass_buf_to_matlab("rhomc", rhohost, ereso3);
	gpu_manager_t::pass_buf_to_matlab("bgnodex", bgnode_x.data(), ereso3);
	gpu_manager_t::pass_buf_to_matlab("bgnodey", bgnode_y.data(), ereso3);
	gpu_manager_t::pass_buf_to_matlab("bgnodez", bgnode_z.data(), ereso3);
#endif

	cudaFree(node_value);
	cudaFree(nodex);
	cudaFree(nodey);
	cudaFree(nodez);
	node_value = nullptr;
	nodex = nullptr;
	nodey = nullptr;
	nodez = nullptr;
	delete[] rhohost;
	rhohost = nullptr;
}

template<typename WeightRadius>
__global__ void filterSensitivity_kernel(int nebitword, gBitSAT<unsigned int> esat, int ereso, const float* g_sens, float* g_dst, float Rfilter, WeightRadius fr, const int* eidmap) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nebitword) return;

	const unsigned int* ebit = esat._bitarray;
	const int* sat = esat._chunksat;

	unsigned int eword = ebit[tid];

	if (eword == 0) return;

	float R2 = Rfilter * Rfilter;
	int eidoffset = sat[tid];
	int ewordoffset = 0;
	for (int j = 0; j < BitCount<unsigned int>::value; j++) {
		if (read_gbit(eword, j)) {
			int bid = tid * BitCount<unsigned int>::value + j;
			int bpos[3] = { bid % ereso, bid % (ereso*ereso) / ereso, bid / (ereso * ereso) };
			int eid = eidoffset + ewordoffset;
			// traverse its spatial neighbors
			int R = Rfilter + 0.5;
			int L = -R;

			int npos[3];

			float weightSum = 0;
			double g_sum = 0;

			// DEBUG

			for (int x = L; x <= R; x++) {

				int x2 = x * x;
				npos[0] = bpos[0] + x;
				if (npos[0] < 0 || npos[0] >= ereso) continue;

				for (int y = L; y <= R; y++) {

					int y2 = y * y;
					npos[1] = bpos[1] + y;
					if (npos[1] < 0 || npos[1] >= ereso) continue;

					for (int z = L; z <= R; z++) {

						int z2 = z * z;
						npos[2] = bpos[2] + z;

						// spatial neighbor position
						if (npos[2] < 0 || npos[2] >= ereso) continue;

						float r2 = x2 + y2 + z2;
						if (r2 > R2) continue;

						// spatial neighbor bit id
						int n_bid = npos[0] + npos[1] * ereso + npos[2] * ereso * ereso;

						// spatial neighbor element id
						int n_eid = esat(n_bid);

						// if neighbor element is not valid
						if (n_eid == -1) continue;

						if (eidmap != nullptr) { n_eid = eidmap[n_eid]; }

						// weighted sum
						float w = fr(sqrtf(r2 / R2));

						g_sum += w * g_sens[n_eid];
						weightSum += w;

					}
				}
			} // traverse all spatial neighbor elements

			g_sum /= weightSum;

			if (eidmap != nullptr) eid = eidmap[eid];

			g_dst[eid] = g_sum;

			ewordoffset++;
		}
	}

	
}

void Grid::filterSensitivity(double radii)
{
	if (_layer != 0) return;
	
	size_t grid_size, block_size;

	make_kernel_param(&grid_size, &block_size, _gbuf.nword_ebits, 512);

	auto fr = [=] __device__(float r) {
		float r2 = r * r;
		return 1 - 6 * r2 + 8 * r2 * r - 3 * r2 * r2;
	};

	gBitSAT<unsigned int> esat(_gbuf.eActiveBits, _gbuf.eActiveChunkSum);

	float* g_sens_copy = (float*)getTempBuf(sizeof(float)* n_gselements);

	cudaMemcpy(g_sens_copy, _gbuf.g_sens, sizeof(float) * n_gselements, cudaMemcpyDeviceToDevice);

	init_array(_gbuf.g_sens, float{ 0 }, n_gselements);

	filterSensitivity_kernel << <grid_size, block_size >> > (_gbuf.nword_ebits, esat, _ereso, g_sens_copy, _gbuf.g_sens, radii, fr, _gbuf.eidmap);

	cudaDeviceSynchronize();

	clearBuf();

	cuda_error_check;
}

void Grid::filterVolSensitivity(double radii)
{
	if (_layer != 0) return;

	size_t grid_size, block_size;

	make_kernel_param(&grid_size, &block_size, _gbuf.nword_ebits, 512);

	auto fr = [=] __device__(float r) {
		float r2 = r * r;
		return 1 - 6 * r2 + 8 * r2 * r - 3 * r2 * r2;
	};

	gBitSAT<unsigned int> esat(_gbuf.eActiveBits, _gbuf.eActiveChunkSum);

	float* vol_sens_copy = (float*)getTempBuf(sizeof(float) * n_gselements);

	cudaMemcpy(vol_sens_copy, _gbuf.vol_sens, sizeof(float) * n_gselements, cudaMemcpyDeviceToDevice);

	init_array(_gbuf.vol_sens, float{ 0 }, n_gselements);

	filterSensitivity_kernel << <grid_size, block_size >> > (_gbuf.nword_ebits, esat, _ereso, vol_sens_copy, _gbuf.vol_sens, radii, fr, _gbuf.eidmap);

	cudaDeviceSynchronize();

	clearBuf();

	cuda_error_check;
}

__global__ void applyK_OTFA_kernel(int nv, devArray_t<double*, 3> u, devArray_t<double*, 3> f, float* rholist, bool use_support = true) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//int mode = gmode[0];

	__shared__ double KE[24][24];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	int vid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= nv) return;

	double KeU[3] = { 0. };

	double* pU[3] = { u[0],u[1],u[2] };

	float power = power_penalty[0];

	bool vifix = false;

	int viflag;

	if (!isValidNode(vid)) goto __writef;

	if (use_support) {
		viflag = gVflag[0][vid];
		vifix = viflag & grid::Grid::Bitmask::mask_supportnodes;
		if (vifix) {
			goto __writef;
		}
	}

	for (int e = 0; e < 8; e++) {

		int vi = 7 - e;

		int eid = gV2E[e][vid];

		if (eid == -1) continue;

		double penalty = powf(rholist[eid], power);

		for (int vj = 0; vj < 8; vj++) {
			int vjpos[3] = {
				vj % 2 + e % 2,
				vj % 4 / 2 + e % 4 / 2,
				vj / 4 + e / 4
			};
			int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
			int vj_vid = gV2V[vj_lid][vid];
			if (vj_vid == -1) continue;

			double u_vj[3] = { pU[0][vj_vid],pU[1][vj_vid],pU[2][vj_vid] };

			if (use_support) {
				int vjflag = gVflag[0][vj_vid];
				if (vjflag & grid::Grid::Bitmask::mask_supportnodes) {
					u_vj[0] = 0; u_vj[1] = 0; u_vj[2] = 0;
				}
			}

			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < 3; j++) {
					KeU[k] += penalty * KE[k + vi * 3][j + vj * 3] * u_vj[j];
				}
			}
		}
	}

__writef:
	for (int i = 0; i < 3; i++) {
		if (use_support && vifix) {
			KeU[i] = gU[i][vid];
		}
		f[i][vid] = KeU[i];
	}
}

void Grid::applyK(double* u[3], double* f[3])
{
	use_grid();
	if (_layer == 0) {
		devArray_t<double*, 3> ulist{ u[0],u[1],u[2] };
		devArray_t<double*, 3> flist{ f[0],f[1],f[2] };
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		applyK_OTFA_kernel << <grid_size, block_size >> > (n_gsvertices, ulist, flist, _gbuf.rho_e);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	
}

void grid::Grid::resetDirchlet(double* v_dev[3])
{
	use_grid();
	if (_layer == 0) {
		devArray_t<double*, 3> vlist{ v_dev[0],v_dev[1],v_dev[2] };
		auto kernel = [=] __device__(int tid) {
			int flag = gVflag[0][tid];
			if ((flag & Grid::Bitmask::mask_supportnodes) && !(flag & Grid::Bitmask::mask_invalid)) {
				for (int i = 0; i < 3; i++) {
					vlist[i][tid] = 0;
				}
			}
		};	
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		traverse_noret << <grid_size, block_size >> > (n_gsvertices, kernel);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

__global__ void cubeGridSetSolidVertices_kernel(int ereso, const unsigned int* ebits, unsigned int* vbits) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int ne = ereso * ereso * ereso;
	int vreso = ereso + 1;
	int nv = vreso * vreso * vreso;

	if (tid >= nv) return;

	int vpos[3] = { tid % vreso, tid % (vreso * vreso) / vreso, tid / (vreso * vreso) };

	bool has_valid = false;
	for (int i = 0; i < 8; i++) {
		int epos[3] = { vpos[0] + i % 2 - 1, vpos[1] + (i % 4 / 2) - 1, vpos[2] + i / 4 - 1 };
		if (
			epos[0] >= ereso || epos[1] >= ereso || epos[2] >= ereso ||
			epos[0] < 0 || epos[1] < 0 || epos[2] < 0
			) continue;
		int eid = epos[0] + epos[1] * ereso + epos[2] * ereso * ereso;
		if (read_gbit(ebits, eid)) {
			has_valid = true;
			break;
		}
	}

	if (has_valid) {
		//set_gbit(vbits, tid);
		atomic_set_gbit(vbits, tid);
	}
}

void grid::cubeGridSetSolidVertices_g(int reso, const std::vector<unsigned int>& solid_ebit, std::vector<unsigned int>& solid_vbit)
{
	int vreso = reso + 1;
	int nv = pow(vreso, 3);
	int n_vword = snippet::Round< BitCount<unsigned int>::value >(nv) / BitCount<unsigned int>::value;

	unsigned int* g_ebits, *g_vbits;
	cudaMalloc(&g_ebits, sizeof(unsigned int)*solid_ebit.size());
	cudaMalloc(&g_vbits, n_vword * sizeof(unsigned int));

	cudaMemcpy(g_ebits, solid_ebit.data(), sizeof(unsigned int) * solid_ebit.size(), cudaMemcpyHostToDevice);
	init_array(g_vbits, (unsigned int)(0), n_vword);

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv, 512);

	cubeGridSetSolidVertices_kernel << <grid_size, block_size >> > (reso, g_ebits, g_vbits);
	cudaDeviceSynchronize();
	cuda_error_check;

	solid_vbit.resize(n_vword, 0);
	cudaMemcpy(solid_vbit.data(), g_vbits, sizeof(unsigned int) * n_vword, cudaMemcpyDeviceToHost);

	cudaFree(g_ebits);
	g_ebits = nullptr;
	cudaFree(g_vbits);
	g_vbits = nullptr;
}

__global__ void setSolidElementFromFineGrid_kernel(int finereso, const unsigned int* ebitsfine, unsigned int* ebitscoarse) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	int nvfine = finereso * finereso * finereso;

	int coarsereso = finereso >> 1;

	if (tid >= nvfine) return;

	// solid fine elements encountered
	if (read_gbit(ebitsfine, tid)) {
		// fine coarse element position
		int epos[3] = { tid %finereso, tid % (finereso*finereso) / finereso, tid / (finereso*finereso) };
		// coarse element position
		for (int i = 0; i < 3; i++) epos[i] >>= 1;
		// coarse element id
		int vcoarse = epos[0] + epos[1] * coarsereso + epos[2] * coarsereso * coarsereso;
		// set solid bit flag
		atomic_set_gbit(ebitscoarse, vcoarse);
	}
}

void grid::setSolidElementFromFineGrid_g(int finereso, const std::vector<unsigned int>& ebits_fine, std::vector<unsigned int>& ebits_coarse)
{
	int nefine = pow(finereso, 3);
	int necoarse = pow(finereso / 2, 3);
	int nword_coarse = snippet::Round<BitCount<unsigned int>::value>(necoarse) / BitCount<unsigned int>::value;

	unsigned int* g_fine, *g_coarse;
	cudaMalloc(&g_fine, snippet::Round<BitCount<unsigned int>::value>(nefine) / 8);
	cudaMalloc(&g_coarse, snippet::Round<BitCount<unsigned int>::value>(necoarse) / 8);

	cudaMemcpy(g_fine, ebits_fine.data(), snippet::Round<BitCount<unsigned int>::value>(nefine) / 8, cudaMemcpyHostToDevice);
	init_array(g_coarse, (unsigned int)(0), nword_coarse);

	int ne_fine = pow(finereso, 3);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, ne_fine, 512);
	setSolidElementFromFineGrid_kernel << <grid_size, block_size >> > (finereso, g_fine, g_coarse);
	cudaDeviceSynchronize();
	cuda_error_check;

	ebits_coarse.resize(nword_coarse);

	cudaMemcpy(ebits_coarse.data(), g_coarse, sizeof(unsigned int) * nword_coarse, cudaMemcpyDeviceToHost);
	
	cudaFree(g_fine);
	cudaFree(g_coarse);
}

__global__ void wordReverse_kernel(size_t nword, unsigned int* g_words) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= nword) return;

	unsigned int word = g_words[tid];

	// reverse the word
	g_words[tid] = __brev(word);
}

void grid::wordReverse_g(size_t nword, unsigned int* wordlist)
{
	unsigned int* g_words;
	cudaMalloc(&g_words, nword * sizeof(unsigned int));
	cudaMemcpy(g_words, wordlist, nword * sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nword, 512);
	wordReverse_kernel << <grid_size, block_size >> > (nword, g_words);
	cudaDeviceSynchronize();
	cuda_error_check;

	cudaMemcpy(wordlist, g_words, sizeof(unsigned int)*nword, cudaMemcpyDeviceToHost);

	cudaFree(g_words);
}

__global__ void setV2VCoarse_kernel(
	int nvword,
	int skip, int vresofine, gBitSAT<unsigned int> vsatfine,
	gBitSAT<unsigned int> vsatcoarse, devArray_t<int*, 8> v2vcoarse
) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= nvword) return;

	int vresocoarse = ((vresofine - 1) >> skip) + 1;
	int vresofine2 = vresofine * vresofine;
	int vresocoarse2 = vresocoarse * vresocoarse;
	int nvbfine = vresofine2 * vresofine;

	unsigned int coarseRatio = (1 << skip) ;
	double cr3 = coarseRatio * coarseRatio * coarseRatio;
	auto word = vsatfine._bitarray[tid];
	if (word == 0) return;
	for (int ji = 0; ji < grid::BitCount<unsigned int>::value; ji++) {
		if (!read_gbit(word, ji)) continue;
		int vbidfine = tid * grid::BitCount<unsigned int>::value + ji;
		if (vbidfine >= nvbfine) continue;
		int vposfine[3] = { vbidfine % vresofine, vbidfine / vresofine % vresofine, vbidfine / vresofine2 };
		int vposInE[3] = { (vposfine[0] % coarseRatio), (vposfine[1] % coarseRatio), (vposfine[2] % coarseRatio) };
		int vidfine = vsatfine[vbidfine];
		// traverse coarse element vertex
		for (int i = 0; i < 8; i++) {
			int vcoarsepos[3] = { i % 2 * coarseRatio, i % 4 / 2 * coarseRatio, i / 4 * coarseRatio };
			int wpos[3] = { abs(vcoarsepos[0] - vposInE[0]),abs(vcoarsepos[1] - vposInE[1]),abs(vcoarsepos[2] - vposInE[2]) };
			int vidcoarse = -1;
			if (wpos[0] < coarseRatio && wpos[1] < coarseRatio && wpos[2] < coarseRatio) {
				int vcoarsebitpos[3] = {
					(vposfine[0] - vposInE[0]) / coarseRatio + i % 2 ,
					(vposfine[1] - vposInE[1]) / coarseRatio + i % 4 / 2,
					(vposfine[2] - vposInE[2]) / coarseRatio + i / 4
				};
				int vcoarsebitid = vcoarsebitpos[0] + vcoarsebitpos[1] * vresocoarse + vcoarsebitpos[2] * vresocoarse2;
				vidcoarse = vsatcoarse(vcoarsebitid);
			}
			v2vcoarse[i][vidfine] = vidcoarse;
			//double weight = (coarseRatio - wpos[0]) * (coarseRatio - wpos[1]) * (coarseRatio - wpos[2]) / cr3;
		}
	}


}


void Grid::setV2VCoarse_g(
	int skip, int vresofine,
	grid::BitSAT<unsigned int>& vsatfine, grid::BitSAT<unsigned int>& vsatcoarse,
	int* v2vcoarse[8]
) {
	int nvword = vsatfine._bitArray.size();

	unsigned int* g_vbfine, *g_vbcoarse;
	int *g_vbfinesat, *g_vbcoarsesat;
	int* gv2vcoarse[8];

	for (int i = 0; i < 8; i++) {
		cudaMalloc(&gv2vcoarse[i], sizeof(int) * vsatfine.total());
		init_array(gv2vcoarse[i], -1, vsatfine.total());
	}

	// copy vertex SAT from host to device
	gBitSAT<unsigned int> g_vsatfine(vsatfine._bitArray, vsatfine._chunkSat);
	gBitSAT<unsigned int> g_vsatcoarse(vsatcoarse._bitArray, vsatcoarse._chunkSat);

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nvword, 512);

	devArray_t<int*, 8> gv2vc_dev;
	for (int i = 0; i < 8; i++) gv2vc_dev[i] = gv2vcoarse[i];
	
	setV2VCoarse_kernel << <grid_size, block_size >> > (nvword, skip, vresofine, g_vsatfine, g_vsatcoarse, gv2vc_dev);
	cudaDeviceSynchronize();
	cuda_error_check;

	for (int i = 0; i < 8; i++) {
		cudaMemcpy(v2vcoarse[i], gv2vcoarse[i], sizeof(int) * vsatfine.total(), cudaMemcpyDeviceToHost);
	}

	g_vsatfine.destroy();
	g_vsatcoarse.destroy();

	for (int i = 0; i < 8; i++) cudaFree(gv2vcoarse[i]);

	cuda_error_check;
}

__global__ void setV2VFine_kernel(int nvcoarseword,
	int skip, int vresocoarse, gBitSAT<unsigned int> vsatfine,
	gBitSAT<unsigned int> vsatcoarse, devArray_t<int*, 27> v2vfine

) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= nvcoarseword) return;

	int ncoarse = 1 << skip;

	int vresocoarse2 = vresocoarse * vresocoarse;

	int vresofine = (vresocoarse - 1) * ncoarse + 1;

	int nvbit = vresocoarse * vresocoarse * vresocoarse;

	unsigned int coarseword = vsatcoarse._bitarray[tid];

	if (coarseword == 0) return;

	for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
		if (!read_gbit(coarseword, ji)) continue;
		int vcoarsebid = tid * BitCount<unsigned int>::value + ji;

		if (vcoarsebid >= nvbit) continue;

		int vidcoarse = vsatcoarse[vcoarsebid];

		int vcoarsepos[3] = { vcoarsebid % vresocoarse, vcoarsebid / vresocoarse % vresocoarse, vcoarsebid / vresocoarse2 };

		if (vcoarsepos[0] < 0 || vcoarsepos[0] >= vresocoarse ||
			vcoarsepos[1] < 0 || vcoarsepos[1] >= vresocoarse ||
			vcoarsepos[2] < 0 || vcoarsepos[2] >= vresocoarse
			) {
			continue;
		}

		int vfinepos[3] = { vcoarsepos[0] * ncoarse, vcoarsepos[1] * ncoarse, vcoarsepos[2] * ncoarse };

		for (int k = 0; k < 27; k++) {
			int vfineneipos[3] = { vfinepos[0] + k % 3 - 1, vfinepos[1] + (k / 3 % 3) - 1, vfinepos[2] + k / 9 - 1 };

			if (vfineneipos[0] < 0 || vfineneipos[0] >= vresofine ||
				vfineneipos[1] < 0 || vfineneipos[1] >= vresofine ||
				vfineneipos[2] < 0 || vfineneipos[2] >= vresofine
				) {
				continue;
			}

			int vfinenei_id = vfineneipos[0] + vfineneipos[1] * vresofine + vfineneipos[2] * vresofine * vresofine;

			//if (!read_gbit(vsatcoarse._bitarray, vfinenei_id)) continue;
			//int vidfine = vsatcoarse[vfinenei_id];
			int vidfine = vsatfine(vfinenei_id);

			v2vfine[k][vidcoarse] = vidfine;
		}
	}
}

void Grid::setV2VFine_g(
	int skip, int vresocoarse,
	grid::BitSAT<unsigned int>& vsatfine,
	grid::BitSAT<unsigned int>& vsatcoarse,
	int* v2vfine[27]
) {
	if (skip != 1) {
		printf("\033[31mV2VFine do not support non-dyadic coarse\033[0m\n");
		exit(-1);
	}

	int nvfineword = vsatfine._bitArray.size();
	int nvcoarseword = vsatcoarse._bitArray.size();

	unsigned int* g_vbfine, *g_vbcoarse;
	int *g_vbfinesat, *g_vbcoarsesat;

	devArray_t<int*, 27> gv2vfinelist;

	// copy host SAT to device
	gBitSAT<unsigned int> g_vsatfine(vsatfine._bitArray, vsatfine._chunkSat);
	gBitSAT<unsigned int> g_vsatcoarse(vsatcoarse._bitArray, vsatcoarse._chunkSat);

	for (int i = 0; i < 27; i++) {
		cudaMalloc(&gv2vfinelist[i], sizeof(int) * vsatcoarse.total());
		init_array(gv2vfinelist[i], -1, vsatcoarse.total());
	}

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nvcoarseword, 512);

	setV2VFine_kernel << <grid_size, block_size >> > (nvcoarseword, skip, vresocoarse, g_vsatfine, g_vsatcoarse, gv2vfinelist);
	cudaDeviceSynchronize();
	cuda_error_check;

	for (int i = 0; i < 27; i++) {
		cudaMemcpy(v2vfine[i], gv2vfinelist[i], sizeof(int) * vsatcoarse.total(), cudaMemcpyDeviceToHost);
	}

	g_vsatfine.destroy();
	g_vsatcoarse.destroy();
	gv2vfinelist.destroy();

	cuda_error_check;
}

__global__ void setV2VFineC_kernel(int nvcoarseword,int vresocoarse, gBitSAT<unsigned int> vsatfine2, gBitSAT<unsigned int> vsatcoarse, devArray_t<int*, 64> g_v2vfinec) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nvcoarseword) return;
	
	int vresofinefine = (vresocoarse - 1) * 4 + 1;
	int vresofinefine2 = vresofinefine * vresofinefine;

	unsigned int word = vsatcoarse._bitarray[tid];

	int nvbit = vresocoarse * vresocoarse * vresocoarse;

	if (word == 0) return;

	for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
		if (!read_gbit(word, ji)) continue;
		int vcoarsebid = tid * BitCount<unsigned int>::value + ji;
		if (vcoarsebid >= nvbit) continue;
		int vidcoarse = vsatcoarse[vcoarsebid];
		
		int vfinepos[3] = { vcoarsebid % vresocoarse * 4, vcoarsebid / vresocoarse % vresocoarse * 4, vcoarsebid / (vresocoarse * vresocoarse) * 4 };
		if (vfinepos[0] >= vresofinefine || vfinepos[1] >= vresofinefine || vfinepos[2] >= vresofinefine) continue;

		for (int k = 0; k < 64; k++) {
			int vfcpos[3] = { k % 4 * 2 + vfinepos[0] - 3, k / 4 % 4 * 2 + vfinepos[1] - 3, k / 16 * 2 + vfinepos[2] - 3 };

			if (vfcpos[0] < 0 || vfcpos[0] >= vresofinefine ||
				vfcpos[1] < 0 || vfcpos[1] >= vresofinefine ||
				vfcpos[2] < 0 || vfcpos[2] >= vresofinefine) {
				continue;
			}

			int vfcid = vfcpos[0] + vfcpos[1] * vresofinefine + vfcpos[2] * vresofinefine2;
			int vidfc = vsatfine2(vfcid);
			g_v2vfinec[k][vidcoarse] = vidfc;
		}
	}
}

void Grid::setV2VFineC_g(int vresocoarse, grid::BitSAT<unsigned int>& vsatfine2, grid::BitSAT<unsigned int>& vsatcoarse, int* v2vfinec[64])
{
	// copy host SAT to device
	gBitSAT<unsigned int> satfine2(vsatfine2._bitArray, vsatfine2._chunkSat);
	gBitSAT<unsigned int> satcoarse(vsatcoarse._bitArray, vsatcoarse._chunkSat);

	int nvcoarseword = vsatcoarse._bitArray.size();

	// allocate v2vfine buffer on device
	devArray_t<int*, 64> gv2vfc;
	for (int i = 0; i < 64; i++) {
		cudaMalloc(&gv2vfc[i], sizeof(int) * vsatcoarse.total());
		init_array(gv2vfc[i], -1, vsatcoarse.total());
	}

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nvcoarseword, 512);

	// lauch kernel
	setV2VFineC_kernel << <grid_size, block_size >> > (nvcoarseword, vresocoarse, satfine2, satcoarse, gv2vfc);
	cudaDeviceSynchronize();
	cuda_error_check;

	// copy result from device to host
	for (int i = 0; i < 64; i++) {
		cudaMemcpy(v2vfinec[i], gv2vfc[i], sizeof(int)* vsatcoarse.total(), cudaMemcpyDeviceToHost);
	}

	// free GPU memory
	satfine2.destroy();
	satcoarse.destroy();
	gv2vfc.destroy();
	cuda_error_check;
}

__global__ void setV2E_kernel(int nvword, int nvvalid, int nevalid, int vreso, gBitSAT<unsigned int> vrtsat, gBitSAT<unsigned int> elsat, devArray_t<int*, 8> v2elist) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ereso = vreso - 1;
	if (tid >= nvword) return;

	int nvbit = vreso * vreso * vreso;

	unsigned int vbitword = vrtsat._bitarray[tid];
	if (vbitword == 0) return;

	for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
		if (!read_gbit(vbitword, ji)) continue;
		int vbitid = tid * BitCount<unsigned int>::value + ji;
		if (vbitid >= nvbit) continue;
		int vid = vrtsat[vbitid];
		int vpos[3] = { vbitid % vreso, vbitid / vreso % vreso, vbitid / (vreso*vreso) };
		for (int k = 0; k < 8; k++) {
			int epos[3] = { vpos[0] + k % 2 - 1,vpos[1] + k / 2 % 2 - 1,vpos[2] + k / 4 - 1 };

			if (epos[0] < 0 || epos[0] >= ereso ||
				epos[1] < 0 || epos[1] >= ereso ||
				epos[2] < 0 || epos[2] >= ereso) {
				continue;
			}

			int ebitid = epos[0] + epos[1] * ereso + epos[2] * ereso * ereso;

			int eid = elsat(ebitid);

			v2elist[k][vid] = eid;
		}
	}
}

void Grid::setV2E_g(int vreso, BitSAT<unsigned int>& vrtsat, BitSAT<unsigned int>& elsat, int* v2e[8])
{
	gBitSAT<unsigned int> g_vsat(vrtsat._bitArray, vrtsat._chunkSat);
	gBitSAT<unsigned int> g_esat(elsat._bitArray, elsat._chunkSat);
	devArray_t<int*, 8> g_v2e;
	for (int i = 0; i < 8; i++) {
		cudaMalloc(&g_v2e[i], vrtsat.total() * sizeof(int));
		init_array(g_v2e[i], -1, vrtsat.total());
	}
	cuda_error_check;

	int n_vword = vrtsat._bitArray.size();

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_vword, 512);
	setV2E_kernel << <grid_size, block_size >> > (n_vword, vrtsat.total(), elsat.total(), vreso, g_vsat, g_esat, g_v2e);
	cudaDeviceSynchronize();
	cuda_error_check;

	for (int i = 0; i < 8; i++) {
		cudaMemcpy(v2e[i], g_v2e[i], sizeof(int) * vrtsat.total(), cudaMemcpyDeviceToHost);
	}
	
	g_v2e.destroy();
	g_vsat.destroy();
	g_esat.destroy();

	cuda_error_check;
}

__global__ void setV2V_kernel(int n_vword, int vreso, gBitSAT<unsigned int> vrtsat, devArray_t<int*, 27> g_v2v) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= n_vword) return;

	int nvbit = vreso * vreso * vreso;

	unsigned int vbitword = vrtsat._bitarray[tid];
	if (vbitword == 0) return;

	for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
		if (!read_gbit(vbitword, ji)) continue;
		int vibid = tid * BitCount<unsigned int>::value + ji;
		if (vibid >= nvbit) continue;
		int viid = vrtsat[vibid];
		int vipos[3] = { vibid % vreso, vibid / vreso % vreso, vibid / (vreso * vreso) };

		for (int k = 0; k < 27; k++) {
			int vjpos[3] = { vipos[0] + k % 3 - 1,vipos[1] + k / 3 % 3 - 1,vipos[2] + k / 9 - 1 };

			if (vjpos[0] < 0 || vjpos[0] >= vreso ||
				vjpos[1] < 0 || vjpos[1] >= vreso ||
				vjpos[2] < 0 || vjpos[2] >= vreso) {
				continue;
			}

			int vjbid = vjpos[0] + vjpos[1] * vreso + vjpos[2] * vreso * vreso;

			int vjid = vrtsat(vjbid);

			g_v2v[k][viid] = vjid;
		}
	}
}

void Grid::setV2V_g(int vreso, BitSAT<unsigned int>& vrtsat, int* v2v[27])
{
	int n_vword = vrtsat._bitArray.size();

	devArray_t<int*, 27> g_v2v;
	for (int i = 0; i < 27; i++) {
		cudaMalloc(&g_v2v[i], sizeof(int) * vrtsat.total());
		init_array(g_v2v[i], -1, vrtsat.total());
	}

	gBitSAT<unsigned int> g_vrtsat(vrtsat._bitArray, vrtsat._chunkSat);

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_vword, 512);
	setV2V_kernel << <grid_size, block_size >> > (n_vword, vreso, g_vrtsat, g_v2v);
	cudaDeviceSynchronize();
	cuda_error_check;

	for (int i = 0; i < 27; i++) {
		cudaMemcpy(v2v[i], g_v2v[i], sizeof(int) * vrtsat.total(), cudaMemcpyDeviceToHost);
	}

	g_v2v.destroy();
	g_vrtsat.destroy();
	cuda_error_check;
}

void Grid::init_rho(double rh0)
{
	init_array(_gbuf.rho_e, float(rh0), n_rho());
}

void Grid::init_coeff(double coeff)
{
	int coeff_size = n_im * n_in * n_il;
	std::cout << "coeff_size: " << coeff_size << "( " << n_im << ", " << n_in << ", " << n_il << " )" << std::endl;
	init_array(_gbuf.coeffs, float(coeff), coeff_size);
}

void Grid::init_volsens(double ratio)
{
	init_array(_gbuf.vol_sens, float(ratio), n_rho());
}

void Grid::init_volcsens(double ratio)
{
	init_array(_gbuf.volc_sens, float(ratio), n_cijk());
}

void Grid::init_sscsens(double sens)
{
	init_array(_gbuf.ssc_sens, float(sens), n_surf_points());
}

void Grid::init_dripcsens(double sens)
{
	init_array(_gbuf.dripc_sens, float(sens), n_surf_points());
}

//void TestSuit::setDensity(float* newrho)
//{
//	cudaMemcpy(grids[0]->getRho(), newrho, sizeof(float) * grids[0]->n_rho(), cudaMemcpyDeviceToDevice);
//	cuda_error_check;
//}

void grid::Grid::init_rholist(float* rh0)
{
	float* tmp_buf;
	cudaMalloc(&tmp_buf, n_rho() * sizeof(float));
	cudaMemcpy(tmp_buf, rh0, n_rho() * sizeof(float), cudaMemcpyHostToDevice);
	init_arraylist(_gbuf.rho_e, tmp_buf, n_rho());
	cudaFree(tmp_buf);
}

void grid::Grid::init_coefflist(float* coeff)
{
	float* tmp_buf;
	cudaMalloc(&tmp_buf, n_cijk() * sizeof(float));
	cudaMemcpy(tmp_buf, coeff, n_cijk() * sizeof(float), cudaMemcpyHostToDevice);
	init_arraylist(_gbuf.coeffs, tmp_buf, n_cijk());
	cudaFree(tmp_buf);
}

__global__ void computeNodePos_kernel(int n_word, int vreso, gBitSAT<unsigned int> vrtsat, devArray_t<double, 3> orig, double eh, devArray_t<double*, 3> pos) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= n_word) return;
	
	auto word = vrtsat._bitarray[tid];

	for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
		if (!read_gbit(word, ji)) continue;
		int vbid = tid * BitCount<unsigned int>::value + ji;
		int vpos[3] = { vbid % vreso, vbid / vreso % vreso, vbid / vreso / vreso };
		int vid = vrtsat[vbid];
		for (int k = 0; k < 3; k++) {
			pos[k][vid] = orig[k] + eh * vpos[k];
		}
	}
}

void HierarchyGrid::getNodePos(Grid& g, std::vector<double>& p3host)
{
	int lay = g._layer;
	auto& vsat = vrtsatlist[lay];
	if (vsat.total() != g.n_vertices) printf("-- error on get node pos\n");
	int vreso = g._ereso + 1;
	devArray_t<double*, 3> p;
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&p[i], sizeof(double)*g.n_vertices);
	}

	double eh = elementLength() * (1 << g._layer);
	devArray_t<double, 3> orig;
	for (int i = 0; i < 3; i++) orig[i] = _gridlayer[0]->_box[0][i];

	int nword = vsat._bitArray.size();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nword, 512);

	gBitSAT<unsigned int> vrtsat(vsat._bitArray, vsat._chunkSat);

	computeNodePos_kernel << <grid_size, block_size >> > (nword, vreso, vrtsat, orig, eh, p);
	cudaDeviceSynchronize();
	cuda_error_check;


	p3host.resize(g.n_gsvertices * 3);
	double* gspos;
	cudaMalloc(&gspos, sizeof(double) * g.n_gsvertices);
	for (int i = 0; i < 3; i++) {
		std::vector<double> hostpos;
		init_array(gspos, std::numeric_limits<double>::quiet_NaN(), g.n_gsvertices);
		int* vidmap = g._gbuf.vidmap;
		auto reorder = [=] __device__(int tid) {
			gspos[vidmap[tid]] = p[i][tid];
		};
		make_kernel_param(&grid_size, &block_size, g.n_vertices, 512);
		traverse_noret << <grid_size, block_size >> > (g.n_vertices, reorder);
		cudaDeviceSynchronize();
		cuda_error_check;
		hostpos.resize(g.n_gsvertices);
		cudaMemcpy(hostpos.data(), gspos, sizeof(double) * g.n_gsvertices, cudaMemcpyDeviceToHost);
		for (int j = 0; j < g.n_gsvertices; j++) {
			p3host[j * 3 + i] = hostpos[j];
		}
	}
}

__global__ void computeElementPos_kernel(int n_word, int vreso, gBitSAT<unsigned int> elesat, devArray_t<double, 3> orig, double eh, devArray_t<double*, 3> pos) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= n_word) return;
	
	auto word = elesat._bitarray[tid];
	int ereso = vreso - 1;

	for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
		if (!read_gbit(word, ji)) continue;
		int ebid = tid * BitCount<unsigned int>::value + ji;
		int epos[3] = { ebid % ereso, ebid / ereso % ereso, ebid / ereso / ereso };
		int eid = elesat[ebid];
		for (int k = 0; k < 3; k++) {
			pos[k][eid] = orig[k] + eh * epos[k] + 0.5 * eh;
		}
	}
}

void HierarchyGrid::getElementPos(Grid& g, std::vector<double>& p3host)
{
	int lay = g._layer;
	auto& esat = elesatlist[lay];
	if (esat.total() != g.n_elements) printf("-- error on get element pos\n");
	int vreso = g._ereso + 1;
	devArray_t<double*, 3> p;
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&p[i], sizeof(double) * g.n_elements);
	}

	double eh = elementLength() * (1 << g._layer);
	devArray_t<double, 3> orig;
	for (int i = 0; i < 3; i++) orig[i] = _gridlayer[0]->_box[0][i];

	int nword = esat._bitArray.size();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nword, 512);

	gBitSAT<unsigned int> elesat(esat._bitArray, esat._chunkSat);

	computeElementPos_kernel << <grid_size, block_size >> > (nword, vreso, elesat, orig, eh, p);
	cudaDeviceSynchronize();
	cuda_error_check;

	p3host.resize(g.n_gselements * 3);
	double* gspos;
	cudaMalloc(&gspos, sizeof(double) * g.n_gselements);
	for (int i = 0; i < 3; i++) {
		std::vector<double> hostpos;
		init_array(gspos, std::numeric_limits<double>::quiet_NaN(), g.n_gselements);
		int* eidmap = g._gbuf.eidmap;
		auto reorder = [=] __device__(int tid) {
			gspos[eidmap[tid]] = p[i][tid];
		};
		make_kernel_param(&grid_size, &block_size, g.n_elements, 512);
		traverse_noret << <grid_size, block_size >> > (g.n_elements, reorder);
		cudaDeviceSynchronize();
		cuda_error_check;
		hostpos.resize(g.n_gselements);
		cudaMemcpy(hostpos.data(), gspos, sizeof(double) * g.n_gselements, cudaMemcpyDeviceToHost);
		for (int j = 0; j < g.n_gselements; j++) {
			p3host[j * 3 + i] = hostpos[j];
		}
	}
}

void HierarchyGrid::fillShell(void)
{
	_gridlayer[0]->use_grid();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, _gridlayer[0]->n_gsvertices, 512);
	int nv = _gridlayer[0]->n_gsvertices;
	float* rholist = _gridlayer[0]->_gbuf.rho_e;
	auto fillkernel = [=] __device__(int tid) {
		if (tid >= nv) return;
		int flag = gVflag[0][tid];
		if (flag & Grid::Bitmask::mask_invalid) return;
		int eid = gV2E[0][tid];
		if (eid == -1) return;
		int eflag = gEflag[0][eid];
		if (eflag & Grid::Bitmask::mask_shellelement) rholist[eid] = 1;
	};
	traverse_noret << <grid_size, block_size >> > (_gridlayer[0]->n_gsvertices, fillkernel);
	cudaDeviceSynchronize();
	cuda_error_check;
}


float* Grid::getlexiEbuf(float* gs_src)
{
	float* dst = (float*)getTempBuf(sizeof(float) * n_elements);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_elements, 512);
	int* eidmap = _gbuf.eidmap;

	auto kernel = [=] __device__(int tid) {
		dst[tid] = gs_src[eidmap[tid]];
	};
	traverse_noret << <grid_size, block_size >> > (n_elements, kernel);
	cudaDeviceSynchronize();
	cuda_error_check;
	return dst;
}

double* Grid::getlexiVbuf(double* gs_src)
{
	double* dst = (double*)getTempBuf(sizeof(double) * n_vertices);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_vertices, 512);
	int* vidmap = _gbuf.vidmap;

	auto kernel = [=] __device__(int tid) {
		dst[tid] = gs_src[vidmap[tid]];
	};
	traverse_noret << <grid_size, block_size >> > (n_vertices, kernel);
	cudaDeviceSynchronize();
	cuda_error_check;
	return dst;
}

__global__ void apply_adjointK_kernel(int nv, float* rholist,
	devArray_t<double*, 3> usrc, devArray_t<double*, 3> fdst,
	gBitSAT<unsigned int> vloadsat, bool use_support, bool constrain_force
) {
	__shared__ double KE[24][24];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	loadTemplateMatrix(KE);

	if (tid >= nv) return;

	int vid = tid;

	int mode = gmode[0];

	// load flag and neighbor ids
	bool vfix[27], vload[27];
	int v2v[27];
	loadNeighborNodesAndFlags(vid, v2v, vfix, vload);

	double vitan[2][3] = { 0. };

	double KU[3] = { 0.,0.,0. };
	float power = power_penalty[0];
	for (int i = 0; i < 8; i++) {
		int eid = gV2E[i][vid];
		if (eid == -1) continue;
		double penalty = powf(rholist[eid], power);

		// vertex id in i-th neighbor element
		int vi = 7 - i;

		//double KeU[3] = { 0. };
		// traverse other vertex of neighbor element, and compute KeU
		for (int vj = 0; vj < 8; vj++) {
			int vjpos[3] = {
				vj % 2 + i % 2,
				vj % 4 / 2 + i % 4 / 2,
				vj / 4 + i / 4
			};
			int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
			int vj_vid = v2v[vj_lid];
			if (vj_vid == -1) {
				// DEBUG
				printf("-- error in update residual otfa\n");
				continue;
			}

			// check if vj is a load node, and load the tangent vectors if it its
			bool vjisload = vload[vj_lid];
			bool vjisfix = vfix[vj_lid];
			double vtan[2][3];
			if (vjisload) {
				int vjloadid = vloadsat(vj_vid); if (vjloadid == -1) printf("-- error on node %d\n", vj_vid);
				for (int k1 = 0; k1 < 2; k1++)
					for (int k2 = 0; k2 < 3; k2++) vtan[k1][k2] = gLoadtangent[k1][k2][vjloadid];
				// set viload if vj is vi
				if (vj_lid == 13) {
					for (int k1 = 0; k1 < 2; k1++)
						for (int k2 = 0; k2 < 3; k2++) vitan[k1][k2] = vtan[k1][k2];
				}
			}

			// fetch displacement
			double u[3] = { usrc[0][vj_vid],usrc[1][vj_vid],usrc[2][vj_vid] };

			if (vjisfix && use_support) {
				u[0] = 0; u[1] = 0; u[2] = 0;
			}

			// multiply N^T on u if vj is load node
			if (vjisload ) {
				if (constrain_force) {
					double Nu[3];
					for (int k = 0; k < 3; k++) Nu[k] = vtan[0][k] * u[0] + vtan[1][k] * u[1];
					for (int k = 0; k < 3; k++) u[k] = Nu[k];
				}
				else {
					u[0] = 0; u[1] = 0; u[2] = 0;
				}
			}

			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 3; col++) {
					KU[row] += penalty * KE[row + vi * 3][col + vj * 3] * u[col];
				}
			}
		}

	}
	// check whether vi is load node, multiply N if true
	if (vload[13] ) {
		if (constrain_force) {
			double ku[2] = { 0. };
			for (int k = 0; k < 3; k++) {
				ku[0] += vitan[0][k] * KU[k];
				ku[1] += vitan[1][k] * KU[k];
			}
			KU[0] = ku[0]; KU[1] = ku[1]; KU[2] = 0;
		}
		else {
			KU[0] = 0; KU[1] = 0; KU[2] = 0;
		}
	}

	// vi is fix
	if (vfix[13] && use_support) { KU[0] = 0; KU[1] = 0; KU[2] = 0; }

	for (int i = 0; i < 3; i++) { fdst[i][vid] = KU[i]; }
}

void grid::Grid::applyAjointK(double* usrc[3], double* fdst[3])
{
	use_grid();
	bool use_support = hasSupport();
	bool constrain_force = !isForceFree();
	devArray_t<double*, 3> us{ usrc[0],usrc[1],usrc[2] };
	devArray_t<double*, 3> fd{ fdst[0],fdst[1],fdst[2] };
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
	apply_adjointK_kernel<<<grid_size,block_size>>>(n_gsvertices, _gbuf.rho_e,
		us, fd, vid2loadid, use_support, constrain_force
	);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void grid::Grid::v3_create(double* dstv[3])
{
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&dstv[i], sizeof(double) * n_gsvertices);
	}
}

bool grid::Grid::checkV2V(void)
{
	use_grid();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
	auto kernel = [=] __device__(int tid) {
		//int v2v[27];
		//loadNeighborNodes(tid, v2v);
		for (int i = 0; i < 8; i++) {
			int eid = gV2E[i][tid];
			if (eid == -1) continue;
			for (int j = 0; j < 8; j++) {
				int vjpos[3] = {
					i % 2 + j % 2,
					i / 2 % 2 + j / 2 % 2,
					i / 4 + j / 4
				};
				int vjlid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
				if (gV2V[vjlid][tid] == -1) {
					printf("-- v[%d] e[%d](%d), v[%d]\n", tid, i, eid, vjlid);
				}
			}
		}
	};
	traverse_noret << <grid_size, block_size >> > (n_gsvertices, kernel);
	cudaDeviceSynchronize();
	cuda_error_check;
	return false;
}

void grid::Grid::v3_pertub(double* v[3], double ratio)
{
	double oldnorm = v3_norm(v);

	// generate a pertubation
	devArray_t<double*, 3> pertub;
	pertub.create(n_gsvertices);
	v3_rand(pertub._data, -1, 1);
	double pertubnorm = v3_norm(pertub._data);
	v3_scale(pertub._data, 1.0 / pertubnorm * (oldnorm * ratio));

	// apply pertubation
	v3_add(v, 1, pertub._data);

	// new norm
	double newnorm = v3_norm(v);

	// scale new v3 to old norm 
	v3_scale(v, oldnorm / newnorm);

	// destroy temp buf
	pertub.destroy();
}

void grid::Grid::pertubForce(double ratio)
{
	// compute current force norm
	getForceSupport(_gbuf.F, getSupportForce());
	double** fsptr = getSupportForce();
	devArray_t<double*, 3> fs{ fsptr[0],fsptr[1],fsptr[2] };
	double oldnorm = norm(fs[0], fs[1], fs[2], (double*)getTempBuf(n_loadnodes() / 100 * sizeof(double)), n_loadnodes());

	// compute required noise norm
	double noisyNormRequare = oldnorm * ratio;

	// compute noise
	devArray_t<double*, 3> fsNoise;
	fsNoise.create(n_loadnodes());
	for (int i = 0; i < 3; i++) {
		randArray<double>(fsNoise._data, 3, n_loadnodes(), -1, 1);
	}
	double noiseNorm = norm(fsNoise[0], fsNoise[1], fsNoise[2], (double*)getTempBuf(n_loadnodes() / 100 * sizeof(double)), n_loadnodes());

	// add scaled noise
	double scaleRatio = noisyNormRequare / noiseNorm;
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_loadnodes(), 512);
	for (int i = 0; i < 3; i++) {
		map<<<grid_size,block_size>>>(n_loadnodes(), [=]__device__(int tid) {
			for (int j = 0; j < 3; j++) {
				fs[j][tid] += fsNoise[j][tid] * scaleRatio;
			}
		});
	}
	cudaDeviceSynchronize();
	cuda_error_check;

	setForceSupport(fsptr, _gbuf.F);
}


__global__ void elementCompliance_kernel(int nv, devArray_t<double*, 3> ulist, devArray_t<double*, 3> flist, float* rholist, float* clist) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ double KE[24][24];

	loadTemplateMatrix(KE);

	if (tid >= nv) return;

	int v2v[27];

	loadNeighborNodes(tid, v2v);

	float power = power_penalty[0];

	for (int e = 0; e < 8; e++) {
		int vi = 7 - e;
		int eid = gV2E[e][tid];
		if (eid == -1) continue;
		float penal = powf(rholist[eid], power);
		double KeU[3] = { 0,0,0 };
		for (int vj = 0; vj < 8; vj++) {
			int vjpos[3] = { e % 2 + vj % 2, e / 2 % 2 + vj / 2 % 2, e / 4 + vj / 4 };
			int vjlid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
			int vjid = v2v[vjlid];
			if (vjid == -1) continue;
			double Uj[3] = { gU[0][vjid], gU[1][vjid], gU[2][vjid] };
			for (int krow = 0; krow < 3; krow++) {
				for (int kcol = 0; kcol < 3; kcol++) {
					KeU[krow] += penal * KE[vi * 3 + krow][vj * 3 + kcol] * Uj[kcol];
				}
			}
		}

		double Ui[3] = { gU[0][tid],gU[1][tid],gU[2][tid] };

		double uKeu = Ui[0] * KeU[0] + Ui[1] * KeU[1] + Ui[2] * KeU[2];

		atomicAdd(clist + eid, uKeu);
	}
	
}

void grid::Grid::elementCompliance(double* u[3], double* f[3], float* dst)
{
	devArray_t<double*, 3> ulist, flist;
	for (int i = 0; i < 3; i++) {
		ulist[i] = u[i]; flist[i] = f[i];
	}

	init_array(dst, float{ 0 }, n_gselements);

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
	elementCompliance_kernel << <grid_size, block_size >> > (n_gsvertices, ulist, flist, _gbuf.rho_e, dst);
	cudaDeviceSynchronize();
	cuda_error_check;
}

double  grid::Grid::compliance(double* u[3], double* f[3])
{
	// create vector
	devArray_t<double*, 3> ku;
	ku.create(n_gsvertices);

	// compute f * K * u
	applyK(u, ku._data);

	double c = v3_dot(ku._data, f);
	ku.destroy();
	return c;
}

float grid::Grid::volumeRatio(void)
{
	cuda_error_check;
	float* tmp = (float*)getTempBuf(sizeof(float)* n_gselements / 100);
	float v = parallel_sum(_gbuf.rho_e, tmp, n_gselements);
	cudaDeviceSynchronize();
	cuda_error_check;
	return v / n_gselements;
}

void grid::HierarchyGrid::test_kernels(void)
{
	_gridlayer[0]->use_grid();
	int nv = _gridlayer[0]->n_gsvertices;
	size_t grid_size, block_size;
	auto t0 = tictoc::getTag();
	float* rholist = _gridlayer[0]->_gbuf.rho_e;
	if (_mode == no_support_constrain_force_direction || _mode == no_support_free_force) {
		make_kernel_param(&grid_size, &block_size, nv, 512);
		update_residual_OTFA_NS_kernel << <grid_size, block_size >> > (nv, rholist);
	}
	else if (_mode == with_support_constrain_force_direction || _mode == with_support_free_force) {
		make_kernel_param(&grid_size, &block_size, nv, 256);
		update_residual_OTFA_WS_kernel << <grid_size, block_size >> > (nv, rholist);
	}
	cudaDeviceSynchronize();
	cuda_error_check;
	auto t1 = tictoc::getTag();
	double t_duration = tictoc::Duration<tictoc::ms>(t0, t1);
	printf("[Routine1] time %6.2lf ms\n", t_duration);

	t0 = tictoc::getTag();
	if (_mode == no_support_constrain_force_direction || _mode == no_support_free_force) {
		make_kernel_param(&grid_size, &block_size, nv, 512);
		update_residual_OTFA_NS_kernel << <grid_size, block_size >> > (nv, rholist);
	}
	else if (_mode == with_support_constrain_force_direction || _mode == with_support_free_force) {
		make_kernel_param(&grid_size, &block_size, nv * 8, 32 * 8);
		update_residual_OTFA_WS_kernel_1 << <grid_size, block_size >> > (nv, rholist);
	}
	cudaDeviceSynchronize();
	cuda_error_check;

	t1 = tictoc::getTag();
	t_duration = tictoc::Duration<tictoc::ms>(t0, t1);
	printf("[Routine2] time %6.2lf ms\n", t_duration);
}

double grid::Grid::densityDiscretiness(void)
{
	float* rholist = _gbuf.rho_e;
	float* pout = _gbuf.g_sens;
	auto disc = [=] __device__(int eid) {
		float rho = rholist[eid];
		pout[eid] = rho * (1 - rho);
	};
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gselements, 512);
	map<<<grid_size,block_size>>>(n_gselements, disc);
	cudaDeviceSynchronize();
	cuda_error_check;

	double* sum = (double*)grid::Grid::getTempBuf(sizeof(double) * n_gselements / 100);
	double Md = parallel_sum_d(pout, sum, n_gselements) / n_gselements;
	return Md;
}

void grid::Grid::compute_spline_surface_point_normal(void)
{
	auto calc_normal = [=] __device__(int node_id) {
		float p[3] = { 0.f };
		float normal[3] = { 0.f };
		float val;
		int i, j, k, ir, it, is, index;
		int d = 2;

		for (i = 0; i < 3; i++) p[i] = gpu_SurfacePoints[i][node_id];

		float NX[m_iM] = { 0.f };
		float pNX[m_iM] = { 0.f };
		float NY[m_iM] = { 0.f };
		float pNY[m_iM] = { 0.f };
		float NZ[m_iM] = { 0.f };
		float pNZ[m_iM] = { 0.f };

		i = (int)((p[0] - gnBoundMin[0]) / gnstep[0]) + m_iM;
		j = (int)((p[1] - gnBoundMin[1]) / gnstep[1]) + m_iM;
		k = (int)((p[2] - gnBoundMin[2]) / gnstep[2]) + m_iM;

		if ((i < m_iM) || (i > gnbasis[0]) || (j < m_iM) || (j > gnbasis[1]) || (k < m_iM) || (k > gnbasis[2]))
		{
			normal[0] = 0.0f;
			normal[1] = 0.0f;
			normal[2] = 0.0f;
		}
		else
		{
			SplineBasisDeriX(p[0], 1, NX);  // 1 means the original function value
			SplineBasisDeriX(p[0], 2, pNX); // 2 means the first order derivative value

			SplineBasisDeriY(p[1], 1, NY);
			SplineBasisDeriY(p[1], 2, pNY);

			SplineBasisDeriZ(p[2], 1, NZ);
			SplineBasisDeriZ(p[2], 2, pNZ);

			for (ir = i - m_iM; ir < i; ir++)
			{
				for (is = j - m_iM; is < j; is++)
				{
					for (it = k - m_iM; it < k; it++)
					{
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];

						normal[0] += gpu_cijk[index] * pNX[ir - i + m_iM] * NY[is - j + m_iM] * NZ[it - k + m_iM];
						normal[1] += gpu_cijk[index] * NX[ir - i + m_iM] * pNY[is - j + m_iM] * NZ[it - k + m_iM];
						normal[2] += gpu_cijk[index] * NX[ir - i + m_iM] * NY[is - j + m_iM] * pNZ[it - k + m_iM];
					}
				}
			}
		}

		//[MARK]TODO: check the normal direction

		for (i = 0; i < 3; i++)	gpu_surface_normal[i][node_id] = normal[i];
		return;
	};

	size_t grid_dim, block_dim;
	int n = spline_surface_node->size();

	//std::cout << "--[TEST] number of surface node: " << spline_surface_node->size() << "," << _num_surface_points << std::endl;

	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse_noret << <grid_dim, block_dim >> > (n, calc_normal);
	cudaDeviceSynchronize();
	cuda_error_check;

#ifdef ENABLE_MATLAB
	{
		float* spline_surf_node_normal[3];
		for (int i = 0; i < 3; i++)
		{
			spline_surf_node_normal[i] = new float[n];
			cudaMemcpy(spline_surf_node_normal[i], _gbuf.surface_normal[i], sizeof(float)* _num_surface_points, cudaMemcpyDeviceToHost);
			cuda_error_check;
		}
		gpu_manager_t::pass_buf_to_matlab("spline_surface_node_normal_x", spline_surf_node_normal[0], n);
		gpu_manager_t::pass_buf_to_matlab("spline_surface_node_normal_y", spline_surf_node_normal[1], n);
		gpu_manager_t::pass_buf_to_matlab("spline_surface_node_normal_z", spline_surf_node_normal[2], n);
		for (int i = 0; i < 3; i++) {
			delete spline_surf_node_normal[i];
			spline_surf_node_normal[i] = nullptr;
		}
	}
#endif
}

float grid::Grid::correct_spline_surface_point_normal_direction(float beta)
{
	float count = 1;
	float* direction_tmp;
	cudaMalloc(&direction_tmp, sizeof(float) * n_surf_points());
	init_array(direction_tmp, float{ 0 }, n_surf_points());
	cuda_error_check;

	float print_angle = _opt_print_angle;

	auto calc_node = [=] __device__(int node_id) {
		float p[3];
		for (int i = 0; i < 3; i++) 	p[i] = gpu_SurfacePoints[i][node_id];

		float normal_vector[3];
		for (int i = 0; i < 3; i++)		normal_vector[i] = gpu_surface_normal[i][node_id];

		float s = 0.f, tmp = 0.f;
		float delta = 1e-3, p_delta[3] = { 0.f };
		for (int i = 0; i < 3; i++)
		{
			p_delta[i] = p[i] + delta * normal_vector[i];
		}
		float val;
		int i, j, k, ir, it, is, index;

		float NX[m_iM + 1], NX_delta[m_iM + 1];
		float NY[m_iM + 1], NY_delta[m_iM + 1];
		float NZ[m_iM + 1], NZ_delta[m_iM + 1];

		i = (int)((p_delta[0] - gnBoundMin[0]) / gnstep[0]) + m_iM;
		j = (int)((p_delta[1] - gnBoundMin[1]) / gnstep[1]) + m_iM;
		k = (int)((p_delta[2] - gnBoundMin[2]) / gnstep[2]) + m_iM;

		if ((i < m_iM) || (i > gnbasis[0]) || (j < m_iM) || (j > gnbasis[1]) || (k < m_iM) || (k > gnbasis[2]))
		{
			val = -0.1f;
		}
		else
		{
			SplineBasisX(p_delta[0], NX_delta);
			SplineBasisY(p_delta[1], NY_delta);
			SplineBasisZ(p_delta[2], NZ_delta);

			val = 0.0f;
			for (ir = i - m_iM; ir < i; ir++)
			{
				for (is = j - m_iM; is < j; is++)
				{
					for (it = k - m_iM; it < k; it++)
					{
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];
						val += gpu_cijk[index] * NX_delta[ir - i + m_iM] * NY_delta[is - j + m_iM] * NZ_delta[it - k + m_iM];
					}
				}
			}
		}

		s = Heaviside(val, beta);
		if (s < 0.5)   // s < 0.5 --> void
		{
			s = 1;
		}
		else
		{
			s = -1;
		}
		direction_tmp[node_id] = s;
	};

	size_t grid_dim, block_dim;
	int n = grid::Grid::spline_surface_node->size();
	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse_noret << < grid_dim, block_dim >> > (n, calc_node);
	cudaDeviceSynchronize();
	cuda_error_check;

	float* direction_host = new float[n_surf_points()];
	cudaMemcpy(direction_host, direction_tmp, sizeof(float)* n_surf_points(), cudaMemcpyDeviceToHost);
	cuda_error_check;
	init_array(_gbuf.surface_normal_direction, float{ 0 }, n_surf_points());
	// [MARK]
	cudaMemcpy(_gbuf.surface_normal_direction, direction_host, n_surf_points() * sizeof(float), cudaMemcpyHostToDevice);
	cuda_error_check;

#ifdef ENABLE_MATLAB
	gpu_manager_t::pass_buf_to_matlab("spline_surface_normal_direction2", direction_host, n_surf_points());
#endif

	cudaFree(direction_tmp);
	direction_tmp = nullptr;
	delete[] direction_host;
	direction_host = nullptr;

	// MARK[TO CHECK] parallel_sum
	cuda_error_check;
	float* tmp = (float*)getTempBuf(sizeof(float) * n / 100);
	count = parallel_sum(_gbuf.surface_normal_direction, tmp, n);
	cudaDeviceSynchronize();
	cuda_error_check;

	if (count < 0)
	{
		printf("\033[34m We need to correct the direction of the normal !\033[0m\n ");
		//std::cout << "\033[34m-- [Same Coeff cannot extra surface points] --\033[0m" << std::endl;

		float* normal_dirction = (float*)_gbuf.surface_normal_direction;
		auto calc_normal = [=] __device__(int node_id) {
			float p[3] = { 0.f };
			float normal[3] = { 0.f };
			float direction_tmp = normal_dirction[node_id];

			for (int i = 0; i < 3; i++)
			{
				p[i] = gpu_SurfacePoints[i][node_id];
				normal[i] = gpu_surface_normal[i][node_id];
			}

			for (int i = 0; i < 3; i++)	gpu_surface_normal[i][node_id] = -normal[i];
			return;
		};
		//size_t grid_dim, block_dim;
		//int n = grid::Grid::spline_surface_node->size();
		//make_kernel_param(&grid_dim, &block_dim, n, 256);
		traverse_noret << <grid_dim, block_dim >> > (n, calc_normal);
		cudaDeviceSynchronize();
		cuda_error_check;

#ifdef ENABLE_MATLAB
		float* spline_surf_node_normal[3];
		for (int i = 0; i < 3; i++)
		{
			spline_surf_node_normal[i] = new float[n];
			cudaMemcpy(spline_surf_node_normal[i], _gbuf.surface_normal[i], sizeof(float) * n, cudaMemcpyDeviceToHost);
			cuda_error_check;
		}
		gpu_manager_t::pass_buf_to_matlab("corrected_spline_surface_node_normal_x", spline_surf_node_normal[0], n);
		gpu_manager_t::pass_buf_to_matlab("corrected_spline_surface_node_normal_y", spline_surf_node_normal[1], n);
		gpu_manager_t::pass_buf_to_matlab("corrected_spline_surface_node_normal_z", spline_surf_node_normal[2], n);
		for (int i = 0; i < 3; i++)		delete spline_surf_node_normal[i];
#endif     		
	}
	return count;
}

void grid::Grid::compute_selfsupp_flag_actual(void)
{
	float default_print_angle = _default_print_angle;

	auto calc_node = [=] __device__(int node_id) {
		float p[3] = { 0.f };
		for (int i = 0; i < 3; i++)
		{
			p[i] = gpu_SurfacePoints[i][node_id];
		}

		float normal_vector[3] = { 0.f };
		for (int i = 0; i < 3; i++)
		{
			normal_vector[i] = gpu_surface_normal[i][node_id];
		}

		float s = 0.f, tmp = 0.f;
		tmp = normal_vector[2] / norm(normal_vector) / cosf(default_print_angle);
		if (normal_vector[2] < 0)
		{
			if (tmp > 1)
			{
				s = 1;
			}
		}
		else
		{
			s = 0;
		}
		return s;
	};

	size_t grid_dim, block_dim;
	int n = grid::Grid::spline_surface_node->size();
	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse << <grid_dim, block_dim >> > ((float*)_gbuf.surface_points_flag, n, calc_node);
	cudaDeviceSynchronize();
	cuda_error_check;

#ifdef ENABLE_MATLAB 
	float* host_spline_constrain = new float[n];
	cudaMemcpy(host_spline_constrain, _gbuf.surface_points_flag, sizeof(float) * n, cudaMemcpyDeviceToHost);
	gpu_manager_t::pass_buf_to_matlab("spline_constrain_actual", host_spline_constrain, n);
	delete host_spline_constrain;
	cuda_error_check;
#endif
}

void grid::Grid::compute_selfsupp_flag_virtual(void)
{
	float print_angle = _opt_print_angle;

	auto calc_node = [=] __device__(int node_id) {
		float p[3] = { 0.f };
		for (int i = 0; i < 3; i++)
		{
			p[i] = gpu_SurfacePoints[i][node_id];
		}

		float normal_vector[3] = { 0.f };
		for (int i = 0; i < 3; i++)
		{
			normal_vector[i] = gpu_surface_normal[i][node_id];
		}

		float s = 0.f, tmp = 0.f;
		tmp = normal_vector[2] / norm(normal_vector) / cosf(print_angle);
		if (normal_vector[2] < 0)
		{
			if (tmp > 1)
			{
				s = 1;
			}
		}
		else
		{
			s = 0;
		}
		return s;
	};

	size_t grid_dim, block_dim;
	int n = grid::Grid::spline_surface_node->size();
	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse << <grid_dim, block_dim >> > ((float*)_gbuf.surface_points_flag_virtual, n, calc_node);
	cudaDeviceSynchronize();
	cuda_error_check;

#ifdef ENABLE_MATLAB 
	float* host_spline_constrain = new float[n];
	cudaMemcpy(host_spline_constrain, _gbuf.surface_points_flag_virtual, sizeof(float) * n, cudaMemcpyDeviceToHost);
	gpu_manager_t::pass_buf_to_matlab("spline_constrain_virtual", host_spline_constrain, n);
	delete host_spline_constrain;
	cuda_error_check;
#endif
}

void grid::Grid::compute_spline_surface_point_normal_norm_dcoeff(void)
{
	float* normdc_tmp;
	cudaMalloc(&normdc_tmp, sizeof(float) * n_cijk());
	init_array(normdc_tmp, float{ 0 }, n_cijk());
	cuda_error_check;

	// MARK[TO CHECK] parallel_sum
	float* tmp = (float*)getTempBuf(sizeof(float) * grid::Grid::spline_surface_node->size() / 100);
	cuda_error_check;
	float count = parallel_sum(_gbuf.surface_normal_direction, tmp, grid::Grid::spline_surface_node->size());
	cudaDeviceSynchronize();
	cuda_error_check;

	float* normal_dirction = (float*)_gbuf.surface_normal_direction;

	float direction;
	if (count > 0)
	{
		direction = 1.0f;
	}
	else
	{
		direction = -1.0f;
	}

	auto calc_node_value = [=] __device__(int node_id)
	{
		float p[3] = { 0.f };
		for (int i = 0; i < 3; i++)	p[i] = gpu_SurfacePoints[i][node_id];

		float normal_vector[3] = { 0.f };
		float normal_der_cijk[3] = { 0.f };
		for (int i = 0; i < 3; i++) normal_vector[i] = gpu_surface_normal[i][node_id];
		float direction_tmp = normal_dirction[node_id];

		float normal_vector_norm = norm(normal_vector);

		int i, j, k, ir, it, is, index;

		float NX[m_iM] = { 0.f };
		float pNX[m_iM] = { 0.f };
		float NY[m_iM] = { 0.f };
		float pNY[m_iM] = { 0.f };
		float NZ[m_iM] = { 0.f };
		float pNZ[m_iM] = { 0.f };

		i = (int)((p[0] - gnBoundMin[0]) / gnstep[0]) + m_iM;
		j = (int)((p[1] - gnBoundMin[1]) / gnstep[1]) + m_iM;
		k = (int)((p[2] - gnBoundMin[2]) / gnstep[2]) + m_iM;

		if ((i < m_iM) || (i > gnbasis[0]) || (j < m_iM) || (j > gnbasis[1]) || (k < m_iM) || (k > gnbasis[2]))
		{
			normal_der_cijk[0] = 0.0f;
			normal_der_cijk[1] = 0.0f;
			normal_der_cijk[2] = 0.0f;
		}
		else
		{
			SplineBasisDeriX(p[0], 1, NX);  // 1 means the original function value
			SplineBasisDeriX(p[0], 2, pNX); // 2 means the first order derivative value

			SplineBasisDeriY(p[1], 1, NY);
			SplineBasisDeriY(p[1], 2, pNY);

			SplineBasisDeriZ(p[2], 1, NZ);
			SplineBasisDeriZ(p[2], 2, pNZ);

			for (ir = i - m_iM; ir < i; ir++)
			{
				for (is = j - m_iM; is < j; is++)
				{
					for (it = k - m_iM; it < k; it++)
					{
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];
						normal_der_cijk[0] = direction * pNX[ir - i + m_iM] * NY[is - j + m_iM] * NZ[it - k + m_iM];
						normal_der_cijk[1] = direction * NX[ir - i + m_iM] * pNY[is - j + m_iM] * NZ[it - k + m_iM];
						normal_der_cijk[2] = direction * NX[ir - i + m_iM] * NY[is - j + m_iM] * pNZ[it - k + m_iM];

						float s = dot(normal_vector, normal_der_cijk) / normal_vector_norm;
						normdc_tmp[index] += s;
					}
				}
			}
		}
		return;
	};

	size_t grid_dim, block_dim;
	int n = grid::Grid::spline_surface_node->size();
	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse_noret << <grid_dim, block_dim >> > (n, calc_node_value);
	cudaDeviceSynchronize();
	cuda_error_check;

	float* normdc_host = new float[n_cijk()];
	cudaMemcpy(normdc_host, normdc_tmp, sizeof(float)* n_cijk(), cudaMemcpyDeviceToHost);
	cuda_error_check;
	init_array(_gbuf.surface_normal_norm_dc, float{ 0 }, n_cijk());
	// [MARK]
	cudaMemcpy(_gbuf.surface_normal_norm_dc, normdc_host, n_cijk() * sizeof(float), cudaMemcpyHostToDevice);
	cuda_error_check;

#ifdef ENABLE_MATLAB
	gpu_manager_t::pass_buf_to_matlab("norm_dcoeff1", normdc_host, n_cijk());
#endif

	cudaFree(normdc_tmp);
	normdc_tmp = nullptr;
	delete[] normdc_host;
	normdc_host = nullptr;
}

void grid::Grid::compute_spline_surface_point_normal_dcoeff(void)
{
}

float grid::Grid::count_surface_points(void)
{
	auto calc_node = [=] __device__(int node_id) {
		float p[3];
		float s = 0.f;
		for (int i = 0; i < 3; i++)		p[i] = gpu_SurfacePoints[i][node_id];
		float normal_vector[3];
		for (int i = 0; i < 3; i++)		normal_vector[i] = gpu_surface_normal[i][node_id];

		if (normal_vector[2] < 0)
		{
			s = 1;
		}
		else
		{
			s = 0;
		}
		return s;
	};

	size_t grid_dim, block_dim;
	int n = grid::Grid::spline_surface_node->size();
	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse << <grid_dim, block_dim >> > ((float*)_gbuf.surface_point_buf, n, calc_node);
	cudaDeviceSynchronize();
	cuda_error_check;

	float* sum = (float*)grid::Grid::getTempBuf(sizeof(float) * n / 100);
	float count = parallel_sum(_gbuf.surface_point_buf, sum, n);
	cudaDeviceSynchronize();
	cuda_error_check;
	return count;
}

//  Deal with self-supporting 
void grid::Grid::compute_spline_selfsupp_constraint_dcoeff(void)
{
	if (_layer != 0) return;

	float* surface_direction = (float*)_gbuf.surface_normal_direction;

	// MARK[TO CHECK] parallel_sum
	float* tmp = (float*)getTempBuf(sizeof(float) * grid::Grid::spline_surface_node->size() / 100);
	cuda_error_check;
	float count = parallel_sum(_gbuf.surface_normal_direction, tmp, grid::Grid::spline_surface_node->size());
	cudaDeviceSynchronize();
	cuda_error_check;

	float* normal_dirction = (float*)_gbuf.surface_normal_direction;
	float direction;
	if (count > 0)
	{
		direction = 1.0f;
	}
	else
	{
		direction = -1.0f;
	}

	float print_angle = _opt_print_angle;
	int modeid = _ssmode;
	float func_para = sigmoid_c;
	if (modeid == 0 || modeid == 1)
	{
		func_para = p_norm;
	}
	else if (modeid == 2 || modeid == 3)
	{
		func_para = hfunction_c;
	}
	else if (modeid == 4 || modeid == 5)
	{
		func_para = sigmoid_c;
	}

	std::cout << "--[TEST] mode id: " << modeid << std::endl;
	std::cout << "--[TEST] function para: " << func_para << std::endl;

	// computation
	float* dc_tmp;
	cudaMalloc(&dc_tmp, sizeof(float) * n_cijk());
	init_array(dc_tmp, float{ 0 }, n_cijk());
	cuda_error_check;

	auto calc_node_value = [=] __device__(int node_id)
	{
		float p[3] = { 0.f };
		for (int i = 0; i < 3; i++)	p[i] = gpu_SurfacePoints[i][node_id];

		float normal_vector[3] = { 0.f };
		float normal_dcijk[3] = { 0.f };
		float norm_dcijk = 0.f;
		for (int i = 0; i < 3; i++) normal_vector[i] = gpu_surface_normal[i][node_id];
		float direction_tmp = normal_dirction[node_id];

		float normal_vector_norm = norm(normal_vector);

		int i, j, k, ir, it, is, index;

		float NX[m_iM] = { 0.f };
		float pNX[m_iM] = { 0.f };
		float NY[m_iM] = { 0.f };
		float pNY[m_iM] = { 0.f };
		float NZ[m_iM] = { 0.f };
		float pNZ[m_iM] = { 0.f };

		i = (int)((p[0] - gnBoundMin[0]) / gnstep[0]) + m_iM;
		j = (int)((p[1] - gnBoundMin[1]) / gnstep[1]) + m_iM;
		k = (int)((p[2] - gnBoundMin[2]) / gnstep[2]) + m_iM;

		if ((i < m_iM) || (i > gnbasis[0]) || (j < m_iM) || (j > gnbasis[1]) || (k < m_iM) || (k > gnbasis[2]))
		{
			normal_dcijk[0] = 0.0f;
			normal_dcijk[1] = 0.0f;
			normal_dcijk[2] = 0.0f;
		}
		else
		{
			SplineBasisDeriX(p[0], 1, NX);  // 1 means the original function value
			SplineBasisDeriX(p[0], 2, pNX); // 2 means the first order derivative value

			SplineBasisDeriY(p[1], 1, NY);
			SplineBasisDeriY(p[1], 2, pNY);

			SplineBasisDeriZ(p[2], 1, NZ);
			SplineBasisDeriZ(p[2], 2, pNZ);

			for (ir = i - m_iM; ir < i; ir++)
			{
				for (is = j - m_iM; is < j; is++)
				{
					for (it = k - m_iM; it < k; it++)
					{
						float val = 0.f;
						float up, down, dinner, inner;
						index = ir + is * gnbasis[0] + it * gnbasis[0] * gnbasis[1];
						normal_dcijk[0] = direction * pNX[ir - i + m_iM] * NY[is - j + m_iM] * NZ[it - k + m_iM];
						normal_dcijk[1] = direction * NX[ir - i + m_iM] * pNY[is - j + m_iM] * NZ[it - k + m_iM];
						normal_dcijk[2] = direction * NX[ir - i + m_iM] * NY[is - j + m_iM] * pNZ[it - k + m_iM];
						norm_dcijk = dot(normal_vector, normal_dcijk) / normal_vector_norm;

						up = normal_dcijk[2] * normal_vector_norm - normal_vector[2] * norm_dcijk;
						down = normal_vector_norm * normal_vector_norm * cosf(print_angle);
						dinner = up / down;
						inner = normal_vector[2] / normal_vector_norm / cosf(print_angle);

						if (modeid == 0)            // p_norm_ss
						{
							if (normal_vector[2] < 0)
							{
								val = func_para * pow(inner, func_para - 1) * dinner;
							}
						}
						else if (modeid == 1)
						{
							val = func_para * pow(inner, func_para - 1) * dinner;
						}
						else if (modeid == 2)       // h_function_ss
						{
							if (normal_vector[2] < 0)
							{
								val = dh(inner, func_para) * dinner;
							}
						}
						else if (modeid == 3)       // h_function2_ss
						{
							val = dh(inner, func_para) * dinner;
						}
						else if (modeid == 4)       // overhang_ss
						{
							if (normal_vector[2] < 0)
							{
								val = doh(inner, func_para) * dinner;
							}
						}
						else if (modeid == 5)       // overhang2_ss
						{
							val = doh(inner, func_para) * dinner;
						}
						dc_tmp[index] += val;
					}
				}
			}
		}
		return;
	};

	size_t grid_dim, block_dim;
	int n = grid::Grid::spline_surface_node->size();
	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse_noret << <grid_dim, block_dim >> > (n, calc_node_value);
	cudaDeviceSynchronize();
	cuda_error_check;

	float* dc_host = new float[n_cijk()];
	cudaMemcpy(dc_host, dc_tmp, sizeof(float) * n_cijk(), cudaMemcpyDeviceToHost);
	cuda_error_check;
	init_array(_gbuf.ssc_sens, float{ 0 }, n_cijk());
	cudaMemcpy(_gbuf.ssc_sens, dc_host, n_cijk() * sizeof(float), cudaMemcpyHostToDevice);
	cuda_error_check;

#ifdef ENABLE_MATLAB
	gpu_manager_t::pass_buf_to_matlab("ssc_sens2", dc_host, n_cijk());
#endif

	cudaFree(dc_tmp);
	dc_tmp = nullptr;
	delete[] dc_host;
	dc_host = nullptr;
}

void grid::Grid::compute_spline_selfsupp_constraint(void)
{
	float print_angle = _opt_print_angle;
	int modeid = _ssmode;
	float func_para = sigmoid_c;
	if (modeid == 0 || modeid == 1)
	{
		func_para = p_norm;
	}
	else if (modeid == 2 || modeid == 3)
	{
		func_para = hfunction_c;
	}
	else if (modeid == 4 || modeid == 5)
	{
		func_para = sigmoid_c;
	}

	auto calc_node = [=] __device__(int node_id) {
		float p[3];
		for (int i = 0; i < 3; i++)
		{
			p[i] = gpu_SurfacePoints[i][node_id];
		}

		float normal_vector[3];
		for (int i = 0; i < 3; i++)
		{
			normal_vector[i] = gpu_surface_normal[i][node_id];
		}

		float val = 0.f, inner = 0.f;
		inner = normal_vector[2] / norm(normal_vector) / cosf(print_angle);
		// [MARK] TOADD more enum
		if (modeid == 0)            // p_norm_ss
		{
			if (normal_vector[2] < 0)
			{
				val = pow(inner, func_para);
			}
		}
		else if (modeid == 1)
		{
			val = pow(inner, func_para);
		}
		else if (modeid == 2)       // h_function_ss
		{
			if (normal_vector[2] < 0)
			{
				val = h(inner, func_para);
			}
		}
		else if (modeid == 3)       // h_function2_ss
		{
			val = h(inner, func_para);
		}
		else if (modeid == 4)       // overhang_ss
		{
			if (normal_vector[2] < 0)
			{
				val = oh(inner, func_para);
			}
		}
		else if (modeid == 5)       // overhang2_ss
		{
			val = oh(inner, func_para);
		}
		return val;
	};

	size_t grid_dim, block_dim;
	int n = n_surf_points();
	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse << <grid_dim, block_dim >> > ((float*)_gbuf.ss_value, n, calc_node);
	cudaDeviceSynchronize();
	cuda_error_check;

#ifdef ENABLE_MATLAB
	float* host_spline_constrain = new float[n];
	cudaMemcpy(host_spline_constrain, _gbuf.ss_value, sizeof(float)* n, cudaMemcpyDeviceToHost);
	gpu_manager_t::pass_buf_to_matlab("ss_constraint", host_spline_constrain, n);
	delete host_spline_constrain;
	cuda_error_check;
#endif
}

float grid::Grid::global_selfsupp_constraint(void)
{
	float val = 0.f;

	int modeid = _ssmode;
	float func_para = sigmoid_c;
	float count = grid::Grid::spline_surface_node->size();

	if (modeid % 2 == 0)
	{
		count = count_surface_points();
	}

	float* sum = (float*)grid::Grid::getTempBuf(sizeof(float) * n_surf_points() / 100);
	float ss_sum = parallel_sum(_gbuf.ss_value, sum, n_surf_points());
	cudaDeviceSynchronize();
	cuda_error_check;

	std::cout << "--[TEST] total  of SS constraints : " << ss_sum << std::endl;
	std::cout << "--[TEST] number of SS constraints : " << count << std::endl;

	if (modeid == 0 || modeid == 1)
	{
		func_para = p_norm;
		val = pow(ss_sum / count, 1 / func_para) - 1;
	}
	else if (modeid == 2 || modeid == 3)
	{
		func_para = hfunction_c;
		val = ss_sum / count;
	}
	else if (modeid == 4 || modeid == 5)
	{
		func_para = sigmoid_c;
		val = ss_sum / count;
	}
	return val;
}

//  Deal with Dripping
void grid::Grid::compute_spline_drip_constraint_dcoeff(void)
{

}

void grid::Grid::compute_spline_drip_constraint(void)
{
	init_array(_gbuf.drip_value, float{ 1.0 }, n_surf_points());

	float print_angle = _opt_print_angle;
	int modeid = _ssmode;
	float func_para = sigmoid_c;
	if (modeid == 0 || modeid == 1)
	{
		func_para = p_norm;
	}
	else if (modeid == 2 || modeid == 3)
	{
		func_para = hfunction_c;
	}
	else if (modeid == 4 || modeid == 5)
	{
		func_para = sigmoid_c;
	}

	auto calc_node = [=] __device__(int node_id) {
		float p[3];
		for (int i = 0; i < 3; i++)
		{
			p[i] = gpu_SurfacePoints[i][node_id];
		}

		float normal_vector[3];
		for (int i = 0; i < 3; i++)
		{
			normal_vector[i] = gpu_surface_normal[i][node_id];
		}

		float val = 0.f, inner = 0.f;
		inner = normal_vector[2] / norm(normal_vector) / cosf(print_angle);
		// [MARK] TOADD more enum
		if (modeid == 0)            // p_norm_ss
		{
			if (normal_vector[2] < 0)
			{
				val = pow(inner, func_para);
			}
		}
		else if (modeid == 1)
		{
			val = pow(inner, func_para);
		}
		else if (modeid == 2)       // h_function_ss
		{
			if (normal_vector[2] < 0)
			{
				//val = h(inner, func_para);
				val = 1.0;
			}
		}
		else if (modeid == 3)       // h_function2_ss
		{
			//val = h(inner, func_para);
			val = 1.0;
		}
		else if (modeid == 4)       // overhang_ss
		{
			if (normal_vector[2] < 0)
			{
				//val = oh(inner, func_para);
				val = 1.0;
			}
		}
		else if (modeid == 5)       // overhang2_ss
		{
			//val = oh(inner, func_para);
			val = 1.0;
		}
		return val;
	};

	size_t grid_dim, block_dim;
	int n = n_surf_points();
	make_kernel_param(&grid_dim, &block_dim, n, 256);
	traverse << <grid_dim, block_dim >> > ((float*)_gbuf.drip_value, n, calc_node);
	cudaDeviceSynchronize();
	cuda_error_check;

#ifdef ENABLE_MATLAB
	float* host_spline_constrain = new float[n];
	cudaMemcpy(host_spline_constrain, _gbuf.drip_value, sizeof(float) * n, cudaMemcpyDeviceToHost);
	gpu_manager_t::pass_buf_to_matlab("drip_constraint", host_spline_constrain, n);
	delete host_spline_constrain;
	cuda_error_check;
#endif
}

float grid::Grid::global_drip_constraint(void)
{
	float val = 0.f;

	int modeid = _ssmode;
	float func_para = sigmoid_c;
	float count = grid::Grid::spline_surface_node->size();

	if (modeid % 2 == 0)
	{
		count = count_surface_points();
	}

	std::cout << "--[TEST] number of DRIP constraints : " << count << std::endl;

	float* sum = (float*)grid::Grid::getTempBuf(sizeof(float) * n_surf_points() / 100);
	float drip_sum = parallel_sum(_gbuf.drip_value, sum, n_surf_points());
	cudaDeviceSynchronize();
	cuda_error_check;

	if (modeid == 0 || modeid == 1)
	{
		func_para = p_norm;
		val = pow(drip_sum / count, 1 / func_para) - 1;
	}
	else if (modeid == 2 || modeid == 3)
	{
		func_para = hfunction_c;
		val = drip_sum / count;
	}
	else if (modeid == 4 || modeid == 5)
	{
		func_para = sigmoid_c;
		val = drip_sum / count;
	}
	return val;
}
