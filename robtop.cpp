// robtop.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
// mark

#include <iostream>
#include <string>
#include "config_parser.h"
#include "optimization.h"
#include "test_utils.h"
#include "mma_t.h"
#include "marchingcube_test.h"

extern void version_info(void);
std::string version_hash(void);
extern void init_cuda(void);

config_parser_t parser;

int main(int argc, char** argv)
{
	version_info();
	init_cuda();
	//MMA::test_mma();
	//testMarchingCube();

	//cgalTest();
	selfTest();
	parser.parse(argc, argv);

	setParameters(
		FLAGS_volume_ratio, FLAGS_vol_reduction, FLAGS_design_step, FLAGS_filter_radius, FLAGS_damp_ratio, FLAGS_power_penalty, FLAGS_min_density,
		FLAGS_gridreso, FLAGS_youngs_modulus, FLAGS_poisson_ratio, FLAGS_shell_width,
		FLAGS_logdensity, FLAGS_logcompliance, FLAGS_partitionx, FLAGS_partitiony, FLAGS_partitionz, FLAGS_spline_order, FLAGS_min_coeff, FLAGS_max_coeff, FLAGS_default_print_angle, FLAGS_opt_print_angle);

	setOutpurDir(FLAGS_outdir);

	setInputMesh(FLAGS_meshfile);

	setWorkMode(FLAGS_workmode);

	setSSMode(FLAGS_SSmode);

	setDripMode(FLAGS_Dripmode);

	setBoundaryCondition(parser.inFixArea, parser.inLoadArea, parser.loadField);
	
	buildGrids(parser.mesh_vertices, parser.mesh_faces, parser.mesh_);

	uploadTemplateMatrix();

	logParams("cmdline", version_hash(), argc, argv);

	TestSuit::testMain(FLAGS_testname);

	optimization();
}
