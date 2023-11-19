#pragma  once
#include "MCrender.h"
//#include "globalFunctions.h"
#include<fstream>
#include<iomanip>

#undef max
#undef min
//#define min(x,y) ((x)>(y)?(y):(x))
//#define max(x,y) ((x)>(y)?(x):(y))


MCImplicitRender::MCImplicitRender()
{
}

MCImplicitRender::MCImplicitRender(float minV1, float minV2, int X, TriMesh3D *mesh_, string file_)
{
	minValueBoundary = minV1;
	minValuePotential = minV2;
	Point3 minp = mesh_->PointMin();
	Point3 maxp = mesh_->PointMax();
	meshSize = mesh_->MeshSize();
	center = mesh_->PointCenter();

	Point3 step_ = (maxp - minp) / X;
	radius = min(step_.data()[0], min(step_.data()[1], step_.data()[2]));
	Point3 boundingbox = (maxp - minp) / radius;
	//std::cout << nX << ", " << nY << ", " << nZ << std::endl;
	nX = size_t(std::ceil(boundingbox[0]));
	nY = size_t(std::ceil(boundingbox[1]));
	nZ = size_t(std::ceil(boundingbox[2]));
	//std::cout << nX << ", " << nY << ", " << nZ << std::endl;
	surfaceMesh = mesh_;
	voxelMin = center - Point3((double)nX, (double)nY, (double)nZ) / 2.0 * radius;
	voxelMax = voxelMin + Point3((double)nX, (double)nY, (double)nZ) * radius;

	tvSpline_fun = new TVsplineFunction();
	tvSpline_fun->InitKnot(voxelMin.data()[0], voxelMin.data()[1], voxelMin.data()[2], voxelMax.data()[0], voxelMax.data()[1], voxelMax.data()[2]);
	tvSpline_fun->InitCoffi(file_);
}

MCImplicitRender::MCImplicitRender(float minV1, float minV2, int Nodes[3], TriMesh3D* mesh_, string file_)
{
	minValueBoundary = minV1;
	minValuePotential = minV2;
	Point3 minp = mesh_->PointMin();
	Point3 maxp = mesh_->PointMax();
	meshSize = mesh_->MeshSize();
	center = mesh_->PointCenter();

	//Point3 step_ = (maxp - minp) / X;
	//radius = min(step_.data()[0], min(step_.data()[1], step_.data()[2]));
	/*Point3 boundingbox = (maxp - minp) / radius;*/
	//std::cout << nX << ", " << nY << ", " << nZ << std::endl;
	//nX = size_t(std::ceil(boundingbox[0]));
	//nY = size_t(std::ceil(boundingbox[1]));
	//nZ = size_t(std::ceil(boundingbox[2]));
	//std::cout << nX << ", " << nY << ", " << nZ << std::endl;
	radius = (maxp.data()[0] - minp.data()[0]) / Nodes[0];
	nX = Nodes[0];
	nY = Nodes[1];
	nZ = Nodes[2];
	surfaceMesh = mesh_;
	voxelMin = center - Point3((double)nX, (double)nY, (double)nZ) / 2.0 * radius;
	voxelMax = voxelMin + Point3((double)nX, (double)nY, (double)nZ) * radius;

	tvSpline_fun = new TVsplineFunction();
	tvSpline_fun->InitKnot(voxelMin.data()[0], voxelMin.data()[1], voxelMin.data()[2], voxelMax.data()[0], voxelMax.data()[1], voxelMax.data()[2]);
	tvSpline_fun->InitCoffi(file_);
}


MCImplicitRender::~MCImplicitRender(){}

void MCImplicitRender::AddSetting(float minV1, float minV2, int Nodes[3], TriMesh3D* mesh_, float boxmin[3], float boxmax[3])
{
	minValueBoundary = minV1;
	minValuePotential = minV2;
	Point3 minp = mesh_->PointMin();
	Point3 maxp = mesh_->PointMax();
	meshSize = mesh_->MeshSize();
	center = mesh_->PointCenter();

	//Point3 step_ = (maxp - minp) / X;
	//radius = min(step_.data()[0], min(step_.data()[1], step_.data()[2]));
	/*Point3 boundingbox = (maxp - minp) / radius;*/
	//std::cout << nX << ", " << nY << ", " << nZ << std::endl;
	//nX = size_t(std::ceil(boundingbox[0]));
	//nY = size_t(std::ceil(boundingbox[1]));
	//nZ = size_t(std::ceil(boundingbox[2]));
	//std::cout << nX << ", " << nY << ", " << nZ << std::endl;
	radius = (maxp.data()[0] - minp.data()[0]) / Nodes[0];
	nX = Nodes[0];
	nY = Nodes[1];
	nZ = Nodes[2];
	surfaceMesh = mesh_;
	voxelMin = center - Point3((double)nX, (double)nY, (double)nZ) / 2.0 * radius;
	voxelMax = voxelMin + Point3((double)nX, (double)nY, (double)nZ) * radius;
}

void MCImplicitRender::InitSurface()
{
	if (tvSpline_fun->IsNull() == 0)
	{
		return;
	}

	mcBoundary.clear();
	mcBoundary.resize(nX * nY * nZ);
	for (int i = 0; i < nX; i++)
	{
		for (int j = 0; j < nY; j++)
		{
			for (int k = 0; k < nZ; k++)
			{
				mp4vector vert(voxelMin.data()[0] + i * radius, voxelMin.data()[1] + j * radius, voxelMin.data()[2] + k * radius, 0);
				vert.val = PotentialBoundary((Point3)vert);
				mcBoundary[i * (nY) * (nZ)+j * (nZ)+k] = vert;
			}
		}
	}
}

void MCImplicitRender::InitData()
{
	mcPoints.clear();
	mcPoints.resize(nX * nY * nZ);
	//std::cout << " test : " << voxelMin.data()[0] << " , " << voxelMin.data()[1] << " , " << voxelMin.data()[2] << std::endl;
	//std::cout << " test : " << radius << std::endl;
	for (int i = 0; i < nX; i++)
	{
		for (int j = 0; j < nY; j++)
		{
			for (int k = 0; k < nZ; k++)
			{
				mp4vector vert(voxelMin.data()[0] + i * radius, voxelMin.data()[1] + j * radius, voxelMin.data()[2] + k * radius, 0);
				vert.val = Potential((Point3)vert);
				mcPoints[i * (nY) * (nZ)+j * (nZ)+k] = vert;
			}
		}
	}
}

void MCImplicitRender::RunMarchingCubesTestPotential(float& minValuePotential, std::vector<float> bg_node[3], std::vector<float>& mcPoints_inner_val)
{
	mcPoints.clear();
	int n = bg_node->size();
	mcPoints.resize(n);
	for (int i = 0; i < n; i++)
	{
		mp4vector verr(bg_node[0][i], bg_node[1][i], bg_node[2][i], mcPoints_inner_val[i]);
		mcPoints[i] = verr;
	}
	
	if (mcPoints.size() == 0)
	{
		return;
	}
	std::cout << "(" << nX << ", " << nY << ", " << nZ << ") " << nX * nY * nZ << std::endl;
	std::cout << mcPoints_inner_val.size() << std::endl;
	std::cout << bg_node->size() << std::endl;
	//delete[] Triangles;	  //first free the previous allocated memory
	//Triangles = nullptr;
	Triangles = MarchingCubes(nX - 1, nY - 1, nZ - 1, 1.0, 1.0, 1.0, minValuePotential, mcPoints, numOfTriangles);
	//std::cout << "Marching cube test \n" << "Surface node number:" << mcPoints.size() << std::endl;
	int num = 0;
	for (int i = 0; i < mcPoints.size(); i++)
	{
		if (mcPoints[i].val < 1e-6)
		{
			//std::cout << mcPoints[i].x << "," << mcPoints[i].y << "," << mcPoints[i].z << "," << mcPoints[i].val << "," << std::endl;
			num++;
		}
		//std::cout << mcPoints[i].x << "," << mcPoints[i].y << "," << mcPoints[i].z << "," << mcPoints[i].val << "," << std::endl;
	}
	std::cout << "sample nodes number: " << num  << "/" << mcPoints.size() << std::endl;
	std::cout << "-=-=-=-=-=-=-=-=-=-=-=-=-=-" << std::endl;
	mcPoints.clear();
}

void MCImplicitRender::RunMarchingCubesTest()
{
	if (mcPoints.size() == 0)
	{
		std::cout << "marching cube boundary size is zero!" << std::endl;
		return;
	}
	delete[] Triangles;	//first free the previous allocated memory
	Triangles = nullptr;
	Triangles = MarchingCubes(nX - 1, nY - 1, nZ - 1, 1.0, 1.0, 1.0, minValuePotential, mcPoints, numOfTriangles);
	int num = 0;
	for (int i = 0; i < mcPoints.size(); i++)
	{
		if (mcPoints[i].val > 1e-6)
		{
			num++;
		}
	}
}

void MCImplicitRender::RunMarchingCubesTestBoundary()
{
	//delete[] TrianglesBoundary;	     //first free the previous allocated memory
	if (mcBoundary.size() == 0)
	{
		std::cout << "marching cube boundary size is zero!" << std::endl;
		return;
	}
	TrianglesBoundary = MarchingCubes(nX-1, nY-1, nZ-1, 5.0, 10.0, 1.0, minValueBoundary, mcBoundary, numOfTrianglesBoundary);
	std::cout << "Marching cube test \n" << "Surface boundary node number:" << mcBoundary.size() << std::endl;
	for (int i = 0; i < 100; i++)
	{
		std::cout << mcBoundary[i].x << "," << mcBoundary[i].y << "," << mcBoundary[i].z << "," << mcBoundary[i].val << "," << std::endl;
	}
	std::cout << "-=-=-=-=-=-=-=-=-=-=-=-=-=-" << std::endl;
}

float MCImplicitRender::PotentialBoundary(Point3 p)
{
	float f = tvSpline_fun->calculate_cublic_tensor_product(p.data()[0], p.data()[1], p.data()[2]);
	return f;
}

float MCImplicitRender::Potential(Point3 p)
{
	//if (blobs_fun.cell_centers.size() == 0)
	//{
	//	return 0;
	//}
	//float f = blobs_fun.calculate_value(p.data()[0], p.data()[1], p.data()[2]);
	//return f;
	
	//
	// Input Surface
	//
	float x = p.data()[0];
	float y = p.data()[1];
	float z = p.data()[2];
	float f = cosf(2 * M_PI * t * x) + cosf(2 * M_PI * t * y) + cosf(2 * M_PI * t * z) + 0.1;
	return f;
}

void MCImplicitRender::TestMC()
{
	InitData();
	minValuePotential = 0;
	RunMarchingCubesTest();
	InnerTransferToOpenMeshTest();
	TransferToOpenMesh();
	SavePoints("isosurface.obj");

	//InitData();
	//minValueBoundary = 0;
	//RunMarchingCubesTestBoundary();
	//OuterTransferToOpenMesh();
	//TransferToOpenMesh();
}

void MCImplicitRender::DoMC()
{
	//InitData();
	//minValuePotential = 0;
	//RunMarchingCubesTest();
	//InnerTransferToOpenMeshTest();
	//TransferToOpenMesh();
	//SavePoints("isosurface.obj");

	InitData();
	minValueBoundary = 0;
	RunMarchingCubesTestBoundary();
	OuterTransferToOpenMesh();
	TransferToOpenMesh();
}

//void MCImplicitRender::OnKey(QKeyEvent* event)
//{
//	switch (event->key()) {
//	case Qt::Key_F3: minValuePotential += 0.02; RunMarchingCubesTestPotential(); InnerTransferToOpenMesh();  TransferToOpenMesh();
//		cout << "change minValuePotential = " << minValuePotential << endl;  break;
//	case Qt::Key_F4: minValuePotential -= 0.02; RunMarchingCubesTestPotential();  InnerTransferToOpenMesh(); TransferToOpenMesh();
//		cout << "change minValuePotential = " << minValuePotential << endl;  break;
//	case 'Q': nX > 1 ? nX-- : nX; InitData(); RunMarchingCubesTestPotential(); InnerTransferToOpenMesh(); TransferToOpenMesh();
//		cout << "change nX = " << nX << endl;  break;
//	case 'W': nX++; InitData(); RunMarchingCubesTestPotential(); InnerTransferToOpenMesh(); TransferToOpenMesh();
//		cout << "change nX = " << nX << endl;  break;
//	case 'A': nY > 1 ? nY-- : nY; InitData(); RunMarchingCubesTestPotential();  InnerTransferToOpenMesh(); TransferToOpenMesh();
//		cout << "change nY = " << nY << endl;  break;
//	case 'S': nY++; InitData(); RunMarchingCubesTestPotential(); InnerTransferToOpenMesh();  TransferToOpenMesh();
//		cout << "change nY = " << nY << endl;  break;
//	case 'Z': nZ > 1 ? nZ-- : nZ; InitData(); RunMarchingCubesTestPotential();  InnerTransferToOpenMesh(); TransferToOpenMesh();
//		cout << "change nZ = " << nZ << endl; break;
//	case 'X': nZ++; InitData(); RunMarchingCubesTestPotential();  InnerTransferToOpenMesh(); TransferToOpenMesh();
//		cout << "change nZ = " << nZ << endl; break;
//
//	case Qt::Key_F1: minValueBoundary += 0.02; RunMarchingCubesTestBoundary(); OuterTransferToOpenMesh();  TransferToOpenMesh();
//		cout << "change minValueBoundary = " << minValueBoundary << endl;  break;
//	case Qt::Key_F2: minValueBoundary -= 0.02; RunMarchingCubesTestBoundary(); OuterTransferToOpenMesh(); TransferToOpenMesh();
//		cout << "change minValueBoundary = " << minValueBoundary << endl;  break;
//		/*case 'Q': nX > 1 ? nX-- : nX; InitData(); RunMarchingCubesTestBoundary();
//		cout << "change nX = " << nX << endl;  break;
//		case 'W': nX++; InitData(); RunMarchingCubesTestBoundary();
//		cout << "change nX = " << nX << endl;  break;
//		case 'A': nY > 1 ? nY-- : nY; InitData(); RunMarchingCubesTestBoundary();
//		cout << "change nY = " << nY << endl;  break;
//		case 'S': nY++; InitData(); RunMarchingCubesTestBoundary();
//		cout << "change nY = " << nY << endl;  break;
//		case 'Z': nZ > 1 ? nZ-- : nZ; InitData(); RunMarchingCubesTestBoundary();
//		cout << "change nZ = " << nZ << endl; break;
//		case 'X': nZ++; InitData(); RunMarchingCubesTestBoundary();
//		cout << "change nZ = " << nZ << endl; break;*/
//	default: break;
//	}
//}

void MCImplicitRender::Remesh()
{

}

void MCImplicitRender::OuterTransferToOpenMesh()
{
	MyMesh mesh;
	for (int i = 0; i < numOfTrianglesBoundary; i++)
	{
		std::vector<OpenMesh::VertexHandle>  points;
		for (int j = 0; j < 3; j++)
		{
			points.push_back(mesh.add_vertex(TrianglesBoundary[i].p[j]));
		}
		mesh.add_face(points);
	}
	mesh.request_vertex_normals();
	mesh.request_face_normals();
	outer_mesh = mesh;
}

void MCImplicitRender::InnerTransferToOpenMeshTest()
{
	MyMesh mesh;
	//std::cout << numOfTriangles << std::endl;
	for (int i = 0; i < numOfTriangles; i++)
	{
		std::vector<OpenMesh::VertexHandle>  points;
		for (int j = 0; j < 3; j++)
		{
			points.push_back(mesh.add_vertex(Triangles[i].p[j]));
		}
		mesh.add_face(points);
	}
	std::vector<float> vert[3];
	std::vector<float> vert_fine[3];
	Point3 p;

	for (const auto& vh : mesh.vertices())
	{
		p = mesh.point(vh);
		for (int i = 0; i < 3; i++)
		{
			vert[0].push_back(p[0]);
			vert[1].push_back(p[1]);
			vert[2].push_back(p[2]);
		}
	}
	//std::cout << vert->size() << std::endl;
	for (int i = 0; i < vert->size(); i = i + 3)
	{
		vert_fine[0].push_back(vert[0][i]);
		vert_fine[1].push_back(vert[1][i]);
		vert_fine[2].push_back(vert[2][i]);
	}
	//std::cout << vert_fine->size() << std::endl;
	//for (int i = 0; i < 10; i++)
	//{
	//	std::cout << vert[0][3 * i] << ", " << vert[1][3 * i] << ", " << vert[2][3 * i] << std::endl;
	//	std::cout << vert[0][3 * i + 1] << ", " << vert[1][3 * i + 1] << ", " << vert[2][3 * i + 1] << std::endl;
	//	std::cout << vert[0][3 * i + 2] << ", " << vert[1][3 * i + 2] << ", " << vert[2][3 * i + 2] << std::endl;
	//	std::cout << "\033[32m" << vert_fine[0][i] << ", " << vert_fine[1][i] << ", " << vert_fine[2][i] << "\033[0m" << std::endl;
	//}
	mesh.request_vertex_normals();
	mesh.request_face_normals();
	inner_mesh = mesh;

	for (int i = 0; i < 3; i++)
	{
		std::vector<float>().swap(vert[i]);
		std::vector<float>().swap(vert_fine[i]);
	}
}

void MCImplicitRender::InnerTransferToOpenMesh(std::vector<float>& surface_node_x, std::vector<float>& surface_node_y, std::vector<float>& surface_node_z)
{
	MyMesh mesh;
	for (int i = 0; i < numOfTriangles; i++)
	{
		std::vector<OpenMesh::VertexHandle>  points;
		for (int j = 0; j < 3; j++)
		{
			points.push_back(mesh.add_vertex(Triangles[i].p[j]));
		}
		mesh.add_face(points);
	}

	std::vector<float> vert[3];
	std::vector<float> vert_fine[3];
	std::vector<float> vert_final[3];
	Point3 p;

	for (const auto& vh : mesh.vertices())
	{
		p = mesh.point(vh);
		for (int i = 0; i < 3; i++)
		{
			vert[0].push_back(p[0]);
			vert[1].push_back(p[1]);
			vert[2].push_back(p[2]);
		}
	}
	std::cout << vert->size() << std::endl;
	for (int i = 0; i < vert->size(); i = i + 3)
	{
		vert_fine[0].push_back(vert[0][i]);
		vert_fine[1].push_back(vert[1][i]);
		vert_fine[2].push_back(vert[2][i]);
	}
	std::cout << "Surface node number( before sampled ): " << vert_fine->size() << std::endl;
	surface_node_x = vert_fine[0];
	surface_node_y = vert_fine[1];
	surface_node_z = vert_fine[2];

	mesh.request_vertex_normals();
	mesh.request_face_normals();
	inner_mesh = mesh;
	
	for (int i = 0; i < 3; i++)
	{
		std::vector<float>().swap(vert[i]);
		std::vector<float>().swap(vert_fine[i]);
		std::vector<float>().swap(vert_final[i]);
	}

	delete[] Triangles;
	Triangles = nullptr;
	mesh.garbage_collection();
}

void MCImplicitRender::save_to_surface_node(std::vector<float>& surface_node_x, std::vector<float>& surface_node_y, std::vector<float>& surface_node_z)
{
	MyMesh mesh;
	for (int i = 0; i < numOfTriangles; i++)
	{
		std::vector<OpenMesh::VertexHandle>  points;
		for (int j = 0; j < 3; j++)
		{
			points.push_back(mesh.add_vertex(Triangles[i].p[j]));
		}
		mesh.add_face(points);
	}

	std::vector<float> vert[3];
	std::vector<float> vert_fine[3];
	std::vector<float> vert_final[3];
	Point3 p;

	for (const auto& vh : mesh.vertices())
	{
		p = mesh.point(vh);
		for (int i = 0; i < 3; i++)
		{
			vert[0].push_back(p[0]);
			vert[1].push_back(p[1]);
			vert[2].push_back(p[2]);
		}
	}
	for (int i = 0; i < vert->size(); i = i + 3)
	{
		vert_fine[0].push_back(vert[0][i]);
		vert_fine[1].push_back(vert[1][i]);
		vert_fine[2].push_back(vert[2][i]);
	}
	std::cout << "Surface node number( before sampled ): " << vert_fine->size() << std::endl;
	surface_node_x = vert_fine[0];
	surface_node_y = vert_fine[1];
	surface_node_z = vert_fine[2];

	for (int i = 0; i < 3; i++)
	{
		std::vector<float>().swap(vert[i]);
		std::vector<float>().swap(vert_fine[i]);
		std::vector<float>().swap(vert_final[i]);
	}

	delete[] Triangles;
	Triangles = nullptr;
	mesh.garbage_collection();
}

void MCImplicitRender::TransferToOpenMesh()
{
	if (outer_mesh.n_vertices()==0)
	{
		implicit_mesh = inner_mesh;
		return;
	}
	else  if (inner_mesh.n_vertices()==0)
	{
		implicit_mesh = outer_mesh;
		return;
	}
	
	MyMesh mesh;
	mesh = outer_mesh;
	for (int i = 0; i < numOfTriangles; i++)
	{
		std::vector<OpenMesh::VertexHandle>  points;
		for (int j = 2; j >=0; j--)   //to modify
		{
			points.push_back(mesh.add_vertex(Triangles[i].p[j]));
		}
		mesh.add_face(points);
	}
	mesh.request_vertex_normals();
	mesh.request_face_normals();
	implicit_mesh = mesh;
}

void MCImplicitRender::SavePoints(const std::string& output_filename)
{
	//
	// change the filename
	//
	//std::string filename = "isomesh_outer.obj";
	std::string filename = output_filename;
	try
	{
		//if (!OpenMesh::IO::write_mesh(mesh, filename, wopt))
		if (!OpenMesh::IO::write_mesh(implicit_mesh, filename))
		{
			std::cerr << "Cannot write mesh to file!" << std::endl;
		}
		else
		{
			printf("-- Marching cube Test Passed \n");
		}
	}
	catch (std::exception& x)
	{
		std::cerr << x.what() << std::endl;
	}
	
	//for (auto v_it = implicit_mesh.vertices_begin(); v_it != implicit_mesh.vertices_end(); ++v_it)
	//{
	//	std::cout << implicit_mesh.point(*v_it) << std::endl;
	//	float s = 0;
	//	Point point;
	//	for (int i = 0; i < 3; i++)
	//	{
	//		point.data()[i] = implicit_mesh.point(*v_it)[i];
	//	}
	//	s = cosf(2 * M_PI * t * point.data()[0]) + cosf(2 * M_PI * t * point.data()[1]) + cosf(2 * M_PI * t * point.data()[2]) + c;
	//	std::cout << "s" << s << std::endl;

	//	std::cout << (implicit_mesh.point(*v_it) - center) / meshSize << std::endl;
	//	float s1 = 0;
	//	Point point1;
	//	for (int i = 0; i < 3; i++)
	//	{
	//		point1.data()[i] = ((implicit_mesh.point(*v_it) - center) / meshSize)[i];
	//	}
	//	s = cosf(2 * M_PI * t * point1.data()[0]) + cosf(2 * M_PI * t * point1.data()[1]) + cosf(2 * M_PI * t * point1.data()[2]) + c;
	//	std::cout << "s1" << s1 << std::endl;

	//}
}

//render
void MCImplicitRender::DrawBoundar_(void) const
{
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH, GL_NICEST);
	glBegin(GL_POINTS);
	/*glColor3f(1, 0, 0);
	for (int i = 0; i < boundary_.size(); i++)
	{
		glVertex3fv(((boundary_[i] - center) / meshSize).data());
	}
	glColor3f(0, 1, 0);
	for (int i = 0; i < outer_.size(); i++)
	{
		glVertex3fv(((outer_[i] - center) / meshSize).data());
	}
	glColor3f(0, 0, 1);
	for (int i = 0; i < inner_.size(); i++)
	{
		glVertex3fv(((inner_[i] - center) / meshSize).data());
	}
	glColor3f(0, 0, 1);
	for (int i = 0; i < all_.size(); i++)
	{
	glVertex3fv(((all_[i] - center) / meshSize).data());
	}
	glEnd();*/
	
}

void MCImplicitRender::DrawPoints(void) const
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH, GL_NICEST);

	glColor3f(0.3, 0.3, 0.3);
	glBegin(GL_POINTS);
	glPointSize(2.0f);
	for (auto v_it = implicit_mesh.vertices_begin(); v_it != implicit_mesh.vertices_end(); ++v_it)
	{
		glColor3d(0.3, 0.3, 0.3);
		glNormal3fv(implicit_mesh.normal(*v_it).data());
		glVertex3fv(((implicit_mesh.point(*v_it) - center) / meshSize).data());
	}
	glEnd();
	glDisable(GL_BLEND);
}

void MCImplicitRender::DrawEdges(void) const
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);

	glColor3f(0.3, 0.3, 0.3);
	glBegin(GL_LINES);
	for (auto e_it = implicit_mesh.edges_begin(); e_it != implicit_mesh.edges_end(); ++e_it)
	{
		glColor3d(0.0, 0.0, 0.0);
		auto heh = implicit_mesh.halfedge_handle(*e_it, 0);
		auto v0 = implicit_mesh.from_vertex_handle(heh);
		auto v1 = implicit_mesh.to_vertex_handle(heh);
		glNormal3fv(implicit_mesh.normal(v0).data());
		glVertex3fv(((implicit_mesh.point(v0) - center) / meshSize).data());
		glNormal3fv(implicit_mesh.normal(v1).data());
		glVertex3fv(((implicit_mesh.point(v1) - center) / meshSize).data());
	}
	glEnd();
	glDisable(GL_BLEND);
}

void MCImplicitRender::DrawFaces(void) const
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);

	glColor3d(1.0, 0, 0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.5f, 2.0f);
	glBegin(GL_TRIANGLES);
	for (auto f_it = implicit_mesh.faces_begin(); f_it != implicit_mesh.faces_end(); ++f_it)
	{
		glColor3d(1.0, 1.0, 0.5);
		for (auto fv_it = implicit_mesh.cfv_begin(*f_it); fv_it != implicit_mesh.cfv_end(*f_it); ++fv_it)
		{
			glNormal3fv(implicit_mesh.normal(*f_it).data());
			glVertex3fv(((implicit_mesh.point(*fv_it) - center) / meshSize).data());
		}
	}
	glEnd();
	glDisable(GL_POLYGON_OFFSET_FILL);
	glDisable(GL_BLEND);
}

void MCImplicitRender::DrawFlatShading(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);

	const float color[] = { r, g, b, 1.0f };

	glColor3f(r, g, b);
	glEnable(GL_FLAT);
	glShadeModel(GL_FLAT);

	setMatirial(color, 30.0);
	glBegin(GL_TRIANGLES);
	for (auto face = implicit_mesh.faces_begin(); face != implicit_mesh.faces_end(); ++face)
	{
		for (auto it = implicit_mesh.fv_begin(*face); it != implicit_mesh.fv_end(*face); ++it)
		{
			auto vertex = *it;
			glNormal3fv(implicit_mesh.normal(vertex).data());
			auto point = implicit_mesh.point(*it);
			glVertex3fv(((point - center) / meshSize).data());
		}
	}
	implicit_mesh.update_normals();
	implicit_mesh.update_face_normals();
	glEnd();
	glDisable(GL_FLAT);
	glDisable(GL_BLEND);
}

void MCImplicitRender::DrawFlatWire(void)
{
	glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_LIGHTING_BIT);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.0, 1);

	renderFlat();

	glDisable(GL_POLYGON_OFFSET_FILL);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

	renderWire();

	glPopAttrib();
	glDisable(GL_BLEND);
}

void MCImplicitRender::renderFlat(void)
{
	const GLfloat color[] = { r, g, b, 1.0f };
	setMatirial(color, 30.0);
	glBegin(GL_TRIANGLES);
	for (auto face = implicit_mesh.faces_begin(); face != implicit_mesh.faces_end(); ++face)
	{
		for (auto it = implicit_mesh.fv_begin(*face); it != implicit_mesh.fv_end(*face); ++it)
		{
			auto vertex = *it;
			glNormal3fv(implicit_mesh.normal(vertex).data());
			auto point = implicit_mesh.point(*it);
			glVertex3fv(((point - center) / meshSize).data());
		}
	}
	glEnd();
}

void MCImplicitRender::renderWire(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);

	glDisable(GL_LIGHTING);
	glColor3f(r, g, b);
	glDepthRange(0.0, 1.0);
	glDepthFunc(GL_LEQUAL);

	glLineWidth(0.1);
	glColor3f(0, 0, 0);
	glBegin(GL_LINES);
	for (auto it = implicit_mesh.halfedges_begin(); it != implicit_mesh.halfedges_end(); ++it)
	{
		auto fromVertex = implicit_mesh.from_vertex_handle(*it);
		auto toVertex = implicit_mesh.to_vertex_handle(*it);

		auto pointA = implicit_mesh.point(fromVertex);
		glVertex3fv(((pointA - center) / meshSize).data());
		glNormal3fv(implicit_mesh.normal(fromVertex).data());
		auto pointB = implicit_mesh.point(toVertex);
		glVertex3fv(((pointB - center) / meshSize).data());
		glNormal3fv(implicit_mesh.normal(toVertex).data());
	}
	//implicit_mesh.update_vertex_normals();
	implicit_mesh.update_normals();
	//implicit_mesh.update_vertex_normals();
	glEnd();
	glDisable(GL_BLEND);
	glDepthFunc(GL_LESS);
}

void MCImplicitRender::DrawSmoothShading(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);

	const GLfloat color[] = { r, g, b, 1.0f };
	glShadeModel(GL_SMOOTH);

	setMatirial(color, 30.0);
	glBegin(GL_TRIANGLES);
	for (auto face = implicit_mesh.faces_begin(); face != implicit_mesh.faces_end(); ++face)
	{
		for (auto it = implicit_mesh.fv_begin(*face); it != implicit_mesh.fv_end(*face); ++it)
		{
			auto vertex = *it;
			glNormal3fv(implicit_mesh.normal(vertex).data());
			auto point = implicit_mesh.point(*it);
			glVertex3fv(((point - center) / meshSize).data());;
		}
	}
	implicit_mesh.update_normals();
	glEnd();
	glDisable(GL_BLEND);
}

void MCImplicitRender::DrawTransparent(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);

	const GLfloat color[] = { r, g, b, 0.12f };
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDepthMask(GL_FALSE);
	setMatirial(color, 30.0);
	glPushMatrix();

	glBegin(GL_TRIANGLES);
	for (auto face = implicit_mesh.faces_begin(); face != implicit_mesh.faces_end(); ++face)
	{
		for (auto it = implicit_mesh.fv_begin(*face); it != implicit_mesh.fv_end(*face); ++it)
		{
			auto vertex = *it;
			glNormal3fv(implicit_mesh.normal(vertex).data());
			auto point = implicit_mesh.point(*it);
			glVertex3fv(((point - center) / meshSize).data());;
		}
	}
	implicit_mesh.update_normals();
	glEnd();
	glPopMatrix();

	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);
}

void MCImplicitRender::DrawWireFrame(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glLineWidth(0.5);
	glColor3f(r, g, b);
	glBegin(GL_LINES);
	for (auto it = implicit_mesh.halfedges_begin(); it != implicit_mesh.halfedges_end(); ++it)
	{
		auto fromVertex = implicit_mesh.from_vertex_handle(*it);
		auto toVertex = implicit_mesh.to_vertex_handle(*it);

		auto pointA = implicit_mesh.point(fromVertex);
		glVertex3fv(((pointA - center) / meshSize).data());
		glNormal3fv(implicit_mesh.normal(fromVertex).data());
		auto pointB = implicit_mesh.point(toVertex);
		glVertex3fv(((pointB - center) / meshSize).data());
		glNormal3fv(implicit_mesh.normal(toVertex).data());
	}
	//implicit_mesh.update_vertex_normals();
	implicit_mesh.update_normals();
	glEnd();
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_BLEND);
}

void MCImplicitRender::setMatirial(const float mat_diffuse[4], float mat_shininess)
{
	static const float mat_specular[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	static const float mat_emission[] = { 0.0f, 0.0f, 0.0f, 1.0f };

	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);
	glMaterialf(GL_FRONT, GL_SHININESS, mat_shininess);
}

void MCImplicitRender::DrawBoundingBox(void)
{
	GLfloat vertex_list[][3] = {
		(voxelMin.data()[0] - center.data()[0]) / meshSize, (voxelMax.data()[1] - center.data()[1]) / meshSize, (voxelMin.data()[2] - center.data()[2]) / meshSize,
		(voxelMin.data()[0] - center.data()[0]) / meshSize, (voxelMin.data()[1] - center.data()[1]) / meshSize, (voxelMin.data()[2] - center.data()[2]) / meshSize,
		(voxelMax.data()[0] - center.data()[0]) / meshSize, (voxelMin.data()[1] - center.data()[1]) / meshSize, (voxelMin.data()[2] - center.data()[2]) / meshSize,
		(voxelMax.data()[0] - center.data()[0]) / meshSize, (voxelMax.data()[1] - center.data()[1]) / meshSize, (voxelMin.data()[2] - center.data()[2]) / meshSize,
		(voxelMin.data()[0] - center.data()[0]) / meshSize, (voxelMax.data()[1] - center.data()[1]) / meshSize, (voxelMax.data()[2] - center.data()[2]) / meshSize,
		(voxelMin.data()[0] - center.data()[0]) / meshSize, (voxelMin.data()[1] - center.data()[1]) / meshSize, (voxelMax.data()[2] - center.data()[2]) / meshSize,
		(voxelMax.data()[0] - center.data()[0]) / meshSize, (voxelMin.data()[1] - center.data()[1]) / meshSize, (voxelMax.data()[2] - center.data()[2]) / meshSize,
		(voxelMax.data()[0] - center.data()[0]) / meshSize, (voxelMax.data()[1] - center.data()[1]) / meshSize, (voxelMax.data()[2] - center.data()[2]) / meshSize,
	};

	static const GLint index_list[][4] = {
		0, 1, 2, 3,//bottem  
		0, 3, 7, 4,//left  
		2, 3, 7, 6,//front  
		1, 2, 6, 5,//right  
		0, 1, 5, 4,//back  
		4, 5, 6, 7//top  
	};

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor3f(0.3, 0.3, 0.3);
	glLineWidth(3.0f);

	for (int i = 0; i < 6; ++i)      // 有六个面，循环六次  
	{
		glBegin(GL_LINE_LOOP);
		for (int j = 0; j < 4; ++j)     // 每个面有四个顶点，循环四次  
			glVertex3fv(vertex_list[index_list[i][j]]);
		glEnd();
	}

	glFlush();
	glDisable(GL_BLEND);
}

