#include "TriMesh3D.h"
#include <windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>

#if defined(DEBUG) || defined(_DEBUG)
#pragma comment(lib, "OpenMeshCored.lib")
#pragma comment(lib, "OpenMeshToolsd.lib")
#else
#pragma comment(lib, "OpenMeshCore.lib")
#pragma comment(lib, "OpenMeshTools.lib")
#endif
#pragma comment(lib, "opengl32.lib")

TriMesh3D::TriMesh3D(void)
	:meshSize(0.0), r(0.56), g(0.56), b(0.56)
{
}

TriMesh3D::~TriMesh3D(void)
{
}

bool TriMesh3D::LoadFile(const std::string &filename)
{
	//#ifdef _DEBUG
	//std::cout << "  Start loading file " << filename << std::endl;
	//#endif
	mesh.clear();
	mesh.request_vertex_normals();
	OpenMesh::IO::Options opt;
	if (!OpenMesh::IO::read_mesh(mesh, filename, opt))
	{
		return false;
	}
	if (mesh.vertices_empty())
	{
		return false;
	}
	if (!opt.check(OpenMesh::IO::Options::VertexNormal))
	{
		mesh.request_face_normals();
		mesh.update_normals();
	}
	ComputeBoundingbox();
//#ifdef _DEBUG
	//std::cout << "  " << filename << " loaded." << std::endl;
	//std::cout << "  Bounding box: (" << pointMin << ")" << std::endl;
	//std::cout << "                (" << pointMax << ")" << std::endl;
//#endif
	return true;
}

void TriMesh3D::init(void)
{
	OpenMesh::VectorT<float, 3> xm(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
	OpenMesh::VectorT<float, 3> xM = -xm;
	MyMesh test = mesh;
	std::array<float, 3> xmin;
	std::array<float, 3> xmax;

	for (auto itv = test.vertices_begin(); itv != test.vertices_end(); itv++) {
		OpenMesh::VectorT<float, 3> x = test.point(*itv);
		xm.minimize(x);
		xM.maximize(x);
	}
	for (int i = 0; i < 3; i++) {
		xmin[i] = xm[i];
		xmax[i] = xM[i];
	}
	printf("model bounding box : [%f, %f, %f] -- [%f, %f, %f]\n", xmin[0], xmin[1], xmin[2], xmax[0], xmax[1], xmax[2]);
	OpenMesh::VectorT<float, 3> len3 = xM - xm;
	float min_len = *std::min_element(len3.begin(), len3.end());
	float scale_ratio = 2 / min_len;
	for (int i = 0; i < 3; i++) {
		xmin[i] *= scale_ratio;
		xmax[i] *= scale_ratio;
	}

	MyMesh mesh_scaled = test;
	for (auto& vt : mesh_scaled.vertices())
	{
		mesh_scaled.set_point(vt, mesh_scaled.point(vt) * scale_ratio);
	}
	mesh_scaled.request_vertex_normals();
	mesh_scaled.request_face_normals();
	mesh_scaled.update_normals();

	mesh = mesh_scaled;
}


void TriMesh3D::TestRotateMesh(void)
{
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
	{
		Point3 p = mesh.point(*v_it);
		mesh.set_point(*v_it, Point3(p[0], -p[2], p[1]));
	}
	ComputeBoundingbox();
}

void TriMesh3D::TestScaleMesh(void)
{
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
	{
		Point3 p = mesh.point(*v_it);
		mesh.set_point(*v_it, Point3(1000 * p[0], 1000 * p[1], 1000 * p[2]));
	}
	ComputeBoundingbox();
}

bool TriMesh3D::SaveFile(const std::string &filename) const
{
	if (mesh.vertices_empty()) return false;
	if (!OpenMesh::IO::write_mesh(mesh, filename))
	{
		std::cerr << "Cannot write mesh to file" << std::endl;
		return false;
	}
	//#ifdef _DEBUG
	std::cout << "Mesh saved to " << filename << "." << std::endl;
	//#endif
	return true;
}

void TriMesh3D::setMatirial(const float mat_diffuse[4], float mat_shininess)
{
	static const float mat_specular[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	static const float mat_emission[] = { 0.0f, 0.0f, 0.0f, 1.0f };

	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);
	glMaterialf(GL_FRONT, GL_SHININESS, mat_shininess);
}

void TriMesh3D::DrawAxis(void) const
{
	glLineWidth(2.0);
	//x axis
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(1, 0.0, 0.0);
	glEnd();
	glPushMatrix();
	glTranslatef(1, 0, 0);
	glRotatef(90, 0.0, 1, 0.0);
	//glutSolidCone(0.02, 0.06, 20, 10);
	glPopMatrix();

	//y axis
	glColor3f(0.0, 1.0, 0.0);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(0.0, 1, 0.0);
	glEnd();

	glPushMatrix();
	glTranslatef(0.0, 1, 0);
	glRotatef(90, -1, 0.0, 0.0);
	//glutSolidCone(0.02, 0.06, 20, 10);
	glPopMatrix();

	//z axis
	glColor3f(0.0, 0.0, 1.0);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(0.0, 0.0, 1);
	glEnd();
	glPushMatrix();
	glTranslatef(0.0, 0, 1);
	//glutSolidCone(0.02, 0.06, 20, 10);
	glPopMatrix();

	glColor3f(1.0, 1.0, 1.0);
}

void TriMesh3D::DrawPoints(void) const
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH, GL_NICEST);

	glColor3f(0.3, 0.3, 0.3);
	glPointSize(2.0f);
	glBegin(GL_POINTS);
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
	{
		glColor3d(0.3, 0.3, 0.3);
		glNormal3fv(mesh.normal(*v_it).data());
		glVertex3fv(((mesh.point(*v_it) - pointCenter) / meshSize).data());
	}
	glEnd();

	glDisable(GL_BLEND);
}

void TriMesh3D::DrawEdges(void) const
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBegin(GL_LINES);
	for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
	{
		glColor3d(0.0, 0.0, 0.0);
		auto heh = mesh.halfedge_handle(*e_it, 0);
		auto v0 = mesh.from_vertex_handle(heh);
		auto v1 = mesh.to_vertex_handle(heh);
		glNormal3fv(mesh.normal(v0).data());
		glVertex3fv(((mesh.point(v0) - pointCenter) / meshSize).data());
		glNormal3fv(mesh.normal(v1).data());
		glVertex3fv(((mesh.point(v1) - pointCenter) / meshSize).data());
	}
	glEnd();
	glFlush();
	glDisable(GL_BLEND);
}

void TriMesh3D::DrawFaces(void) const
{
	glColor3d(1.0, 1.0, 1.0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.5f, 2.0f);
	glBegin(GL_TRIANGLES);
	for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
	{
		glColor3d(1.0, 1.0, 0.5);
		for (auto fv_it = mesh.cfv_begin(*f_it); fv_it != mesh.cfv_end(*f_it); ++fv_it)
		{
			glNormal3fv(mesh.normal(*f_it).data());
			glVertex3fv(((mesh.point(*fv_it) - pointCenter) / meshSize).data());
		}
	}
	glEnd();
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void TriMesh3D::DrawFlatShading(void)
{
	const float color[] = { r, g, b, 1.0f };

	glColor3f(0.56, 0.56, 0.56);
	glEnable(GL_FLAT);
	glShadeModel(GL_FLAT);

	setMatirial(color, 30.0);
	glBegin(GL_TRIANGLES);
	for (auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face)
	{
		for (auto it = mesh.fv_begin(*face); it != mesh.fv_end(*face); ++it)
		{
			auto vertex = *it;
			glNormal3fv(mesh.normal(vertex).data());
			auto point = mesh.point(*it);
			glVertex3fv(((point - pointCenter) / meshSize).data());
		}
	}
	mesh.update_face_normals();
	mesh.update_normals();
	glEnd();
	glDisable(GL_FLAT);
}

void TriMesh3D::DrawFlatWire(void)
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
}

void TriMesh3D::renderFlat(void)
{
	const GLfloat color[] = { r, g, b, 1.0f };
	setMatirial(color, 30.0);
	glBegin(GL_TRIANGLES);
	for (auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face)
	{
		for (auto it = mesh.fv_begin(*face); it != mesh.fv_end(*face); ++it)
		{
			auto vertex = *it;
			glNormal3fv(mesh.normal(vertex).data());
			auto point = mesh.point(*it);
			glVertex3fv(((point - pointCenter) / meshSize).data());
		}
	}
	glEnd();
}
void TriMesh3D::renderWire(void)
{
	glDisable(GL_LIGHTING);
	glColor3f(0.0, 0.0, 0.0);
	glDepthRange(0.0, 1.0);
	glDepthFunc(GL_LEQUAL);

	glLineWidth(0.1);
	glColor3f(0, 0, 0);
	glBegin(GL_LINES);
	for (auto it = mesh.halfedges_begin(); it != mesh.halfedges_end(); ++it)
	{
		auto fromVertex = mesh.from_vertex_handle(*it);
		auto toVertex = mesh.to_vertex_handle(*it);

		auto pointA = mesh.point(fromVertex);
		glVertex3fv(((pointA - pointCenter) / meshSize).data());
		glNormal3fv(mesh.normal(fromVertex).data());
		auto pointB = mesh.point(toVertex);
		glVertex3fv(((pointB - pointCenter) / meshSize).data());
		glNormal3fv(mesh.normal(toVertex).data());
	}
	//mesh.update_vertex_normals();
	mesh.update_normals();
	glEnd();
	glDisable(GL_BLEND);
	glDepthFunc(GL_LESS);
}

void TriMesh3D::DrawSmoothShading(void)
{
	const GLfloat color[] = { r, g, b, 1.0f };
	glShadeModel(GL_SMOOTH);

	setMatirial(color, 30.0);
	glBegin(GL_TRIANGLES);
	for (auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face)
	{
		for (auto it = mesh.fv_begin(*face); it != mesh.fv_end(*face); ++it)
		{
			auto vertex = *it;
			glNormal3fv(mesh.normal(vertex).data());
			auto point = mesh.point(*it);
			glVertex3fv(((point - pointCenter) / meshSize).data());;
		}
	}
	//mesh.update_face_normals();
	mesh.update_normals();
	glEnd();
}

void TriMesh3D::DrawTransparent(void)
{
	const GLfloat color[] = { r, g, b, 0.12f };
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDepthMask(GL_FALSE);
	setMatirial(color, 50.0);
	glPushMatrix();

	glBegin(GL_TRIANGLES);
	for (auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face)
	{
		for (auto it = mesh.fv_begin(*face); it != mesh.fv_end(*face); ++it)
		{
			auto vertex = *it;
			glNormal3fv(mesh.normal(vertex).data());
			auto point = mesh.point(*it);
			glVertex3fv(((point - pointCenter) / meshSize).data());;
		}
	}
	mesh.update_normals();
	glEnd();
	glPopMatrix();

	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);

	/*glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);// 开启混合模式
	glEnable(GL_LIGHTING);
	glBegin(GL_TRIANGLES);
	for (auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face)
	{
		for (auto it = mesh.fv_begin(face); it != mesh.fv_end(face); ++it)
		{
			auto vertex = it.handle();
			glNormal3fv(mesh.normal(vertex).data());
			auto point = mesh.point(it.handle());
			glVertex3fv(((point - pointCenter) / meshSize).data());;
		}
	}
	mesh.update_normals();
	glEnd();
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);// 关闭混合模式*/
}

void TriMesh3D::DrawWireFrame(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glLineWidth(0.5);
	glColor3f(0, 0, 0);
	glBegin(GL_LINES);
	for (auto it = mesh.halfedges_begin(); it != mesh.halfedges_end(); ++it)
	{
		auto fromVertex = mesh.from_vertex_handle(*it);
		auto toVertex = mesh.to_vertex_handle(*it);

		auto pointA = mesh.point(fromVertex);
		glVertex3fv(((pointA - pointCenter) / meshSize).data());
		glNormal3fv(mesh.normal(fromVertex).data());
		auto pointB = mesh.point(toVertex);
		glVertex3fv(((pointB - pointCenter) / meshSize).data());
		glNormal3fv(mesh.normal(toVertex).data());
	}
	//mesh.update_vertex_normals();
	mesh.update_normals();
	glEnd();
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_BLEND);
}

void TriMesh3D::ComputeBoundingbox(void)
{
	pointMax = pointMin = mesh.point(*(mesh.vertices_begin()));
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
	{
		pointMax.maximize(mesh.point(*v_it));
		pointMin.minimize(mesh.point(*v_it));
	}
	pointCenter = (pointMax + pointMin) / 2;
	//std::cout << "point center " << pointCenter.data()[0] << "   " << pointCenter.data()[1] <<  "   "<< pointCenter.data()[2] << std::endl;
	auto n = (pointMax - pointMin) / 2;
	n += Point3(n.norm() / 10000);
	meshSize = n.norm()/2;
	pointMax += Point3(n.norm() * 1e-6);
	pointMin -= Point3(n.norm() * 1e-6);
	//std::cout << "point max " << pointMax.data()[0] << "   " << pointMax.data()[1] << "   " << pointMax.data()[2]<<std::endl;
	//std::cout << "point min " << pointMin.data()[0] << "   " << pointMin.data()[1] << "   " << pointMin.data()[2]<<std::endl;
}
void TriMesh3D::DrawBoundingBox(void)
{
	GLfloat vertex_list[][3] = {
		(pointMin.data()[0] - pointCenter.data()[0]) / meshSize, (pointMax.data()[1] - pointCenter.data()[1]) / meshSize, (pointMin.data()[2] - pointCenter.data()[2]) / meshSize,
		(pointMin.data()[0] - pointCenter.data()[0]) / meshSize, (pointMin.data()[1] - pointCenter.data()[1]) / meshSize, (pointMin.data()[2] - pointCenter.data()[2]) / meshSize,
		(pointMax.data()[0] - pointCenter.data()[0]) / meshSize, (pointMin.data()[1] - pointCenter.data()[1]) / meshSize, (pointMin.data()[2] - pointCenter.data()[2]) / meshSize,
		(pointMax.data()[0] - pointCenter.data()[0]) / meshSize, (pointMax.data()[1] - pointCenter.data()[1]) / meshSize, (pointMin.data()[2] - pointCenter.data()[2]) / meshSize,
		(pointMin.data()[0] - pointCenter.data()[0]) / meshSize, (pointMax.data()[1] - pointCenter.data()[1]) / meshSize, (pointMax.data()[2] - pointCenter.data()[2]) / meshSize,
		(pointMin.data()[0] - pointCenter.data()[0]) / meshSize, (pointMin.data()[1] - pointCenter.data()[1]) / meshSize, (pointMax.data()[2] - pointCenter.data()[2]) / meshSize,
		(pointMax.data()[0] - pointCenter.data()[0]) / meshSize, (pointMin.data()[1] - pointCenter.data()[1]) / meshSize, (pointMax.data()[2] - pointCenter.data()[2]) / meshSize,
		(pointMax.data()[0] - pointCenter.data()[0]) / meshSize, (pointMax.data()[1] - pointCenter.data()[1]) / meshSize, (pointMax.data()[2] - pointCenter.data()[2]) / meshSize,
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
