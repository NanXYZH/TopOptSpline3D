////////////////////////////////////////////////////////////////////////////
// mesh render :
// Points, Edges, Faces, Axis, FlatWire, transparent
// WireFrame, FlatShading, Smooth shading
// mesh compute:
// Bounding box, voxelization
////////////////////////////////////////////////////////////////////////////

#pragma once
#include <string>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;
typedef MyMesh::Point Point3;
typedef MyMesh::Normal Normal;

// Triangular mesh in 3 dimension using OpenMesh 
class TriMesh3D
{
private:
	Point3 pointCenter, pointMax, pointMin; // Bounding box
	double meshSize; // Size of the mesh
	MyMesh mesh;     // Mesh data
public:
	float		r, g, b, alpha;

public:
	TriMesh3D(void);
	virtual ~TriMesh3D(void);

	// Load mesh from file
	bool LoadFile(const std::string &filename);
	bool SaveFile(const std::string &filename) const;
	void TestRotateMesh(void);
	void TestScaleMesh(void);
	void Clear(void) { mesh.clear(); }

	void init(void);

	MyMesh& Mesh(void) { return mesh; }
	const MyMesh& Mesh(void) const { return mesh; }

	const Point3& PointMax(void) const { return pointMax; }
	const Point3& PointMin(void) const { return pointMin; }
	const Point3& PointCenter(void) const { return pointCenter; }
	const double& MeshSize(void) const { return meshSize; }

	void setMatirial(const float mat_diffuse[4], float mat_shininess);
	bool IsEmpty(void) const { return mesh.vertices_empty(); }
	void DrawAxis(void) const;
	void DrawPoints(void) const;
	void DrawEdges(void) const;
	void DrawFaces(void) const;
	void DrawFlatWire(void);
	void renderWire(void);
	void renderFlat(void);
	void DrawSmoothShading(void);
	void DrawTransparent(void);
	void DrawWireFrame(void);
	void DrawFlatShading(void);
	void DrawBoundingBox(void);
	void ComputeBoundingbox(void);

};