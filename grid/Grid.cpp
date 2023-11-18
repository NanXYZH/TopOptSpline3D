#include "Grid.h"
#include "algorithm"
#include "sstream"
#include "CGALDefinition.h"
#include "projection.h"
#include "matlab_utils.h"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseQR"
#include "Eigen/Eigen"
#include "binaryIO.h"
#include "openvdb_wrapper_t.h"
#include "tictoc.h"
#include <set>
#include <random>

#include <CGAL/Polygon_mesh_processing\distance.h>
#include "CGAL/Surface_mesh.h"
#include "CGAL/Simple_cartesian.h"
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include "CGAL/AABB_segment_primitive.h"

#include <list>
#include <array>
#include <vector>
#include <CGAL/Simple_cartesian.h>

//#include <CGAL/Simple_cartesian.h>
//#include <CGAL/point_generators_3.h>
//#include <CGAL/Orthogonal_k_neighbor_search.h>
//#include <CGAL/Search_traits_3.h>
//#include <tbb/blocked_range.h>
//#include <tbb/parallel_for.h>
//using K = CGAL::Simple_cartesian<double>;
//using Point_3 = K::Point_3;
//using CGTraits = CGAL::Search_traits_3<K>;
//using Neighbor_search = CGAL::Orthogonal_k_neighbor_search<CGTraits>;
//using Tree = Neighbor_search::Tree;
//using Point_with_distance = Neighbor_search::Point_with_transformed_distance;
//using Generator = CGAL::Random_points_in_sphere_3<Point_3>;

#include "MCrender.h"
#include "TriMesh3D.h"

//#include "fmm.h"

using namespace grid;

#ifdef __linux__
#ifndef sprintf_s
#define sprintf_s(buf, ...) snprintf((buf), sizeof(buf), __VA_ARGS__)
#endif
#endif

std::vector<Triangle> aabb_tris;
std::vector<Triangle> tree_tris;
aabb_tree_t aabb_tree;
aabb_tree_t tree;

CGMesh::Property_map<Face_descriptor, Vector3> cgmesh_fnormals;
CGMesh::Property_map<Vertex_descriptor, Vector3> cgmesh_vnormals;

CGMesh cmesh;

Mesh omesh;
Mesh mesh_entity;

extern HierarchyGrid grids;

static Eigen::SparseMatrix<double> Klast;
static Eigen::Matrix<double, -1, -1> fullK;
static Eigen::Matrix<double, -1, -1> Klastkernel;
static std::vector<int> vlastrowid;
static int nvlastrows;
static Eigen::BDCSVD<Eigen::MatrixXd> svd;

namespace Me {
	template<typename T>
	T min(const T& a, const T& b) {
		return a < b ? a : b;
	}
	template<typename T>
	T max(const T& a, const T& b) {
		return a > b ? a : b;
	}
	template<typename T, typename ... Args>
	T min(const T& a, const T& b, const Args& ...args) {
		return min(a, min(b, args...));
	}
	template<typename T, typename ... Args>
	T max(const T& a, const T& b, const Args& ...args) {
		return max(a, max(b, args...));
	}
}

void HierarchyGrid::cgalTest(void) {
	Point a(1.0, 0.0, 0.0);
	Point b(0.0, 1.0, 0.0);
	Point c(0.0, 0.0, 1.0);
	Point d(0.0, 0.0, 0.0);
	//std::list<Triangle> triangles;
	//triangles.push_back(Triangle(a, b, c));
	//triangles.push_back(Triangle(a, b, d));
	//triangles.push_back(Triangle(a, d, c));
	tree_tris.clear();
	tree_tris.emplace_back(a, b, c);
	tree_tris.emplace_back(a, b, d);
	tree_tris.emplace_back(a, d, c);
	// constructs AABB tree
	tree.clear();
	tree.rebuild(tree_tris.begin(), tree_tris.end());
	// counts #intersections
	Ray ray_query(a, b);
	std::cout << tree.number_of_intersected_primitives(ray_query)
		<< " intersections(s) with ray query" << std::endl;
	// compute closest point and squared distance
	Point point_query(2.0, 2.0, 2.0);
	Point closest_point = tree.closest_point(point_query);
	std::cerr << "closest point is: " << closest_point << std::endl;
	FT sqd = tree.squared_distance(point_query);
	std::cout << "squared distance: " << sqd << std::endl;


	Kernel::Ray_3 ray3(point_query, closest_point);
	std::cout << "Type of aabb_tree_t::Primitive_id: " << typeid(aabb_tree_t::Primitive_id).name() << std::endl;
	//std::vector<std::pair<typename aabb_tree_t::Intersection_and_primitive_id<Triangle>::Type, unsigned>> intersections4;
	//std::vector<std::pair<typename aabb_tree_t::Intersection_and_primitive_id<Ray>::Type, unsigned>> intersections4;
	//std::vector<std::pair<typename aabb_tree_t::Point_and_primitive_id, unsigned>> intersections4;
	//std::vector<aabb_tree_t::Point_and_primitive_id> intersections4;
	aabb_tree_t3 tree3(tree_tris.begin(), tree_tris.end());
	closest_point = tree3.closest_point(point_query);
	std::cerr << "closest point is: " << closest_point << std::endl;
	std::cout << "Type of AABB_tree::Primitive: " << typeid(Primitive3).name() << std::endl;

	std::vector<std::pair<Primitive3, FT>> intersections4;
	//tree3.all_intersections(ray3, std::back_inserter(intersections4));
	//std::cout << "Number of intersections: " << intersections4.size() << std::endl;
	//// Output the results
	//for (const auto& intersection : intersections4) {
	//	const Point& intersection_point = intersection.first.reference_point();
	//	FT squared_distance = intersection.second;
	//	std::cout << "Intersection Point: " << intersection_point << ", Squared Distance: " << squared_distance << std::endl;
	//}

	std::list<Triangle> triangles; 
	triangles.emplace_back( Triangle(a, b, c) );
	triangles.emplace_back( Triangle(a, b, d) );
	triangles.emplace_back( Triangle(a, d, c) );
	aabb_tree_t1 tree1(triangles.begin(), triangles.end());
	typedef aabb_tree_t1::Point_and_primitive_id Point_and_primitive_id;
	Point_and_primitive_id pp = tree1.closest_point_and_primitive(point_query);
	//tree1.any_reference_point_and_id(point_query);
	std::cout << "closest point is: " << pp.first << std::endl;
	std::cout << "Iterator value: " << *(pp.second) << std::endl;

	const Triangle& nearest_triangle = *(pp.second);
	for (int i = 0; i < 3; i++)
	{
		const Point& vertex1 = nearest_triangle.vertex(i);
		std::cout << "Coordinates of the first vertex: (" << vertex1.x() << ", " << vertex1.y() << ", " << vertex1.z() << ")" << std::endl;
	}

	Point direction(point_query[0] - pp.first[0], point_query[1] - pp.first[1], point_query[2] - pp.first[2]);
	float small_move = 0.001f;
	Point move_length(direction[0] * small_move, direction[1] * small_move, direction[2] * small_move);
	Point near_closest(pp.first[0] + move_length[0], pp.first[1] + move_length[1], pp.first[2] + move_length[2]);

	Ray ray1(near_closest, point_query);  // point out of the model
	Ray ray2(point_query, near_closest);  // point in the model

	std::cout << tree.do_intersect(ray1) << std::endl;
	std::cout << tree.do_intersect(ray2) << std::endl;

	std::list<aabb_tree_t1::Primitive_id> primitives;
	tree1.all_intersected_primitives(ray1, std::back_inserter(primitives));
	std::cout << "num of intersections: " << primitives.size() << std::endl;
	for (std::list<aabb_tree_t1::Primitive_id>::iterator it = primitives.begin(); it != primitives.end(); ++it)
	{
		const aabb_tree_t1::Primitive_id& primirives_id = *it;
		int index = std::distance(primitives.begin(), it);
		std::cout << "NO." << index << std::endl;
		std::cout << primirives_id->vertex(0) << ", " << primirives_id->vertex(1) << ", " << primirives_id->vertex(2) << std::endl;
	}

	std::list<aabb_tree_t1::Primitive_id> primitives2;
	tree1.all_intersected_primitives(ray2, std::back_inserter(primitives2));
	std::cout << "num of intersections: " << primitives2.size() << std::endl;
	for (std::list<aabb_tree_t1::Primitive_id>::iterator it = primitives2.begin(); it != primitives2.end(); ++it)
	{
		const aabb_tree_t1::Primitive_id& primirives_id = *it;
		int index = std::distance(primitives2.begin(), it);
		std::cout << "NO." << index << std::endl;
		std::cout << primirives_id->vertex(0) << ", " << primirives_id->vertex(1) << ", " << primirives_id->vertex(2) << std::endl;
	}
	std::cout << "num of intersected: "<< tree1.number_of_intersected_primitives(ray1) << std::endl;
	std::cout << "num of intersected: "<< tree1.number_of_intersected_primitives(ray2) << std::endl;

	if (tree1.number_of_intersected_primitives(ray1) % 2 == 0) {
		std::cout << "Point1 is outside the AABB tree." << std::endl;
	}
	else {
		std::cout << "Point1 is inside the AABB tree." << std::endl;
	}
	//Point p(1.0, 2.0, 3.0);
	//CGAL::Object object = CGAL::make_object(p);
	//typedef aabb_tree_t1::Object_and_primitive_id Object_and_primitive_id;
}

void HierarchyGrid::buildAABBTree(const std::vector<float>& pcoords, const std::vector<int>& trifaces, const Mesh& inputmesh)
{
	// build aabb tree
	aabb_tris.clear();
	for (int i = 0; i < trifaces.size(); i += 3) {
		Point v0(pcoords[trifaces[i] * 3], pcoords[trifaces[i] * 3 + 1], pcoords[trifaces[i] * 3 + 2]);
		Point v1(pcoords[trifaces[i + 1] * 3], pcoords[trifaces[i + 1] * 3 + 1], pcoords[trifaces[i + 1] * 3 + 2]);
		Point v2(pcoords[trifaces[i + 2] * 3], pcoords[trifaces[i + 2] * 3 + 1], pcoords[trifaces[i + 2] * 3 + 2]);
		aabb_tris.emplace_back(v0, v1, v2);
	}
	aabb_tree.clear();
	aabb_tree.rebuild(aabb_tris.begin(), aabb_tris.end());

	// build cgal mesh
	std::vector<CGMesh::Vertex_index>  vidlist;
	for (int i = 0; i < pcoords.size(); i += 3) {
		vidlist.emplace_back(cmesh.add_vertex(Point(pcoords[i], pcoords[i + 1], pcoords[i + 2])));
	}
	for (int i = 0; i < trifaces.size(); i += 3) {
		cmesh.add_face(vidlist[trifaces[i]], vidlist[trifaces[i + 1]], vidlist[trifaces[i + 2]]);
	}

	cgmesh_fnormals = cmesh.add_property_map<Face_descriptor, Vector3>("f:normals", CGAL::NULL_VECTOR).first;
	cgmesh_vnormals = cmesh.add_property_map<Vertex_descriptor, Vector3>("v:normals", CGAL::NULL_VECTOR).first;

	PMP::compute_normals(cmesh, cgmesh_vnormals, cgmesh_fnormals);

	// build open-mesh mesh
	Vec3x xm(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
	Vec3x xM = -xm;

	for (auto itv = inputmesh.vertices_begin(); itv != inputmesh.vertices_end(); itv++) {
		auto x1 = inputmesh.point(*itv);
		Vec3x x(x1[0], x1[1], x1[2]);
		xm.minimize(x);
		xM.maximize(x);
	}
	//mesh_entity = inputmesh;
	//float scale_ratio = 1.0f;

	//for (auto& vt : mesh_entity.vertices()) {
	//	mesh_entity.set_point(vt, mesh_entity.point(vt) * scale_ratio);
	//}
	//mesh_entity.request_face_normals();
	//mesh_entity.request_vertex_normals();
	//mesh_entity.update_normals();
	omesh = inputmesh;
	printf("--[TEST] openmesh model bounding box : [%f, %f, %f] -- [%f, %f, %f]\n", xm[0], xm[1], xm[2], xM[0], xM[1], xM[2]);
}

void HierarchyGrid::setSolidShellElement(const std::vector<unsigned int>& ebitfine, BitSAT<unsigned int>& esat, float box[2][3], int ereso, std::vector<int>& eflags) {
	std::vector<CGMesh::Face_index> fidlist;
	for (auto iter = cmesh.faces_begin(); iter != cmesh.faces_end(); iter++) {
		fidlist.emplace_back(*iter);
	}

	double eh = (box[1][0] - box[0][0]) / ereso;

	printf("-- element size h = %lf (%d)\n", eh, ereso);

	std::set<int> shellelements;

	double wshell = _setting.shell_width * eh;

	printf("-- shell width %lf \n", wshell);

	double sh2 = pow(wshell, 2);

#pragma omp parallel for
	for (int i = 0; i < fidlist.size(); i++) {
		Point fv[3];
		int counter = 0;
		for (auto v : cmesh.vertices_around_face(cmesh.halfedge(fidlist[i]))) {
			fv[counter++] = cmesh.point(v);
		}
		auto fbb = PMP::face_bbox(fidlist[i], cmesh);
		int lid[3], rid[3];
		for (int j = 0; j < 3; j++) {
			lid[j] = (fbb.min_coord(j) - wshell * 1.3 - box[0][j] - 0.5 * eh) / eh;
			rid[j] = (fbb.max_coord(j) + wshell * 1.3 - box[0][j] - 0.5 * eh) / eh + 1;
			lid[j] = std::clamp(lid[j], 0, ereso - 1);
			rid[j] = std::clamp(rid[j], 0, ereso - 1);
		}
		double ec[3];
		std::vector<int> eidshell;
		
		for (int x = lid[0]; x < rid[0]; x++) {
			ec[0] = (x + 0.5) * eh + box[0][0];
			for (int y = lid[1]; y < rid[1]; y++) {
				ec[1] = (y + 0.5) * eh + box[0][1];
				for (int z = lid[2]; z < rid[2]; z++) {
					ec[2] = (z + 0.5)* eh + box[0][2];
					double d = aabb_tree.squared_distance(Point(ec[0], ec[1], ec[2]));
					if (d < sh2) {
						int ebid = x + y * ereso + z * ereso * ereso;
						int eid = esat(ebid);
						if (eid != -1) {
							eidshell.emplace_back(eid);
						}
					}
				}
			}
		}
#pragma omp critical
		{
			for (int i = 0; i < eidshell.size(); i++) {
				shellelements.insert(eidshell[i]);
			}
		}
	}

	printf("-- found %d shell elements\n", shellelements.size());

	for (auto iter = shellelements.begin(); iter != shellelements.end(); iter++) {
		int oldword = eflags[*iter];
		oldword |= int(Grid::Bitmask::mask_shellelement);
		eflags[*iter] = oldword;
	}
}

// MARK[TOupdate]
void HierarchyGrid::setinModelVertice(const std::vector<unsigned int>& vbitfine, BitSAT<unsigned int>& vsat, float box[2][3], int ereso, std::vector<int>& vflags) {
	std::vector<CGMesh::Face_index> fidlist;
	for (auto iter = cmesh.faces_begin(); iter != cmesh.faces_end(); iter++) {
		fidlist.emplace_back(*iter);
	}
	int vreso = ereso + 1;
	double eh = (box[1][0] - box[0][0]) / ereso;

	printf("-- vertice size h = %lf (%d)\n", eh, vreso);

	std::set<int> shellelements;

	double wshell = _setting.shell_width * eh;

	printf("-- shell width %lf \n", wshell);

	double sh2 = pow(wshell, 2);

#pragma omp parallel for
	for (int i = 0; i < fidlist.size(); i++) {
		Point fv[3];
		int counter = 0;
		for (auto v : cmesh.vertices_around_face(cmesh.halfedge(fidlist[i]))) {
			fv[counter++] = cmesh.point(v);
		}
		auto fbb = PMP::face_bbox(fidlist[i], cmesh);
		int lid[3], rid[3];
		for (int j = 0; j < 3; j++) {
			lid[j] = (fbb.min_coord(j) - wshell * 1.3 - box[0][j] - 0.5 * eh) / eh;
			rid[j] = (fbb.max_coord(j) + wshell * 1.3 - box[0][j] - 0.5 * eh) / eh + 1;
			lid[j] = std::clamp(lid[j], 0, ereso - 1);
			rid[j] = std::clamp(rid[j], 0, ereso - 1);
		}
		double ec[3];
		std::vector<int> eidshell;

		for (int x = lid[0]; x < rid[0]; x++) {
			ec[0] = (x + 0.5) * eh + box[0][0];
			for (int y = lid[1]; y < rid[1]; y++) {
				ec[1] = (y + 0.5) * eh + box[0][1];
				for (int z = lid[2]; z < rid[2]; z++) {
					ec[2] = (z + 0.5) * eh + box[0][2];
					double d = aabb_tree.squared_distance(Point(ec[0], ec[1], ec[2]));

					// MARK[TODO]
					// UPDATE to nodes in model
					if (d < sh2) {
						int ebid = x + y * ereso + z * ereso * ereso;
						int eid = vsat(ebid);
						if (eid != -1) {
							eidshell.emplace_back(eid);
						}
					}
				}
			}
		}
#pragma omp critical
		{
			for (int i = 0; i < eidshell.size(); i++) {
				shellelements.insert(eidshell[i]);
			}
		}
	}

	printf("-- found %d Model vertices\n", shellelements.size());

	for (auto iter = shellelements.begin(); iter != shellelements.end(); iter++) {
		int oldword = vflags[*iter];
		oldword |= int(Grid::Bitmask::mask_modelnodes);
		vflags[*iter] = oldword;
	}
}

// MARK[TOupdate]
void HierarchyGrid::setinModelElement(const std::vector<unsigned int>& ebitfine, BitSAT<unsigned int>& esat, float box[2][3], int ereso, std::vector<int>& eflags) {
	std::vector<CGMesh::Face_index> fidlist;
	for (auto iter = cmesh.faces_begin(); iter != cmesh.faces_end(); iter++) {
		fidlist.emplace_back(*iter);
	}

	double eh = (box[1][0] - box[0][0]) / ereso;

	printf("-- element size h = %lf (%d)\n", eh, ereso);

	std::set<int> shellelements;

	double wshell = _setting.shell_width * eh;

	printf("-- shell width %lf \n", wshell);

	double sh2 = pow(wshell, 2);

#pragma omp parallel for
	for (int i = 0; i < fidlist.size(); i++) {
		Point fv[3];
		int counter = 0;
		for (auto v : cmesh.vertices_around_face(cmesh.halfedge(fidlist[i]))) {
			fv[counter++] = cmesh.point(v);
		}
		auto fbb = PMP::face_bbox(fidlist[i], cmesh);
		int lid[3], rid[3];
		for (int j = 0; j < 3; j++) {
			lid[j] = (fbb.min_coord(j) - wshell * 1.3 - box[0][j] - 0.5 * eh) / eh;
			rid[j] = (fbb.max_coord(j) + wshell * 1.3 - box[0][j] - 0.5 * eh) / eh + 1;
			lid[j] = std::clamp(lid[j], 0, ereso - 1);
			rid[j] = std::clamp(rid[j], 0, ereso - 1);
		}
		double ec[3];
		std::vector<int> eidshell;

		for (int x = lid[0]; x < rid[0]; x++) {
			ec[0] = (x + 0.5) * eh + box[0][0];
			for (int y = lid[1]; y < rid[1]; y++) {
				ec[1] = (y + 0.5) * eh + box[0][1];
				for (int z = lid[2]; z < rid[2]; z++) {
					ec[2] = (z + 0.5) * eh + box[0][2];
					double d = aabb_tree.squared_distance(Point(ec[0], ec[1], ec[2]));
					if (d < sh2) {
						int ebid = x + y * ereso + z * ereso * ereso;
						int eid = esat(ebid);
						if (eid != -1) {
							eidshell.emplace_back(eid);
						}
					}
				}
			}
		}
#pragma omp critical
		{
			for (int i = 0; i < eidshell.size(); i++) {
				shellelements.insert(eidshell[i]);
			}
		}
	}

	printf("-- found %d Model elements\n", shellelements.size());

	for (auto iter = shellelements.begin(); iter != shellelements.end(); iter++) {
		int oldword = eflags[*iter];
		oldword |= int(Grid::Bitmask::mask_modelelements);
		eflags[*iter] = oldword;
	}
}

void HierarchyGrid::_find_grid_node_close_to_face(vec3& v1, vec3& v2, vec3& v3,
	float spacing, int N[3],
	std::vector<std::array<int, 3>>& boundary_indices, std::vector<float>& boundary_distance,
	float box[2][3])
{
	float origin[3] = { box[0][0], box[0][1], box[0][2] };

#if 0
#elif 1
	Point p1(v1[0], v1[1], v1[2]), p2(v2[0], v2[1], v2[2]), p3(v3[0], v3[1], v3[2]);
	Kernel::Triangle_3 tri(p1, p2, p3);
	std::set<std::array<int, 3>> local_close;
	Point upCorner(Me::max(v1[0], v2[0], v3[0]), Me::max(v1[1], v2[1], v3[1]), Me::max(v1[2], v2[2], v3[2]));
	Point downCorner(Me::min(v1[0], v2[0], v3[0]), Me::min(v1[1], v2[1], v3[1]), Me::min(v1[2], v2[2], v3[2]));
	int xleft = (std::max)(0, int((downCorner[0] - origin[0]) / spacing) - 1);
	int xright = (std::min)(N[0], int((upCorner[0] - origin[0]) / spacing) + 3);
	int yleft = (std::max)(0, int((downCorner[1] - origin[1]) / spacing) - 1);
	int yright = (std::min)(N[1], int((upCorner[1] - origin[1]) / spacing) + 3);
	int zleft = (std::max)(0, int((downCorner[2] - origin[2]) / spacing) - 1);
	int zright = (std::min)(N[2], int((upCorner[2] - origin[2]) / spacing) + 3);
	for (int kx = xleft; kx < xright; kx++) {
		for (int ky = yleft; ky < yright; ky++) {
			for (int kz = zleft; kz < zright; kz++) {
				Point pos(kx * spacing + origin[0], ky * spacing + origin[1], kz * spacing + origin[2]);
				auto sqrdist = CGAL::squared_distance(pos, tri);
				if (sqrdist < 3 * spacing * spacing) local_close.insert({ kx,ky,kz });
			}
		}
	}
	for (auto it = local_close.begin(); it != local_close.end(); it++) {
		boundary_indices.push_back(*it);
	}
#endif
}

void HierarchyGrid::generate_signed_dist_field(std::vector<float>& dst, float spacing, int Nodes[3], float inside_offset, float box[2][3])
{
	std::vector<std::array<int, 3>> boundary_indices;
	std::vector<float> boundary_distance;
	auto grid_spacing = std::array<float, 3>{ {spacing, spacing, spacing}};
	auto uniform_speed = 1.f;

	float origin[3] = { box[0][0], box[0][1], box[0][2] };
	//if (origin == nullptr) {
	//	origin = xmin.data();
	//}
	std::array<float, 3> xm = { origin[0],origin[1], origin[2] };
	std::array<float, 3> xM = { Nodes[0] * spacing + origin[0],Nodes[1] * spacing + origin[1],Nodes[2] * spacing + origin[2] };
	for (auto fh : omesh.faces()) {
		auto itv = omesh.fv_begin(fh);
		double* doublev1 = omesh.point(*itv++).data();
		double* doublev2 = omesh.point(*itv++).data();
		double* doublev3 = omesh.point(*itv).data();
		float* floatv1 = reinterpret_cast<float*>(doublev1);
		float* floatv2 = reinterpret_cast<float*>(doublev2);
		float* floatv3 = reinterpret_cast<float*>(doublev3);
		vec3 v1 = vec3::Map(floatv1);
		vec3 v2 = vec3::Map(floatv2);
		vec3 v3 = vec3::Map(floatv3);
		//vec3 v1 = vec3::Map(omesh.point(*itv++).data());
		//vec3 v2 = vec3::Map(omesh.point(*itv++).data());
		//vec3 v3 = vec3::Map(omesh.point(*itv).data());
		_find_grid_node_close_to_face(v1, v2, v3, spacing, Nodes,
			boundary_indices, boundary_distance, box);
	}
	std::map<std::array<int, 3>, float> id2dist;

	// delete duplicated element
	auto end_iter = std::unique(boundary_indices.begin(), boundary_indices.end());
	boundary_indices.erase(end_iter, boundary_indices.end());
	boundary_distance.resize(boundary_indices.size());
#pragma omp parallel for
	for (int i = 0; i < boundary_indices.size(); i++) {
		Point p(boundary_indices[i][0] * spacing + origin[0], boundary_indices[i][1] * spacing + origin[1], boundary_indices[i][2] * spacing + origin[2]);
		auto close_one = aabb_tree.closest_point_and_primitive(p);
		auto tri_itr = close_one.second;
		auto p_proj = close_one.first;
		Kernel::Vector_3 pp(p[0] - p_proj[0], p[1] - p_proj[1], p[2] - p_proj[2]);
		auto sqrdist = pp.squared_length();
		auto tri_normal = tri_itr->supporting_plane().orthogonal_vector();
		auto product = tri_normal * Kernel::Vector_3(p[0] - p_proj[0], p[1] - p_proj[1], p[2] - p_proj[2]);
		if (product > 0)
			boundary_distance[i] = sqrt(sqrdist);
		else if (product < 0)
			boundary_distance[i] = -sqrt(sqrdist);
		else {
			boundary_distance[i] = 0;
		}
		boundary_distance[i] += inside_offset;
	}
	auto boundary_times = boundary_distance;

	//namespace fmm = thinks::fast_marching_method;
	std::array<size_t, 3> grid_size{ Nodes[0], Nodes[1], Nodes[2] };
}

void HierarchyGrid::compute_nodes_in_model(std::vector<int>& flags, float spacing, int Nodes[3], float box[2][3])
{
	float origin[3] = { box[0][0], box[0][1], box[0][2] };
	std::vector<Scaler> grid_values;

#ifndef   DETERMINE_NODES_IN_MODEL_WITH_RAY_CASTING
	grid_values.clear();
	std::vector<Scaler>().swap(grid_values);
	//generate_signed_dist_field(grid_values, spacing, Nodes, box);
	flags.resize((grid_values.size() + 31) / 32, 0);
	std::cout << "test 3: " << grid_values.size() << std::endl;

#pragma omp parallel for
	for (int i = 0; i < flags.size(); i++) {
		int fdw = flags[i];
		for (int j = 0; j < 32; j++) {
			int node_id = i * 32 + j;
			if (node_id >= grid_values.size()) break;
			//if (grid_values[node_id] < -shell_length) set_bit(fdw, j);
			// Mark
			if (grid_values[node_id] < 0) set_bit(fdw, j);
		}
		flags[i] = fdw;
	}
#else
	Point pfar(1e4, 1e4, 1e4);
	Point pfarop(-1e4, -1e4, -1e4);
	int nodes_num = Nodes[0] * Nodes[1] * Nodes[2];
	flags.resize((nodes_num + 31) / 32, 0);
#pragma omp parallel for
	for (int i = 0; i < nodes_num; i++) {
		int xcoord = i % Nodes[0];
		int ycoord = (i / Nodes[0]) % Nodes[1];
		int zcoord = i / (Nodes[0] * Nodes[1]);
		Point p(xcoord * spacing + origin[0], ycoord * spacing + origin[1], zcoord * spacing + origin[2]);
		Segm ray(p, pfar);
		int nIntersect = aabb_tree.number_of_intersected_primitives(ray);
		int nIntersectOp = aabb_tree.number_of_intersected_primitives(Segm(p, pfarop));
		if (!(nIntersect % 2 == 0 || nIntersectOp % 2 == 0)) {
			set_bit(flags.data(), i);
		}
	}
#endif
}

void HierarchyGrid::testShell(void)
{
	_gridlayer[0]->init_rho(0);
	fillShell();
	writeDensity(getPath("shell.vdb"));
}

std::string HierarchyGrid::getModeStr(Mode mode)
{
	switch (mode) {
	case no_support_constrain_force_direction:
		return "no_fix_constrain_force_direction";
	case no_support_free_force:
		return "no_fix_free_force";
	case with_support_constrain_force_direction:
		return "fix_constrain_force_direction";
	case with_support_free_force:
		return "fix_free_force";
	default:
		printf("-- \033[31munsupported mode\033[0m\n");
		break;
	}
	return "";
}

void grid::HierarchyGrid::log(int itn)
{
	char fn[100];
	if (_logFlag & mask_log_density) {
		sprintf_s(fn, "density%04d.vdb", itn);
		printf("-- writing density to %s\n", fn);
		//writeDensity(getPath(fn));
		writeDensityac(getPath(fn));
	}
	if (_logFlag & mask_log_compliance) {
		sprintf_s(fn, "compliance%04d.vdb", itn);
		printf("-- writing compliance to %s\n", fn);
		writeComplianceDistribution(getPath(fn));
	}
}

void grid::HierarchyGrid::enable_logs(int flag)
{
	_logFlag = flag;
}

// MAEK[USED]
void grid::HierarchyGrid::genFromMesh(const std::vector<float>& pcoords, const std::vector<int>& facevertices, Mesh& inputmesh)
{
	//_pcoords = pcoords;
	//_trifaces = facevertices;

	buildAABBTree(pcoords, facevertices, inputmesh);

	//for (int i = 0; i < facevertices.size(); i++) std::cout << facevertices[i] << std::endl;

	std::vector<unsigned int> solid_bit; // elements info (in solid or not)

	int out_reso[3];
	float out_box[2][3];

	auto voxInfo = voxelize_mesh(pcoords, facevertices, _setting.prefer_reso, solid_bit, out_reso, out_box);
	//write_obj_cubes(solid_bit.data(), voxInfo, "voxels.obj");

#if 1
#ifdef ENABLE_MATLAB
	Eigen::Matrix<int, -1, 1> solid_bit_(solid_bit.size(), 1);
	for (int i = 0; i < solid_bit.size(); i++)
	{
		solid_bit_(i, 1) = solid_bit[i];
	}
	eigen2ConnectedMatlab("solid_bit1", solid_bit_);
#endif

#endif

	// The bits in a word are listed starting from high order in voxelizer, so we reverse the bits in all words 
	wordReverse_g(solid_bit.size(), solid_bit.data());

#if 1
#ifdef ENABLE_MATLAB
	Eigen::Matrix<int, -1, 1> solid_bit2(solid_bit.size(), 1);
	for (int i = 0; i < solid_bit.size(); i++)
	{
		solid_bit2(i, 1) = solid_bit[i];
	}
	eigen2ConnectedMatlab("solid_bit2", solid_bit2);
#endif
#endif

	std::vector<unsigned int> inci_vbit; // vertices info (in solid or not)
	
	int nfineelements = out_reso[0] * out_reso[1] * out_reso[2];
	
	int vreso[3] = { out_reso[0] + 1,out_reso[1] + 1,out_reso[2] + 1 };

	int nfinevertices = vreso[0] * vreso[1] * vreso[2];

	//size_t inci_vsize = nfinevertices / (sizeof(unsigned int) * 8) + 1;
	size_t inci_vsize = snippet::Round<BitCount<unsigned int>::value>(nfinevertices) / BitCount<unsigned int>::value; // make sure inci_vsize / 32 == 0
	
	//cubeGridSetSolidVertices(out_reso[0], solid_bit, inci_vbit);
	cubeGridSetSolidVertices_g(out_reso[0], solid_bit, inci_vbit);

	//array2ConnectedMatlab("solid_vbits", inci_vbit.data(), inci_vbit.size());

	int reso = out_reso[0];

	_nlayer = 1;
	
	elesatlist.emplace_back(std::move(solid_bit));
	vrtsatlist.emplace_back(std::move(inci_vbit));
	std::vector<int> resolist;
	resolist.emplace_back(reso);

	// set coarse layers solid bits (not actual solid elements number, solid elements / 32 = solid bits)
	while (elesatlist.rbegin()->total() > 400) {
		int finereso = reso;
		int finereso2 = pow(reso, 2);
		std::vector<unsigned int>& fine_ebit = elesatlist.rbegin()->_bitArray;
		std::vector<unsigned int>& fine_vbit = vrtsatlist.rbegin()->_bitArray;
		reso >>= 1;
		resolist.emplace_back(reso);
		size_t reso2 = pow(reso, 2);
		_nlayer++;
		std::vector<unsigned int> coarse_bit;
		std::vector<unsigned int> coarse_vbit;
		int nCoarseElements = pow(reso, 3);
		int nCoarseVertices = pow(reso + 1, 3);
		coarse_bit.resize(snippet::Round<BitCount<unsigned int>::value>(nCoarseElements) / BitCount<unsigned int>::value, 0);
		coarse_vbit.resize(snippet::Round<BitCount<unsigned int>::value>(nCoarseVertices) / BitCount<unsigned int>::value, 0);

		setSolidElementFromFineGrid_g(finereso, fine_ebit, coarse_bit);

		cubeGridSetSolidVertices_g(reso, coarse_bit, coarse_vbit);

		elesatlist.emplace_back(std::move(coarse_bit));
		vrtsatlist.emplace_back(std::move(coarse_vbit));
	}

	printf("-- Building %d layers (%s)\n", elesatlist.size(), (_setting.skiplayer1 ? "Non-dyadic" : "Dyadic"));

	std::vector<std::vector<int>> v2ehost[8];
	std::vector<std::vector<int>> v2vfinehost[27];
	std::vector<std::vector<int>> v2vcoarsehost[8];
	std::vector<std::vector<int>> v2vhost[27];
	std::vector<std::vector<double>> rxStencilhost[27][9];
	std::vector<std::vector<int>> vbitflaglist;
	std::vector<std::vector<int>> ebitflaglist;

	std::vector<int> v2vfinec[64];
	int* v2vfineclist[64];

	// generate topology between elements and vertices
	for (int i = 0; i < elesatlist.size(); i++) {
		std::vector<unsigned int>& ebit = elesatlist[i]._bitArray;
		std::vector<unsigned int>& vbit = vrtsatlist[i]._bitArray;
		int elementreso = resolist[i];
		int elementreso2 = pow(elementreso, 2);
		int vertexreso = elementreso + 1;
		int vertexreso2 = pow(vertexreso, 2);
		BitSAT<unsigned int>& elesat = elesatlist[i];
		BitSAT<unsigned int>& vrtsat = vrtsatlist[i];
		int nSolidElement = elesat.total();
		int nSolidVertex = vrtsat.total();

		//printf("--[%d] total valid vertex %d\n", i, vrtsat.total());
		int out_width = 10;
		printf("--[%d] total valid vertex %*d, total element %*d\n", i, out_width, vrtsat.total(), out_width, elesat.total());

		// allocate buffer
		for (int k = 0; k < 8; k++) {
			v2ehost[k].emplace_back(nSolidVertex, -1);
			v2vcoarsehost[k].emplace_back(nSolidVertex, -1);
		}

		for (int k = 0; k < 27; k++) {
			v2vfinehost[k].emplace_back(nSolidVertex, -1);
			v2vhost[k].emplace_back(nSolidVertex, -1);
		}
		
		vbitflaglist.emplace_back(nSolidVertex, 0);
		ebitflaglist.emplace_back(nSolidElement, 0);

		// compute flags
		std::vector<int>& vbitflag = *vbitflaglist.rbegin();
		std::vector<int>& ebitflag = *ebitflaglist.rbegin();

		// set vertices bit flags
		Grid::setVerticesPosFlag(vertexreso, vrtsat, vbitflag.data());

		// set elements bit flags
		Grid::setElementsPosFlag(elementreso, elesat, ebitflag.data());
		// MARK[OR]
		for (int j = 0; j < ebit.size(); j++) {
			
		}

		// generate v2e
		while (1) {
			// only need vertex element topology on first layer
			if (i != 0) break;
			int* v2elist[8];
			for (int j = 0; j < 8; j++) v2elist[j] = v2ehost[j].rbegin()->data();
			//Grid::setV2E(vertexreso, vrtsat, elesat, v2elist);
			Grid::setV2E_g(vertexreso, vrtsat, elesat, v2elist);
			break;
		}

		// generate v2v
		int* v2vlist[27];
		for (int j = 0; j < 27; j++) v2vlist[j] = v2vhost[j].rbegin()->data();
		//Grid::setV2V(vertexreso, vrtsat, v2vlist);
		Grid::setV2V_g(vertexreso, vrtsat, v2vlist);

		/// between layers, generate neighbors on different layers
		// generate v2vcoarse from coarse layer
		while (1) {
			if ((_setting.skiplayer1 && i == 1) || i == resolist.size() - 1) break;
			// v2vfine, v2vcoarse, v2
			BitSAT<unsigned int>* vrtsatfine = &vrtsatlist[i];
			BitSAT<unsigned int>* vrtsatcoarse;
			int vresofine = resolist[i] + 1;
			int skip = 1;
			int finelayer = i - 1;
			if (_setting.skiplayer1 && i == 0) {
				vrtsatcoarse = &vrtsatlist[i + 2];
				skip = 2;
				finelayer = 0;
			}
			else {
				vrtsatcoarse = &vrtsatlist[i + 1];
			}

			int* v2vcoarse[8];
			for (int j = 0; j < 8; j++) v2vcoarse[j] = v2vcoarsehost[j].rbegin()->data();

			Grid::setV2VCoarse_g(skip, vresofine, *vrtsatfine, *vrtsatcoarse, v2vcoarse);

			break;
		}

		// generate v2vfine 
		while (1) {
			if (i == 0) break;
			if (_setting.skiplayer1 && (i == 2 || i == 1)) break;
			int* v2vfine[27];
			for (int j = 0; j < 27; j++) v2vfine[j] = v2vfinehost[j].rbegin()->data();
			BitSAT<unsigned int>& vsatfine = vrtsatlist[i - 1];
			BitSAT<unsigned int>& vsatcoarse = vrtsatlist[i];
			int vresocoase = resolist[i] + 1;
			int vresofine = resolist[i - 1] + 1;
			Grid::setV2VFine_g(1, vresocoase, vsatfine, vsatcoarse, v2vfine);
			break;
		}
		
		// generate v2vfinc for non-dyadic layer 2 
		if (_setting.skiplayer1 && i == 2) {
			int vresocoarse = resolist[i] + 1;
			BitSAT<unsigned int>& vsatfine = vrtsatlist[0];
			BitSAT<unsigned int>& vsatcoarse = vrtsatlist[2];
			for (int j = 0; j < 64; j++) {
				v2vfinec[j].resize(vsatcoarse.total());
				v2vfineclist[j] = v2vfinec[j].data();
			}
			Grid::setV2VFineC_g(vresocoarse, vsatfine, vsatcoarse, v2vfineclist);
		}
		

	} // finished all layers

	// find shell elements
	setSolidShellElement(elesatlist[0]._bitArray, elesatlist[0], out_box, resolist[0], ebitflaglist[0]);

	// upload grid to device
	for (int i = 0; i < elesatlist.size(); i++) {
		auto grd = new Grid();

		if (_setting.skiplayer1&&i == 1) {
			grd->set_dummy();
			//_gridlayer.emplace_back(grd);
		}

		int nv = 0, ne = 0;
			
		int* v2e[8] = { nullptr };
		int* v2vfine[27] = { nullptr };
		int* v2vcoarse[8] = { nullptr };
		int* v2v[27] = { nullptr };
		int* vbitflag = vbitflaglist[i].data();
		int* ebitflag = ebitflaglist[i].data();

		nv = v2vhost[0][i].size();

		ne = ebitflaglist[i].size();

		// list v2e
		for (int j = 0; j < 8; j++) {
			v2e[j] = v2ehost[j][i].data();
		}

		// list v2vcoarse
		for (int j = 0; j < 8; j++) {
			if (i < elesatlist.size() - 1) v2vcoarse[j] = v2vcoarsehost[j][i].data();
		}

		// list v2v
		for (int j = 0; j < 27; j++) {
			v2v[j] = v2vhost[j][i].data();
		}
		
		// select finer grid
		Grid* finer = nullptr;
		if (i != 0) {
			if (_setting.skiplayer1 && i == 2) { finer = _gridlayer[0]; }
			else { finer = _gridlayer[i - 1]; }
		}

		// list v2vfine
		for (int j = 0; j < 27; j++) {
			if (finer != nullptr) v2vfine[j] = v2vfinehost[j][i].data();
		}

		
		if (_setting.skiplayer1) grd->set_skip();

		grd->_inFixedArea = _inFixedArea;
		grd->_inLoadArea = _inLoadArea;
		grd->_loadField = _loadField;

		grd->_min_density = _min_density;

		for (int i = 0; i < 6; i++) {
			(&grd->_box[0][0])[i] = (&out_box[0][0])[i];
		}

		grd->build(get_gmem(), vrtsatlist[i], elesatlist[i], finer, resolist[i] + 1, i, nv, ne, _setting.min_coeff, _setting.max_coeff, v2e, v2vfine, v2vcoarse, v2v, v2vfineclist, vbitflag, ebitflag);

		if (i == elesatlist.size() - 1) grd->getV2V();

		_gridlayer.emplace_back(grd);
	}

}

void grid::HierarchyGrid::genFromMesh(const std::vector<unsigned int> &solid_bit, int out_reso[3])
{
	float out_box[2][3];
	for (int i = 0; i < 3; i++) {
		out_box[0][i] = 0;
		out_box[1][i] = out_reso[i];
	}
	std::vector<unsigned int> inci_vbit;

	int nfineelements = out_reso[0] * out_reso[1] * out_reso[2];
	
	int vreso[3] = { out_reso[0] + 1,out_reso[1] + 1,out_reso[2] + 1 };

	int nfinevertices = vreso[0] * vreso[1] * vreso[2];

	//size_t inci_vsize = nfinevertices / (sizeof(unsigned int) * 8) + 1;
	size_t inci_vsize = snippet::Round<BitCount<unsigned int>::value>(nfinevertices) / BitCount<unsigned int>::value;

	//cubeGridSetSolidVertices(out_reso[0], solid_bit, inci_vbit);
	cubeGridSetSolidVertices_g(out_reso[0], solid_bit, inci_vbit);

	//array2ConnectedMatlab("solid_vbits", inci_vbit.data(), inci_vbit.size());

	int reso = out_reso[0];

	_nlayer = 1;
	
	elesatlist.emplace_back(std::move(solid_bit));
	vrtsatlist.emplace_back(std::move(inci_vbit));
	std::vector<int> resolist;
	resolist.emplace_back(reso);

	// set coarse layers solid bits
	while (elesatlist.rbegin()->total() > 400) {
		int finereso = reso;
		int finereso2 = pow(reso, 2);
		std::vector<unsigned int>& fine_ebit = elesatlist.rbegin()->_bitArray;
		std::vector<unsigned int>& fine_vbit = vrtsatlist.rbegin()->_bitArray;
		reso >>= 1;
		resolist.emplace_back(reso);
		size_t reso2 = pow(reso, 2);
		_nlayer++;
		std::vector<unsigned int> coarse_bit;
		std::vector<unsigned int> coarse_vbit;
		int nCoarseElements = pow(reso, 3);
		int nCoarseVertices = pow(reso + 1, 3);
		coarse_bit.resize(snippet::Round<BitCount<unsigned int>::value>(nCoarseElements) / BitCount<unsigned int>::value, 0);
		coarse_vbit.resize(snippet::Round<BitCount<unsigned int>::value>(nCoarseVertices) / BitCount<unsigned int>::value, 0);

		setSolidElementFromFineGrid_g(finereso, fine_ebit, coarse_bit);

		cubeGridSetSolidVertices_g(reso, coarse_bit, coarse_vbit);

		elesatlist.emplace_back(std::move(coarse_bit));
		vrtsatlist.emplace_back(std::move(coarse_vbit));
	}

	printf("-- Building %d layers (%s)\n", elesatlist.size(), (_setting.skiplayer1 ? "Non-dyadic" : "Dyadic"));

	std::vector<std::vector<int>> v2ehost[8];
	std::vector<std::vector<int>> v2vfinehost[27];
	std::vector<std::vector<int>> v2vcoarsehost[8];
	std::vector<std::vector<int>> v2vhost[27];
	std::vector<std::vector<double>> rxStencilhost[27][9];
	std::vector<std::vector<int>> vbitflaglist;
	std::vector<std::vector<int>> ebitflaglist;

	std::vector<int> v2vfinec[64];
	int* v2vfineclist[64];


	// generate topology between elements and vertices
	for (int i = 0; i < elesatlist.size(); i++) {
		std::vector<unsigned int>& ebit = elesatlist[i]._bitArray;
		std::vector<unsigned int>& vbit = vrtsatlist[i]._bitArray;
		int elementreso = resolist[i];
		int elementreso2 = pow(elementreso, 2);
		int vertexreso = elementreso + 1;
		int vertexreso2 = pow(vertexreso, 2);
		BitSAT<unsigned int>& elesat = elesatlist[i];
		BitSAT<unsigned int>& vrtsat = vrtsatlist[i];
		int nSolidElement = elesat.total();
		int nSolidVertex = vrtsat.total();

		//printf("--[%d] total valid vertex %*d\n, ", i, vrtsat.total());

		int out_width = 10;
		printf("--[%d] total valid vertex %*d, total element %*d\n ", i, out_width, vrtsat.total(), out_width, elesat.total());

		// allocate buffer
		for (int k = 0; k < 8; k++) {
			v2ehost[k].emplace_back(nSolidVertex, -1);
			v2vcoarsehost[k].emplace_back(nSolidVertex, -1);
		}

		for (int k = 0; k < 27; k++) {
			v2vfinehost[k].emplace_back(nSolidVertex, -1);
			v2vhost[k].emplace_back(nSolidVertex, -1);
		}
		
		vbitflaglist.emplace_back(nSolidVertex, 0);
		ebitflaglist.emplace_back(nSolidElement, 0);

		// compute flags
		std::vector<int>& vbitflag = *vbitflaglist.rbegin();
		std::vector<int>& ebitflag = *ebitflaglist.rbegin();

		// set vertices bit flags
		Grid::setVerticesPosFlag(vertexreso, vrtsat, vbitflag.data());

		// set elements bit flags
		for (int j = 0; j < ebit.size(); j++) {
			
		}

		// generate v2e
		while (1) {
			// only need vertex element topology on first layer
			if (i != 0) break;
			int* v2elist[8];
			for (int j = 0; j < 8; j++) v2elist[j] = v2ehost[j].rbegin()->data();
			//Grid::setV2E(vertexreso, vrtsat, elesat, v2elist);
			Grid::setV2E_g(vertexreso, vrtsat, elesat, v2elist);
			break;
		}

		// generate v2v
		int* v2vlist[27];
		for (int j = 0; j < 27; j++) v2vlist[j] = v2vhost[j].rbegin()->data();
		//Grid::setV2V(vertexreso, vrtsat, v2vlist);
		Grid::setV2V_g(vertexreso, vrtsat, v2vlist);

		/// between layers, generate neighbors on different layers
		// generate v2vcoarse from coarse layer
		while (1) {
			if ((_setting.skiplayer1 && i == 1) || i == resolist.size() - 1) break;
			// v2vfine, v2vcoarse, v2
			BitSAT<unsigned int>* vrtsatfine = &vrtsatlist[i];
			BitSAT<unsigned int>* vrtsatcoarse;
			int vresofine = resolist[i] + 1;
			int skip = 1;
			int finelayer = i - 1;
			if (_setting.skiplayer1 && i == 0) {
				vrtsatcoarse = &vrtsatlist[i + 2];
				skip = 2;
				finelayer = 0;
			}
			else {
				vrtsatcoarse = &vrtsatlist[i + 1];
			}

			int* v2vcoarse[8];
			for (int j = 0; j < 8; j++) v2vcoarse[j] = v2vcoarsehost[j].rbegin()->data();

			Grid::setV2VCoarse_g(skip, vresofine, *vrtsatfine, *vrtsatcoarse, v2vcoarse);

			break;
		}

		// generate v2vfine 
		while (1) {
			if (i == 0) break;
			if (_setting.skiplayer1 && (i == 2 || i == 1)) break;
			int* v2vfine[27];
			for (int j = 0; j < 27; j++) v2vfine[j] = v2vfinehost[j].rbegin()->data();
			BitSAT<unsigned int>& vsatfine = vrtsatlist[i - 1];
			BitSAT<unsigned int>& vsatcoarse = vrtsatlist[i];
			int vresocoase = resolist[i] + 1;
			int vresofine = resolist[i - 1] + 1;
			Grid::setV2VFine_g(1, vresocoase, vsatfine, vsatcoarse, v2vfine);
			break;
		}
		
		// generate v2vfinc for non-dyadic layer 2 
		if (_setting.skiplayer1 && i == 2) {
			int vresocoarse = resolist[i] + 1;
			BitSAT<unsigned int>& vsatfine = vrtsatlist[0];
			BitSAT<unsigned int>& vsatcoarse = vrtsatlist[2];
			for (int j = 0; j < 64; j++) {
				v2vfinec[j].resize(vsatcoarse.total());
				v2vfineclist[j] = v2vfinec[j].data();
			}
			Grid::setV2VFineC_g(vresocoarse, vsatfine, vsatcoarse, v2vfineclist);
		}
		

	} // finished all layers

	// find shell elements
	setSolidShellElement(elesatlist[0]._bitArray, elesatlist[0], out_box, resolist[0], ebitflaglist[0]);


	// upload grid to device
	for (int i = 0; i < elesatlist.size(); i++) {
		auto grd = new Grid();

		if (_setting.skiplayer1&&i == 1) {
			grd->set_dummy();
			//_gridlayer.emplace_back(grd);
		}

		int nv = 0, ne = 0;
			
		int* v2e[8] = { nullptr };
		int* v2vfine[27] = { nullptr };
		int* v2vcoarse[8] = { nullptr };
		int* v2v[27] = { nullptr };
		int* vbitflag = vbitflaglist[i].data();
		int* ebitflag = ebitflaglist[i].data();

		nv = v2vhost[0][i].size();

		ne = ebitflaglist[i].size();

		// list v2e
		for (int j = 0; j < 8; j++) {
			v2e[j] = v2ehost[j][i].data();
		}

		// list v2vcoarse
		for (int j = 0; j < 8; j++) {
			if (i < elesatlist.size() - 1) v2vcoarse[j] = v2vcoarsehost[j][i].data();
		}

		// list v2v
		for (int j = 0; j < 27; j++) {
			v2v[j] = v2vhost[j][i].data();
		}
		
		// select finer grid
		Grid* finer = nullptr;
		if (i != 0) {
			if (_setting.skiplayer1 && i == 2) { finer = _gridlayer[0]; }
			else { finer = _gridlayer[i - 1]; }
		}

		// list v2vfine
		for (int j = 0; j < 27; j++) {
			if (finer != nullptr) v2vfine[j] = v2vfinehost[j][i].data();
		}

		
		if (_setting.skiplayer1) grd->set_skip();

		grd->_inFixedArea = _inFixedArea;
		grd->_inLoadArea = _inLoadArea;
		grd->_loadField = _loadField;

		for (int i = 0; i < 6; i++) {
			(&grd->_box[0][0])[i] = (&out_box[0][0])[i];
		}

		grd->build(get_gmem(), vrtsatlist[i], elesatlist[i], finer, resolist[i] + 1, i, nv, ne, _setting.min_coeff, _setting.max_coeff, v2e, v2vfine, v2vcoarse, v2v, v2vfineclist, vbitflag, ebitflag);

		if (i == elesatlist.size() - 1) grd->getV2V();

		_gridlayer.emplace_back(grd);
	}

}

void HierarchyGrid::writeSupportForce(const std::string& filename)
{
	double* fs[3];
	Grid::getTempBufArray(fs, 3, n_loadnodes());
	getForceSupport(_gridlayer[0]->_gbuf.F, fs);
	std::vector<double> hostfs[3];
	for (int i = 0; i < 3; i++) {
		hostfs[i].resize(n_loadnodes());
		gpu_manager_t::download_buf(hostfs[i].data(), fs[i], sizeof(double) * n_loadnodes());
	}
	bio::write_vectors<double, 3>(filename, hostfs);
}

void HierarchyGrid::writeForce(const std::string& filename)
{
	double* fs[3];
	Grid::getTempBufArray(fs, 3, _gridlayer[0]->n_gsvertices);
	getForceSupport(_gridlayer[0]->_gbuf.F, fs);
	std::vector<double> hostfs[3];
	for (int i = 0; i < 3; i++) {
		hostfs[i].resize(_gridlayer[0]->n_gsvertices);
		gpu_manager_t::download_buf(hostfs[i].data(), fs[i], sizeof(double) * _gridlayer[0]->n_gsvertices);
	}
	bio::write_vectors<double, 3>(filename, hostfs);
}


// id pos
void HierarchyGrid::writeDensity(const std::string& filename)
{
	printf("-- writing vdb to %s\n", filename.c_str());

	std::vector<int> eidmaphost(_gridlayer[0]->n_elements);
	gpu_manager_t::download_buf(eidmaphost.data(), _gridlayer[0]->_gbuf.eidmap, sizeof(int) * _gridlayer[0]->n_elements);
	std::vector<float> rhohost(_gridlayer[0]->n_gselements);
	gpu_manager_t::download_buf(rhohost.data(), _gridlayer[0]->_gbuf.rho_e, sizeof(float) * _gridlayer[0]->n_gselements);

	std::vector<int> epos[3];
	for (int i = 0; i < 3; i++) epos[i].resize(_gridlayer[0]->n_elements);

	std::vector<float> evalue;
	evalue.resize(_gridlayer[0]->n_elements);

	int reso = _gridlayer[0]->_ereso;

	auto& esat = elesatlist[0];

	for (int i = 0; i < esat._bitArray.size(); i++) {
		int eword = esat._bitArray[i];
		int eidbase = esat._chunkSat[i];

		int eidoffset = 0;
		for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
			if (!read_bit(eword, ji)) continue;
			int bitid = i * BitCount<unsigned int>::value + ji;
			int bitpos[3] = { bitid % reso, bitid / reso % reso, bitid / reso / reso };
			int eid = eidoffset + eidbase;
			int rhoid = eidmaphost[eid];
			for (int k = 0; k < 3; k++) epos[k][eid] = bitpos[k];
			evalue[eid] = rhohost[rhoid];
			eidoffset++;
		}
	}
	
	openvdb_wrapper_t<float>::grid2openVDBfile(filename, epos, evalue);
}

// actual pos
void HierarchyGrid::writeDensityac(const std::string& filename)
{
	printf("-- writing vdb to %s\n", filename.c_str());

	std::vector<int> eidmaphost(_gridlayer[0]->n_elements);
	gpu_manager_t::download_buf(eidmaphost.data(), _gridlayer[0]->_gbuf.eidmap, sizeof(int) * _gridlayer[0]->n_elements);
	std::vector<float> rhohost(_gridlayer[0]->n_gselements);
	gpu_manager_t::download_buf(rhohost.data(), _gridlayer[0]->_gbuf.rho_e, sizeof(float) * _gridlayer[0]->n_gselements);

	std::vector<int> epos[3];
	std::vector<double> eposf[3];
	for (int i = 0; i < 3; i++) epos[i].resize(_gridlayer[0]->n_elements);
	for (int i = 0; i < 3; i++) eposf[i].resize(_gridlayer[0]->n_elements);

	std::vector<float> evalue;
	evalue.resize(_gridlayer[0]->n_elements);

	double eh = elementLength();

	double boxOrigin[3] = { _gridlayer[0]->_box[0][0],_gridlayer[0]->_box[0][1],_gridlayer[0]->_box[0][2] };

	int ereso = _gridlayer[0]->_ereso;

	auto& esat = elesatlist[0];
	
	for (int i = 0; i < esat._bitArray.size(); i++) {
		int eword = esat._bitArray[i];
		int eidbase = esat._chunkSat[i];

		int eidoffset = 0;
		for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
			if (!read_bit(eword, ji)) continue;
			int bitid = i * BitCount<unsigned int>::value + ji;
			int bitpos[3] = { bitid % ereso, bitid / ereso % ereso, bitid / ereso / ereso };
			int eid = eidoffset + eidbase;
			int rhoid = eidmaphost[eid];
			for (int k = 0; k < 3; k++)
			{
				epos[k][eid] = bitpos[k];
				eposf[k][eid] = bitpos[k] * eh + 0.5 * eh + boxOrigin[k];
			}
			evalue[eid] = rhohost[rhoid];
			eidoffset++;
		}
	}
#ifdef ENABLE_MATLAB
	Eigen::Matrix<int, -1, -1> eidx_;
	eidx_.resize(_gridlayer[0]->n_elements, 3);

	for (int i = 0; i < _gridlayer[0]->n_elements; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			eidx_(i, j) = epos[j][i];
		}
	}
	eigen2ConnectedMatlab("eid", eidx_);

	Eigen::Matrix<float, -1, 1> evalue_;
	evalue_.resize(_gridlayer[0]->n_elements, 1);

	for (int i = 0; i < _gridlayer[0]->n_elements; i++)
	{
		evalue_(i, 1) = evalue[i];
	}
	eigen2ConnectedMatlab("rhoe", evalue_);

	Eigen::Matrix<double, -1, -1> epos_;
	epos_.resize(_gridlayer[0]->n_elements, 3);

	for (int i = 0; i < _gridlayer[0]->n_elements; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			epos_(i, j) = eposf[j][i];
		}
	}
	eigen2ConnectedMatlab("epos", epos_);
#endif
	openvdb_wrapper_t<float>::grid2openVDBfile(filename, epos, evalue);
}


void grid::HierarchyGrid::writeSurfaceElement(const std::string& filename)
{
	_gridlayer[0]->mark_surface_elements_g(
		_gridlayer[0]->n_gsvertices, _gridlayer[0]->n_gselements,
		_gridlayer[0]->_gbuf.v2e, _gridlayer[0]->_gbuf.vBitflag, _gridlayer[0]->_gbuf.eBitflag
	);

	std::vector<int> eflags(_gridlayer[0]->n_gselements);
	gpu_manager_t::download_buf(eflags.data(), _gridlayer[0]->_gbuf.eBitflag, sizeof(int) * _gridlayer[0]->n_gselements);
	std::vector<int> eidmap(_gridlayer[0]->n_gselements);
	gpu_manager_t::download_buf(eidmap.data(), _gridlayer[0]->_gbuf.eidmap, sizeof(int) * _gridlayer[0]->n_elements);
	auto& esat = elesatlist[0];

	std::vector<float> surfpos[3];

	float eh = elementLength();

	float boxOrigin[3] = { _gridlayer[0]->_box[0][0],_gridlayer[0]->_box[0][1],_gridlayer[0]->_box[0][2] };

	int ereso = _gridlayer[0]->_ereso;

	for (int i = 0; i < esat._bitArray.size(); i++) {
		if (esat._bitArray[i] == 0) continue;
		int eid = esat._chunkSat[i];
		int bitword = esat._bitArray[i];
		for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
			int ebitid = i * BitCount<unsigned int>::value + ji;
			int epos[3] = { ebitid % ereso, ebitid / ereso % ereso, ebitid / ereso / ereso };
			if (!read_bit(bitword, ji))  continue;
			int egsid = eidmap[eid];
			if (egsid == -1) printf("-- error on eidmap\n");
			int efw = eflags[egsid];
			if (efw & Grid::Bitmask::mask_surfaceelements) {
				for (int k = 0; k < 3; k++)  surfpos[k].emplace_back(epos[k] * eh + boxOrigin[k]);
			}
			eid++;
		}
	}
	
	printf("-- writing surface element pos to file %s\n", filename.c_str());
	bio::write_vectors(filename, surfpos);
}

void grid::HierarchyGrid::readDensity(const std::string& filename)
{
	std::vector<int> eidmaphost(_gridlayer[0]->n_elements);
	gpu_manager_t::download_buf(eidmaphost.data(), _gridlayer[0]->_gbuf.eidmap, sizeof(int) * _gridlayer[0]->n_elements);
	std::vector<float> rhohost(_gridlayer[0]->n_gselements, 0);

	std::vector<int> epos[3];
	for (int i = 0; i < 3; i++) epos[i].resize(_gridlayer[0]->n_elements);
	std::vector<float> evalue;
	openvdb_wrapper_t<float>::openVDBfile2grid(filename, epos, evalue);
	
	int ereso = _gridlayer[0]->_ereso;
	auto& esat = elesatlist[0];

	for (int i = 0; i < epos->size(); i++) {
		if (epos[0][i] >= ereso || epos[1][i] >= ereso || epos[2][i] >= ereso) {
			printf("\033[31m-- unmatched grid and file \033[0m\n");
			exit(-1);
		}
		int ebid = epos[0][i] + epos[1][i] * ereso + epos[2][i] * ereso * ereso;
		int eid = esat(ebid);
		if (eid == -1 || eid >= eidmaphost.size()) {
			printf("\033[31m-- unmatched grid and file\033[0m\n");
			exit(-1);
		}
		int egsid = eidmaphost[eid];
		rhohost[egsid] = evalue[i];
	}

	gpu_manager_t::upload_buf(_gridlayer[0]->_gbuf.rho_e, rhohost.data(), sizeof(float) * _gridlayer[0]->n_gselements);
}

void grid::HierarchyGrid::writeSensitivity(const std::string& filename)
{
	std::vector<int> eidmaphost(_gridlayer[0]->n_elements);
	gpu_manager_t::download_buf(eidmaphost.data(), _gridlayer[0]->_gbuf.eidmap, sizeof(int) * _gridlayer[0]->n_elements);
	std::vector<float> senshost(_gridlayer[0]->n_gselements);
	gpu_manager_t::download_buf(senshost.data(), _gridlayer[0]->_gbuf.g_sens, sizeof(float) * _gridlayer[0]->n_gselements);

	std::vector<int> epos[3];
	for (int i = 0; i < 3; i++) epos[i].resize(_gridlayer[0]->n_elements);

	std::vector<float> evalue;
	evalue.resize(_gridlayer[0]->n_elements);

	int reso = _gridlayer[0]->_ereso;

	auto& esat = elesatlist[0];

	for (int i = 0; i < esat._bitArray.size(); i++) {
		int eword = esat._bitArray[i];
		int eidbase = esat._chunkSat[i];

		int eidoffset = 0;
		for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
			if (!read_bit(eword, ji)) continue;
			int bitid = i * BitCount<unsigned int>::value + ji;
			int bitpos[3] = { bitid % reso, bitid / reso % reso, bitid / reso / reso };
			int eid = eidoffset + eidbase;
			int rhoid = eidmaphost[eid];
			for (int k = 0; k < 3; k++) epos[k][eid] = bitpos[k];
			evalue[eid] = senshost[rhoid];
			eidoffset++;
		}
	}
	
	openvdb_wrapper_t<float>::grid2openVDBfile(filename, epos, evalue);

}

void grid::HierarchyGrid::readCoeff(const std::string& filename)
{
	std::vector<int> eidmaphost(_gridlayer[0]->n_elements);
	gpu_manager_t::download_buf(eidmaphost.data(), _gridlayer[0]->_gbuf.eidmap, sizeof(int) * _gridlayer[0]->n_elements);
	std::vector<float> rhohost(_gridlayer[0]->n_gselements, 0);

	std::vector<int> epos[3];
	for (int i = 0; i < 3; i++) epos[i].resize(_gridlayer[0]->n_elements);
	std::vector<float> evalue;
	openvdb_wrapper_t<float>::openVDBfile2grid(filename, epos, evalue);

	int ereso = _gridlayer[0]->_ereso;
	auto& esat = elesatlist[0];

	for (int i = 0; i < epos->size(); i++) {
		if (epos[0][i] >= ereso || epos[1][i] >= ereso || epos[2][i] >= ereso) {
			printf("\033[31m-- unmatched grid and file \033[0m\n");
			exit(-1);
		}
		int ebid = epos[0][i] + epos[1][i] * ereso + epos[2][i] * ereso * ereso;
		int eid = esat(ebid);
		if (eid == -1 || eid >= eidmaphost.size()) {
			printf("\033[31m-- unmatched grid and file\033[0m\n");
			exit(-1);
		}
		int egsid = eidmaphost[eid];
		rhohost[egsid] = evalue[i];
	}

	gpu_manager_t::upload_buf(_gridlayer[0]->_gbuf.rho_e, rhohost.data(), sizeof(float) * _gridlayer[0]->n_gselements);
}


void grid::HierarchyGrid::writeCoeff(const std::string& filename)
{
	int coeff_size = _gridlayer[0]->n_im * _gridlayer[0]->n_in * _gridlayer[0]->n_il;
	std::vector<float> coeffhost(coeff_size);
	gpu_manager_t::download_buf(coeffhost.data(), _gridlayer[0]->_gbuf.coeffs, sizeof(float) * coeff_size);

	bio::write_vector(filename, coeffhost);

#ifdef ENABLE_MATLAB
	Eigen::Matrix<double, -1, 1> coeff2host(coeff_size, 1);
	int f_total = 0;
	for (int i = 0; i < coeff_size; i++)
	{
		coeff2host(i, 1) = coeffhost[i];
	}
	eigen2ConnectedMatlab("coeff", coeff2host);
#endif
}

void grid::HierarchyGrid::writeComplianceDistribution(const std::string& filename)
{
	// compute element compliance
	_gridlayer[0]->elementCompliance(_gridlayer[0]->getDisplacement(), _gridlayer[0]->getForce(), _gridlayer[0]->getSens());

	// write computed element compliance
	writeSensitivity(filename);
}

void grid::HierarchyGrid::writeDisplacement(const std::string& filename)
{
	/*writeNodePos(getPath("nodepos"), *_gridlayer[0]);*/
	std::vector<double> u[3];
	for (int i = 0; i < 3; i++) {
		u[i].resize(_gridlayer[0]->n_nodes());
		gpu_manager_t::download_buf(u[i].data(), _gridlayer[0]->_gbuf.U[i], sizeof(double) * _gridlayer[0]->n_nodes());
	}

	bio::write_vectors(filename, u);
}

void HierarchyGrid::resetAllResidual(void)
{
	for (int i = 0; i < _gridlayer.size(); i++) {
		if (_gridlayer[i]->is_dummy()) continue;
		_gridlayer[i]->reset_residual();
	}
}

void HierarchyGrid::writeV2V(const std::string& filename, Grid& g)
{
	std::vector<int> v2v(27 * g.n_gsvertices);
	for (int i = 0; i < 27; i++) {
		std::vector<int> v(g.n_gsvertices);
		gpu_manager_t::download_buf(v.data(), g._gbuf.v2v[i], sizeof(int) * g.n_gsvertices);
		for (int j = 0; j < g.n_gsvertices; j++) {
			v2v[j * 27 + i] = v[j];
		}
	}
	bio::write_vector(filename, v2v);
}

void HierarchyGrid::writeV2Vfine(const std::string& filename, Grid& g)
{
	std::vector<int> v2vf(27 * g.n_gsvertices);
	for (int i = 0; i < 27; i++) {
		std::vector<int> v(g.n_gsvertices);
		gpu_manager_t::download_buf(v.data(), g._gbuf.v2vfine[i], sizeof(int) * g.n_gsvertices);
		for (int j = 0; j < g.n_gsvertices; j++) {
			v2vf[j * 27 + i] = v[j];
		}
	}
	bio::write_vector(filename, v2vf);
}

void HierarchyGrid::writeV2Vcoarse(const std::string& filename, Grid& g)
{
	std::vector<int> v2vc(8 * g.n_gsvertices);
	for (int i = 0; i < 8; i++) {
		std::vector<int> v(g.n_gsvertices);
		gpu_manager_t::download_buf(v.data(), g._gbuf.v2vcoarse[i], sizeof(int) * g.n_gsvertices);
		for (int j = 0; j < g.n_gsvertices; j++) {
			v2vc[j * 8 + i] = v[j];
		}
	}
	bio::write_vector(filename, v2vc);
}

void HierarchyGrid::writeVidMap(const std::string& filename, Grid& g)
{
	std::vector<int> vidhost(g.n_vertices);
	gpu_manager_t::download_buf(vidhost.data(), g._gbuf.vidmap, sizeof(int)*g.n_vertices);
	bio::write_vector(filename, vidhost);
}

void HierarchyGrid::writeEidMap(const std::string& filename, Grid& g)
{
	std::vector<int> eidhost(g.n_elements);
	gpu_manager_t::download_buf(eidhost.data(), g._gbuf.eidmap, sizeof(int) * g.n_elements);
	bio::write_vector(filename, eidhost);
}

void HierarchyGrid::writeNodePos(const std::string& nam, Grid& g)
{
	std::vector<double> p3host;
	getNodePos(g, p3host);
	bio::write_vector(nam, p3host);
#ifdef ENABLE_MATLAB
	Eigen::Matrix<double, -1, -1> p3host_;
	p3host_.resize(g.n_gsvertices, 3);
	for (int i = 0; i < 3; i++)
	{
		
		for (int j = 0; j < g.n_gsvertices; j++)
		{
			p3host_(j, i) = p3host[3 * j + i];
		}
	}
	eigen2ConnectedMatlab("nodepos", p3host_);
#endif // ENABLE_MATLAB
}

void HierarchyGrid::writeElementPos(const std::string& filename, Grid& g)
{
	std::vector<double> p3host;
	getElementPos(g, p3host);
	bio::write_vector(filename, p3host);
#ifdef ENABLE_MATLAB
	Eigen::Matrix<double, -1, -1> p3host_;
	p3host_.resize(g.n_gselements, 3);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < g.n_gselements; j++)
		{
			p3host_(j, i) = p3host[3 * j + i];
		}
	}
	eigen2ConnectedMatlab("elepos", p3host_);
#endif // ENABLE_MATLAB
}

void HierarchyGrid::test_vcycle(void)
{
	printf("-- testing v cycles \n");
	double rel_res = 1;

	resetAllResidual();

	_gridlayer[0]->randForce();
	_gridlayer[0]->reset_displacement();

	//while (rel_res > 1e-3) {
	//	//rel_res = v_cycle();
	//	//_gridlayer[0]->force2matlab("f");
	//	_gridlayer[0]->gs_relax();
	//	//_gridlayer[0]->displacement2matlab("u");
	//	_gridlayer[0]->update_residual();
	//	//_gridlayer[0]->residual2matlab("r");
	//	rel_res = _gridlayer[0]->relative_residual();
	//	printf("-- r = %4.2lf%%\n", rel_res * 100);

	//	_gridlayer[2]->restrict_residual();
	//	//_gridlayer[2]->residual2matlab("f2");
	//	_gridlayer[2]->reset_displacement();
	//	_gridlayer[2]->gs_relax();

	//	//_gridlayer[2]->displacement2matlab("u2");
	//	_gridlayer[0]->prolongate_correction();
	//	_gridlayer[0]->displacement2matlab("uc");
	//	_gridlayer[0]->gs_relax();
	//}


	int itn = 0;
	while (rel_res > 1e-4 && itn++ < 20) {
		int depth = n_grid() - 1;
		for (int i = 0; i < depth + 1; i++) {
			if (_gridlayer[i]->is_dummy()) { continue; }
			if (i > 0) {
				_TIC("updater");
				_gridlayer[i]->fineGrid->update_residual();
				_TOC;
				_TIC("restrictr");
				_gridlayer[i]->restrict_residual();
				_TOC;
				_gridlayer[i]->reset_displacement();
			}
			if (i < n_grid() - 1) {
				_TIC("downrelax");
				_gridlayer[i]->gs_relax();
				_TOC;
			}
			else {
				_TIC("hostfem");
				_gridlayer[i]->solve_fem_host();
				_TOC;
			}
		}
		for (int i = depth - 1; i >= 0; i--) {
			if (_gridlayer[i]->is_dummy()) { continue; }
			_TIC("prolong");
			_gridlayer[i]->prolongate_correction();
			_TOC;
			_TIC("uprelax");
			_gridlayer[i]->gs_relax();
			_TOC;
		}
		//_gridlayer[0]->gs_relax(2);
		_gridlayer[0]->update_residual();
		rel_res = _gridlayer[0]->relative_residual();
		printf("-- r_rel = %4.2lf%%\n", rel_res * 100);
	}

	printf("-- time downrelax : %6.2f ms\n", tictoc::get_record("downrelax"));
	printf("-- time updater   : %6.2f ms\n", tictoc::get_record("updater"));
	printf("-- time restrictr : %6.2f ms\n", tictoc::get_record("restrictr"));
	printf("-- time hostfem   : %6.2f ms\n", tictoc::get_record("hostfem"));
	printf("-- time prolong   : %6.2f ms\n", tictoc::get_record("prolong"));
	printf("-- time uprelax   : %6.2f ms\n", tictoc::get_record("uprelax"));
	exit(-1);

	_gridlayer[0]->displacement2matlab("u1");
	_gridlayer[0]->residual2matlab("r1");
	while (rel_res > 1e-5 && itn++ < 200) {
		_gridlayer[0]->gs_relax();
		_gridlayer[0]->update_residual();
		rel_res = _gridlayer[0]->relative_residual();
		printf("---r_rel = %4.2lf%%\n", rel_res * 100);
	}
	_gridlayer[0]->displacement2matlab("u2");
	_gridlayer[0]->residual2matlab("r2");

	// two layers
	exit(-1);
}

double HierarchyGrid::elementLength(void)
{
	return _gridlayer[0]->elementLength();
}

double HierarchyGrid::v_cycle(int pre_relax, int post_relax)
{
	int depth = n_grid() - 1;
	// downside
	for (int i = 0; i < depth + 1; i++) {
		if (_gridlayer[i]->is_dummy()) { continue; }
		if (i > 0) {
			//_gridlayer[i]->stencil2matlab("rxcoarse");
			_gridlayer[i]->fineGrid->update_residual();
			//_gridlayer[i]->fineGrid->residual2matlab("rfine");
			_gridlayer[i]->restrict_residual(); 
			//_gridlayer[i]->force2matlab("fcoarse");
			_gridlayer[i]->reset_displacement();
		}
		if (i < n_grid() - 1) {
			_gridlayer[i]->gs_relax(pre_relax);
			//_gridlayer[i]->displacement2matlab("u");
		}
		else {
			//_gridlayer[i]->force2matlab("f");
			_gridlayer[i]->solve_fem_host();
			//_gridlayer[i]->displacement2matlab("u");
		}
	}
	// DEBUG
	//_gridlayer[0]->displacement2matlab("u");
	// upside
	for (int i = depth - 1; i >= 0; i--) {
		if (_gridlayer[i]->is_dummy()) { continue; }
		//_gridlayer[i]->displacement2matlab("u");
		//_gridlayer[i]->update_residual();
		//printf("-- [%d] r = %lf%%\n", i, _gridlayer[i]->relative_residual() * 100);
		_gridlayer[i]->prolongate_correction();
		//_gridlayer[i]->update_residual();
		//printf("-- [%d] rc=  %lf%%\n", i, _gridlayer[i]->relative_residual() * 100);
		//_gridlayer[i]->displacement2matlab("uc");
		//_gridlayer[i]->force2matlab("fc");
		_gridlayer[i]->gs_relax(post_relax);
		//_gridlayer[i]->update_residual();
		//printf("-- [%d] rr=  %lf%%\n", i, _gridlayer[i]->relative_residual() * 100);
		//_gridlayer[i]->displacement2matlab("ur");
	}

	_gridlayer[0]->update_residual();
	return _gridlayer[0]->relative_residual();
}

double grid::HierarchyGrid::v_halfcycle(int depth, int pre_relax /*= 1*/, int post_relax /*= 1*/)
{
	// downside
	for (int i = 0; i < depth + 1; i++) {
		if (_gridlayer[i]->is_dummy()) { continue; }
		if (i > 0) {
			_gridlayer[i]->fineGrid->update_residual();
			_gridlayer[i]->restrict_residual(); 
			_gridlayer[i]->reset_displacement();
		}
		if (i < n_grid() - 1) {
			_gridlayer[i]->gs_relax(pre_relax);
		}
		else {
			_gridlayer[i]->solve_fem_host();
		}
	}
	// upside
	for (int i = depth - 1; i >= 0; i--) {
		if (_gridlayer[i]->is_dummy()) { continue; }
		if (depth >= 2) {
			_gridlayer[i]->prolongate_correction();
		}
		_gridlayer[i]->gs_relax(post_relax);
	}

	_gridlayer[0]->update_residual();
	return _gridlayer[0]->relative_residual();

}

void grid::cubeGridSetSolidVertices(int reso, const std::vector<unsigned int>& solid_ebit, std::vector<unsigned int>& solid_vbit)
{
	size_t nelements = pow(reso, 3);

	size_t reso2 = pow(reso, 2);

	size_t vreso = reso + 1;

	size_t nvertices = pow(reso + 1, 3);

	size_t vreso2 = pow(reso + 1, 2);

	size_t inci_vsize = nvertices / (sizeof(unsigned int) * 8) + 1;

	solid_vbit.clear();
	solid_vbit.resize(inci_vsize, 0);

	for (int j = 0; j < 8; j++) {
		int loc[3] = { j % 2,(j % 4) / 2,j / 4 };
#pragma omp parallel for
		for (int i = 0; i < nelements; i++) {
			// MARK[TODO] add solid_ebit selection of i
			// read(solid_ebit.data(), i)
			// may return 1 --> in the model
			if (read_bit(solid_ebit.data(), i))
			{
				int eloc[3] = { i % reso ,(i / reso) % reso,i / reso2 };
				int vloc[3] = { eloc[0] + loc[0],eloc[1] + loc[1],eloc[2] + loc[2] };
				int vid = vloc[0] + vloc[1] * vreso + vloc[2] * vreso2;
#pragma omp critical
				{
					set_bit(solid_vbit.data(), vid);
				}
			}
		}
	}
}

grid::DispatchCubeVertex::vertex_type grid::DispatchCubeVertex::vtype[27];

grid::DispatchCubeVertex::DispatchCubeVertex(void)
{
	// volume vertex
	for (int j : {0, 2, 6, 8, 18, 20, 24, 26}) {
		vtype[j] = corner_vertex;
	}

	// face center
	for (int j : {4, 10, 12, 14, 16, 22}) {
		vtype[j] = face_center;
	}

	// edge center
	for (int j : {1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25}) {
		vtype[j] = edge_center;
	}
	// volume center
	{
		vtype[13] = volume_center;
	}
}

grid::DispatchCubeVertex::vertex_type grid::DispatchCubeVertex::dispatch(int id)
{
	if (id >= 27) throw std::string("index overflow!");
	return vtype[id];
}

std::ostream& Grid::msg(void)
{
	std::cout << "\033[32m" << "[" << _name << "] " << "\033[0m";
	return std::cout;
}

void Grid::setOutDir(const std::string& outdir)
{
	_outdir = outdir;
}

void Grid::setMeshFile(const std::string& meshfile)
{
	_meshfile = meshfile;
}

std::string grid::Grid::_outdir;
std::string grid::Grid::_meshfile;

void* grid::Grid::_tmp_buf = nullptr;
size_t grid::Grid::_tmp_buf_size = 0;
void* grid::Grid::_tmp_buf1 = nullptr;
size_t grid::Grid::_tmp_buf1_size = 0;
void* grid::Grid::_tmp_buf2 = nullptr;
size_t grid::Grid::_tmp_buf2_size = 0;

grid::Mode grid::Grid::_mode;
grid::GlobalSSMode grid::Grid::_ssmode;
grid::GlobalDripMode grid::Grid::_dripmode;

float grid::Grid::_default_print_angle = 3 * M_PI / 4;
float grid::Grid::_opt_print_angle = 3 * M_PI / 4;

int grid::Grid::n_order = 0;       // The order of implicit spline
size_t grid::Grid::n_partitionx = 0;  // The number of partition of X,Y,Z direction
size_t grid::Grid::n_partitiony = 0;
size_t grid::Grid::n_partitionz = 0;
size_t grid::Grid::n_im;              // The number of knot series of X,Y,Z direction
size_t grid::Grid::n_in;
size_t grid::Grid::n_il;
size_t grid::Grid::n_knotspanx;       // The number of knot of X,Y,Z direction
size_t grid::Grid::n_knotspany;
size_t grid::Grid::n_knotspanz;
int grid::Grid::sppartition[3];
int grid::Grid::spbasis[3];
int grid::Grid::spknotspan[3];

float grid::Grid::m_sStep[3];
float grid::Grid::m_3sBoundMin[3];
float grid::Grid::m_3sBoundMax[3];

const std::string& Grid::getOutDir(void)
{
	return _outdir;
}

snippet::Loger Grid::subLog(const std::string& sublog)
{
	if (_logHistory.find(sublog) == _logHistory.end()) {
		// clear old log
		std::ofstream ofs(getOutDir() + sublog, std::ios::trunc);
		ofs.close();
		_logHistory.insert(sublog);
	}
	return snippet::Loger(_logStack, getOutDir() + sublog);
}

std::ofstream Grid::msglog(void)
{
	std::ofstream ofs(_logStack.top(), std::ios::app);
	ofs << "[" << _name << "] ";
	return ofs;
}

void Grid::clearMsglog(void)
{
	while (!_logStack.empty()) {
		std::string fi = getOutDir() + _logStack.top();
		std::ofstream os(fi, std::ios::trunc);
		os.close();
		_logStack.pop();
	}
	std::string outfile = getOutDir() + "log.txt";
	_logStack.push(outfile);
	std::ofstream ofs(outfile, std::ios::trunc);
	ofs.close();
}

void Grid::lexico2gsorder(int* idmap, int n_id, int* ids, int n_mapid, int* mapped_ids, int* valuemap)
{
	std::vector<int> oldids;
	int* pid = ids;
	if (ids == mapped_ids) {
		oldids.resize(n_id);
		oldids.assign(ids, ids + n_id);
		pid = oldids.data();
	}
	for (int i = 0; i < n_mapid; i++) mapped_ids[i] = -1;
	if (valuemap != nullptr) {
		for (int i = 0; i < n_id; i++) {
			if (idmap != nullptr) {
				mapped_ids[idmap[i]] = valuemap[pid[i]];
			}
			else {
				mapped_ids[i] = valuemap[pid[i]];
			}
		}
	}
	else {
		for (int i = 0; i < n_id; i++) {
			mapped_ids[idmap[i]] = pid[i];
		}
	}
}

void Grid::computeProjectionMatrix(int nv, int nv_gs, int vreso, const std::vector<int>& lexi2gs, const int* lexi2gs_dev, BitSAT<unsigned int>& vsat, int* vflaghost, int* vflagdev)
{
	double eh = (_box[1][0] - _box[0][0]) / (vreso - 1);

	std::cout << " bounding box min : (" << _box[0][0] << ", " << _box[0][1] << ", " << _box[0][2] << ")" << std::endl;
	std::cout << " bounding box max : (" << _box[1][0] << ", " << _box[1][1] << ", " << _box[1][2] << ")" << std::endl;
	int vreso2 = pow(vreso, 2);

	std::vector<int> loadvid;
	std::vector<Eigen::Matrix<double, 3, 1>> loadpos;
	std::vector<Eigen::Matrix<double, 3, 1>> loadnormal;

	std::vector<Eigen::Matrix<double, 3, 1>> loadforce;

	std::vector<int> supportvids;

	std::vector<Eigen::Matrix<double, 3, 1>> supportpos;

	int nfixnodes = 0;

	// compute surface load nodes positions and set corresponding flag
#pragma omp parallel for
	for (int i = 0; i < vsat._bitArray.size(); i++) {
		unsigned int word = vsat._bitArray[i];
		int vidoffset = vsat._chunkSat[i];
		int nv_word = 0;
		for (int j = 0; j < BitCount<unsigned int>::value; j++) {
			if (read_bit(word, j)) {
				int flag = vflaghost[vidoffset + nv_word];
				if (flag & Bitmask::mask_surfacenodes) {
					int bitid = i * BitCount<unsigned int>::value + j;
					int id[3] = { bitid % vreso, bitid % vreso2 / vreso, bitid / vreso2 };
					double vpos[3] = { _box[0][0] + id[0] * eh,_box[0][1] + id[1] * eh,_box[0][2] + id[2] * eh };
					bool isFix = false, isLoad = false;
					// set support nodes flag
					if (_inFixedArea(vpos)) {
						isFix = true;
						flag |= mask_supportnodes;
#pragma omp critical
						{
							supportvids.emplace_back(vidoffset + nv_word);
							supportpos.emplace_back(vpos[0], vpos[1], vpos[2]);
							nfixnodes++;
						}
					}

					if (isFix) { goto _setFlag; }

					// set load nodes flag
					if (_inLoadArea(vpos)) {
						isLoad = true;
						flag |= mask_loadnodes;
#pragma omp critical
						{
							loadvid.emplace_back(vidoffset + nv_word);
							loadpos.emplace_back(vpos[0], vpos[1], vpos[2]);
							loadforce.emplace_back(_loadField(vpos));
						}
					}

					_setFlag:
					vflaghost[vidoffset + nv_word] = flag;
				}
				nv_word++;
			}
		}
	}

	printf("-- found %d fixed nodes, %d load nodes\n", nfixnodes, loadvid.size());

	//array2ConnectedMatlab("loadpos", loadpos.data()->data(), loadpos.size() * 3);
#ifdef ENABLE_MATLAB
	array2ConnectedMatlab("supportnodes", supportvids.data(), supportvids.size());

	Eigen::Matrix<double, -1, -1> loadpos_(loadpos.size(), 3);
	for (int i = 0; i < loadpos.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			loadpos_(i, j) = loadpos[i][j];
		}
	}
	eigen2ConnectedMatlab("loadpos_", loadpos_);	
#endif
	

	// write load flags back to device
	gpu_manager_t::upload_buf(vflagdev, vflaghost, sizeof(int) * nv);

	// [ n1_x^T;n1_y^T, n2_x^T;n2_y^T, ... , nk_x^T;nk_y^T ]
	loadnormal.resize(loadvid.size());

	for (int i = 0; i < loadvid.size(); i++) {
		loadnormal[i] = outwardNormal(loadpos[i].data());
	}

	for (int i = 0; i < loadvid.size(); i++) {
		int oldid = loadvid[i];
		loadvid[i] = lexi2gs[oldid];
	}

	// DEBUG
	{
		//std::vector<double> p3;
		//for (int i = 0; i < loadpos.size(); i++) {
		//	p3.emplace_back(loadpos[i][0]);
		//	p3.emplace_back(loadpos[i][1]);
		//	p3.emplace_back(loadpos[i][2]);
		//}
		//std::vector<double> f3;
		//for (int i = 0; i < loadforce.size(); i++) {
		//	f3.emplace_back(loadforce[i][0]);
		//	f3.emplace_back(loadforce[i][1]);
		//	f3.emplace_back(loadforce[i][2]);
		//}
		//bio::write_vector("./out/p3", p3);
		//bio::write_vector("./out/f3", f3);
		std::vector<double> s3;
		for (int i = 0; i < supportpos.size(); i++) {
			s3.emplace_back(supportpos[i][0]);
			s3.emplace_back(supportpos[i][1]);
			s3.emplace_back(supportpos[i][2]);
		}
		bio::write_vector(grids.getPath("supportpos"), s3);

		// here the pos are actual input model scale
#ifdef ENABLE_MATLAB
		Eigen::Matrix<double, -1, -1> supppos_(supportpos.size(), 3);
		for (int i = 0; i < supportpos.size(); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				supppos_(i, j) = supportpos[i][j];				
			}
		}
		eigen2ConnectedMatlab("supppos_", supppos_);
#endif

	}

	setNodes(vsat, vreso, lexi2gs, lexi2gs_dev, vflagdev, nv_gs);
	setLoadNodes(loadvid, loadpos, loadnormal, loadforce);
}

Eigen::Matrix<double, 3, 1> Grid::outwardNormal(double p[3])
{
	Eigen::Matrix<double, 3, 1> normal;
	Point pos(p[0], p[1], p[2]);
	auto ptri = aabb_tree.closest_point_and_primitive(pos);
	auto trinormal = ptri.second->supporting_plane().orthogonal_vector();
	Kernel::Vector_3 pos2close(pos, ptri.first);
	double len = sqrt(pos2close.squared_length());
	if (len < 1e-6) {
		for (int i = 0; i < 3; i++) normal[i] = trinormal[i];
		return normal.normalized();
	}
	double dotpro = CGAL::scalar_product(pos2close, trinormal);
	if (dotpro < 0) {
		len = -len;
	}
	for (int i = 0; i < 3; i++) normal[i] = pos2close[i] / len;
	return normal;
}

void Grid::setVerticesPosFlag(int vreso, BitSAT<unsigned int>& vrtsat, int* flags)
{
	auto& vbit = vrtsat._bitArray;
	int vreso2 = vreso * vreso;
	for (int j = 0; j < vbit.size(); j++) {
		auto word = vbit[j];
		if (word == 0) continue;
		for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
			if (!read_bit(word, ji)) continue;
			int vbitid = j * BitCount<unsigned int>::value + ji;
			int flagword = 0;

			// position mod 8 flag
			int vpos[3] = { vbitid % vreso, vbitid / vreso % vreso, vbitid / vreso2 };
			flagword |= vpos[0] % 8;
			flagword |= (vpos[1] % 8) << 3;
			flagword |= (vpos[2] % 8) << 6;

			int vid = vrtsat[vbitid];

			// write flag word to memory
			flags[vid] = flagword;
		}
	}
}

void Grid::setElementsPosFlag(int ereso, BitSAT<unsigned int>& elesat, int* flags)
{
	auto& ebit = elesat._bitArray;
	int ereso2 = ereso * ereso;
	for (int j = 0; j < ebit.size(); j++)
	{
		auto word = ebit[j];
		if (word == 0) continue;
		for (int ji = 0; ji < BitCount<unsigned int>::value; ji++)
		{
			if (!read_bit(word, ji)) continue;
			int ebitid = j * BitCount<unsigned int>::value + ji;
			int flagword = 0;

			// position mod 8 flag
			int epos[3] = { ebitid * ereso, ebitid / ereso % ereso, ebitid / ereso2 };
			flagword |= epos[0] % 8;
			flagword |= (epos[1] % 8) << 3;
			flagword |= (epos[2] % 8) << 6;

			int eid = elesat[ebitid];

			// write flag word to memory 
			flags[eid] = flagword;
		}
	}
}

void Grid::setV2E(int vreso, BitSAT<unsigned int>& vrtsat, BitSAT<unsigned int>& elsat, int* v2elist[8])
{
	int vreso2 = pow(vreso, 2);
	int elementreso = vreso - 1;
	int elementreso2 = pow(elementreso, 2);
	auto& ebit = elsat._bitArray;
	for (int k = 0; k < 8; k++) {
		int* v2e = v2elist[k];
		int loc[3] = { k % 2 ,(k % 4) / 2 ,k / 4 };
		for (int j = 0; j < ebit.size(); j++) {
			auto word = ebit[j];
			if (word == 0) continue;
			for (int ji = 0; ji < sizeof(unsigned int) * 8; ji++) {
				if (!read_bit(word, ji)) continue;
				int eid = j * sizeof(unsigned int) * 8 + ji;
				int epos[3] = { eid % elementreso, (eid % elementreso2) / elementreso, eid / elementreso2 };
				int vloc[3] = { epos[0] + loc[0],epos[1] + loc[1],epos[2] + loc[2] };
				int vid = vloc[0] + vloc[1] * vreso + vloc[2] * vreso2;
				v2e[vrtsat[vid]] = elsat[eid];
			}
		}
	}

}

void Grid::setV2V(int vreso, BitSAT<unsigned int>& vrtsat, int* v2vlist[27])
{
	int vertexreso = vreso;
	int vertexreso2 = pow(vreso, 2);
	auto& vbit = vrtsat._bitArray;
	for (int k = 0; k < 27; k++) {
		int* v2v = v2vlist[k];
		int loc[3] = { k % 3 - 1,k / 3 % 3 - 1,k / 9 - 1 };
		for (int j = 0; j < vbit.size(); j++) {
			auto word = vbit[j];
			if (word == 0) continue;
			for (int ji = 0; ji < BitCount<unsigned int>::value; ji++) {
				if (!read_bit(word, ji)) continue;
				int vid = BitCount<unsigned int>::value*j + ji;
				int vloc[3] = { vid % vertexreso + loc[0], (vid % vertexreso2) / vertexreso + loc[1], vid / vertexreso2 + loc[2] };
				if (vloc[0] < 0 || vloc[1] < 0 || vloc[2] < 0) continue;
				int neighid = vloc[0] + vloc[1] * vertexreso + vloc[2] * vertexreso2;
				v2v[vrtsat[vid]] = vrtsat[neighid];
			}
		}
	}
}

double Grid::elementLength(void)
{
	return (_box[1][0] - _box[0][0]) / _ereso;
}

void Grid::getV2V(void)
{
	for (int i = 0; i < 27; i++) {
		_v2v[i].resize(n_gsvertices);
		gpu_manager_t::download_buf(_v2v[i].data(), _gbuf.v2v[i], sizeof(int) * n_gsvertices);
	}
}

size_t grid::Grid::build(
	gpu_manager_t& gm,
	BitSAT<unsigned int>& vbit,
	BitSAT<unsigned int>& ebit,
	Grid* finer,
	//Grid* coarser,
	int vreso,
	int layer,
	int nv, int ne,
	float mincoeff, float maxcoeff,
	int * v2ehost[8],
	int * v2vfinehost[27],
	int * v2vcoarsehost[8],
	int * v2vhost[27],
	int * v2vfinec[64],
	int * vbitflags,
	int * ebitflags
) {
	_dummy |= v2vhost[0] == nullptr;

	fineGrid = finer;

	if (finer != nullptr) finer->coarseGrid = this;

	_ereso = vreso - 1;

	_layer = layer;

	if (_dummy) return -1;

	char sbuf[1000];

	sprintf_s(sbuf, "[%d]", layer);

	_name = sbuf;

	// mark GS color id in element and vertex bitflag word
	compute_gscolor(gm, vbit, ebit, vreso, vbitflags, ebitflags);

	// lexicographical order to GS colored order 
	// gs[lexi] = Gs colored vid
	int nv_gs = 0;
	int ne_gs = 0;
	std::vector<int> vlexi2gs;
	std::vector<int> elexi2gs;
	enumerate_gs_subset(nv, ne, vbitflags, ebitflags, nv_gs, ne_gs, vlexi2gs, elexi2gs);

	n_vertices = nv;
	n_elements = ne;
	n_gselements = ne_gs; // correct to mod32 == 0
	n_gsvertices = nv_gs; // correct to mod32 == 0

	
	size_t gbuf_size = 0;
	
	int* vidmap = (int*)gm.add_buf(_name + " vid map ", sizeof(int) * nv, vlexi2gs.data(), sizeof(int) * nv); gbuf_size += sizeof(int) * nv;
	int* eidmap = (int*)gm.add_buf(_name + " eid map ", sizeof(int) * ne, elexi2gs.data(), sizeof(int) * ne); gbuf_size += sizeof(int) * ne;

	_gbuf.eidmap = eidmap;
	_gbuf.vidmap = vidmap;

	// allocate GPU memory for FEM displacement, force, residual 
	for (int i = 0; i < 3; i++) {
		_gbuf.U[i] = (double*)gm.add_buf(_name + " U " + std::to_string(i), sizeof(double)* nv_gs); gbuf_size += sizeof(double) * nv_gs;
		_gbuf.F[i] = (double*)gm.add_buf(_name + " F " + std::to_string(i), sizeof(double)* nv_gs); gbuf_size += sizeof(double) * nv_gs;
		_gbuf.R[i] = (double*)gm.add_buf(_name + " R " + std::to_string(i), sizeof(double)* nv_gs); gbuf_size += sizeof(double) * nv_gs;
		//if (_layer == 0) {
		//	_gbuf.Uworst[i] = (double*)gm.add_buf(_name + " Uworst " + std::to_string(i), sizeof(double)* nv_gs); gbuf_size += sizeof(double) * nv_gs;
		//	_gbuf.Fworst[i] = (double*)gm.add_buf(_name + " Fworst " + std::to_string(i), sizeof(double)* nv_gs); gbuf_size += sizeof(double) * nv_gs;
		//}
	}

	// finest layer
	if (layer == 0) {
		_min_coeff = mincoeff;
		_max_coeff = maxcoeff;
		_isosurface_value = (_min_coeff + _max_coeff) / 2;

		_gbuf.rho_e = (float*)gm.add_buf(_name + "rho_e ", sizeof(float) * ne_gs); gbuf_size += sizeof(float) * ne_gs;		
		_gbuf.eActiveBits = (unsigned int*)gm.add_buf(_name + "eActiveBits", sizeof(unsigned int)*ebit._bitArray.size(), ebit._bitArray.data()); gbuf_size += sizeof(unsigned int) * ebit._bitArray.size();
		_gbuf.eActiveChunkSum = (int*)gm.add_buf(_name + "eActiveChunkSum", sizeof(int)*ebit._chunkSat.size(), ebit._chunkSat.data()); gbuf_size += sizeof(int) * ebit._chunkSat.size();
		_gbuf.nword_ebits = ebit._bitArray.size();

		_gbuf.coeffs = (float*)gm.add_buf(_name + " coeff ", sizeof(float) * n_im * n_in * n_il); gbuf_size += sizeof(float) * n_im * n_in * n_il;
		_gbuf.ss_sens = (float*)gm.add_buf(_name + " ss_sens ", sizeof(float) * n_im * n_in * n_il); gbuf_size += sizeof(float) * n_im * n_in * n_il;
		_gbuf.drip_sens = (float*)gm.add_buf(_name + " drip_sens ", sizeof(float) * n_im * n_in * n_il); gbuf_size += sizeof(float) * n_im * n_in * n_il;

		_gbuf.ss_value = (float*)gm.add_buf(_name + " ss_value ", sizeof(float) * _num_surface_points); gbuf_size += sizeof(float) * _num_surface_points;
		_gbuf.drip_value = (float*)gm.add_buf(_name + " drip_value ", sizeof(float) * _num_surface_points); gbuf_size += sizeof(float) * _num_surface_points;

		for (int i = 0; i < 3; i++)
		{
			_gbuf.KnotSer[i] = (float*)gm.add_buf(_name + "KnotSer " + std::to_string(i), sizeof(float) * spknotspan[i]); gbuf_size += sizeof(float) * spknotspan[i];
		}

		for (int i = 0; i < 3; i++)
		{
			_gbuf.surface_points[i] = (float*)gm.add_buf(_name + " SurfacePoints " + std::to_string(i), sizeof(int) * _num_surface_points); 
			gbuf_size += sizeof(int) * _num_surface_points;
		}

		/* may not use */
		for (int i = 0; i < 3; i++)
		{
			_gbuf.surface_normal[i] = (float*)gm.add_buf(_name + " SurfaceNormal " + std::to_string(i), sizeof(float) * _num_surface_points);
			_gbuf.surface_normal_dc[i] = (float*)gm.add_buf(_name + " SurfaceNormaldc " + std::to_string(i), sizeof(float) * _num_surface_points);
			gbuf_size += sizeof(float) * _num_surface_points;
			gbuf_size += sizeof(float) * _num_surface_points;
		} 
		_gbuf.surface_normal_norm_dc = (float*)gm.add_buf(_name + " SurfaceNormalnormdc ", sizeof(float) * _num_surface_points);
		gbuf_size += sizeof(float) * _num_surface_points;
		
		_gbuf.surface_normal_direction = (float*)gm.add_buf(_name + " SurfaceNormaldirection ", sizeof(float) * _num_surface_points);
		gbuf_size += sizeof(float) * _num_surface_points;
		_gbuf.surface_points_flag = (float*)gm.add_buf(_name + " SurfacePointsflag ", sizeof(float) * _num_surface_points);
		gbuf_size += sizeof(float) * _num_surface_points;
		_gbuf.surface_points_flag_virtual = (float*)gm.add_buf(_name + " SurfacePointsflagVirtual ", sizeof(float) * _num_surface_points);
		gbuf_size += sizeof(float) * _num_surface_points;

	}

	// allocate v2e topology buffer 
	if (layer == 0) {
		for (int i = 0; i < 8; i++) {
			_gbuf.v2e[i] = (int*)gm.add_buf(_name + " v2e " + std::to_string(i), sizeof(int) * nv_gs, v2ehost[i], sizeof(int) * nv); gbuf_size += sizeof(int) * nv_gs;
		}
	}

	// reorder v2vfine in current grid
	if (layer >= (_skiplayer ? 3 : 1)) {
		for (int i = 0; i < 27; i++) {
			if (fineGrid != nullptr) {
				_gbuf.v2vfine[i] = (int*)gm.add_buf(_name + " v2vfine " + std::to_string(i), sizeof(int) * nv_gs, v2vfinehost[i], sizeof(int) * nv); gbuf_size += sizeof(int) * nv_gs;
				lexico2gsorder_g(vidmap, nv, _gbuf.v2vfine[i], nv_gs, _gbuf.v2vfine[i], fineGrid->_gbuf.vidmap);
			}
		}
	}

	// reorder v2vcoarse in fine grid
	for (int i = 0; i < 8; i++) {
		if (fineGrid != nullptr) {
			lexico2gsorder_g(fineGrid->_gbuf.vidmap, fineGrid->n_vertices, fineGrid->_gbuf.v2vcoarse[i], fineGrid->n_gsvertices, fineGrid->_gbuf.v2vcoarse[i], vidmap);
		}
	}

	// allocate V2Vcoarse topology buffer on current grid
	for (int i = 0; i < 8; i++) {
		if (v2vcoarsehost[i] != nullptr) {
			_gbuf.v2vcoarse[i] = (int*)gm.add_buf(_name + " v2vcoarse " + std::to_string(i), sizeof(int) * nv_gs, v2vcoarsehost[i], sizeof(int) * nv); gbuf_size += sizeof(int) * nv_gs;
		}
	}

	// allocate V2V and reordering
	for (int i = 0; i < 27; i++) {
		_gbuf.v2v[i] = (int*)gm.add_buf(_name + " v2v " + std::to_string(i), sizeof(int) * nv_gs, v2vhost[i], sizeof(int) * nv);  gbuf_size += sizeof(int) * nv_gs;
		lexico2gsorder_g(vidmap, nv, _gbuf.v2v[i], nv_gs, _gbuf.v2v[i], vidmap);
	}

	// allocate V2Vfinecenter buffer and reordering
	if (_skiplayer && layer == 2) {
		for (int i = 0; i < 64; i++) {
			_gbuf.v2vfinecenter[i] = (int*)gm.add_buf(_name + " v2vfinecenter " + std::to_string(i), sizeof(int) * nv_gs, v2vfinec[i], sizeof(int) * nv); gbuf_size += sizeof(int) * nv_gs;
			lexico2gsorder_g(vidmap, nv, _gbuf.v2vfinecenter[i], nv_gs, _gbuf.v2vfinecenter[i], fineGrid->_gbuf.vidmap);
		}
	}

	// allocate sensitivity buffer on first grid
	if (_layer == 0) {
		_gbuf.g_sens = (float*)gm.add_buf(_name + " g_sens ", sizeof(float) * ne_gs); gbuf_size += sizeof(float) * ne_gs;
		_gbuf.c_sens = (float*)gm.add_buf(_name + " c_sens ", sizeof(float) * n_im * n_in * n_il); gbuf_size += sizeof(float) * n_im * n_in * n_il;
		//_gbuf.de2dc = (float*)gm.add_buf(_name + " de2dc ", sizeof(float) * ne_gs * n_im * n_in * n_il); gbuf_size += sizeof(float) * ne_gs * n_im * n_in * n_il;

	}

	// allocate bitflag buffer for vertex and element
	_gbuf.vBitflag = (int*)gm.add_buf(_name + " vbitflag ", sizeof(int) * nv_gs, vbitflags, sizeof(int) * nv); gbuf_size += sizeof(int) * nv_gs;
	_gbuf.eBitflag = (int*)gm.add_buf(_name + " ebitflag ", sizeof(int) * ne_gs, ebitflags, sizeof(int) * ne); gbuf_size += sizeof(int) * ne_gs;


	// find surface load nodes 
	if (layer == 0) {
		mark_surface_nodes_g(n_vertices, _gbuf.v2e, _gbuf.vBitflag);
		getVflags(n_vertices, vbitflags);
		computeProjectionMatrix(nv, nv_gs, vreso, vlexi2gs, vidmap, vbit, vbitflags, _gbuf.vBitflag);
		_gsLoadNodes = getLoadNodes();
		array2ConnectedMatlab("loadnodes", _gsLoadNodes.data(), _gsLoadNodes.size());
		//lexico2gsorder(nullptr, _gsLoadNodes.size(), _gsLoadNodes.data(), _gsLoadNodes.size(), _gsLoadNodes.data(), vidmap);
		
		// add force support buf
		for (int i = 0; i < 3; i++) _gbuf.Fsupport[i] = (double*)gm.add_buf(_name + " fsupport " + std::to_string(i), sizeof(double) * n_loadnodes());
	}

	// reorder v2e list
	if (layer == 0) {
		for (int i = 0; i < 8; i++) lexico2gsorder_g(vidmap, nv, _gbuf.v2e[i], nv_gs, _gbuf.v2e[i], eidmap);
	}

	// reorder bit flags
	lexico2gsorder_g(vidmap, nv, _gbuf.vBitflag, nv_gs, _gbuf.vBitflag);
	lexico2gsorder_g(eidmap, ne, _gbuf.eBitflag, ne_gs, _gbuf.eBitflag);

	if (_layer != 0) {
		_gbuf.rxStencil = (double*)gm.add_buf(_name + " rxStencil ", sizeof(double) * nv_gs * 27 * 9); gbuf_size += sizeof(double) * nv_gs * 27 * 9;
		gpu_manager_t::initMem(_gbuf.rxStencil, sizeof(double) * nv_gs * 27 * 9);
	}

	printf("-- Allocate %d MB buffer\n", gbuf_size / 1024 / 1024);

	return nv_gs;
}

void grid::Grid::readForce(std::string forcefile)
{
	std::vector<double> fhost;
	bool suc = bio::read_vector(forcefile, fhost);
	if (!suc) {
		printf("\033[31mFailed to open file %s \n\033[0m", forcefile.c_str());
		throw std::runtime_error("error open file");
	}
	if (fhost.size() != n_gsvertices * 3) {
		printf("\033[31mForce Size does not match\033[0m\n");
		printf("\033[31mForce Size Load: %i, Force size: %i\033[0m\n", fhost.size() / 3, n_gsvertices);
		throw std::runtime_error("invalid size");
	}
	std::vector<double> f[3];
	for (int i = 0; i < fhost.size(); i += 3) {
		f[0].emplace_back(fhost[i]);
		f[1].emplace_back(fhost[i + 1]);
		f[2].emplace_back(fhost[i + 2]);
	}
	
	for (int i = 0; i < 3; i++) {
		gpu_manager_t::upload_buf(_gbuf.F[i], f[i].data(), sizeof(double) * n_gsvertices);
	}
#ifdef ENABLE_MATLAB
	std::vector<double> fhos[3];
	Eigen::Matrix<double, -1, -1> f2fhost(n_gsvertices, 3);
	for (int i = 0; i < 3; i++) {
		fhos[i].resize(n_gsvertices);
		gpu_manager_t::download_buf(fhos[i].data(), _gbuf.F[i], sizeof(double) * n_gsvertices);
	}

	int f_total = 0;
	for (int i = 0; i < n_gsvertices; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			f2fhost(i, j) = fhos[j][i];
			f_total += fhos[j][i];
		}
	}
	eigen2ConnectedMatlab("f", f2fhost);
#endif
}

void grid::Grid::readSupportForce(std::string fsfile)
{
	std::vector<double> fhost;
	bool suc = bio::read_vector(fsfile, fhost);
	if (!suc) {
		printf("\033[31mFailed to open file %s \n\033[0m", fsfile.c_str());
		throw std::runtime_error("error open file");
	}
	if (fhost.size() != n_loadnodes() * 3) {
		printf("\033[31mForce Size does not match\033[0m\n");
		printf("\033[31mForce Size Load: %i, Force size: %i\033[0m\n", fhost.size()/3, n_loadnodes());
		throw std::runtime_error("invalid size");
	}
	std::vector<double> f[3];
	for (int i = 0; i < fhost.size(); i += 3) {
		f[0].emplace_back(fhost[i]);
		f[1].emplace_back(fhost[i + 1]);
		f[2].emplace_back(fhost[i + 2]);
	}

	double* pload[3] = { f[0].data(),f[1].data(),f[2].data() };
	uploadLoadForce(pload);
	setForceSupport(getPreloadForce(), getForce());

#ifdef ENABLE_MATLAB
	std::vector<double> fhos[3];
	Eigen::Matrix<double, -1, -1> f2fhost(n_loadnodes(), 3);
	for (int i = 0; i < 3; i++) {
		fhos[i].resize(n_loadnodes());
		gpu_manager_t::download_buf(fhos[i].data(), _gbuf.F[i], sizeof(double) * n_loadnodes());
	}

	int fs_total = 0;
	for (int i = 0; i < n_loadnodes(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			f2fhost(i, j) = fhos[j][i];
			fs_total += fhos[j][i];
		}
	}
	eigen2ConnectedMatlab("fs", f2fhost);
#endif

}

void grid::Grid::set_spline_knot_series(void)
{
	int i, j, k;
	float tmp;
	float scale_bound = 2.0f;
	int sorder = n_order;
	int spartx = n_partitionx;
	int sparty = n_partitiony;
	int spartz = n_partitionz;

	for (int i = 0; i < 3; i++)
	{
		m_3sBoundMin[i] = _box[0][i];
		m_3sBoundMax[i] = _box[1][i];
	}

	// enlarge the bound
	float deltax = m_3sBoundMax[0] - m_3sBoundMin[0];	
	m_3sBoundMax[0] += scale_bound * deltax / (float)(spartx + 1);
	m_3sBoundMin[0] -= scale_bound * deltax / (float)(spartx + 1);
	float deltay = m_3sBoundMax[1] - m_3sBoundMin[1];
	m_3sBoundMax[1] += scale_bound * deltay / (float)(sparty + 1);
	m_3sBoundMin[1] -= scale_bound * deltay / (float)(sparty + 1);
	float deltaz = m_3sBoundMax[2] - m_3sBoundMin[2];
	m_3sBoundMax[2] += scale_bound * deltaz / (float)(spartz + 1);
	m_3sBoundMin[2] -= scale_bound * deltaz / (float)(spartz + 1);

	// calculate the step
	m_sStepX = (m_3sBoundMax[0] - m_3sBoundMin[0]) / (float)(spartx + 1);
	m_sStepY = (m_3sBoundMax[1] - m_3sBoundMin[1]) / (float)(sparty + 1);
	m_sStepZ = (m_3sBoundMax[2] - m_3sBoundMin[2]) / (float)(spartz + 1);

	m_sStep[0] = m_sStepX;
	m_sStep[1] = m_sStepY;
	m_sStep[2] = m_sStepZ;

	{
		// X knot series
		tmp = m_3sBoundMin[0];
		for (i = 0; i < sorder; i++)
		{
			KnotSer[0].push_back(m_3sBoundMin[0]);
		}
		for (i = sorder; i < sorder + spartx; i++)
		{
			tmp += m_sStepX;
			KnotSer[0].push_back(tmp);
		}
		for (i = sorder + spartx; i < 2 * sorder + spartx; i++)
		{
			KnotSer[0].push_back(m_3sBoundMax[0]);
		}

		// Y knot series 
		tmp = m_3sBoundMin[1];
		for (i = 0; i < sorder; i++)
		{
			KnotSer[1].push_back(m_3sBoundMin[1]);
		}
		for (i = sorder; i < sorder + sparty; i++)
		{
			tmp += m_sStepY;
			KnotSer[1].push_back(tmp);
		}
		for (i = sorder + sparty; i < 2 * sorder + sparty; i++)
		{
			KnotSer[1].push_back(m_3sBoundMax[1]);
		}

		// Z knot series
		tmp = m_3sBoundMin[2];
		for (i = 0; i < sorder; i++)
		{
			KnotSer[2].push_back(m_3sBoundMin[2]);
		}
		for (i = sorder; i < sorder + spartz; i++)
		{
			tmp += m_sStepZ;
			KnotSer[2].push_back(tmp);
		}
		for (i = sorder + spartz; i < 2 * sorder + spartz; i++)
		{
			KnotSer[2].push_back(m_3sBoundMax[2]);
		}
	}

	//std::cout << "Step: " << m_sStep[0] << ", " << m_sStep[1] << ", " << m_sStep[2] << std::endl;
	//std::cout << "Boundmin: " << m_3sBoundMin[0] << ", " << m_3sBoundMin[1] << ", " << m_3sBoundMin[2] << std::endl;
	//std::cout << "Boundmax: " << m_3sBoundMax[0] << ", " << m_3sBoundMax[1] << ", " << m_3sBoundMax[2] << std::endl;

	// upload to device
	for (int i = 0; i < 3; i++)
	{
		gpu_manager_t::upload_buf(_gbuf.KnotSer[i], KnotSer[i].data(), sizeof(float) * spknotspan[i]);
	}

#if 0
#ifdef ENABLE_MATLAB
	// download to host
	std::vector<float> KnotSer_host[3];

	for (int i = 0; i < 3; i++)
	{
		KnotSer_host[i].resize(spknotspan[i]);
		gpu_manager_t::download_buf(KnotSer_host[i].data(), _gbuf.KnotSer[i], sizeof(float) * spknotspan[i]);
	}
	Eigen::VectorXf KnotSer_x;
	KnotSer_x.resize(spknotspan[0]);
	std::copy(KnotSer_host[0].begin(), KnotSer_host[0].end(), KnotSer_x.data());
	eigen2ConnectedMatlab("Knotspanx", KnotSer_x);

	Eigen::VectorXf KnotSer_y;
	KnotSer_y.resize(spknotspan[1]);
	std::copy(KnotSer_host[1].begin(), KnotSer_host[1].end(), KnotSer_y.data());
	eigen2ConnectedMatlab("Knotspany", KnotSer_y);

	Eigen::Matrix<float, -1, 1> KnotSer_z(spknotspan[2], 1);
	for (int i = 0; i < spknotspan[2]; i++)
	{
		KnotSer_z(i, 1) = KnotSer_host[2][i];
	}
	eigen2ConnectedMatlab("Knotspanz", KnotSer_z);
#endif
#endif
}

void grid::Grid::uploadSurfacePoints(void)
{
	// upload to device
	for (int i = 0; i < 3; i++)
	{
		gpu_manager_t::upload_buf(_gbuf.surface_points[i], spline_surface_node[i].data(), sizeof(float) * _num_surface_points);
	}

#if 1
#ifdef ENABLE_MATLAB
	std::vector<float> surface_points_host[3];
	for (int i = 0; i < 3; i++)
	{
		surface_points_host[i].resize(_num_surface_points);
		gpu_manager_t::download_buf(surface_points_host[i].data(), _gbuf.surface_points[i], sizeof(float) * _num_surface_points);
	}
	Eigen::VectorXf surface_points_x(_num_surface_points);
	Eigen::VectorXf surface_points_y(_num_surface_points);
	Eigen::VectorXf surface_points_z(_num_surface_points);
	std::copy(surface_points_host[0].begin(), surface_points_host[0].end(), surface_points_x.data());
	std::copy(surface_points_host[1].begin(), surface_points_host[1].end(), surface_points_y.data());
	std::copy(surface_points_host[2].begin(), surface_points_host[2].end(), surface_points_z.data());
	eigen2ConnectedMatlab("gpu_surf_x", surface_points_x);
	eigen2ConnectedMatlab("gpu_surf_y", surface_points_y);
	eigen2ConnectedMatlab("gpu_surf_z", surface_points_z);
#endif
#endif
}

void grid::Grid::readDisplacement(std::string displacementfile)
{
	std::vector<double> uhost;
	bool suc = bio::read_vector(displacementfile, uhost);
	if (!suc) {
		printf("\033[31mFailed to open file %s \n\033[0m", displacementfile.c_str());
		throw std::runtime_error("error open file");
	}

	if (uhost.size() != n_gsvertices * 3) {
		printf("\033[31mDisplacement Size does not match\033[0m\n");
		throw std::runtime_error("invalid size");
	}

	std::vector<double> u[3];
	for (int i = 0; i < uhost.size(); i += 3) {
		u[0].emplace_back(uhost[i]);
		u[1].emplace_back(uhost[i + 1]);
		u[2].emplace_back(uhost[i + 2]);
	}

	for (int i = 0; i < 3; i++) {
		gpu_manager_t::upload_buf(_gbuf.U[i], u[i].data(), sizeof(double) * n_gsvertices);
	}
}

void grid::Grid::pertubDisplacement(double ratio)
{
	v3_pertub(getDisplacement(), ratio);
}

bool grid::Grid::checkV2Vhost(int nv, int ne, int* v2e[8], int* v2v[27])
{
	for (int i = 0; i < nv; i++) {
		for (int k = 0; k < 8; k++) {
			int eid = v2e[k][i];
			if (eid == -1) continue;
			if (eid == 3259449) {
				printf("-- e: v[%d](i)\n", 7 - k, i);
			}
			for (int j = 0; j < 8; j++) {
				int vjpos[3] = {
					k % 2 + j % 2,
					k / 2 % 2 + j / 2 % 2,
					k / 4 + j / 4
				};
				int vjlid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
				if (v2v[vjlid][i] == -1) {
					printf("-- v[%d] e[%d](%d) v[%d]\n", i, k, eid, j);
				}
			}
		}
	}

	std::set<int> eset;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < nv; j++) {
			int eid = v2e[i][j];
			if (eid != -1) eset.insert(eid);
		}
	}
	printf("-- ne(v2e) %d ; ne(solid) %d\n", eset.size(), ne);

	return false;
}

grid::hostbufbackup_t<double, 3> grid::Grid::v3_backup(double* vdata[3])
{
	return hostbufbackup_t(vdata, n_gsvertices);
}

bool grid::Grid::v3_hasNaN(double* v[3])
{
	std::vector<double> hostv[3];
	for (int i = 0; i < 3; i++) {
		hostv[i].resize(n_gsvertices);
		gpu_manager_t::download_buf(hostv[i].data(), v[i], sizeof(double)*n_gsvertices);
	}
	bool hasNaN = false;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < n_gsvertices; j++) {
			if (isnan(hostv[i][j])) {
				hasNaN = true;
				break;
			}
		}
		if (hasNaN) break;
	}
	return hasNaN;
}

void Grid::elexibuf2matlab(const std::string& nam, float* p_gsbuf)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<float, -1, 1> hostgs(n_elements, 1);
	float* plexi = getlexiEbuf(p_gsbuf);
	gpu_manager_t::download_buf(hostgs.data(), plexi, sizeof(float)*n_elements);
	eigen2ConnectedMatlab(nam, hostgs);
#endif
}

void Grid::vlexibuf2matlab(const std::string& nam, double* p_gsbuf)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<double, -1, 1> hostgs(n_vertices, 1);
	double* plexi = getlexiVbuf(p_gsbuf);
	gpu_manager_t::download_buf(hostgs.data(), plexi, sizeof(double) * n_vertices);
	eigen2ConnectedMatlab(nam, hostgs);
#endif
}


void Grid::sens2matlab(const std::string& nam)
{
	gpu_manager_t::pass_dev_buf_to_matlab(nam.c_str(), _gbuf.g_sens, n_rho());
}

void Grid::csens2matlab(const std::string& nam)
{
	gpu_manager_t::pass_dev_buf_to_matlab(nam.c_str(), _gbuf.c_sens, n_cijk());
}

void Grid::v2vcoarse2matlab(const std::string& nam)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<int, -1, -1> v2vcoarsehost(n_gsvertices, 8);
	for (int i = 0; i < 8; i++) {
		gpu_manager_t::download_buf(v2vcoarsehost.col(i).data(), _gbuf.v2vcoarse[i], sizeof(int) * n_gsvertices);
	}
	eigen2ConnectedMatlab(nam, v2vcoarsehost);
#endif
}

void Grid::v2v2matlab(const std::string& nam)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<int, -1, -1> v2vhost(n_gsvertices, 27);
	for (int i = 0; i < 27; i++) {
		gpu_manager_t::download_buf(v2vhost.col(i).data(), _gbuf.v2v[i], sizeof(int)*n_gsvertices);
	}
	eigen2ConnectedMatlab(nam, v2vhost);
#endif
}

void Grid::vidmap2matlab(const std::string & nam)
{
#ifdef ENABLE_MATLAB
	std::vector<int> vidmaphost(n_vertices);
	gpu_manager_t::download_buf(vidmaphost.data(), _gbuf.vidmap, sizeof(int)*vidmaphost.size());
	array2ConnectedMatlab(nam, vidmaphost.data(), vidmaphost.size());
#endif
}

void Grid::eidmap2matlab(const std::string & nam)
{
#ifdef ENABLE_MATLAB
	std::vector<int> eidmaphost(n_elements);
	gpu_manager_t::download_buf(eidmaphost.data(), _gbuf.eidmap, sizeof(int)*eidmaphost.size());
	array2ConnectedMatlab(nam, eidmaphost.data(), eidmaphost.size());
#endif
}

void Grid::buildCoarsestSystem(void)
{
	std::vector<double> rxdata(n_gsvertices * 27 * 9);
	gpu_manager_t::download_buf(rxdata.data(), _gbuf.rxStencil, sizeof(double) * n_gsvertices * 27 * 9);

	std::vector<Eigen::Triplet<double>> triplist;

	vlastrowid.resize(n_gsvertices);
	int rowid = 0;
	for (int i = 0; i < n_gsvertices; i++) {
		if (_v2v[13][i] != -1) {
			vlastrowid[i] = rowid;
			rowid++;
		}
		else {
			vlastrowid[i] = -1;
		}
	}

	nvlastrows = rowid;

	Klast.resize(rowid * 3, rowid * 3);
	fullK.resize(rowid * 3, rowid * 3);
	fullK.fill(0);

	for (int i = 0; i < rxdata.size(); i++) {
		double rxvalue = rxdata[i];
		int vid = i % n_gsvertices;
		// stencil is stored in row major order
		int ke_id = i / n_gsvertices % 9;
		int krow = ke_id / 3;
		int kcol = ke_id % 3;
		int nei = i / (n_gsvertices * 9);
		int nid = _v2v[nei][vid];
		if (nid == -1) continue;
		triplist.emplace_back(vlastrowid[vid] * 3 + krow, vlastrowid[nid] * 3 + kcol, rxvalue);
		fullK(vlastrowid[vid] * 3 + krow, vlastrowid[nid] * 3 + kcol) = rxvalue;
	}

	Klast.setFromTriplets(triplist.begin(), triplist.end());

	//fullK = Klast;
	//svd.setThreshold(1e-11);
	svd.compute(fullK, Eigen::ComputeFullU | Eigen::ComputeFullV);
	int lossrank = fullK.rows() - svd.rank();
	printf("-- degenerate rank = %d\n", lossrank);
	if (lossrank > 0) {
		Klastkernel = svd.matrixV().block(0, fullK.cols() - lossrank, fullK.rows(), lossrank);
	}
	else {
		Klastkernel = Eigen::VectorXd::Zero(fullK.rows(), 1);
	}

	eigen2ConnectedMatlab("Klast", fullK);
	eigen2ConnectedMatlab("Klastker", Klastkernel);
}

void Grid::stencil2matlab(const std::string& nam)
{
	Eigen::Matrix<double, -1, -1> stencilarray;
	stencilarray.resize(n_gsvertices, 27 * 9);
	gpu_manager_t::download_buf(stencilarray.data(), _gbuf.rxStencil, sizeof(double) * 27 * 9 * n_gsvertices);
	eigen2ConnectedMatlab(nam, stencilarray);
}

void Grid::force2matlab(const std::string& nam) { v3_toMatlab(nam, _gbuf.F); }

void Grid::displacement2matlab(const std::string& nam) { v3_toMatlab(nam, _gbuf.U); }

void Grid::residual2matlab(const std::string& nam) { v3_toMatlab(nam, _gbuf.R); }

void Grid::v3_toMatlab(const std::string& nam, double* v[3])
{
#ifdef ENABLE_MATLAB
	std::vector<double> vhost[3];
	for (int i = 0; i < 3; i++) {
		vhost[i].resize(n_gsvertices);
		gpu_manager_t::download_buf(vhost[i].data(), v[i], sizeof(double) * n_gsvertices);
	}

	Eigen::Matrix<double, 3, -1> vmat;
	vmat.resize(3, n_gsvertices);
	for (int i = 0; i < n_gsvertices; i++) {
		for (int j = 0; j < 3; j++) {
			vmat(j, i) = vhost[j][i];
		}
	}

	eigen2ConnectedMatlab(nam, vmat);
#endif
}

void::Grid::pass_spline_surf_node2matlab(void)
{
	Eigen::Matrix<float, -1, 1> node_x;
	Eigen::Matrix<float, -1, 1> node_y;
	Eigen::Matrix<float, -1, 1> node_z;
	node_x.resize(spline_surface_node->size(), 1);
	node_y.resize(spline_surface_node->size(), 1);
	node_z.resize(spline_surface_node->size(), 1);
	std::copy(spline_surface_node[0].begin(), spline_surface_node[0].end(), node_x.begin());
	std::copy(spline_surface_node[1].begin(), spline_surface_node[1].end(), node_y.begin());
	std::copy(spline_surface_node[2].begin(), spline_surface_node[2].end(), node_z.begin());
	eigen2ConnectedMatlab("mc_spline_node_x", node_x);
	eigen2ConnectedMatlab("mc_spline_node_y", node_y);
	eigen2ConnectedMatlab("mc_spline_node_z", node_z);
}

void Grid::generate_surface_nodes_by_MC(const std::string& fileName, int Nodes[3], std::vector<float>& surface_node_x, std::vector<float>& surface_node_y, std::vector<float>& surface_node_z, std::vector<float> bg_node[3], std::vector<float> mcPoints_in)
{
	MCImplicitRender* mc_render;
	TriMesh3D* ptr_mesh_;
	ptr_mesh_ = new TriMesh3D;
	std::string filename = fileName;
	//std::cout << filename << std::endl;
	std::string key_file;
	ptr_mesh_->LoadFile(filename.data());
	//mc_render = new MCImplicitRender(0, 1e-6, Nodes, ptr_mesh_, key_file);
	mc_render = new MCImplicitRender();

	float boxOrigin[3] = { _box[0][0],_box[0][1],_box[0][2] };
	float boxEnd[3] = { _box[1][0],_box[1][1],_box[1][2] };

	mc_render->AddSetting(0, 1e-6, Nodes, ptr_mesh_, boxOrigin, boxEnd);
	//mc_render = new MCImplicitRender();
	//mc_render->DoMC();
	//mc_render->InitData();
	float minValue = 0.f;
	//std::vector<mp4vector> mcPoints;

	mc_render->RunMarchingCubesTestPotential(minValue, bg_node, mcPoints_in);
	mc_render->save_to_surface_node(surface_node_x, surface_node_y, surface_node_z);
}

std::vector<int> generateEquidistantIntegers(int range, int num_sample) {
	std::vector<int> points;

	if (num_sample >= range) {
		std::cerr << "Error: Number of points to generate (m) must be less than N." << std::endl;
		return points;
	}

	double step = static_cast<double>(range) / (num_sample - 1);

	for (int i = 0; i < num_sample; ++i) {
		int point = static_cast<int>(i * step);
		points.push_back(point);
	}

	return points;
}

void Grid::compute_surface_nodes_in_model(std::vector<float>& surface_node_x, std::vector<float>& surface_node_y, std::vector<float>& surface_node_z)
{
	//#ifdef  ENABLE_MATLAB
	//	Eigen::Matrix<float, -1, 1> surf_x3, surf_y3, surf_z3;
	//	surf_x3.resize(surface_node_x.size());
	//	surf_y3.resize(surface_node_y.size());
	//	surf_z3.resize(surface_node_z.size());
	//	memcpy(surf_x3.data(), surface_node_x.data(), sizeof(float) * surface_node_x.size());
	//	memcpy(surf_y3.data(), surface_node_y.data(), sizeof(float) * surface_node_y.size());
	//	memcpy(surf_z3.data(), surface_node_z.data(), sizeof(float) * surface_node_z.size());
	//	eigen2ConnectedMatlab("surf_x3", surf_x3);
	//	eigen2ConnectedMatlab("surf_y3", surf_y3);
	//	eigen2ConnectedMatlab("surf_z3", surf_z3);
	//#endif

	std::vector<float> tmp_x;
	std::vector<float> tmp_y;
	std::vector<float> tmp_z;
	std::vector<float> vert_final[3];
	std::vector<float> reduced_surf_nodes[3];
	const unsigned int num = surface_node_x.size();
	const unsigned int num_surface_points = _num_surface_points;
	float smallmove = 1e-3f;
	std::vector<Point> init_surface_nodes;
	std::vector<Point> closest_points_set;
	init_surface_nodes.reserve(num);
	closest_points_set.reserve(num);
	for (int i = 0; i < 3; i++)
	{
		reduced_surf_nodes[i].reserve(num_surface_points * 3);
	}

	int count_interior = 0;

	if (num / num_surface_points > 2)
	{
#if 0  // random picking
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, num - 1);
		std::set<int> generated_numbers;

		while (generated_numbers.size() < num_surface_points * 3) {
			int random_number = dis(gen);
			if (generated_numbers.find(random_number) == generated_numbers.end()) {
				generated_numbers.insert(random_number);
				reduced_surf_nodes[0].push_back(surface_node_x[random_number]);
				reduced_surf_nodes[1].push_back(surface_node_y[random_number]);
				reduced_surf_nodes[2].push_back(surface_node_z[random_number]);
			}
		}
#else  // equal distance picking
		std::vector<int> equidistantIntegers = generateEquidistantIntegers(num, num_surface_points * 3);
		for (int j = 0; j < num_surface_points * 3; j++)
		{
			// TODO[UPDATE COPY]
			reduced_surf_nodes[0].push_back(surface_node_x[equidistantIntegers[j]]);
			reduced_surf_nodes[1].push_back(surface_node_y[equidistantIntegers[j]]);
			reduced_surf_nodes[2].push_back(surface_node_z[equidistantIntegers[j]]);
		}
		Eigen::Matrix<int, -1, 1> random_list;
		random_list.resize(num_surface_points * 3);
		memcpy(random_list.data(), equidistantIntegers.data(), sizeof(int) * num_surface_points * 3);
		eigen2ConnectedMatlab("step_list1", random_list);
#endif
	}
	else
	{
		for (int j = 0; j < surface_node_x.size(); j++)
		{
			// TODO[UPDATE COPY]
			reduced_surf_nodes[0].push_back(surface_node_x[j]);
			reduced_surf_nodes[1].push_back(surface_node_y[j]);
			reduced_surf_nodes[2].push_back(surface_node_z[j]);
		}
	}
#ifdef  ENABLE_MATLAB
	Eigen::Matrix<float, -1, 1> surf_x1, surf_y1, surf_z1;
	surf_x1.resize(reduced_surf_nodes[0].size());
	surf_y1.resize(reduced_surf_nodes[0].size());
	surf_z1.resize(reduced_surf_nodes[0].size());
	memcpy(surf_x1.data(), reduced_surf_nodes[0].data(), sizeof(float) * reduced_surf_nodes[0].size());
	memcpy(surf_y1.data(), reduced_surf_nodes[1].data(), sizeof(float) * reduced_surf_nodes[1].size());
	memcpy(surf_z1.data(), reduced_surf_nodes[2].data(), sizeof(float) * reduced_surf_nodes[2].size());
	eigen2ConnectedMatlab("surf_x1", surf_x1);
	eigen2ConnectedMatlab("surf_y1", surf_y1);
	eigen2ConnectedMatlab("surf_z1", surf_z1);
#endif

#pragma omp parallel for
	for (int i = 0; i < reduced_surf_nodes->size(); i++)
	{
		Point current_point(reduced_surf_nodes[0][i], reduced_surf_nodes[1][i], reduced_surf_nodes[2][i]);
		init_surface_nodes.push_back(current_point);

		Point current_closest = aabb_tree.closest_point(current_point);
		closest_points_set.push_back(current_closest);

		Point move_direction(current_point[0] - current_closest[0], current_point[1] - current_closest[1], current_point[2] - current_closest[2]);
		Point move_length(smallmove * move_direction[0], smallmove * move_direction[1], smallmove * move_direction[2]);
		Point near_closest(current_closest[0] + move_length[0], current_closest[1] + move_length[1], current_closest[2] * move_length[2]);

		Ray ray1(near_closest, current_point);  // point out of the model
		Ray ray2(current_point, near_closest);  // point in the model

#pragma omp critical
		if (aabb_tree.do_intersect(ray1))
		{
			tmp_x.push_back(reduced_surf_nodes[0][i]);
			tmp_y.push_back(reduced_surf_nodes[1][i]);
			tmp_z.push_back(reduced_surf_nodes[2][i]);
			count_interior++;
		}
	}

	std::cout << "--[TEST] surface points in model: " << count_interior << std::endl;

#ifdef  ENABLE_MATLAB
	Eigen::Matrix<float, -1, 1> surf_x2, surf_y2, surf_z2;
	surf_x2.resize(tmp_x.size());
	surf_y2.resize(tmp_y.size());
	surf_z2.resize(tmp_z.size());
	memcpy(surf_x2.data(), tmp_x.data(), sizeof(float) * tmp_x.size());
	memcpy(surf_y2.data(), tmp_y.data(), sizeof(float) * tmp_y.size());
	memcpy(surf_z2.data(), tmp_z.data(), sizeof(float) * tmp_z.size());
	eigen2ConnectedMatlab("surf_x2", surf_x2);
	eigen2ConnectedMatlab("surf_y2", surf_y2);
	eigen2ConnectedMatlab("surf_z2", surf_z2);
#endif

	if (tmp_x.size() == 0)                   // no marching cube point
	{
		for (int index = 0; index < num_surface_points; ++index)
		{
			vert_final[0].push_back(0);
			vert_final[1].push_back(0);
			vert_final[2].push_back(0);
		}
	}
	else if (tmp_x.size() < num_surface_points)  // marching cube points less than sample points
	{
		for (int index = 0; index < tmp_x.size(); ++index)
		{
			vert_final[0].push_back(tmp_x[index]);
			vert_final[1].push_back(tmp_y[index]);
			vert_final[2].push_back(tmp_z[index]);
		}
		for (int index = tmp_x.size(); index < num_surface_points; ++index)
		{
			vert_final[0].push_back(tmp_x[0]);
			vert_final[1].push_back(tmp_y[0]);
			vert_final[2].push_back(tmp_z[0]);
		}
	}
	else
	{
#if 0
		for (int j = 0; j < num_surface_points; ++j)
		{
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> dis(0, tmp_x.size() - 1);
			std::set<int> generated_numbers;

			while (generated_numbers.size() < num_surface_points) {
				int random_number = dis(gen);
				if (generated_numbers.find(random_number) == generated_numbers.end()) {
					generated_numbers.insert(random_number);
					vert_final[0].push_back(tmp_x[random_number]);
					vert_final[1].push_back(tmp_y[random_number]);
					vert_final[2].push_back(tmp_z[random_number]);
				}
			}
		}
	}
#else  // equal distance picking
		std::vector<int> equidistantIntegers = generateEquidistantIntegers(tmp_x.size(), num_surface_points);
		for (int j = 0; j < num_surface_points; j++)
		{
			// TODO[UPDATE COPY]
			vert_final[0].push_back(tmp_x[equidistantIntegers[j]]);
			vert_final[1].push_back(tmp_y[equidistantIntegers[j]]);
			vert_final[2].push_back(tmp_z[equidistantIntegers[j]]);
		}
		Eigen::Matrix<int, -1, 1> random_list;
		random_list.resize(num_surface_points);
		memcpy(random_list.data(), equidistantIntegers.data(), sizeof(int) * num_surface_points);
		eigen2ConnectedMatlab("step_list2", random_list);
#endif
		surface_node_x = vert_final[0];
		surface_node_y = vert_final[1];
		surface_node_z = vert_final[2];

#ifdef  ENABLE_MATLAB
		Eigen::Matrix<float, -1, 1> surf_x, surf_y, surf_z;
		surf_x.resize(surface_node_x.size());
		surf_y.resize(surface_node_y.size());
		surf_z.resize(surface_node_z.size());
		memcpy(surf_x.data(), surface_node_x.data(), sizeof(float) * surface_node_x.size());
		memcpy(surf_y.data(), surface_node_y.data(), sizeof(float) * surface_node_y.size());
		memcpy(surf_z.data(), surface_node_z.data(), sizeof(float) * surface_node_z.size());
		eigen2ConnectedMatlab("surf_x", surf_x);
		eigen2ConnectedMatlab("surf_y", surf_y);
		eigen2ConnectedMatlab("surf_z", surf_z);
#endif

		std::vector<float>().swap(tmp_x);
		std::vector<float>().swap(tmp_y);
		std::vector<float>().swap(tmp_z);
		for (int i = 0; i < 3; i++)
		{
			std::vector<float>().swap(vert_final[i]);
		}
	}
}

void Grid::generate_spline_surface_nodes(void)
{
	std::vector<float> mcPoints_val;
	std::vector<float> bgnodex;
	std::vector<float> bgnodey;
	std::vector<float> bgnodez;
	int cur_ereso = _ereso / 2;
	compute_background_mcPoints_value(bgnodex, bgnodey, bgnodez, mcPoints_val, cur_ereso);

	std::vector<float> bgnode[3];
	bgnode[0].insert(bgnode[0].end(), bgnodex.begin(), bgnodex.end());
	bgnode[1].insert(bgnode[1].end(), bgnodey.begin(), bgnodey.end());
	bgnode[2].insert(bgnode[2].end(), bgnodez.begin(), bgnodez.end());
	int ereso3 = cur_ereso * cur_ereso * cur_ereso;
#ifdef ENABLE_MATLAB
	Eigen::Matrix<float, -1, 1> node_value;
	node_value.resize(ereso3, 1);
	std::copy(mcPoints_val.begin(), mcPoints_val.end(), node_value.begin());
	eigen2ConnectedMatlab("rhomc1", node_value);
#endif

	int N[3] = { cur_ereso, cur_ereso, cur_ereso };
	// MARK[TOCHECK] sometime crash in mc points generation
	generate_surface_nodes_by_MC(_meshfile, N, spline_surface_node[0], spline_surface_node[1], spline_surface_node[2], bgnode, mcPoints_val);
	std::cout << "spline surface node number (after marching cube): " << spline_surface_node->size() << std::endl;
	pass_spline_surf_node2matlab();

	compute_surface_nodes_in_model(spline_surface_node[0], spline_surface_node[1], spline_surface_node[2]);
	std::cout << "spline surface node number (in model): " << spline_surface_node->size() << std::endl;
}


double Grid::compliance(void)
{
	return v3_dot(_gbuf.F, _gbuf.U);
}

void Grid::solve_fem_host(void)
{
	static Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solverhost;
	//auto& solverhost = svd;
	//static Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solverhost;
	static Eigen::Matrix<double, -1, 1> fhost;
	static std::vector<double> v3host[3];
	static Eigen::Matrix<double, -1, 1> uhost;

	int nrow = nvlastrows;
	// copy data from device to host
	for (int i = 0; i < 3; i++) {
		v3host[i].resize(n_gsvertices);
		gpu_manager_t::download_buf(v3host[i].data(), _gbuf.F[i], sizeof(double) * n_gsvertices);
	}
	fhost.resize(nrow * 3, 1);
	for (int i = 0; i < v3host->size(); i++) {
		if (vlastrowid[i] == -1) continue;
		fhost[vlastrowid[i] * 3] = v3host[0][i];
		fhost[vlastrowid[i] * 3 + 1] = v3host[1][i];
		fhost[vlastrowid[i] * 3 + 2] = v3host[2][i];
	}

	// solver compute matrix
	//solverhost.compute(fullK);
	//solverhost.compute(Klast);

	//// remove degenerate eigenvectors
	//fhost = fhost - Klastkernel * (Klastkernel.transpose() * fhost);

	//// solve
	//uhost = solverhost.solve(fhost);
	uhost = svd.solve(fhost);

	// remove degenerate eigenvectors
	//uhost = uhost - Klastkernel * (Klastkernel.transpose() * uhost);

	// DEBUG
	eigen2ConnectedMatlab("uhost", uhost);
	eigen2ConnectedMatlab("fhost", fhost);
	//printf("-- coarse system error %lf\n", (fullK*uhost - fhost).norm());

	// if preffered solver failed, try alternative solver
	if (solverhost.info() != Eigen::Success) {
		printf("-- \033[31mHost solver failed \033[0m\n");
		uhost.fill(0);
	}

	// pass solved displacement back to device
	for (int j = 0; j < n_gsvertices; j++) {
		int rowid = vlastrowid[j];
		for (int i = 0; i < 3; i++) {
			if (rowid == -1)
				v3host[i][j] = 0;
			else
				v3host[i][j] = uhost[rowid * 3 + i];
		}
	}

	for (int i = 0; i < 3; i++) {
		gpu_manager_t::upload_buf(_gbuf.U[i], v3host[i].data(), sizeof(double) * n_gsvertices);
	}
}

void Grid::enumerate_gs_subset(
	int nv, int ne,
	int* vflags, int* eflags,
	int& nv_gs, int& ne_gs,
	std::vector<int>& vlexi2gs, std::vector<int>& elexi2gs
) {
	printf("[%d] Enumerating GS subset...\n", _layer);
	int nv_gsset[8] = { 0 };
	int ne_gsset[8] = { 0 };
	for (int i = 0; i < nv; i++) {
		int gsid = (vflags[i] & Bitmask::mask_gscolor) >> Bitmask::offset_gscolor;
		if (gsid >= 8) printf("\033[31merror id\033[0m\n");
		nv_gsset[gsid]++;
	}
	for (int i = 0; i < ne; i++) {
		int gsid = (eflags[i] & Bitmask::mask_gscolor) >> Bitmask::offset_gscolor;
		if (gsid >= 8) printf("\033[31merror id\033[0m\n");
		ne_gsset[gsid]++;
	}
	for (int i = 0; i < 8; i++) {
	}

	// lexicographical order to GS colored order 
	// gs[lexi] = Gs colored vid
	nv_gs = 0; ne_gs = 0;
	for (int i = 0; i < 8; i++) {
		int oldvset = nv_gsset[i];
		nv_gsset[i] = snippet::Round<32>(nv_gsset[i]);
		int oldeset = ne_gsset[i];
		ne_gsset[i] = snippet::Round<32>(ne_gsset[i]);
		nv_gs += nv_gsset[i];
		ne_gs += ne_gsset[i];
		printf("--  #%d v %d (%d) | e %d (%d)\n", i, oldvset, nv_gsset[i], oldeset, ne_gsset[i]);
		gs_num[i] = nv_gsset[i];
	}

	vlexi2gs.resize(nv, -1);
	elexi2gs.resize(ne, -1);

	int gs_vidaccu[8] = { 0 };
	int gs_eidaccu[8] = { 0 };
	for (int i = 0; i < nv; i++) {
		int gsid = (vflags[i] & Bitmask::mask_gscolor) >> Bitmask::offset_gscolor;
		if (gsid < 0 || gsid >= 8) printf("-- error on gs id computation\n");
		int vgsid = 0;
		// add offset of other subset vertices
		for (int j = 0; j < gsid; j++)  vgsid += nv_gsset[j];
		vgsid += gs_vidaccu[gsid];
		gs_vidaccu[gsid]++;
		vlexi2gs[i] = vgsid;
	}
	for (int i = 0; i < ne; i++) {
		int gsid = (eflags[i] & Bitmask::mask_gscolor) >> Bitmask::offset_gscolor;
		if (gsid < 0 || gsid >= 8) printf("-- error on gs id computation\n");
		int egsid = 0;
		// add offset of other subset vertices
		for (int j = 0; j < gsid; j++)  egsid += ne_gsset[j];
		egsid += gs_eidaccu[gsid];
		gs_eidaccu[gsid]++;
		elexi2gs[i] = egsid;
	}
}

void HierarchyGrid::update_stencil(void)
{
	for (int i = 0; i < _gridlayer.size(); i++) {
		if (_gridlayer[i]->is_dummy()) continue;
		if (i == 0) continue;
		restrict_stencil(*_gridlayer[i], *_gridlayer[i]->fineGrid);
		// last layer build host system
		if (i == _gridlayer.size() - 1) {
			_gridlayer[i]->buildCoarsestSystem();
		}
	}
}


