#pragma once

#ifndef __CGAL_DEFINITION_H
#define __CGAL_DEFINITION_H

#include <list>

#include "CGAL/Surface_mesh/Surface_mesh.h"
#include "CGAL/Simple_cartesian.h"
#include "CGAL/Side_of_triangle_mesh.h"
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include "CGAL/AABB_segment_primitive.h"

#include "CGAL/point_generators_3.h"
#include "CGAL/Orthogonal_k_neighbor_search.h"
#include "CGAL/Search_traits_3.h"
#include "CGAL/Search_traits_adapter.h"
#include "boost/iterator/zip_iterator.hpp"
#include "CGAL/Plane_3.h"

#include "CGAL/Polygon_mesh_processing/corefinement.h"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/AABB_segment_primitive.h>
#include <typeinfo>
//#include <CGAL/AABB_intersection_and_primitive_id.h>
#include <CGAL/intersections.h>
#include <CGAL/Object.h>

//#include "CGAL/bary"

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector3;
typedef Kernel::Segment_3 Segment;
typedef CGAL::Surface_mesh<Point> CGMesh;
typedef CGAL::Triangle_3<Kernel> Triangle;
typedef CGAL::Segment_3<Kernel> Segment;


typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel1;
typedef Kernel1::Point_3 Point1;

typedef CGAL::AABB_tree< CGAL::AABB_traits<
	Kernel, CGAL::AABB_triangle_primitive<
	Kernel, std::vector<Triangle>::iterator
	>
>
> aabb_tree_t;

typedef CGAL::AABB_tree< CGAL::AABB_traits<
	Kernel, CGAL::AABB_triangle_primitive<
	Kernel, std::list<Triangle>::iterator
	>
>
> aabb_tree_t1;

typedef std::vector<std::array<Point, 3>> TriangleList;
typedef CGAL::AABB_triangle_primitive<Kernel, std::vector<std::array<Point, 3>>::iterator> Primitive2;
typedef CGAL::AABB_traits<Kernel, Primitive2> Traits2;
typedef CGAL::AABB_tree<Traits2> aabb_tree_t2;

typedef std::vector<Triangle>::iterator Iterator3;
typedef CGAL::AABB_triangle_primitive<Kernel, Iterator3> Primitive3;
typedef CGAL::AABB_traits<Kernel, Primitive3> Traits3;
typedef CGAL::AABB_tree<Traits3> aabb_tree_t3;

typedef typename aabb_tree_t::Primitive_id Primitive_id;

typedef typename aabb_tree_t::Intersection_and_primitive_id<Triangle> Inersection_and_primitive_id;

typedef Kernel::FT FT;
typedef Kernel::Ray_3 Ray;
typedef Kernel::Line_3 Line;
typedef Kernel::Point_3 Point;
typedef Kernel::Triangle_3 Triangle;
typedef Kernel::Plane_3 Plane;

typedef boost::optional< aabb_tree_t::Intersection_and_primitive_id<Ray>::Type > Ray_intersetion;
typedef boost::optional< aabb_tree_t::Intersection_and_primitive_id<Segment>::Type > Segment_intersection;
typedef boost::optional< aabb_tree_t::Intersection_and_primitive_id<Plane>::Type > Plane_intersection;

typedef boost::tuple<Point, int> PointInt;
typedef CGAL::Search_traits_3<Kernel> KdTreeTraits;
typedef CGAL::Search_traits_adapter<PointInt, CGAL::Nth_of_tuple_property_map<0, PointInt>, KdTreeTraits> Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits> KdTreeSearch;
typedef KdTreeSearch::Tree KdTree;

typedef boost::graph_traits<CGMesh>::vertex_descriptor Vertex_descriptor;
typedef boost::graph_traits<CGMesh>::face_descriptor Face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;

extern CGMesh cgmesh_container, cgmesh_object;

extern std::vector<Triangle> aabb_tris;
extern aabb_tree_t aabb_tree;

extern std::vector<Point> kdpoints;
extern std::vector<int> kdpointids;
extern KdTree kdtree;

//CGMesh gmesh;
extern CGMesh::Property_map<Face_descriptor, Vector3> cgmesh_fnormals;
extern CGMesh::Property_map<Vertex_descriptor, Vector3> cgmesh_vnormals;

//void mesh2cgalmesh(Mesh& mesh, CGMesh& cgmesh);
#endif


