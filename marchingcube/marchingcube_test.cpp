/* ========================================================================= *
 *                                                                           *
 *                               OpenMesh                                    *
 *           Copyright (c) 2001-2015, RWTH-Aachen University                 *
 *           Department of Computer Graphics and Multimedia                  *
 *                          All rights reserved.                             *
 *                            www.openmesh.org                               *
 *                                                                           *
 *---------------------------------------------------------------------------*
 * This file is part of OpenMesh.                                            *
 *---------------------------------------------------------------------------*
 *                                                                           *
 * Redistribution and use in source and binary forms, with or without        *
 * modification, are permitted provided that the following conditions        *
 * are met:                                                                  *
 *                                                                           *
 * 1. Redistributions of source code must retain the above copyright notice, *
 *    this list of conditions and the following disclaimer.                  *
 *                                                                           *
 * 2. Redistributions in binary form must reproduce the above copyright      *
 *    notice, this list of conditions and the following disclaimer in the    *
 *    documentation and/or other materials provided with the distribution.   *
 *                                                                           *
 * 3. Neither the name of the copyright holder nor the names of its          *
 *    contributors may be used to endorse or promote products derived from   *
 *    this software without specific prior written permission.               *
 *                                                                           *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              *
 *                                                                           *
 * ========================================================================= */
 /*===========================================================================*\
  *                                                                           *
  *   $Revision$                                                         *
  *   $Date$                   *
  *                                                                           *
 \*===========================================================================*/
#include <iostream>
 // -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include "MCrender.h"
#include "TriMesh3D.h"
// ----------------------------------------------------------------------------
//typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;
// ----------------------------------------------------------------------------
// Build a simple cube and write it to std::cout

void testMarchingCube(void)
{
    MCImplicitRender* mc_render;
    TriMesh3D* ptr_mesh_;
    ptr_mesh_ = new TriMesh3D;
    //std::string filename = "../mesh/tmp.obj";
    std::string filename = "../mesh/after_scale.obj";
    //std::cout << filename << std::endl;
    std::string key_file;
    ptr_mesh_->LoadFile(filename);
    std::string file = filename;
    int k1 = file.rfind("/");
    int k2 = file.rfind(".");
    key_file = file.substr(k1 + 1, k2 - 1 - k1);
    //std::cout << key_file << std::endl;

    mc_render = new MCImplicitRender(0, 1e-6, 100, ptr_mesh_, key_file);
    mc_render->TestMC();
    //mc_render = new MCImplicitRender();
    //mc_render->DoMC();
    //ptr_mesh_->SaveFile("./output/isosurface.obj");
}