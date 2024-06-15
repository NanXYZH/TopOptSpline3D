# Topology Optimization of Self-Supporting Structures for Additive Manufacturing Via Implicit B-Spline Representations



## Compilation

The source can be compiled using CMake, while most of the dependcies has be packed into a conda environment. You can activate it by :

```bash
conda env create -f environment.yml
conda activate robtop
```

Then configure and build the project in conda environment:

```bash
mkdir build
cd buld
cmake .. -DUSE_CONDA=ON 
make -j4
```





## Usage

### Option

* `-meshfile`:  The input mesh model.
* `-jsonfile`:  The input boundary condition in Json format.

* `-gridreso`:  default=`200`, set the grid resolution along the longest axis (**must<1024**)
* `-volume_ratio`: The  goal volume ratio of optimized model
* `-outdir`: The output directory of the results.
* `-workmode`: 4 alternative mode (`wscf`/`wsff`/`nscf`/`nsff`), `ws/ns` means with/no support(fixed) boundary, `cf/ff` means constrain force direction to surface normal or not.
* `-filter_radius`: default=`2`, the sensitivity filter radius in the unit of the voxel length. 
* `-damp_ratio`:  default=`0.5`, the damp ratio of the  Optimality Criteria method
* `-design_step`:  default=`0.03`, the change limit (maximal step length) when updating the density.
* `-shell_width`: default=`3`, the shell width to be enforced during the optimization, given in the unit of the voxel length.
* `-poisson_ratio`:default=`0.4`, Poisson ratio of the solid material.
* `-youngs_modulus`: default=`1e6`, Young's Modulus of the solid material.
* `-min_density`: default=`1e-3`,  minimal density value to avoid numerical problem.
* `-power_penalty`: default=`3`, the power exponent of density penalty.
* `-[no]logdensity`: output density field in each iteration or not.
* `-testname`: additional test suits. For example, `-testname=testordtop` will do general topology optimization, not worst-case optimization.



### example

1. Copy the compiled executable to `./bench` 

2. Copy the dependent shared library to `./bench` .

3. Run the following command in CMD or shell

   * ##### Bunny Example

     ###### Optimization with with spline

     ```
     ./robtop -jsonfile=./bunnysmooth/config2.json -meshfile=./bunnysmooth/bunny.obj -outdir=./result/bunnysmooth/ordsplinetopmma/ -power_penalty=3 -volume_ratio=0.4 -filter_radius=2 -gridreso=511 -damp_ratio=0.5 -shell_width=0 -workmode=wscf -poisson_ratio=0.4 -design_step=0.06 -vol_reduction=0.05 -min_density=1e-3 -logdensity -nologcompliance -testname=testordsplinetopmma
     ```

## Dependency

* CGAL
* Eigen3
* Trimesh2 (https://github.com/Forceflow/trimesh2.git)
* CUDA 11
* OpenVDB 
* OpenMesh
* glm (https://glm.g-truc.net/0.9.9/)
* Boost
* RapidJson
* gflags
* Spectra (https://github.com/yixuan/spectra.git)



## Note 

To see our original source of paper "Topology Optimization of Self-supporting Structures for Additive Manufacturing using
Implicit B-splines Representation".



## Reference

* This repository incorporates a voxelization routine that credits the original source, [cuda_voxelizer](https://github.com/Forceflow/cuda_voxelizer).
