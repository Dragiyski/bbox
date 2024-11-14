# Bounding Box Index

Prototype for the bounding box index. Bounding box index is a development of algorithm (not an application or library). The purpose of this algorithm is:

* To be efficiently implementable on the GPU;
* To generate an index for raytracing - generating two structures - a tree that organizes elements (a.k.a. up-structure) and a list of elements organized by types (a.k.a. down structure). Currently there will be two types of elements: triangles (to be displayed) and bounding boxes (to organize triangles);
* To measure the amount of tracing reduction this type of accelerated structure provides, including when using recursive raytracing;

Since this is written in python, it uses `numpy` and `pandas` to optimize the loops. However, all aspects of the loops should be parallelizable.

* Sorting: Using bitonic sort on GPU: Complexity `O(N * log(N)^2)`, but the `N` part is in parallel;
* Grouping: Using tree;
* Tree: mapped onto binary tree, `left_child` will be `first_child`, `right_child` will be `next_sibling`; There will be maximum `K` (configurable) children per node, currently set to `6`, as raytracing bounding box seems comparable to raytracing 6 triangles (or 6 planar shapes);

## Structure

`index.py` - generates the up and down structures containing bounding boxes and triangles (for now);
`raytracer.py` - test the generated up and down structures, allowing analyzing how many actual traces are fired;

## Goals

The index generating up- and down- structures seems to be dimensionally agnostic. It is possible to remap all elements for each ray for the recursive part of the raytracing and divide the tree not just on the bounding box `xyz`, but on the ray direction `xyz` and ray origin `xyz`. While this seems redundant for a single ray, since we know exact coordinates for the ray origin and ray direction, for large number of rays it will optimize the search;

## Algorithm

The algorithm takes a set of elements and project them into 1D model space for each dimension. A triangle projected onto 1D is a line. Given a huge set of lines, we can split the space in 3: toward the minimum value (min-child), toward the maximum value (max-child) and whatever cannot be easily split (the center). The algorithm computes the bounding box given the minimum and maximum value. Then create 3 children: the balanced min-child, max-child and as little as possible elements assigned to the center without causing the min-child and max-child overlap too much. This create ternary tree, with balanced binary tree within.

Firing `R` rays on `N` objects will actually require `O*(R log(N))` intersection checks instead of `O(R * N)` intersection checks.

While the idea of such bounding box accelerated structure is not new (see `VkAccelerationStructureKHR`), the generation of the structure in `O(N * log(N)^2)` with `N` parallizable on the GPU is new. This would allow the acceleration structure to be build for every frame, and even for each raytracing recursion level, similar to how vertex shaders depends on `N` while fragment shaders depeends on `O(R)`, making the entire process `O(R + N)` instead of `O(R * N)`.

## Roadmap

* Recursion: figure out how to create the three on the set of cartesian product of `R` rays and `N` objects without using `R*N` memory;
* Generate a tree including `ray_direction` and `ray_origin` as potential dimensions for splitting the tree;
* Transition from `pandas` and `numpy` to GPU based on VulkanAPI:
  * Sort: use bitonic sort (proof of concept already implemented for WebGPU);
  * GroupBy: TODO;
