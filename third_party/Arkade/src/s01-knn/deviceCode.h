// All rights reserved by 
// Durga Keerthi Mandarapu, Vani Nagarajan, Artem Pelenitsyn, and Milind Kulkarni. 2024. 
// Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing. 

#pragma once

#include <owl/owl.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/random.h>

#define FLOAT_MIN 1.175494351e-38
#define FLOAT_MAX 3.402823466e+38
// #define KN 10
using namespace owl;
using namespace std;

namespace owl {

	typedef struct maximumDistanceLogger
	{
		int ind;
		float dist;
	}Neigh;
	
  struct Sphere {
    vec3f center;
  };

  struct SpheresGeom {
    Sphere *data_pts;
    float rad;
  };

  struct RayGenData
  {
    OptixTraversableHandle world;
    Sphere *query_pts; 
  };

  struct MyGlobals 
  {	
    Neigh *frameBuffer;
    int k;
  };

  struct NeighGroup
  {
		Neigh res[KN];
	};

}
