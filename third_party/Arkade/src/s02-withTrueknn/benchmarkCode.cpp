// Reproducible Arkade benchmark frontend: binary point input, repeated
// synchronous TrueKNN queries, separated BVH/query timings, and binary indices.

#include <owl/owl.h>
#include <owl/DeviceMemory.h>

#include "deviceCode.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" char deviceCode_ptx[];

namespace {

using Clock = std::chrono::steady_clock;
using owl::Neigh;
using owl::Sphere;

double secondsSince(const Clock::time_point& begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

std::vector<Sphere> readPoints(
    const std::string& path, std::size_t pointCount) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open input file: " + path);
  }
  std::vector<float> coordinates(pointCount * 3);
  input.read(
      reinterpret_cast<char*>(coordinates.data()),
      static_cast<std::streamsize>(coordinates.size() * sizeof(float)));
  if (input.gcount() !=
      static_cast<std::streamsize>(coordinates.size() * sizeof(float))) {
    throw std::runtime_error("binary input does not contain the requested points");
  }
  std::vector<Sphere> points;
  points.reserve(pointCount);
  for (std::size_t index = 0; index < pointCount; ++index) {
    points.push_back(Sphere{owl::vec3f(
        coordinates[index * 3], coordinates[index * 3 + 1],
        coordinates[index * 3 + 2])});
  }
  return points;
}

void writeIndices(
    const std::string& path, const Neigh* neighbors,
    std::size_t queryCount) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("cannot open output file: " + path);
  }
  for (std::size_t query = 0; query < queryCount; ++query) {
    for (int neighbor = 0; neighbor < KN; ++neighbor) {
      const std::int32_t index = neighbors[query * KN + neighbor].ind;
      output.write(
          reinterpret_cast<const char*>(&index), sizeof(index));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7 || argc > 8) {
    std::cerr << "usage: arkade-benchmark INPUT.bin SUPPORT QUERY "
                 "INITIAL_RADIUS WARMUPS REPEATS [INDICES.bin]\n";
    return 2;
  }
  try {
    const std::string inputPath = argv[1];
    const std::size_t supportCount = std::stoull(argv[2]);
    const std::size_t queryCount = std::stoull(argv[3]);
    const float initialRadius = std::stof(argv[4]);
    const int warmups = std::stoi(argv[5]);
    const int repeats = std::stoi(argv[6]);
    if (supportCount == 0 || queryCount == 0 || initialRadius <= 0.0f ||
        warmups < 0 || repeats <= 0) {
      throw std::runtime_error("counts, radius, warmups, or repeats are invalid");
    }
    const auto points = readPoints(inputPath, supportCount + queryCount);
    std::vector<Sphere> support(points.begin(), points.begin() + supportCount);
    std::vector<Sphere> queries(points.begin() + supportCount, points.end());

    OWLContext context = owlContextCreate(nullptr, 1);
    OWLModule module = owlModuleCreate(context, deviceCode_ptx);
    OWLVarDecl geometryVariables[] = {
        {"data_pts", OWL_BUFPTR, OWL_OFFSETOF(owl::SpheresGeom, data_pts)},
        {"rad", OWL_FLOAT, OWL_OFFSETOF(owl::SpheresGeom, rad)},
        {}};
    OWLGeomType geometryType = owlGeomTypeCreate(
        context, OWL_GEOMETRY_USER, sizeof(owl::SpheresGeom),
        geometryVariables, -1);
    owlGeomTypeSetIntersectProg(geometryType, 0, module, "Spheres");
    owlGeomTypeSetBoundsProg(geometryType, module, "Spheres");
    owlBuildPrograms(context);

    OWLBuffer frameBuffer = owlHostPinnedBufferCreate(
        context, OWL_USER_TYPE(Neigh), queryCount * KN);
    OWLBuffer neighborCountBuffer = owlHostPinnedBufferCreate(
        context, OWL_INT, queryCount);
    OWLBuffer supportBuffer = owlDeviceBufferCreate(
        context, OWL_USER_TYPE(Sphere), support.size(), support.data());
    OWLBuffer queryBuffer = owlDeviceBufferCreate(
        context, OWL_USER_TYPE(Sphere), queries.size(), queries.data());

    OWLGeom geometry = owlGeomCreate(context, geometryType);
    owlGeomSetPrimCount(geometry, support.size());
    owlGeomSetBuffer(geometry, "data_pts", supportBuffer);
    owlGeomSet1f(geometry, "rad", initialRadius);

    int round = 0;
    OWLVarDecl launchVariables[] = {
        {"frameBuffer", OWL_BUFPTR, OWL_OFFSETOF(owl::MyGlobals, frameBuffer)},
        {"num_neighbors", OWL_BUFPTR,
         OWL_OFFSETOF(owl::MyGlobals, num_neighbors)},
        {"round", OWL_INT, OWL_OFFSETOF(owl::MyGlobals, round)},
        {}};
    OWLParams launchParams = owlParamsCreate(
        context, sizeof(owl::MyGlobals), launchVariables, -1);
    owlParamsSetBuffer(launchParams, "frameBuffer", frameBuffer);
    owlParamsSetBuffer(launchParams, "num_neighbors", neighborCountBuffer);
    owlParamsSet1i(launchParams, "round", round);

    OWLGeom geometries[] = {geometry};
    const auto buildBegin = Clock::now();
    OWLGroup geometryGroup = owlUserGeomGroupCreate(
        context, 1, geometries, OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlGroupBuildAccel(geometryGroup);
    OWLGroup world = owlInstanceGroupCreate(
        context, 1, &geometryGroup, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL,
        OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlGroupBuildAccel(world);
    const double buildSeconds = secondsSince(buildBegin);

    OWLVarDecl rayGenerationVariables[] = {
        {"query_pts", OWL_BUFPTR, OWL_OFFSETOF(owl::RayGenData, query_pts)},
        {"world", OWL_GROUP, OWL_OFFSETOF(owl::RayGenData, world)},
        {}};
    OWLRayGen rayGeneration = owlRayGenCreate(
        context, module, "rayGen", sizeof(owl::RayGenData),
        rayGenerationVariables, -1);
    owlRayGenSetBuffer(rayGeneration, "query_pts", queryBuffer);
    owlRayGenSetGroup(rayGeneration, "world", world);
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);

    std::vector<double> querySeconds;
    std::vector<int> queryRounds;
    querySeconds.reserve(repeats);
    queryRounds.reserve(repeats);
    float previousRadius = initialRadius;
    for (int iteration = 0; iteration < warmups + repeats; ++iteration) {
      if (previousRadius != initialRadius) {
        owlGeomSet1f(geometry, "rad", initialRadius);
        owlGroupRefitAccel(geometryGroup);
        owlGroupRefitAccel(world);
      }
      owlBufferClear(frameBuffer);
      owlBufferClear(neighborCountBuffer);
      round = 0;
      owlParamsSet1i(launchParams, "round", round);
      float radius = initialRadius;
      bool incomplete = true;
      const auto queryBegin = Clock::now();
      while (incomplete) {
        owlLaunch2D(
            rayGeneration, static_cast<int>(queryCount), 1, launchParams);
        const int* neighborCounts = static_cast<const int*>(
            owlBufferGetPointer(neighborCountBuffer, 0));
        incomplete = false;
        for (std::size_t query = 0; query < queryCount; ++query) {
          if (neighborCounts[query] < KN) {
            incomplete = true;
            break;
          }
        }
        if (incomplete) {
          ++round;
          radius *= 2.0f;
          owlParamsSet1i(launchParams, "round", round);
          owlGeomSet1f(geometry, "rad", radius);
          owlGroupRefitAccel(geometryGroup);
          owlGroupRefitAccel(world);
        }
      }
      const double elapsed = secondsSince(queryBegin);
      previousRadius = radius;
      if (iteration >= warmups) {
        querySeconds.push_back(elapsed);
        queryRounds.push_back(round + 1);
      }
    }

    if (argc == 8) {
      const Neigh* neighbors = static_cast<const Neigh*>(
          owlBufferGetPointer(frameBuffer, 0));
      writeIndices(argv[7], neighbors, queryCount);
    }
    std::cout << "{\"support\":" << supportCount
              << ",\"query\":" << queryCount
              << ",\"k\":" << KN
              << ",\"norm\":" << NORM
              << ",\"initial_radius\":" << initialRadius
              << ",\"build_s\":" << buildSeconds
              << ",\"query_s\":[";
    for (std::size_t index = 0; index < querySeconds.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << querySeconds[index];
    }
    std::cout << "],\"rounds\":[";
    for (std::size_t index = 0; index < queryRounds.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << queryRounds[index];
    }
    std::cout << "]}" << std::endl;
    owlContextDestroy(context);
  } catch (const std::exception& error) {
    std::cerr << "arkade-benchmark: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
