/*!
 * \file pcm_cuda_tests.cu
 * \brief Contains the tests for the code in pcm_cuda.h and pcm_cuda.cu
 */

// STL Includes
#include <random>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../io/io.h"
#include "../reconstruction/reconstruction.h"
#include "../utils/basic_structs.h"
#include "../utils/testing_utilities.h"
TEST(tAllReconstructInterfaceStates, PcmCorrectInputExpectCorrectOutput)
{
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx      = 7;
  size_t const ny      = 7;
  size_t const nz      = 7;
  size_t const xid     = 3;
  size_t const yid     = 3;
  size_t const zid     = 3;
  size_t const n_cells = nx * ny * nz;
  double const dx      = doubleRand(prng);
  double const dt      = doubleRand(prng);
  double const gamma   = 5.0 / 3.0;

  // Setup host grid. Fill host grid with random values and randomly assign values
  std::vector<double> host_grid(n_cells * grid_enum::num_fields);
  for (Real &val : host_grid) {
    val = doubleRand(prng);
  }

  // Test each direction
  reconstruction::InterfaceState test_interface_pcm_l_0, test_interface_pcm_r_0, test_interface_pcm_l_1,
      test_interface_pcm_r_1, test_interface_pcm_l_2, test_interface_pcm_r_2;

  reconstruction::Reconstruct_Interface_States<reconstruction::Kind::pcm, 0>(
      host_grid.data(), xid, yid, zid, nx, ny, n_cells, gamma, test_interface_pcm_l_0, test_interface_pcm_r_0);
  reconstruction::Reconstruct_Interface_States<reconstruction::Kind::pcm, 1>(
      host_grid.data(), xid, yid, zid, nx, ny, n_cells, gamma, test_interface_pcm_l_1, test_interface_pcm_r_1);
  reconstruction::Reconstruct_Interface_States<reconstruction::Kind::pcm, 2>(
      host_grid.data(), xid, yid, zid, nx, ny, n_cells, gamma, test_interface_pcm_l_2, test_interface_pcm_r_2);

  // Fiducial values
  reconstruction::InterfaceState fiducial_interface_pcm_l_0{
      1.6206985712721595,     {1.9275471960012214, 2.0380692774425846, 1.9771827902007457}, 4.5791453055608384,
      9.9999999999999995e-21, {4.1622274705137627, 2.1906071705977261, 3.1997462690190144}, 16.180636739137334},
      fiducial_interface_pcm_r_0{
          1.5162490166443841,     {0.74079082506491523, 1.4295471037207337, 0.49525487240256766}, 1.6382470722683291,
          9.9999999999999995e-21, {2.6539699941465473, 2.6775840565878508, 2.4794718891665037},   10.180396979545293},
      fiducial_interface_pcm_l_1{
          1.6206985712721595,     {2.0380692774425846, 1.9771827902007457, 1.9275471960012214}, 4.5791453055608384,
          9.9999999999999995e-21, {2.1906071705977261, 3.1997462690190144, 4.1622274705137627}, 16.180636739137338},
      fiducial_interface_pcm_r_1{
          3.8412847012400144,     {1.1260155024295584, 0.37985902941387084, 0.31356489904284668}, 4.1037970369599064,
          9.9999999999999995e-21, {2.7361340285756826, 4.5077114382460621, 3.2694920805403553},   19.247735148770136},
      fiducial_interface_pcm_l_2{
          1.6206985712721595,     {1.9771827902007457, 1.9275471960012214, 2.0380692774425846}, 4.5791453055608384,
          9.9999999999999995e-21, {3.1997462690190144, 4.1622274705137627, 2.1906071705977261}, 16.180636739137334},
      fiducial_interface_pcm_r_2{
          0.75619040256911529,    {4.3870709307030475, 0.53201818469160067, 3.0376042247856248}, 4.181424078824616,
          9.9999999999999995e-21, {3.0291890161755175, 3.4009589976457022, 2.1042141607876181},  12.585112716922399};

  // Check correctness
  testing_utilities::Check_Interface(test_interface_pcm_l_0, fiducial_interface_pcm_l_0, 0);
  testing_utilities::Check_Interface(test_interface_pcm_r_0, fiducial_interface_pcm_r_0, 0);
  testing_utilities::Check_Interface(test_interface_pcm_l_1, fiducial_interface_pcm_l_1, 0);
  testing_utilities::Check_Interface(test_interface_pcm_r_1, fiducial_interface_pcm_r_1, 0);
  testing_utilities::Check_Interface(test_interface_pcm_l_2, fiducial_interface_pcm_l_2, 0);
  testing_utilities::Check_Interface(test_interface_pcm_r_2, fiducial_interface_pcm_r_2, 0);
}
