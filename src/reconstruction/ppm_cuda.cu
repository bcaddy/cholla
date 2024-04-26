/*! \file ppm_cuda.cu
 *  \brief Functions definitions for the ppm kernels, using characteristic
 tracing. Written following Stone et al. 2008. */

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/ppm_cuda.h"
#include "../reconstruction/reconstruction_internals.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif

// =====================================================================================================================
template <int dir>
__global__ __launch_bounds__(TPB) void PPM_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                                int ny, int nz, Real dx, Real dt, Real gamma)
{
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Ensure that we are only operating on cells that will be used
  if (reconstruction::Thread_Guard<3>(nx, ny, nz, xid, yid, zid)) {
    return;
  }

  // Compute the total number of cells
  int const n_cells = nx * ny * nz;

  // Set the field indices for the various directions
  int o1, o2, o3;
  if constexpr (dir == 0) {
    o1 = grid_enum::momentum_x;
    o2 = grid_enum::momentum_y;
    o3 = grid_enum::momentum_z;
  } else if constexpr (dir == 1) {
    o1 = grid_enum::momentum_y;
    o2 = grid_enum::momentum_z;
    o3 = grid_enum::momentum_x;
  } else if constexpr (dir == 2) {
    o1 = grid_enum::momentum_z;
    o2 = grid_enum::momentum_x;
    o3 = grid_enum::momentum_y;
  }

  // load the 5-cell stencil into registers
  // cell i
  hydro_utilities::Primitive const cell_i =
      hydro_utilities::Load_Cell_Primitive<dir>(dev_conserved, xid, yid, zid, nx, ny, n_cells, gamma);

  // cell i-1. The equality checks the direction and will subtract one from the correct direction
  // im1 stands for "i minus 1"
  hydro_utilities::Primitive const cell_im1 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny, n_cells, gamma);

  // cell i+1.  The equality checks the direction and add one to the correct direction
  // ip1 stands for "i plus 1"
  hydro_utilities::Primitive const cell_ip1 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid + int(dir == 0), yid + int(dir == 1), zid + int(dir == 2), nx, ny, n_cells, gamma);

  // cell i-2. The equality checks the direction and will subtract two from the correct direction
  // im2 stands for "i minus 2"
  hydro_utilities::Primitive const cell_im2 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid - 2 * int(dir == 0), yid - 2 * int(dir == 1), zid - 2 * int(dir == 2), nx, ny, n_cells, gamma);

  // cell i+2.  The equality checks the direction and add two to the correct direction
  // ip2 stands for "i plus 2"
  hydro_utilities::Primitive const cell_ip2 = hydro_utilities::Load_Cell_Primitive<dir>(
      dev_conserved, xid + 2 * int(dir == 0), yid + 2 * int(dir == 1), zid + 2 * int(dir == 2), nx, ny, n_cells, gamma);

#ifdef PPMC
  // Compute the eigenvectors
  reconstruction::EigenVecs const eigenvectors = reconstruction::Compute_Eigenvectors(cell_i, gamma);

  // Cell i
  reconstruction::Characteristic const cell_i_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_i, eigenvectors, gamma);

  // Cell i-1
  reconstruction::Characteristic const cell_im1_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_im1, eigenvectors, gamma);

  // Cell i-2
  reconstruction::Characteristic const cell_im2_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_im2, eigenvectors, gamma);

  // Cell i+1
  reconstruction::Characteristic const cell_ip1_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_ip1, eigenvectors, gamma);

  // Cell i+2
  reconstruction::Characteristic const cell_ip2_characteristic =
      reconstruction::Primitive_To_Characteristic(cell_i, cell_ip2, eigenvectors, gamma);

  // Compute the interface states for each field
  auto const [interface_L_iph_characteristic, interface_R_imh_characteristic] =
      reconstruction::PPM_Interfaces(cell_im2_characteristic, cell_im1_characteristic, cell_i_characteristic,
                                     cell_ip1_characteristic, cell_ip2_characteristic);

  // Convert back to primitive variables
  hydro_utilities::Primitive interface_L_iph =
      reconstruction::Characteristic_To_Primitive(cell_i, interface_L_iph_characteristic, eigenvectors, gamma);
  hydro_utilities::Primitive interface_R_imh =
      reconstruction::Characteristic_To_Primitive(cell_i, interface_R_imh_characteristic, eigenvectors, gamma);

  // Compute the interfaces for the variables that don't have characteristics
  #ifdef DE
  reconstruction::PPM_Single_Variable(cell_im2.gas_energy_specific, cell_im1.gas_energy_specific,
                                      cell_i.gas_energy_specific, cell_ip1.gas_energy_specific,
                                      cell_ip2.gas_energy_specific, interface_L_iph.gas_energy_specific,
                                      interface_R_imh.gas_energy_specific);
  #endif  // DE
  #ifdef SCALAR
  for (int i = 0; i < NSCALARS; i++) {
    reconstruction::PPM_Single_Variable(cell_im2.scalar_specific[i], cell_im1.scalar_specific[i],
                                        cell_i.scalar_specific[i], cell_ip1.scalar_specific[i],
                                        cell_ip2.scalar_specific[i], interface_L_iph.scalar_specific[i],
                                        interface_R_imh.scalar_specific[i]);
  }
  #endif  // SCALAR
#else     // PPMC
  auto [interface_L_iph, interface_R_imh] =
      reconstruction::PPM_Interfaces(cell_im2, cell_im1, cell_i, cell_ip1, cell_ip2);
#endif    // PPMC

  // Do the characteristic tracing
#ifndef VL
  PPM_Characteristic_Evolution(cell_i, dt, dx, gamma, interface_R_imh, interface_L_iph);
#endif  // VL

  // enforce minimum values
  interface_R_imh.density  = fmax(interface_R_imh.density, (Real)TINY_NUMBER);
  interface_L_iph.density  = fmax(interface_L_iph.density, (Real)TINY_NUMBER);
  interface_R_imh.pressure = fmax(interface_R_imh.pressure, (Real)TINY_NUMBER);
  interface_L_iph.pressure = fmax(interface_L_iph.pressure, (Real)TINY_NUMBER);

  // Step 11 - Send final values back from kernel

  // Convert the left and right states in the primitive to the conserved variables send final values back from kernel
  // bounds_R refers to the right side of the i-1/2 interface
  size_t id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
  reconstruction::Write_Data(interface_L_iph, dev_bounds_L, dev_conserved, id, n_cells, o1, o2, o3, gamma);

  id = cuda_utilities::compute1DIndex(xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny);
  reconstruction::Write_Data(interface_R_imh, dev_bounds_R, dev_conserved, id, n_cells, o1, o2, o3, gamma);
}
// Instantiate the relevant template specifications
template __global__ __launch_bounds__(TPB) void PPM_cuda<0>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                            int nx, int ny, int nz, Real dx, Real dt, Real gamma);
template __global__ __launch_bounds__(TPB) void PPM_cuda<1>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                            int nx, int ny, int nz, Real dx, Real dt, Real gamma);
template __global__ __launch_bounds__(TPB) void PPM_cuda<2>(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
                                                            int nx, int ny, int nz, Real dx, Real dt, Real gamma);
// =====================================================================================================================
