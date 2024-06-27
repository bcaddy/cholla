#pragma once

#include <stdio.h>

#include "../global/global.h"
#include "../grid/grid3D.h"

// Specifies minimum density threshold for identifying cells used in
// feedback-related analysis. In cgs, this is 0.01 cm^{-3}.
#define FB_ANALYSIS_CUTOFF_DENSITY (0.01 * MU * MP / DENSITY_UNIT)

class FeedbackAnalysis
{
  Real *h_circ_vel_x, *h_circ_vel_y;
  Real r_max, z_max;

#ifdef PARTICLES_GPU
  Real *d_circ_vel_x, *d_circ_vel_y;
  void Compute_Gas_Velocity_Dispersion_GPU(Grid3D& G);
#endif

 public:
  int countSN{0};
  int countResolved{0};
  int countUnresolved{0};
  Real totalEnergy{0};
  Real totalMomentum{0};
  Real totalUnresEnergy{0};
  Real totalWindMomentum{0};
  Real totalWindEnergy{0};

  FeedbackAnalysis(Grid3D& G, struct Parameters* P);
  ~FeedbackAnalysis();

  void Compute_Gas_Velocity_Dispersion(Grid3D& G);
  void Reset();
};
