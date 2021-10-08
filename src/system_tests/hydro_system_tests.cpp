/*!
 * \file hydro_system_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains all the system tests for the HYDRO build type
 *
 */


// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../system_tests/system_tester.h"

#ifndef PI
#define PI 3.141592653589793
#endif

TEST(tHYDROSYSTEMSodShockTube,
     CorrectInputExpectCorrectOutput)
{
    systemTest::systemTestRunner();
}

TEST(tHYDROSYSTEMConstant,
     CorrectInputExpectCorrectOutput)
{
  H5::H5File testDataFile;
  systemTest::systemTestRunAndLoad(testDataFile);
  systemTest::systemTestDatasetIsConstant(testDataFile,"density",1.0);
  systemTest::systemTestDatasetIsConstant(testDataFile,"momentum_x",0.0);
  systemTest::systemTestDatasetIsConstant(testDataFile,"momentum_y",0.0);
  systemTest::systemTestDatasetIsConstant(testDataFile,"momentum_z",0.0);
  systemTest::systemTestDatasetIsConstant(testDataFile,"Energy",1.5e-5);  

}

TEST(tHYDROSYSTEMSoundWave3D,
     CorrectInputExpectCorrectOutput)
{
  H5::H5File testDataFile;
  double time = 0.05;
  double amplitude = 1e-5;
  double dx = 1./64.;
    
  double real_kx = 2*PI;//kx of the physical problem
  
  double kx = real_kx * dx;
  double speed = 1;//speed of wave is 1 since P = 0.6 and gamma = 1.666667
  double phase = kx*0.5 - speed * time * real_kx; //kx*0.5 for half-cell offset
  double tolerance = 1e-7;
  systemTest::systemTestRunAndLoad(testDataFile);
  systemTest::systemTestDatasetIsSinusoid(testDataFile,"density",1.0,amplitude,kx,0.0,0.0,phase,tolerance);
}
