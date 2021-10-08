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
