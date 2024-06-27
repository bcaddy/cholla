/*!
 * \file cosmology_system_tests.cpp
 * \brief Contains all the system tests for the GRAVITY build type
 *
 */

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../system_tests/system_tester.h"

TEST(tCOSMOLOGYSYSTEM50Mpc, CorrectInputExpectCorrectOutput)
{
  system_test::SystemTestRunner cosmo(true, true, true, true, true);
  cosmo.runTest(true, 1.0e-07, 0.0006);
}
