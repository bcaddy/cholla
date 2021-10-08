
/*!
 * \file testing_utilites.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Implementation file for various testing related utility functions
 *
 */

// STL includes
#include <limits>
#include <cmath>
#include <cstring>

// Local includes
#include "../utils/testing_utilities.h" // Include the header file

namespace testingUtilities
{
    // =========================================================================
    int64_t ulpsDistanceDbl(double const &a, double const &b)
    {
        // Save work if the floats are equal.
        // Also handles +0 == -0
        if (a == b) return 0;

        const auto maxInt = std::numeric_limits<int64_t>::max();

        // If either one is NaN then they are not equal, max distance.
        if (std::isnan(a) || std::isnan(b)) return maxInt;

        // If one's infinite and they're not equal, max distance.
        if (std::isinf(a) || std::isinf(b)) return maxInt;

        int64_t ia, ib;
        std::memcpy(&ia, &a, sizeof(double));
        std::memcpy(&ib, &b, sizeof(double));

        // Don't compare differently-signed floats.
        if ((ia < 0) != (ib < 0)) return maxInt;

        // Return the absolute value of the distance in ULPs.
        int64_t distance = ia - ib;
        if (distance < 0) distance = -distance;

        return distance;
    }
    // =========================================================================

    // =========================================================================
    bool nearlyEqualDbl(double  const &a,
                        double  const &b,
                        double  &absoluteDiff,
                        int64_t &ulpsDiff,
                        int     const &ulpsEpsilon,  // = 4 by default
                        double  const &fixedEpsilon) // = 1E-14 by default
    {
        // Handle the near-zero case and pass back the absolute difference
        absoluteDiff = std::abs(a - b);
        if (absoluteDiff <= fixedEpsilon)
            return true;

        // Handle all other cases and pass back the difference in ULPs
        ulpsDiff = ulpsDistanceDbl(a, b);
        return ulpsDiff <= ulpsEpsilon;
    }
    // =========================================================================
}