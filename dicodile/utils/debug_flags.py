
# Set the debug flags to True when testing dicod.
import os
TESTING_DICOD = os.environ.get("TESTING_DICOD", "0") == "1"


# Start interactive child processes when set to True
INTERACTIVE_PROCESSES = (
    os.environ.get("DICODILE_INTERACTIVE_WORKERS", "0") == "1"
)

# If set to True, check that inactive segments do not have any coefficient
# with update over tol.
CHECK_ACTIVE_SEGMENTS = TESTING_DICOD


# If set to True, check that the updates selected have indeed an impact only
# on the coefficients that are contained in the worker.
CHECK_UPDATE_CONTAINED = TESTING_DICOD


# If set to True, check that beta is consistent with z_hat after each update
# from a neighbor.
CHECK_BETA = TESTING_DICOD


# If set to True, request the full z_hat from each worker. It should not change
# the resulting solution.
GET_OVERLAP_Z_HAT = TESTING_DICOD


# If set to True, check that the computed beta are consistent on neighbor
# workers when initiated with z_0 != 0
CHECK_WARM_BETA = TESTING_DICOD


# If set to True, check that the computed beta are consistent on neighbor
# workers at the end of the algorithm
CHECK_FINAL_BETA = TESTING_DICOD
