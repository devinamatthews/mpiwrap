AC_INIT(mpiwrap, 1.0, dmatthews@utexas.edu, mpiwrap, http://www.github.com/devinamatthews/mpiwrap)

AC_CONFIG_MACRO_DIR([m4])

AX_PROG_CXX_MPI([true], [], [AC_MSG_FAILURE([No MPI C++ compiler found.])])
AC_LANG([C++])

AC_ARG_ENABLE([c++11],
              [AS_HELP_STRING([--disable-c++11],
                              [Disable the use of C++11 features.
                               These features are enable by default
                               if the compiler supports them.])])
                               
AS_IF([test "x$enable_c__11" != "xno"],
[
    AX_CXX_COMPILE_STDCXX_11([noext], [optional])
], [HAVE_CXX11=0])

AC_CHECK_TYPE([MPI_Count], [HAVE_MPI_COUNT=1], [HAVE_MPI_COUNT=0], [[#include "mpi.h"]])
AC_SUBST([HAVE_MPI_COUNT])

AC_CONFIG_FILES([
include/mpiwrap.hpp
include/internal/mpiwrap_common.hpp:include/internal/mpiwrap_common.hpp
include/internal/mpiwrap_datatype.hpp:include/internal/mpiwrap_datatype.hpp
include/internal/mpiwrap_status.hpp:include/internal/mpiwrap_status.hpp
include/internal/mpiwrap_request.hpp:include/internal/mpiwrap_request.hpp
include/internal/mpiwrap_comm.hpp:include/internal/mpiwrap_comm.hpp
include/internal/mpiwrap_intracomm.hpp:include/internal/mpiwrap_intracomm.hpp
include/internal/mpiwrap_intercomm.hpp:include/internal/mpiwrap_intercomm.hpp
])
AC_OUTPUT
