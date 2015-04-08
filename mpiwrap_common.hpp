#ifndef _MPIWRAP_COMMON_HPP_
#define _MPIWRAP_COMMON_HPP_

#include "mpi.h"

#include <complex>
#include <vector>
#include <cassert>
#include <numeric>
#include <cwchar>

#ifndef MPIWRAP_CPP11
#if __cplusplus >= 201103L
#define MPIWRAP_CPP11 1
#else
#define MPIWRAP_CPP11 0
#endif
#endif

#if MPIWRAP_CPP11
#include <type_traits>
#define MPIWRAP_CONSTEXPR constexpr
#else
#define MPIWRAP_CONSTEXPR const
#endif

#ifndef MPIWRAP_ASSERT
#define MPIWRAP_ASSERT(cond,message) assert(cond)
#endif

#ifndef MPIWRAP_MPI_INT
#define MPIWRAP_MPI_INT int
#endif

#ifndef MPIWRAP_INT
#define MPIWRAP_INT long
#endif

typedef MPIWRAP_MPI_INT MPI_Int;
typedef MPIWRAP_INT MPIWrap_Int;

//TODO: actually use MPIWrap_Int

#define MPIWRAP_CALL(...) \
{ \
    MPI_Int info = __VA_ARGS__; \
    MPIWRAP_ASSERT(info == 0, "Non-zero return from MPI function."); \
}

#define MPIWRAP_VERSION_AT_LEAST(major,minor) \
    ((MPI_VERSION > major) || \
    (MPI_VERSION == major && MPI_SUBVERSION >= minor))

namespace MPIWrap
{
    template <typename T>
    T* nc(const T* x)
    {
        return const_cast<T*>(x);
    }

    template <typename T>
    typename std::enable_if<!std::is_pointer<typename std::remove_reference<T>::type>::value,T&&>::type
    nc(const T&& x)
    {
        return const_cast<T&&>(x);
    }

    enum Collective
    {
        ALLGATHER,
        ALLGATHERV,
        ALLTOALL,
        ALLTOALLV,
        ALLTOALLW,
        ALLREDUCE,
        BARRIER,
        BCAST,
        GATHER,
        GATHERV,
        REDUCE,
        REDUCE_SCATTER,
        REDUCE_SCATTER_BLOCK,
        SCATTER,
        SCATTERV,
        IALLGATHER,
        IALLGATHERV,
        IALLTOALL,
        IALLTOALLV,
        IALLTOALLW,
        IALLREDUCE,
        IBARRIER,
        IBCAST,
        IGATHER,
        IGATHERV,
        IREDUCE,
        IREDUCE_SCATTER,
        IREDUCE_SCATTER_BLOCK,
        ISCATTER,
        ISCATTERV,
        NUM_COLLECTIVES
    };

    inline MPI_Int Tag_UB()
    {
        static const MPI_Int* ub = NULL;
        if (!ub)
        {
            MPI_Int flag;
            MPIWRAP_CALL(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &flag));
            MPIWRAP_ASSERT(flag, "Error in MPI_Comm_get_attr(MPI_TAG_UB)");
        }
        return *ub-NUM_COLLECTIVES;
    }
}

#endif
