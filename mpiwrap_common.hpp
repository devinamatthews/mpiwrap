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
    T* nconst(const T* x)
    {
        return const_cast<T*>(x);
    }

#if MPIWRAP_CPP11

    template <typename T>
    typename std::enable_if<!std::is_pointer<typename std::remove_reference<T>::type>::value,T&&>::type
    nconst(const T&& x)
    {
        return const_cast<T&&>(x);
    }

#else

    template <typename T>
    T& nconst(const T& x)
    {
        return const_cast<T&>(x);
    }

#endif
}

#endif
