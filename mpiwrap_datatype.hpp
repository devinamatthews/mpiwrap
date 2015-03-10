#ifndef _MPIWRAP_DATATYPE_HPP_
#define _MPIWRAP_DATATYPE_HPP_

#include "mpiwrap_common.hpp"

namespace MPIWrap
{

template <typename T>
struct MPI_TYPE_ {};

template <>
struct MPI_TYPE_<float>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_FLOAT; }
};

template <>
struct MPI_TYPE_<double>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_DOUBLE; }
};

template <>
struct MPI_TYPE_<long double>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_LONG_DOUBLE; }
};

#ifdef MPIWRAP_COMPLEX

template <>
struct MPI_TYPE_< std::complex<float> >
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_COMPLEX; }
};

template <>
struct MPI_TYPE_< std::complex<double> >
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_DOUBLE_COMPLEX; }
};

#endif

template <>
struct MPI_TYPE_<void>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_BYTE; }
};

template <>
struct MPI_TYPE_<char>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_CHAR; }
};

#if MPIWRAP_VERSION_AT_LEAST(2,0)
template <>
struct MPI_TYPE_<signed char>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_SIGNED_CHAR; }
};
#endif

template <>
struct MPI_TYPE_<unsigned char>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_UNSIGNED_CHAR; }
};

#if MPIWRAP_VERSION_AT_LEAST(2,0)
template <>
struct MPI_TYPE_<wchar_t>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_WCHAR; }
};
#endif

template <>
struct MPI_TYPE_<short>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_SHORT; }
};

template <>
struct MPI_TYPE_<unsigned short>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_UNSIGNED_SHORT; }
};

template <>
struct MPI_TYPE_<int>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_INT; }
};

template <>
struct MPI_TYPE_<unsigned int>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_UNSIGNED; }
};

template <>
struct MPI_TYPE_<long>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_LONG; }
};

template <>
struct MPI_TYPE_<unsigned long>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_UNSIGNED_LONG; }
};

#ifdef MPIWRAP_LONG_LONG

template <>
struct MPI_TYPE_<long long>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_LONG_LONG; }
};

template <>
struct MPI_TYPE_<unsigned long long>
{
    static MPIWRAP_CONSTEXPR MPI_Datatype& value() { return MPI_UNSIGNED_LONG_LONG; }
};

#endif

}

#endif
