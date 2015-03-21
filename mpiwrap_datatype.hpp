#ifndef _MPIWRAP_DATATYPE_HPP_
#define _MPIWRAP_DATATYPE_HPP_

#include "mpiwrap_common.hpp"

namespace MPIWrap
{

class Datatype
{
    protected:
        MPI_Datatype type;

    public:
        Datatype() : type(MPI_DATATYPE_NULL) {}

        explicit Datatype(const MPI_Datatype& type) : type(type) {}

        operator MPI_Datatype&() { return type; }

        operator const MPI_Datatype&() const { return type; }

        operator MPI_Datatype*() { return &type; }

        operator const MPI_Datatype*() const { return &type; }
};

template <typename T>
struct MPI_TYPE_ {};

template <>
struct MPI_TYPE_<float>
{
    static const Datatype value() { return Datatype(MPI_FLOAT); }
};

template <>
struct MPI_TYPE_<double>
{
    static const Datatype value() { return Datatype(MPI_DOUBLE); }
};

template <>
struct MPI_TYPE_<long double>
{
    static const Datatype value() { return Datatype(MPI_LONG_DOUBLE); }
};

#ifdef MPIWRAP_COMPLEX

template <>
struct MPI_TYPE_< std::complex<float> >
{
    static const Datatype value() { return Datatype(MPI_COMPLEX); }
};

template <>
struct MPI_TYPE_< std::complex<double> >
{
    static const Datatype value() { return Datatype(MPI_DOUBLE_COMPLEX); }
};

#endif

template <>
struct MPI_TYPE_<void>
{
    static const Datatype value() { return Datatype(MPI_BYTE); }
};

template <>
struct MPI_TYPE_<char>
{
    static const Datatype value() { return Datatype(MPI_CHAR); }
};

#if MPIWRAP_VERSION_AT_LEAST(2,0)
template <>
struct MPI_TYPE_<signed char>
{
    static const Datatype value() { return Datatype(MPI_SIGNED_CHAR); }
};
#endif

template <>
struct MPI_TYPE_<unsigned char>
{
    static const Datatype value() { return Datatype(MPI_UNSIGNED_CHAR); }
};

#if MPIWRAP_VERSION_AT_LEAST(2,0)
template <>
struct MPI_TYPE_<wchar_t>
{
    static const Datatype value() { return Datatype(MPI_WCHAR); }
};
#endif

template <>
struct MPI_TYPE_<short>
{
    static const Datatype value() { return Datatype(MPI_SHORT); }
};

template <>
struct MPI_TYPE_<unsigned short>
{
    static const Datatype value() { return Datatype(MPI_UNSIGNED_SHORT); }
};

template <>
struct MPI_TYPE_<int>
{
    static const Datatype value() { return Datatype(MPI_INT); }
};

template <>
struct MPI_TYPE_<unsigned int>
{
    static const Datatype value() { return Datatype(MPI_UNSIGNED); }
};

template <>
struct MPI_TYPE_<long>
{
    static const Datatype value() { return Datatype(MPI_LONG); }
};

template <>
struct MPI_TYPE_<unsigned long>
{
    static const Datatype value() { return Datatype(MPI_UNSIGNED_LONG); }
};

#ifdef MPIWRAP_LONG_LONG

template <>
struct MPI_TYPE_<long long>
{
    static const Datatype value() { return Datatype(MPI_LONG_LONG); }
};

template <>
struct MPI_TYPE_<unsigned long long>
{
    static const Datatype value() { return Datatype(MPI_UNSIGNED_LONG_LONG); }
};

#endif

}

#endif
