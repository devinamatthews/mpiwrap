#ifndef _MPIWRAP_STATUS_HPP_
#define _MPIWRAP_STATUS_HPP_

#include "mpiwrap_common.hpp"
#include "mpiwrap_datatype.hpp"

namespace MPIWrap
{

class Status
{
    friend class Intracomm;

    protected:
        MPI_Status status;

        Status(const MPI_Status& status) : status(status) {}

    public:
        operator MPI_Status&() { return status; }

        operator const MPI_Status&() const { return status; }

        operator MPI_Status*() { return &status; }

        operator const MPI_Status*() const { return &status; }

        MPI_Int source() const
        {
            return status.MPI_SOURCE;
        }

        MPI_Int tag() const
        {
            return status.MPI_TAG;
        }

        template <typename T>
        MPI_Int count() const
        {
            return count(MPI_TYPE_<T>::value());
        }

        MPI_Int count(const MPI_Datatype& type) const
        {
            MPI_Int n;
            MPIWRAP_CALL(MPI_Get_count(&status, type, &n));
            return n;
        }

        bool cancelled() const
        {
            MPI_Int flag;
            MPIWRAP_CALL(MPI_Test_cancelled(&status, &flag));
            return flag;
        }
};

}

#endif
