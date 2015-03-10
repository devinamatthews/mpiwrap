#ifndef _MPIWRAP_COMM_HPP_
#define _MPIWRAP_COMM_HPP_

#include "mpiwrap_common.hpp"
#include "mpiwrap_group.hpp"

namespace MPIWrap
{

class Comm
{
    protected:
        MPI_Comm comm;

        static int getRank(const MPI_Comm& comm)
        {
            MPI_Int rank;
            MPIWRAP_CALL(MPI_Comm_rank(comm, &rank));
            return rank;
        }

        static int getSize(const MPI_Comm& comm)
        {
            MPI_Int size;
            MPIWRAP_CALL(MPI_Comm_size(comm, &size));
            return size;
        }

        std::vector<MPI_Int> displacements(const MPI_Int* counts)
        {
            std::vector<MPI_Int> displs(size);
            std::partial_sum(counts, counts+size-1, displs.begin()+1);
            return displs;
        }

        std::vector<MPI_Int> displacements(const std::vector<MPI_Int>& counts)
        {
            return displacements(&counts.front());
        }

        MPI_Int sum(const std::vector<MPI_Int>& counts)
        {
            return std::accumulate(counts.begin(), counts.end(), 0);
        }

    public:
        const MPI_Int rank;
        const MPI_Int size;

        Comm()
        : comm(MPI_COMM_NULL), rank(0), size(0) {}

        Comm(const MPI_Comm& comm)
        : comm(comm), rank(getRank(comm)), size(getSize(comm)) {}

        operator MPI_Comm&() { return comm; }

        operator const MPI_Comm&() const { return comm; }

        operator MPI_Comm*() { return &comm; }

        operator const MPI_Comm*() const { return &comm; }

        Group group() const
        {
            Group g;
            MPIWRAP_CALL(MPI_Comm_group(comm, g));
            return g;
        }

        MPI_Int compare(const Comm& other) const
        {
            MPI_Int result;
            MPIWRAP_CALL(MPI_Comm_compare(comm, other, &result));
            return result;
        }

        Comm duplicate() const
        {
            Comm c;
            MPIWRAP_CALL(MPI_Comm_dup(comm, c));
            return c;
        }

        Comm subset(const Group& group) const
        {
            Comm c;
            MPIWRAP_CALL(MPI_Comm_create(comm, group, c));
            return c;
        }

        Comm split(MPI_Int color, MPI_Int key)
        {
            Comm c;
            MPIWRAP_CALL(MPI_Comm_split(comm, color, key, c));
            return c;
        }

        void free()
        {
            MPIWRAP_ASSERT(comm != MPI_COMM_NULL,
                           "You cannot free MPI_COMM_NULL.");
            MPIWRAP_ASSERT(comm != MPI_COMM_WORLD,
                           "You cannot free MPI_COMM_WORLD.");
            MPIWRAP_ASSERT(comm != MPI_COMM_SELF,
                           "You cannot free MPI_COMM_SELF.");
            MPIWRAP_CALL(MPI_Comm_free(&comm));
        }

        bool isIntercommunicator() const
        {
            MPI_Int flag;
            MPIWRAP_CALL(MPI_Comm_test_inter(comm, &flag));
            return flag;
        }

        Comm intercomm(MPI_Int leader) const
        {
            Comm c;
            MPIWRAP_CALL(MPI_Intercomm_create(comm, leader, MPI_COMM_NULL, MPI_PROC_NULL, MPI_ANY_TAG, c));
            return c;
        }

        Comm intercomm(MPI_Int leader, const Comm& peerComm, MPI_Int remoteLeader, MPI_Int tag) const
        {
            Comm c;
            MPIWRAP_CALL(MPI_Intercomm_create(comm, leader, peerComm, remoteLeader, tag, c));
            return c;
        }

        //TODO: attributes
};

}

#endif
