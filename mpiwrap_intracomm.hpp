#ifndef _MPIWRAP_INTRACOMM_HPP_
#define _MPIWRAP_INTRACOMM_HPP_

#include "mpiwrap_common.hpp"
#include "mpiwrap_datatype.hpp"
#include "mpiwrap_status.hpp"
#include "mpiwrap_request.hpp"
#include "mpiwrap_comm.hpp"

namespace MPIWrap
{

class Intracomm : public Comm
{
    public:
        Intracomm() {}

        Intracomm(const MPI_Comm& comm) : Comm(comm) {}

        /*
         * MPI_Allgather
         */

        template <typename T>
        void Allgather(const T* sendbuf, T* recvbuf, MPI_Int count) const
        {
            Allgather(sendbuf, recvbuf, count, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf) const
        {
            Allgather(sendbuf, recvbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Allgather(sendbuf, count, type, recvbuf, count, type, comm));
        }

        template <typename T>
        void Allgather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == recvbuf.size(),
                           "Send and receive buffers must be the same size.");
            Allgather(&sendbuf.front(), &recvbuf.front(), sendbuf.size(), type);
        }

        /*
         * MPI_Allgather in-place
         */

        template <typename T>
        void Allgather(T* recvbuf, MPI_Int count) const
        {
            Allgather(recvbuf, count, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(std::vector<T>& recvbuf) const
        {
            Allgather(recvbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(T* recvbuf, MPI_Int count, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Allgather(MPI_IN_PLACE, 0, type, recvbuf, count, type, comm));
        }

        template <typename T>
        void Allgather(std::vector<T>& recvbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvbuf.size()%size == 0,
                           "Receive buffer size must be a multiple of the communicator size.");
            Allgather(&recvbuf.front(), recvbuf.size()/size, type);
        }

        /*
         * MPI_Allgatherv
         */

        template <typename T>
        void Allgather(const T* sendbuf, MPI_Int sendcount, T* recvbuf, const MPI_Int* recvcounts) const
        {
            Allgather(sendbuf, sendcount, recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf,
                       const std::vector<MPI_Int>& recvcounts) const
        {
            Allgather(sendbuf, recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(const T* sendbuf, MPI_Int sendcount, T* recvbuf, const MPI_Int* recvcounts,
                       const MPI_Datatype& type) const
        {
            std::vector<MPI_Int> displs = displacements(recvcounts);
            Allgather(sendbuf, sendcount, recvbuf, recvcounts, recvdispls, type);
        }

        template <typename T>
        void Allgather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts,
                       const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvbuf.size() == sum(recvcounts),
                           "The receive buffer size must equal the sum of the receive counts.");
            Allgather(sendbuf, recvbuf, recvcounts, displacements(recvcounts), type);
        }

        template <typename T>
        void Allgather(const T* sendbuf, MPI_Int sendcount, T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls) const
        {
            Allgather(sendbuf, sendcount, recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf,
                       const std::vector<MPI_Int>& recvcounts, const std::vector<MPI_Int>& recvdispls) const
        {
            Allgather(sendbuf, recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(const T* sendbuf, MPI_Int sendcount, T* recvbuf, const MPI_Int* recvcounts,
                       const MPI_Int* recvdispls, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Allgatherv(sendbuf, sendcount, type, recvbuf, recvcounts, recvdispls, type, comm));
        }

        template <typename T>
        void Allgather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts,
                       const std::vector<MPI_Int>& recvdispls, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvcounts.size() == size,
                           "There must be exactly one receive count for each process.");
            MPIWRAP_ASSERT(recvcounts.size() == size,
                           "There must be exactly one receive displacement for each process.");
            MPIWRAP_ASSERT(recvbuf.size() >= recvdispls[size-1]+recvcounts[size-1],
                           "The receive buffer size must be at least as large the sum of last receive count and the last receive displacement.");
            Allgather(&sendbuf.front(), sendbuf.size(), &recvbuf.front(), &recvcounts.front(). &recvdispls.front(), type);
        }

        /*
         * MPI_Allgatherv in-place
         */

        template <typename T>
        void Allgather(T* recvbuf, const MPI_Int* recvcounts) const
        {
            Allgather(recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts) const
        {
            Allgather(recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(T* recvbuf, const MPI_Int* recvcounts, const MPI_Datatype& type) const
        {
            std::vector<MPI_Int> displs = displacements(recvcounts);
            Allgather(recvbuf, recvcounts, displs, type);
        }

        template <typename T>
        void Allgather(std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvbuf.size() == sum(recvcounts),
                           "The receive buffer size must equal the sum of the receive counts.");
            Allgather(recvbuf, recvcounts, displacements(recvcounts), type);
        }

        template <typename T>
        void Allgather(T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls) const
        {
            Allgather(recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts, const std::vector<MPI_Int>& recvdispls) const
        {
            Allgather(recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allgather(T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Allgatherv(MPI_IN_PLACE, 0, type, recvbuf, recvcounts, recvdispls, type, comm));
        }

        template <typename T>
        void Allgather(std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts, const std::vector<MPI_Int>& recvdispls, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvcounts.size() == size,
                           "There must be exactly one receive count for each process.");
            MPIWRAP_ASSERT(recvdispls.size() == size,
                           "There must be exactly one receive displacement for each process.");
            MPIWRAP_ASSERT(recvbuf.size() >= recvdispls[size-1]+recvcounts[size-1],
                           "The receive buffer size must be at least as large the sum of last receive count and the last receive displacement.");
            Allgather(&recvbuf.front(), &recvcounts.front(), &recvdispls.front(), type);
        }

        /*
         * MPI_Allreduce
         */

        template <typename T>
        void Allreduce(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Op& op) const
        {
            Allreduce(sendbuf, recvbuf, count, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allreduce(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Op& op) const
        {
            Allreduce(sendbuf, recvbuf, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allreduce(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Allreduce(sendbuf, recvbuf, count, type, op, comm));
        }

        template <typename T>
        void Allreduce(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Op& op,
                       const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == recvbuf.size(),
                           "Send and receive buffers must be the same size.");
            Allreduce(&sendbuf.front(), &recvbuf.front(), sendbuf.size(), op, type);
        }

        /*
         * MPI_Allreduce in-place
         */

        template <typename T>
        void Allreduce(T* buf, MPI_Int count, const MPI_Op& op) const
        {
            Allreduce(buf, count, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allreduce(std::vector<T>& buf, const MPI_Op& op) const
        {
            Allreduce(buf, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Allreduce(T* buf, MPI_Int count, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Allreduce(MPI_IN_PLACE, buf, count, type, op, comm));
        }

        template <typename T>
        void Allreduce(std::vector<T>& buf, const MPI_Op& op, const MPI_Datatype& type) const
        {
            Allreduce(&buf.front(), buf.size(), op, type));
        }

        /*
         * MPI_Alltoall
         */

        template <typename T>
        void Alltoall(const T* sendbuf, T* recvbuf, MPI_Int count) const
        {
            Alltoall(sendbuf, recvbuf, count, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Alltoall(const std::vector<T>& sendbuf, std::vector<T>& recvbuf) const
        {
            Alltoall(sendbuf, recvbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Alltoall(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Alltoall(sendbuf, count, type, recvbuf, count, type, comm));
        }

        template <typename T>
        void Alltoall(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == recvbuf.size(),
                           "Send and receive buffers must be the same size.");
            MPIWRAP_ASSERT(recvbuf.size()%size == 0,
                           "Receive buffer size must be a multiple of the communicator size.");
            Alltoall(&sendbuf.front(), &recvbuf.front(), recvbuf.size()/size, type);
        }

        /*
         * MPI_Alltoallv
         */

        template <typename T>
        void Alltoall(const T* sendbuf, const MPI_Int* sendcounts, T* recvbuf, const MPI_Int* recvcounts) const
        {
            Alltoall(sendbuf, sendcounts, recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Alltoall(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts,
                            std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts) const
        {
            Alltoall(sendbuf, sendcounts, recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Alltoall(const T* sendbuf, const MPI_Int* sendcounts,
                            T* recvbuf, const MPI_Int* recvcounts, const MPI_Datatype& type) const
        {
            std::vector<MPI_Int> senddispls = displacements(sendcounts);
            std::vector<MPI_Int> recvdispls = displacements(recvcounts);
            Alltoall(sendbuf, sendcounts, senddispls,
                     recvbuf, recvcounts, recvdispls, type);
        }

        template <typename T>
        void Alltoall(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts,
                            std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts,
                      const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == sum(sendcounts),
                           "The send buffer size must equal the sum of the send counts.");
            MPIWRAP_ASSERT(recvbuf.size() == sum(recvcounts),
                           "The receive buffer size must equal the sum of the receive counts.");
            Alltoall(sendbuf, sendcounts, displacements(sendcounts),
                     recvbuf, recvcounts, displacements(recvcounts), type);
        }

        template <typename T>
        void Alltoall(const T* sendbuf, const MPI_Int* sendcounts, const MPI_Int* senddispls,
                            T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls) const
        {
            Alltoall(sendbuf, sendcounts, senddispls,
                     recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Alltoall(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts, const std::vector<MPI_Int>& senddispls,
                            std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts, const std::vector<MPI_Int>& recvdispls) const
        {
            Alltoall(sendbuf, sendcounts, senddispls,
                     recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Alltoall(const T* sendbuf, const MPI_Int* sendcounts, const MPI_Int* senddispls,
                            T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Alltoallv(sendbuf, sendcounts, senddispls, type,
                                       recvbuf, recvcounts, recvdispls, type, comm));
        }

        template <typename T>
        void Alltoall(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts, const std::vector<MPI_Int>& senddispls,
                            std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts, const std::vector<MPI_Int>& recvdispls,
                      const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendcounts.size() == size,
                           "There must be exactly one send count for each process.");
            MPIWRAP_ASSERT(recvcounts.size() == size,
                           "There must be exactly one receive count for each process.");
            MPIWRAP_ASSERT(senddispls.size() == size,
                           "There must be exactly one send displacement for each process.");
            MPIWRAP_ASSERT(recvdispls.size() == size,
                           "There must be exactly one receive displacement for each process.");
            MPIWRAP_ASSERT(sendbuf.size() >= senddispls[size-1]+sendcounts[size-1],
                           "The send buffer size must be at least as large the sum of last send count and the last send displacement.");
            MPIWRAP_ASSERT(recvbuf.size() >= recvdispls[size-1]+recvcounts[size-1],
                           "The receive buffer size must be at least as large the sum of last receive count and the last receive displacement.");
            Alltoall(&sendbuf.front(), &sendcounts.front(), &senddispls.front(),
                     &recvbuf.front(), &recvcounts.front(), &recvdispls.front(), type);
        }

        /*
         * MPI_Barrier
         */

        void Barrier() const
        {
            MPIWRAP_CALL(MPI_Barrier(comm));
        }

        /*
         * MPI_Bcast
         */

        template <typename T>
        void Bcast(T* buffer, MPI_Int count, MPI_Int root) const
        {
            Bcast(buffer, count, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Bcast(std::vector<T>& buffer, MPI_Int root) const
        {
            Bcast(buffer, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Bcast(T* buffer, MPI_Int count, MPI_Int root, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Bcast(buffer, count, type, root, comm));
        }

        template <typename T>
        void Bcast(std::vector<T>& buffer, MPI_Int root, const MPI_Datatype& type) const
        {
            Bcast(&buffer.front(), buffer.size(), root, type);
        }

        /*
         * MPI_Exscan
         */

        template <typename T>
        void Exscan(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Op& op) const
        {
            Exscan(sendbuf, recvbuf, count, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Exscan(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Op& op) const
        {
            Exscan(sendbuf, recvbuf, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Exscan(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Exscan(sendbuf, recvbuf, count, type, op, comm));
        }

        template <typename T>
        void Exscan(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == recvbuf.size(),
                           "Send and receive buffers must be the same size.");
            Exscan(&sendbuf.front(), &recvbuf.begin(), sendbuf.size(), op, type);
        }

        /*
         * MPI_Gather non-root
         */

        template <typename T>
        void Gather(const T* sendbuf, MPI_Int count, MPI_Int root) const
        {
            Gather(sendbuf, count, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(const std::vector<T>& sendbuf, MPI_Int root) const
        {
            Gather(sendbuf, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(const T* sendbuf, MPI_Int count, MPI_Int root, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Gather(sendbuf, count, type, NULL, 0, type, root, comm));
        }

        template <typename T>
        void Gather(const std::vector<T>& sendbuf, MPI_Int root, const MPI_Datatype& type) const
        {
            Gather(&sendbuf.front(), sendbuf.size(), root, type);
        }

        /*
         * MPI_Gather root
         */

        template <typename T>
        void Gather(const T* sendbuf, T* recvbuf, MPI_Int count) const
        {
            Gather(sendbuf, recvbuf, count, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf) const
        {
            Gather(sendbuf, recvbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Gather(sendbuf, count, type, recvbuf, count, type, rank, comm));
        }

        template <typename T>
        void Gather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(size*sendbuf.size() == recvbuf.size(),
                           "Receive buffers size must be equal communicator size times send buffer size.");
            Gather(&sendbuf.front(), &recvbuf.front(), sendbuf.size(), type);
        }

        /*
         * MPI_Gather root in-place
         */

        template <typename T>
        void Gather(T* recvbuf, MPI_Int count) const
        {
            Gather(recvbuf, count, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(std::vector<T>& recvbuf) const
        {
            Gather(recvbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(T* recvbuf, MPI_Int count, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Gather(MPI_IN_PLACE, 0, type, recvbuf, count, type, rank, comm));
        }

        template <typename T>
        void Gather(std::vector<T>& recvbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvbuf.size()%size == 0,
                           "Receive buffer size must be a multiple of the communicator size.");
            Gather(&recvbuf.front(), recvbuf.size()/size, type);
        }

        /*
         * MPI_Gatherv non-root
         */

        template <typename T>
        void Gatherv(const T* sendbuf, MPI_Int sendcount, MPI_Int root) const
        {
            Gatherv(sendbuf, sendcount, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gatherv(const std::vector<T>& sendbuf, MPI_Int root) const
        {
            Gatherv(sendbuf, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gatherv(const T* sendbuf, MPI_Int sendcount, MPI_Int root, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Gatherv(sendbuf, sendcount, type, NULL, NULL, NULL, type, root, comm));
        }

        template <typename T>
        void Gatherv(const std::vector<T>& sendbuf, MPI_Int root, const MPI_Datatype& type) const
        {
            Gatherv(&sendbuf.front(), sendbuf.size(), root, type);
        }

        /*
         * MPI_Gatherv root
         */

        template <typename T>
        void Gather(const T* sendbuf, MPI_Int sendcount, T* recvbuf, const MPI_Int* recvcounts) const
        {
            Gather(sendbuf, sendcount, recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf,
                    const std::vector<MPI_Int>& recvcounts) const
        {
            Gather(sendbuf, recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(const T* sendbuf, MPI_Int sendcount, T* recvbuf, const MPI_Int* recvcounts,
                     const MPI_Datatype& type) const
        {
            std::vector<MPI_Int> recvdispls = displacements(recvcounts);
            Gather(sendbuf, recvbuf, recvcounts, &recvdispls.front(), type);
        }

        template <typename T>
        void Gather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts,
                    const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvbuf.size() == sum(recvcounts),
                           "The receive buffer size must equal the sum of the receive counts.");
            Gather(sendbuf, recvbuf, recvcounts, displacements(recvcounts), type);
        }

        template <typename T>
        void Gather(const T* sendbuf, MPI_Int sendcount, T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls) const
        {
            Gather(sendbuf, sendcount, recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf,
                    const std::vector<MPI_Int>& recvcounts, const std::vector<MPI_Int>& recvdispls) const
        {
            Gather(sendbuf, recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(const T* sendbuf, MPI_Int sendcount, T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls,
                     const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Gatherv(sendbuf, sendcount, type, recvbuf, recvcounts, recvdispls, type, rank, comm));
        }

        template <typename T>
        void Gather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts,
                    const std::vector<MPI_Int>& recvdispls, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvcounts.size() == size,
                           "There must be exactly one receive count for each process.");
            MPIWRAP_ASSERT(recvdispls.size() == size,
                           "There must be exactly one receive displacement for each process.");
            MPIWRAP_ASSERT(recvbuf.size() >= recvdispls[size-1]+recvcounts[size-1],
                           "The receive buffer size must be at least as large the sum of last receive count and the last receive displacement.");
            MPIWRAP_ASSERT(sendbuf.size() == recvcounts[rank],
                           "The send buffer size must equal the receive count for this process.");
            Gather(&sendbuf.front(), sendbuf.size(), &recvbuf.front(), &recvcounts.front(), &recvdispls.front(), type);
        }

        /*
         * MPI_Gatherv root in-place
         */

        template <typename T>
        void Gather(T* recvbuf, const MPI_Int* recvcounts) const
        {
            Gather(recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts) const
        {
            Gather(recvbuf, recvcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(T* recvbuf, const MPI_Int* recvcounts, const MPI_Datatype& type) const
        {
            std::vector<MPI_Int> recvdispls = displacements(recvcounts);
            Gather(recvbuf, recvcounts, recvdispls, type);
        }

        template <typename T>
        void Gather(std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvbuf.size() == sum(recvcounts),
                           "The receive buffer size must equal the sum of the receive counts.");
            Gather(recvbuf, recvcounts, displacements(recvcounts), type);
        }

        template <typename T>
        void Gather(T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls) const
        {
            Gather(recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts, const std::vector<MPI_Int>& recvdispls) const
        {
            Gather(recvbuf, recvcounts, recvdispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Gather(T* recvbuf, const MPI_Int* recvcounts, const MPI_Int* recvdispls, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Gatherv(MPI_IN_PLACE, 0, type, recvbuf, recvcounts, recvdispls, type, rank, comm));
        }

        template <typename T>
        void Gather(std::vector<T>& recvbuf, const std::vector<MPI_Int>& recvcounts, const std::vector<MPI_Int>& recvdispls, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvcounts.size() == size,
                           "There must be exactly one receive count for each process.");
            MPIWRAP_ASSERT(recvdispls.size() == size,
                           "There must be exactly one receive displacement for each process.");
            MPIWRAP_ASSERT(recvbuf.size() >= recvdispls[size-1]+recvcounts[size-1],
                           "The receive buffer size must be at least as large the sum of last receive count and the last receive displacement.");
            Gather(&recvbuf.front(), &recvcounts.front(), &recvdispls.front(), type);
        }

        /*
         * MPI_Iprobe
         */

        bool Iprobe(MPI_Int source, MPI_Int tag, Status& status) const
        {
            MPI_Int flag;
            MPIWRAP_CALL(MPI_Iprobe(source, tag, comm, &flag, status));
            return flag;
        }

        bool Iprobe(MPI_Int source, MPI_Int tag) const
        {
#if MPIWRAP_VERSION_AT_LEAST(2,0)
            MPI_Int flag;
            MPIWRAP_CALL(MPI_Iprobe(source, tag, comm, &flag, MPI_STATUS_IGNORE));
            return flag;
#else
            Status status;
            return Iprobe(source, tag, status);
#endif
        }

        /*
         * MPI_Irecv
         */

        template <typename T>
        Request Irecv(T* buf, MPI_Int count, MPI_Int source, MPI_Int tag) const
        {
            return Irecv(buf, count, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Irecv(std::vector<T>& buf, MPI_Int source, MPI_Int tag) const
        {
            return Irecv(buf, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Irecv(T* buf, MPI_Int count, MPI_Int source, MPI_Int tag, const MPI_Datatype& type) const
        {
            Request req;
            MPIWRAP_CALL(MPI_Irecv(buf, count, type, source, tag, comm, &req));
            return req;
        }

        template <typename T>
        Request Irecv(std::vector<T>& buf, MPI_Int source, MPI_Int tag, const MPI_Datatype& type) const
        {
            return Irecv(&buf.front(), buf.size(), source, tag, type);
        }

        /*
         * MPI_Irsend
         */

        template <typename T>
        Request Irsend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            return Irsend(buf, count, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Irsend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            return Irsend(buf, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Irsend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Request req;
            MPIWRAP_CALL(MPI_Irsend(buf, count, type, dest, tag, comm, &req));
            return req;
        }

        template <typename T>
        Request Irsend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            return Irsend(&buf.front(), buf.size(), dest, tag, type);
        }

        /*
         * MPI_Isend
         */

        template <typename T>
        Request Isend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            return Isend(buf, count, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Isend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            return Isend(buf, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Isend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Request req;
            MPIWRAP_CALL(MPI_Isend(buf, count, type, dest, tag, comm, &req));
            return req;
        }

        template <typename T>
        Request Isend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            return Isend(&buf.front(), buf.size(), dest, tag, type);
        }

        /*
         * MPI_Issend
         */

        template <typename T>
        Request Issend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            return Issend(buf, count, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Issend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            return Issend(buf, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Issend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Request req;
            MPIWRAP_CALL(MPI_Issend(buf, count, type, dest, tag, comm, &req));
            return req;
        }

        template <typename T>
        Request Issend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            return Issend(&buf.front(), buf.size(), dest, tag, type);
        }

        /*
         * MPI_Probe
         */

        void Probe(MPI_Int source, MPI_Int tag, Status& status) const
        {
            MPIWRAP_CALL(MPI_Probe(source, tag, comm, status));
        }

        void Probe(MPI_Int source, MPI_Int tag) const
        {
#if MPIWRAP_VERSION_AT_LEAST(2,0)
            MPIWRAP_CALL(MPI_Probe(source, tag, comm, MPI_STATUS_IGNORE));
#else
            Status status;
            Probe(source, tag, status);
#endif
        }

        /*
         * MPI_Recv
         */

        template <typename T>
        void Recv(T* buf, MPI_Int count, MPI_Int source, MPI_Int tag) const
        {
            Recv(buf, count, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Recv(std::vector<T>& buf, MPI_Int source, MPI_Int tag) const
        {
            Recv(buf, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Recv(T* buf, MPI_Int count, MPI_Int source, MPI_Int tag, const MPI_Datatype& type) const
        {
#if MPIWRAP_VERSION_AT_LEAST(2,0)
            MPIWRAP_CALL(MPI_Recv(buf, count, type, source, tag, comm, MPI_STATUS_IGNORE));
#else
            Status status;
            Recv(buf, count, source, tag, status, type);
#endif
        }

        template <typename T>
        void Recv(std::vector<T>& buf, MPI_Int source, MPI_Int tag, const MPI_Datatype& type) const
        {
            Recv(&buf.front(), buf.size(), source, tag, type);
        }

        template <typename T>
        void Recv(T* buf, MPI_Int count, MPI_Int source, MPI_Int tag, Status& status) const
        {
            Recv(buf, count, source, tag, status, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Recv(std::vector<T>& buf, MPI_Int source, MPI_Int tag, Status& status) const
        {
            Recv(buf, source, tag, status, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Recv(T* buf, MPI_Int count, MPI_Int source, MPI_Int tag, Status& status, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Recv(buf, count, type, source, tag, comm, status));
        }

        template <typename T>
        void Recv(std::vector<T>& buf, MPI_Int source, MPI_Int tag, Status& status, const MPI_Datatype& type) const
        {
            Recv(&buf.front(), buf.size(), source, tag, status, type);
        }

        /*
         * MPI_Recv_init
         */

        template <typename T>
        Request Recv_init(T* buf, MPI_Int count, MPI_Int source, MPI_Int tag) const
        {
            return Recv_init(buf, count, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Recv_init(std::vector<T>& buf, MPI_Int source, MPI_Int tag) const
        {
            return Recv_init(buf, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Recv_init(T* buf, MPI_Int count, MPI_Int source, MPI_Int tag, const MPI_Datatype& type) const
        {
            Request req;
            MPIWRAP_CALL(MPI_Recv_init(buf, count, type, source, tag, comm, &req));
            return req;
        }

        template <typename T>
        Request Recv_init(std::vector<T>& buf, MPI_Int source, MPI_Int tag, const MPI_Datatype& type) const
        {
            return Recv_init(&buf.front(), buf.size(), source, tag, type);
        }

        /*
         * MPI_Reduce non-root
         */

        template <typename T>
        void Reduce(const T* sendbuf, MPI_Int count, const MPI_Op& op, MPI_Int root) const
        {
            Reduce(sendbuf, count, op, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce(const std::vector<T>& sendbuf, const MPI_Op& op, MPI_Int root) const
        {
            Reduce(sendbuf, op, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce(const T* sendbuf, MPI_Int count, const MPI_Op& op, MPI_Int root, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Reduce(sendbuf, NULL, count, type, op, root, comm));
        }

        template <typename T>
        void Reduce(const std::vector<T>& sendbuf, const MPI_Op& op, MPI_Int root, const MPI_Datatype& type) const
        {
            Reduce(&sendbuf.front(), sendbuf.size(), op, root, type);
        }

        /*
         * MPI_Reduce root
         */

        template <typename T>
        void Reduce(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Op& op) const
        {
            Reduce(sendbuf, recvbuf, count, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Op& op) const
        {
            Reduce(sendbuf, recvbuf, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Reduce(sendbuf, recvbuf, count, type, op, rank, comm));
        }

        template <typename T>
        void Reduce(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == recvbuf.size(),
                           "Send and receive buffers must be the same size.");
            Reduce(&sendbuf.front(), &recvbuf.front(), sendbuf.size(), op, type);
        }

        /*
         * MPI_Reduce root in-place
         */

        template <typename T>
        void Reduce(T* recvbuf, MPI_Int count, const MPI_Op& op) const
        {
            Reduce(recvbuf, count, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce(std::vector<T>& recvbuf, const MPI_Op& op) const
        {
            Reduce(recvbuf, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce(T* recvbuf, MPI_Int count, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Reduce(MPI_IN_PLACE, recvbuf, count, type, op, rank, comm));
        }

        template <typename T>
        void Reduce(std::vector<T>& recvbuf, const MPI_Op& op, const MPI_Datatype& type) const
        {
            Reduce(&recvbuf.front(), recvbuf.size(), op, type);
        }

        /*
         * MPI_Reduce_scatter
         */

        template <typename T>
        void Reduce_scatter(const T* sendbuf, T* recvbuf, MPI_Int* recvcounts, const MPI_Op& op) const
        {
            Reduce_scatter(sendbuf, recvbuf, recvcounts, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce_scatter(const std::vector<T>& sendbuf, std::vector<T>& recvbuf,
                            std::vector<MPI_Int>& recvcounts, const MPI_Op& op) const
        {
            Reduce_scatter(sendbuf, recvbuf, recvcounts, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce_scatter(const T* sendbuf, T* recvbuf, MPI_Int* recvcounts, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, type, op, comm));
        }

        template <typename T>
        void Reduce_scatter(const std::vector<T>& sendbuf, std::vector<T>& recvbuf,
                            std::vector<MPI_Int>& recvcounts, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvcounts.size() == size,
                           "There must be exactly one receive count for each process.");
            MPIWRAP_ASSERT(sendbuf.size() == sum(recvcounts),
                           "The send buffer size must equal the sum of the receive counts.");
            MPIWRAP_ASSERT(recvbuf.size() == recvcounts[rank],
                           "The receive buffer size must equal the receive count for this process.");
            Reduce_scatter(&sendbuf.front(), &recvbuf.front(), &recvcounts.front(), op, type);
        }

        /*
         * MPI_Reduce_scatter in-place
         */

        template <typename T>
        void Reduce_scatter(T* recvbuf, MPI_Int* recvcounts, const MPI_Op& op) const
        {
            Reduce_scatter(recvbuf, recvcounts, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce_scatter(std::vector<T>& recvbuf, std::vector<MPI_Int>& recvcounts, const MPI_Op& op) const
        {
            Reduce_scatter(recvbuf, recvcounts, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Reduce_scatter(T* recvbuf, MPI_Int* recvcounts, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Reduce_scatter(MPI_IN_PLACE, recvbuf, recvcounts, type, op, comm));
        }

        template <typename T>
        void Reduce_scatter(std::vector<T>& recvbuf, std::vector<MPI_Int>& recvcounts, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(recvcounts.size() == size,
                           "There must be exactly one receive count for each process.");
            MPIWRAP_ASSERT(recvbuf.size() == sum(recvcounts),
                           "The send/receive buffer size must equal the sum of the receive counts.");
            Reduce_scatter(&recvbuf.front(), &recvcounts.front(), op, type);
        }

        /*
         * MPI_Rsend
         */

        template <typename T>
        void Rsend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            Rsend(buf, count, dest, tag);
        }

        template <typename T>
        void Rsend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            Rsend(buf, dest, tag);
        }

        template <typename T>
        void Rsend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Rsend(buf, count, type, dest, tag, comm));
        }

        template <typename T>
        void Rsend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Rsend(&buf.front(), buf.size(), dest, tag, type);
        }

        /*
         * MPI_Rsend_init
         */

        template <typename T>
        Request Rsend_init(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            return Rsend_init(buf, count, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Rsend_init(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            return Rsend_init(buf, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Rsend_init(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Request req;
            MPIWRAP_CALL(MPI_Rsend_init(buf, count, type, dest, tag, comm, &req));
            return req;
        }

        template <typename T>
        Request Rsend_init(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            return Rsend_init(&buf.front(), buf.size(), dest, tag, type);
        }

        /*
         * MPI_Scan
         */

        template <typename T>
        void Scan(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Op& op) const
        {
            Scan(sendbuf, recvbuf, count, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scan(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Op& op) const
        {
            Scan(sendbuf, recvbuf, op, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scan(const T* sendbuf, T* recvbuf, MPI_Int count, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Scan(sendbuf, recvbuf, count, type, op, comm));
        }

        template <typename T>
        void Scan(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Op& op, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == recvbuf.size(),
                           "Send and receive buffers must be the same size.");
            Scan(&sendbuf.front(), &recvbuf.front(), sendbuf.size(), op, type);
        }

        /*
         * MPI_Scatter non-root
         */

        template <typename T>
        void Scatter(T* recvbuf, MPI_Int recvcount, MPI_Int root) const
        {
            Scatter(recvbuf, recvcount, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(std::vector<T>& recvbuf, MPI_Int root) const
        {
            Scatter(recvbuf, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(T* recvbuf, MPI_Int recvcount, MPI_Int root, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Scatter(NULL, 0, type, recvbuf, recvcount, type, root, comm));
        }

        template <typename T>
        void Scatter(std::vector<T>& recvbuf, MPI_Int root, const MPI_Datatype& type) const
        {
            Scatter(&recvbuf.front(), recvbuf.size(), root, type);
        }

        /*
         * MPI_Scatter root
         */

        template <typename T>
        void Scatter(const T* sendbuf, T* recvbuf, MPI_Int recvcount) const
        {
            Scatter(sendbuf, recvbuf, recvcount, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, std::vector<T>& recvbuf) const
        {
            Scatter(sendbuf, recvbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const T* sendbuf, T* recvbuf, MPI_Int recvcount, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Scatter(sendbuf, recvcount, type, recvbuf, recvcount, type, rank, comm));
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(size*recvbuf.size() == sendbuf.size(),
                           "Send buffer size must be equal communicator size times send receive size.");
            Scatter(&sendbuf.front(), &recvbuf.front(), recvbuf.size(), type);
        }

        /*
         * MPI_Scatter root in-place
         */

        template <typename T>
        void Scatter(const T* sendbuf, MPI_Int sendcount) const
        {
            Scatter(sendbuf, sendcount, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf) const
        {
            Scatter(sendbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const T* sendbuf, MPI_Int sendcount, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Scatter(sendbuf, sendcount, type, MPI_IN_PLACE, 0, type, rank, comm));
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size()%size == 0,
                           "Send buffer size must be a multiple of the communicator size.");
            Scatter(&sendbuf.front(), sendbuf.size()/size, type);
        }

        /*
         * MPI_Scatterv non-root
         */

        template <typename T>
        void Scatterv(T* recvbuf, MPI_Int recvcount, MPI_Int root) const
        {
            Scatterv(recvbuf, recvcount, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatterv(std::vector<T>& recvbuf, MPI_Int root) const
        {
            Scatterv(recvbuf, root, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatterv(T* recvbuf, MPI_Int recvcount, MPI_Int root, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Scatterv(NULL, NULL, NULL, type, recvbuf, recvcount, type, root, comm));
        }

        template <typename T>
        void Scatterv(std::vector<T>& recvbuf, MPI_Int root, const MPI_Datatype& type) const
        {
            Scatterv(&recvbuf.front(), recvbuf.size(), root, type);
        }

        /*
         * MPI_Scatterv root
         */

        template <typename T>
        void Scatter(const T* sendbuf, const MPI_Int* sendcounts, T* recvbuf, MPI_Int recvcount) const
        {
            Scatter(sendbuf, sendcounts, recvbuf, recvcount, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts,
                     std::vector<T>& recvbuf) const
        {
            Scatter(sendbuf, sendcounts, recvbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const T* sendbuf, const MPI_Int* sendcounts, T* recvbuf, MPI_Int recvcount,
                      const MPI_Datatype& type) const
        {
            std::vector<MPI_Int> senddispls = displacements(sendcounts);
            Scatter(sendbuf, sendcounts, &senddispls.front(), recvbuf, recvcount, type);
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts,
                     std::vector<T>& recvbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == sum(sendcounts),
                           "Send buffer size must equal the sum of the send counts.");
            Scatter(sendbuf, sendcounts, displacements(sendcounts), recvbuf, type);
        }

        template <typename T>
        void Scatter(const T* sendbuf, const MPI_Int* sendcounts, const MPI_Int* senddispls, T* recvbuf, MPI_Int recvcount) const
        {
            Scatter(sendbuf, sendcounts, senddispls, recvbuf, recvcount, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts,
                     const std::vector<MPI_Int>& senddispls, std::vector<T>& recvbuf) const
        {
            Scatter(sendbuf, sendcounts, senddispls, recvbuf, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const T* sendbuf, const MPI_Int* sendcounts, const MPI_Int* senddispls, T* recvbuf, MPI_Int recvcount,
                      const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Scatterv(sendbuf, sendcounts, senddispls, type, recvbuf, recvcount, type, rank, comm));
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts,
                     const std::vector<MPI_Int>& senddispls, std::vector<T>& recvbuf, const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendcounts.size() == size,
                           "There must be exactly one send count for each process.");
            MPIWRAP_ASSERT(senddispls.size() == size,
                           "There must be exactly one send displacement for each process.");
            MPIWRAP_ASSERT(recvbuf.size() == sendcounts[rank],
                           "Receive buffer size must equal send count for this process.");
            MPIWRAP_ASSERT(sendbuf.size() >= senddispls[size-1]+sendcounts[size-1],
                           "Send buffer size must be at least the sum of the last send count and the last send displacement.");
            Scatter(&sendbuf.front(), &sendcounts.front(), &senddispls.front(), &recvbuf.front(), recvbuf.size(), type);
        }

        /*
         * MPI_Scatterv root in-place
         */

        template <typename T>
        void Scatter(const T* sendbuf, const MPI_Int* sendcounts) const
        {
            Scatter(sendbuf, sendcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts) const
        {
            Scatter(sendbuf, sendcounts, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const T* sendbuf, const MPI_Int* sendcounts, const MPI_Datatype& type) const
        {
            std::vector<MPI_Int> senddispls = displacements(sendcounts);
            Scatter(sendbuf, sendcounts, &senddispls.front(), type);
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts,
                     const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendbuf.size() == sum(sendcounts),
                           "Send buffer size must equal the sum of the send counts.");
            Scatter(sendbuf, sendcounts, displacements(sendcounts), type);
        }

        template <typename T>
        void Scatter(const T* sendbuf, const MPI_Int* sendcounts, const MPI_Int* senddispls) const
        {
            Scatter(sendbuf, sendcounts, senddispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts, const std::vector<MPI_Int>& senddispls) const
        {
            Scatter(sendbuf, sendcounts, senddispls, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Scatter(const T* sendbuf, const MPI_Int* sendcounts, const MPI_Int* senddispls, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Scatterv(sendbuf, sendcounts, senddispls, type, MPI_IN_PLACE, 0, type, rank, comm));
        }

        template <typename T>
        void Scatter(const std::vector<T>& sendbuf, const std::vector<MPI_Int>& sendcounts, const std::vector<MPI_Int>& senddispls,
                     const MPI_Datatype& type) const
        {
            MPIWRAP_ASSERT(sendcounts.size() == size,
                           "There must be exactly one send count for each process.");
            MPIWRAP_ASSERT(senddispls.size() == size,
                           "There must be exactly one send displacement for each process.");
            MPIWRAP_ASSERT(sendbuf.size() >= senddispls[size-1]+sendcounts[size-1],
                           "Send buffer size must be at least the sum of the last send count and the last send displacement.");
            Scatter(&sendbuf.front(), &sendcounts.front(), &senddispls.front(), type);
        }

        /*
         * MPI_Send
         */

        template <typename T>
        void Send(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            Send(buf, count, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Send(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            Send(buf, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Send(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Send(buf, count, type, dest, tag, comm));
        }

        template <typename T>
        void Send(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Send(&buf.front(), buf.size(), dest, tag, type);
        }

        /*
         * MPI_Send_init
         */

        template <typename T>
        Request Send_init(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            return Send_init(buf, count, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Send_init(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            return Send_init(buf, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Send_init(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Request req;
            MPIWRAP_CALL(MPI_Send_init(buf, count, type, dest, tag, comm, &req));
            return req;
        }

        template <typename T>
        Request Send_init(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            return Send_init(&buf.front(), buf.size(), dest, tag, type);
        }

        /*
         * MPI_Sendrecv
         */

        template <typename T>
        void Sendrecv(const T* sendbuf, MPI_Int sendcount, MPI_Int dest,
                            T* recvbuf, MPI_Int recvcount, MPI_Int source, MPI_Int tag) const
        {
            Sendrecv(sendbuf, sendcount, dest,
                     recvbuf, recvcount, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Sendrecv(const std::vector<T>& sendbuf, MPI_Int dest,
                            std::vector<T>& recvbuf, MPI_Int source, MPI_Int tag) const
        {
            Sendrecv(sendbuf, dest,
                     recvbuf, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Sendrecv(const T* sendbuf, MPI_Int sendcount, MPI_Int dest,
                            T* recvbuf, MPI_Int recvcount, MPI_Int source, MPI_Int tag,
                      const MPI_Datatype& type) const
        {
#if MPIWRAP_VERSION_AT_LEAST(2,0)
            MPIWRAP_CALL(MPI_Sendrecv(sendbuf, sendcount, type,   dest, tag,
                                      recvbuf, recvcount, type, source, tag, comm, MPI_STATUS_IGNORE));
#else
            Status status;
            Sendrecv(sendbuf, sendcount, dest,
                     recvbuf, recvcount, source, tag, status, type);
#endif
        }

        template <typename T>
        void Sendrecv(const std::vector<T>& sendbuf, MPI_Int dest,
                            std::vector<T>& recvbuf, MPI_Int source, MPI_Int tag,
                      const MPI_Datatype& type) const
        {
            Sendrecv(&sendbuf.front(), sendbuf.size(), dest,
                     &recvbuf.front(), recvbuf.size(), source, tag, type);
        }

        template <typename T>
        void Sendrecv(const T* sendbuf, MPI_Int sendcount, MPI_Int dest,
                            T* recvbuf, MPI_Int recvcount, MPI_Int source, MPI_Int tag, Status& status) const
        {
            Sendrecv(sendbuf, sendcount, dest,
                     recvbuf, recvcount, source, tag, status, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Sendrecv(const std::vector<T>& sendbuf, MPI_Int dest,
                            std::vector<T>& recvbuf, MPI_Int source, MPI_Int tag, Status& status) const
        {
            Sendrecv(sendbuf, dest,
                     recvbuf, source, tag, status, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Sendrecv(const T* sendbuf, MPI_Int sendcount, MPI_Int dest,
                            T* recvbuf, MPI_Int recvcount, MPI_Int source, MPI_Int tag, Status& status,
                      const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Sendrecv(sendbuf, sendcount, type,   dest, tag,
                                      recvbuf, recvcount, type, source, tag, comm, status));
        }

        template <typename T>
        void Sendrecv(const std::vector<T>& sendbuf, MPI_Int dest,
                            std::vector<T>& recvbuf, MPI_Int source, MPI_Int tag, Status& status,
                      const MPI_Datatype& type) const
        {
            Sendrecv(&sendbuf.front(), sendbuf.size(), dest,
                     &recvbuf.front(), recvbuf.size(), source, tag, status, type);
        }

        /*
         * MPI_Sendrecv_replace
         */

        template <typename T>
        void Sendrecv(T* buf, MPI_Int count, MPI_Int dest, MPI_Int source, MPI_Int tag) const
        {
            Sendrecv(buf, count, dest, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Sendrecv(std::vector<T>& buf, MPI_Int dest, MPI_Int source, MPI_Int tag) const
        {
            Sendrecv(buf, dest, source, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Sendrecv(T* buf, MPI_Int count, MPI_Int dest, MPI_Int source, MPI_Int tag, const MPI_Datatype& type) const
        {
#if MPIWRAP_VERSION_AT_LEAST(2,0)
            MPIWRAP_CALL(MPI_Sendrecv_replace(buf, count, type, dest, tag, source, tag, comm, MPI_STATUS_IGNORE));
#else
            Status status;
            Sendrecv(buf, count, dest, source, tag, status, type);
#endif
        }

        template <typename T>
        void Sendrecv(std::vector<T>& buf, MPI_Int dest, MPI_Int source, MPI_Int tag, const MPI_Datatype& type) const
        {
            Sendrecv(&buf.front(), buf.size(), dest, source, tag, type);
        }

        template <typename T>
        void Sendrecv(T* buf, MPI_Int count, MPI_Int dest, MPI_Int source, MPI_Int tag, Status& status) const
        {
            Sendrecv(buf, count, dest, source, tag, status, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Sendrecv(std::vector<T>& buf, MPI_Int dest, MPI_Int source, MPI_Int tag, Status& status) const
        {
            Sendrecv(buf, dest, source, tag, status, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Sendrecv(T* buf, MPI_Int count, MPI_Int dest, MPI_Int source, MPI_Int tag, Status& status,
                      const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Sendrecv_replace(buf, count, type, dest, tag, source, tag, comm, status));
        }

        template <typename T>
        void Sendrecv(std::vector<T>& buf, MPI_Int dest, MPI_Int source, MPI_Int tag, Status& status,
                      const MPI_Datatype& type) const
        {
            Sendrecv(&buf.front(), buf.size(), dest, source, tag, status, type);
        }

        /*
         * MPI_Ssend
         */

        template <typename T>
        void Ssend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            Ssend(buf, count, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Ssend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            Ssend(buf, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        void Ssend(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            MPIWRAP_CALL(MPI_Ssend(buf, count, type, dest, tag, comm));
        }

        template <typename T>
        void Ssend(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Ssend(&buf.front(). buf.size(), dest, tag, type);
        }

        /*
         * MPI_Ssend_init
         */

        template <typename T>
        Request Ssend_init(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag) const
        {
            return Ssend_init(buf, count, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Ssend_init(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag) const
        {
            return Ssend_init(buf, dest, tag, MPI_TYPE_<T>::value());
        }

        template <typename T>
        Request Ssend_init(const T* buf, MPI_Int count, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            Request req;
            MPIWRAP_CALL(MPI_Ssend_init(buf, count, type, dest, tag, comm, &req));
            return req;
        }

        template <typename T>
        Request Ssend_init(const std::vector<T>& buf, MPI_Int dest, MPI_Int tag, const MPI_Datatype& type) const
        {
            return Ssend_init(&buf.front(), buf.size(), dest, tag, type);
        }
};

}

#endif
