#ifndef _MPIWRAP_INTERCOMM_HPP_
#define _MPIWRAP_INTERCOMM_HPP_

#include "mpiwrap_common.hpp"
#include "mpiwrap_datatype.hpp"
#include "mpiwrap_status.hpp"
#include "mpiwrap_request.hpp"
#include "mpiwrap_comm.hpp"

namespace MPIWrap
{

class Intercomm : public Comm
{
    protected:
        static int getRemoteSize(const MPI_Comm& comm)
        {
            MPI_Int size;
            MPIWRAP_CALL(MPI_Comm_remote_size(comm, &size));
            return size;
        }

    public:
        const MPI_Int remoteSize;

        Intercomm()
        : remoteSize(getRemoteSize(comm)) {}

        Intercomm(const MPI_Comm& comm)
        : Comm(comm), remoteSize(getRemoteSize(comm)) {}

        Group remoteGroup() const
        {
            Group g;
            MPIWRAP_CALL(MPI_Comm_remote_group(comm, g));
            return g;
        }

        Comm merge(bool high) const
        {
            Comm c;
            MPIWRAP_CALL(MPI_Intercomm_merge(comm, high, c));
            return c;
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
