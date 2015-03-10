#if !defined(_MPIWRAP_INTRACOMM_HPP_) && !defined(_MPIWRAP_INTERCOMM_HPP_)
#error "This file should only be included through mpiwrap_intercomm.hpp or mpiwrap_intracomm.hpp."
#endif

/*
 * MPI_Exscan
 */

#if MPIWRAP_VERSION_AT_LEAST(2,0)

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

#endif

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
