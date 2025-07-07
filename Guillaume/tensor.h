//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <numeric>
#include <initializer_list>
#include <stdexcept>
#include <functional>
#include <algorithm>

namespace utec::algebra {
    template<typename T, size_t N>
    class Tensor {
    public:
        Tensor() noexcept = default;
        template<typename... Dims>
        explicit Tensor(Dims... dims) {
            static_assert(sizeof...(Dims) == N,
                "Tensor constructor must be given exactly N dimensions");
            std::array<size_t,N> list{ static_cast<size_t>(dims)... };
            dims_ = list;
            data_.resize(total_size());
        }

        explicit Tensor(const std::array<size_t, N>& dims) : dims_(dims), data_(total_size()) {}

        std::array<size_t, N> shape() const noexcept {
            return dims_;
        }

        void fill (const T& value) {
            std::fill(data_.begin(), data_.end(), value);
        }

        template<typename... Idxs>
        T& operator()(Idxs... idxs) {
            static_assert(sizeof...(Idxs) == N, "Número de índices debe ser N");
            std::array<size_t, N> idx = {static_cast<size_t>(idxs)...};
            for (size_t i = 0; i < N; ++i) {
                if (idx[i] >= dims_[i])
                    throw std::out_of_range("Índice fuera de rango");
            }
            return data_[compute_index(idx)];
        }

        template<typename... Idxs>
        const T& operator()(Idxs... idxs) const {
            static_assert(sizeof...(Idxs) == N, "Número de índices debe ser N");
            std::array<size_t, N> idx = {static_cast<size_t>(idxs)...};
            for (size_t i = 0; i < N; ++i) {
                if (idx[i] >= dims_[i])
                    throw std::out_of_range("Índice fuera de rango");
            }
            return data_[compute_index(idx)];
        }

        Tensor& operator=(std::initializer_list<T> list) {
            if (list.size() != data_.size())
                throw std::invalid_argument("Data size does not match tensor size");
            std::copy(list.begin(), list.end(), data_.begin());
            return *this;
        }

        template<typename U>
        Tensor& operator=(std::initializer_list<std::initializer_list<U>> list) {
            static_assert(N == 2,
                "Initializer-list assignment only supported for 2D tensors");
            if (list.size() != dims_[0])
                throw std::invalid_argument("Data size does not match tensor size");
            size_t pos = 0;
            for (auto const& row : list) {
                if (row.size() != dims_[1])
                    throw std::invalid_argument("Data size does not match tensor size");
                for (auto const& v : row)
                    data_[pos++] = v;
            }
            return *this;
        }
        friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
            const auto& dims = t.dims_;
            const auto& data = t.data_;
            if constexpr (N == 1) {
                for (size_t i = 0; i < dims[0]; ++i) {
                    os << data[i];
                    if (i + 1 < dims[0]) os << ' ';
                }
                return os;
            }
            std::array<size_t, N> strides;
            strides[N-1] = 1;
            for (int i = static_cast<int>(N)-2; i >= 0; --i) {
                strides[i] = strides[i+1] * dims[i+1];
            }
            std::function<void(size_t, size_t)> rec = [&](size_t depth, size_t offset) {
                if (depth == N-1) {
                    for (size_t i = 0; i < dims[depth]; ++i) {
                        os << data[offset + i];
                        if (i + 1 < dims[depth]) os << ' ';
                    }
                    os << '\n';
                } else {
                    os << "{\n";
                    for (size_t i = 0; i < dims[depth]; ++i) {
                        rec(depth + 1, offset + i * strides[depth]);
                    }
                    os << "}\n";
                }
            };
            os << "{\n";
            for (size_t i = 0; i < dims[0]; ++i) {
                rec(1, i * strides[0]);
            }
            os << "}";
            return os;
        }
        // Pregunta 2
        template<typename... Dims>
        void reshape(Dims... dims) {
            std::initializer_list<size_t> list{ static_cast<size_t>(dims)... };

            if (list.size() != N)
                throw std::invalid_argument(
                    "Number of dimensions do not match with " + std::to_string(N)
                );

            size_t new_total = 1;
            for (auto d : list) new_total *= d;

            std::copy(list.begin(), list.end(), dims_.begin());

            data_.resize(new_total);
        }

        // Pregunta 3
        Tensor operator+(const Tensor& other) const {
            const auto A = this-> shape();
            const auto B = other.shape();
            if (!is_compatible(A, B))
                throw std::invalid_argument(
                    "Shapes do not match and they are not compatible for broadcasting"
                    );

            auto Rshape =broadcast_shape(A, B);
            Tensor result(Rshape);

            std::array<size_t, N> strides;
            strides[N-1] = 1;
            for(int i = static_cast<int>(N)-2; i >= 0; --i)
                strides[i] = strides[i+1] * Rshape[i+1];

            for (size_t lin = 0; lin < result.data_.size(); ++lin) {
                size_t rem = lin;
                std::array<size_t, N> idx;
                for (size_t i = 0; i < N; ++i) {
                    idx[i] = rem / strides[i];
                    rem %= strides[i];
                }
                std::array<size_t, N> idxA, idxB;
                for (size_t i = 0; i < N; ++i) {
                    idxA[i] = (A[i] == 1 ? 0 : idx[i]);
                    idxB[i] = (B[i] == 1 ? 0 : idx[i]);
                }
                result.data_[lin] =
                    this->data_[ compute_index(idxA) ] +
                        other.data_[ other.compute_index(idxB) ];
            }
            return result;
        }

        Tensor operator-(const Tensor& other) const {
            auto A = this->shape();
            auto B = other.shape();
            if (!is_compatible(A,B))
                throw std::invalid_argument(
                    "Shapes do not match and they are not compatible for broadcasting"
                );
            auto Rshape = broadcast_shape(A,B);
            Tensor result(Rshape);

            std::array<size_t,N> strides;
            strides[N-1] = 1;
            for (int i = static_cast<int>(N)-2; i >= 0; --i)
                strides[i] = strides[i+1] * Rshape[i+1];

            for (size_t lin = 0; lin < result.data_.size(); ++lin) {
                size_t rem = lin;
                std::array<size_t,N> idx, idxA, idxB;
                for (size_t i = 0; i < N; ++i) {
                    idx[i] = rem / strides[i];
                    rem %= strides[i];
                    idxA[i] = (A[i] == 1 ? 0 : idx[i]);
                    idxB[i] = (B[i] == 1 ? 0 : idx[i]);
                }
                result.data_[lin] =
                    this->data_[ compute_index(idxA) ] -
                    other .data_[ other.compute_index(idxB) ];
            }
            return result;
        }

        auto begin()        { return data_.begin(); }
        auto end()          { return data_.end(); }
        auto begin()  const { return data_.begin(); }
        auto end()    const { return data_.end(); }
        auto cbegin() const { return data_.cbegin(); }
        auto cend()   const { return data_.cend(); }

        [[nodiscard]] size_t size() const noexcept { return data_.size(); }

        // Pregunta 4
        Tensor operator*(const Tensor& other) const {
            auto A = this->shape();
            auto B = other.shape();
            if (!is_compatible(A,B))
                throw std::invalid_argument(
                    "Shapes do not match and they are not compatible for broadcasting"
                );
            auto Rshape = broadcast_shape(A,B);
            Tensor result(Rshape);

            std::array<size_t,N> strides;
            strides[N-1] = 1;
            for (int i = static_cast<int>(N)-2; i >= 0; --i)
                strides[i] = strides[i+1] * Rshape[i+1];

            for (size_t lin = 0; lin < result.data_.size(); ++lin) {
                size_t rem = lin;
                std::array<size_t,N> idx, idxA, idxB;
                for (size_t i = 0; i < N; ++i) {
                    idx[i] = rem / strides[i];
                    rem %= strides[i];
                    idxA[i] = (A[i] == 1 ? 0 : idx[i]);
                    idxB[i] = (B[i] == 1 ? 0 : idx[i]);
                }
                result.data_[lin] =
                    this->data_[ compute_index(idxA) ] *
                    other.data_[ other.compute_index(idxB) ];
            }
            return result;
        }

        Tensor operator*(const T& scalar) const {
            Tensor result(*this);
            for (size_t i = 0; i < result.data_.size(); ++i)
                result.data_[i] = result.data_[i] * scalar;
            return result;
        }

        Tensor operator/(const T& scalar) const {
            Tensor result(*this);
            for (size_t i = 0; i < result.data_.size(); ++i)
                result.data_[i] = result.data_[i] / scalar;
            return result;
        }

        Tensor operator+(const T& scalar) const {
            Tensor result(*this);
            for (size_t i = 0; i < result.data_.size(); ++i)
                result.data_[i] = result.data_[i] + scalar;
            return result;
        }

        Tensor operator-(const T& scalar) const {
            Tensor result(*this);
            for (size_t i = 0; i < result.data_.size(); ++i)
                result.data_[i] = result.data_[i] - scalar;
            return result;
        }

        // Pregunta 5
        static bool is_compatible(const std::array<size_t, N> & A,
                                  const std::array<size_t, N> & B) {
            for (size_t i = 0; i < N; ++i) {
                if (A[i] != B[i] && A[i] != 1 && B[i] != 1)
                    return false;
            }
            return true;
        }
        static std::array<size_t, N> broadcast_shape(
            const std::array<size_t,N> & A,
            const std::array<size_t,N> & B) {
            std::array<size_t, N> R;
            for (size_t i = 0; i < N; ++i)
                R[i] = std::max(A[i], B[i]);
            return R;
        }

    private:
        std::array<size_t, N> dims_;
        std::vector<T> data_;

        [[nodiscard]] size_t total_size() const noexcept {
            return std::accumulate(dims_.begin(), dims_.end(), size_t{1}, std::multiplies<>());
        }

        size_t compute_index(const std::array<size_t, N>& idx) const {
            size_t offset = 0;
            for (size_t i = 0; i < N; ++i) {
                offset = offset * dims_[i] + idx[i];
            }
            return offset;
        }
    };

    template<typename T, size_t N>
    Tensor<T, N> operator*(const T& scalar, const Tensor<T, N>& t) {
        return t * scalar;
    }

    template<typename T, size_t N>
    Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& t) {
        return t + scalar;
    }

    // Pregunta 6

    template<typename T, size_t N, size_t... Is>
    T get_at(const Tensor<T,N>& t,
             const std::array<size_t,N>& idx,
             std::index_sequence<Is...>)
    {
        return t(idx[Is]...);
    }

    template<typename T, size_t N, size_t... Is>
    void set_at(Tensor<T,N>& t,
                const std::array<size_t,N>& idx,
                const T& val,
                std::index_sequence<Is...>)
    {
        t(idx[Is]...) = val;
    }

    template<typename T, size_t N>
    Tensor<T, N> transpose_2d(const Tensor<T,N>& t) {
        if (N < 2) {
            throw std::invalid_argument(
                "Cannot transpose 1D tensor: need at least 2 dimensions"
                );
        }

        auto in_dims = t.shape();
        std::array<size_t,N> out_dims = in_dims;
        std::swap(out_dims[N-2], out_dims[N-1]);

        Tensor<T,N> result(out_dims);

        std::array<size_t,N> in_strides, out_strides;
        in_strides[N-1]  = out_strides[N-1] = 1;
        for (int i = static_cast<int>(N)-2; i >= 0; --i) {
            in_strides[i]  = in_strides[i+1]  * in_dims[i+1];
            out_strides[i] = out_strides[i+1] * out_dims[i+1];
        }

        const size_t total = result.cend() - result.cbegin();
        constexpr auto Seq = std::make_index_sequence<N>{};
        for (size_t lin = 0; lin < total; ++lin) {
            size_t rem = lin;
            std::array<size_t,N> idx_out;
            for (size_t i = 0; i < N; ++i) {
                idx_out[i] = rem / out_strides[i];
                rem %= out_strides[i];
            }
            auto idx_in = idx_out;
            std::swap(idx_in[N-2], idx_in[N-1]);
            T value = get_at<T,N>(t, idx_in, Seq);
            set_at<T,N>(result, idx_out, value, Seq);
        }
        return result;
    }

    template<typename T, size_t N, typename F>
    auto apply(const Tensor<T,N>& t, F func) {
        using U = std::invoke_result_t<F, T>;
        Tensor<U, N> result(t.shape());
        std::transform(t.begin(), t.end(), result.begin(),
                       [&](const T& x){ return func(x); });
        return result;
    }
    // Pregunta 7
    template<typename T, size_t N>
    Tensor<T, N> matrix_product(const Tensor<T, N>& A,
                                const Tensor<T, N>& B) {
        auto a = A.shape();
        auto b = B.shape();

        const size_t batchA = (N > 2 ? a[0] : 1);
        const size_t batchB = (N > 2 ? b[0] : 1);
        const size_t rowsA = (N > 2 ? a[1] : a[0]);
        const size_t colsA = (N > 2 ? a[2] : a[1]);
        const size_t rowsB = (N > 2 ? b[1] : b[0]);
        const size_t colsB = (N > 2 ? b[2] : b[1]);

        if (colsA != rowsB) {
            throw std::invalid_argument(
                "Matrix dimensions are incompatible for multiplication"
                );
        }
        if constexpr (N > 2) {
            if (batchA != batchB) {
                throw std::invalid_argument(
                    "Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match"
                    );
            }
        }

        std::array<size_t,N> out_dims;
        if constexpr(N > 2) {
            out_dims = a;
            out_dims[2] = colsB;
        } else {
            out_dims = { rowsA, colsB };
        }
        Tensor<T,N> result(out_dims);


        const size_t BATCH = (N > 2 ? batchA : 1);
        for (size_t bat = 0; bat < BATCH; ++bat) {
            for (size_t i = 0; i < rowsA; ++i) {
                for (size_t j = 0; j < colsB; ++j) {
                    T acc = T{};
                    for (size_t k = 0; k < colsA; ++k) {
                        if constexpr (N > 2) {
                            acc += A(bat, i, k) * B(bat, k, j);
                        } else {
                            acc += A(i, k) * B(k, j);
                        }
                    }
                    if constexpr (N > 2) {
                        result(bat, i, j) = acc;
                    } else {
                        result(i, j) = acc;
                    }
                }
            }
        }
        return result;
    }
}

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
