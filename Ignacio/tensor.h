//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>

namespace utec::algebra
{

    template <typename T, size_t N>
    class Tensor
    {
        std::array<size_t, N> forma;

    public:
        std::vector<T> datos;

        static size_t size(const std::array<size_t, N> &shape)
        {
            size_t prod = 1;
            for (size_t i = 0; i < N; ++i)
                prod *= shape[i];
            return prod;
        }

        Tensor(const std::array<size_t, N> &shape) : forma(shape)
        {
            datos.resize(size(forma));
        }

        template <typename... Dims>
        Tensor(Dims... dims)
        {
            size_t temp[N] = {static_cast<size_t>(dims)...};
            std::copy(temp, temp + N, forma.begin());
            datos.resize(size(forma));
        }

        void fill(const T &value) noexcept
        {
            std::fill(datos.begin(), datos.end(), value);
        }

        size_t size() const noexcept {
            return datos.size();
        }

        template <typename... Args>
        void reshape(Args... dims)
        {
            std::vector<size_t> temp = {static_cast<size_t>(dims)...};
            if (temp.size() != N)
                throw std::invalid_argument("Number of dimensions do not match with 2");

            size_t new_total = 1;
            for (auto d : temp) new_total *= d;

            std::vector<T> new_datos(new_total, T{});
            size_t min_size = std::min(datos.size(), new_total);
            for (size_t i = 0; i < min_size; ++i)
                new_datos[i] = datos[i];

            datos = std::move(new_datos);
            for (size_t i = 0; i < N; ++i)
                forma[i] = temp[i];
        }

        const std::array<size_t, N> &shape() const noexcept
        {
            return forma;
        }

        template <typename... Idxs>
        T &operator()(Idxs... idxs)
        {
            if (sizeof...(Idxs) != N)
                throw std::invalid_argument("Número de índices incorrecto");
            array<size_t, N> indices{static_cast<size_t>(idxs)...};
            return datos[flat_index(indices)];
        }

        template <typename... Idxs>
        const T &operator()(Idxs... idxs) const
        {
            if (sizeof...(Idxs) != N)
                throw std::invalid_argument("Número de índices incorrecto");
            array<size_t, N> indices{static_cast<size_t>(idxs)...};
            return datos[flat_index(indices)];
        }

        T &operator[](size_t idx) { return datos[idx]; }
        const T &operator[](size_t idx) const { return datos[idx]; }

        Tensor &operator=(std::initializer_list<T> lista)
        {
            if (lista.size() != datos.size())
                throw std::runtime_error("Data size does not match tensor size");
            auto it = lista.begin();
            for (size_t i = 0; i < datos.size(); ++i, ++it) {
                datos[i] = *it;
            }
            return *this;
        }

        auto begin() { return datos.begin(); }
        auto end() { return datos.end(); }
        auto begin() const { return datos.begin(); }
        auto end() const { return datos.end(); }
        auto cbegin() const { return datos.cbegin(); }
        auto cend() const { return datos.cend(); }

        Tensor operator+(const Tensor &other) const
        {
            if (!shapes_compatible_for_multiplication(forma, other.forma))
                throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            std::array<size_t, N> result_shape;
            for (size_t i = 0; i < N; ++i)
                result_shape[i] = std::max(forma[i], other.forma[i]);

            Tensor result(result_shape);
            for (size_t i = 0; i < result.datos.size(); ++i)
            {
                auto idx = unflatten_index(i, result_shape);
                result.datos[i] = this->at_broadcast(idx) + other.at_broadcast(idx);
            }
            return result;
        }

        Tensor operator-(const Tensor &other) const
        {
            if (!shapes_compatible_for_multiplication(forma, other.forma))
                throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            std::array<size_t, N> result_shape;
            for (size_t i = 0; i < N; ++i)
                result_shape[i] = std::max(forma[i], other.forma[i]);

            Tensor result(result_shape);
            for (size_t i = 0; i < result.datos.size(); ++i)
            {
                auto idx = unflatten_index(i, result_shape);
                result.datos[i] = this->at_broadcast(idx) - other.at_broadcast(idx);
            }
            return result;
        }

        Tensor operator*(const T &scalar) const
        {
            Tensor result(forma);
            for (size_t i = 0; i < datos.size(); ++i)
                result.datos[i] = datos[i] * scalar;
            return result;
        }

        friend Tensor operator*(const T &scalar, const Tensor &t)
        {
            return t * scalar;
        }

        Tensor operator+(const T &scalar) const
        {
            Tensor result(forma);
            for (size_t i = 0; i < datos.size(); ++i)
                result.datos[i] = datos[i] + scalar;
            return result;
        }

        friend Tensor operator+(const T &scalar, const Tensor &t)
        {
            return t + scalar;
        }

        Tensor operator-(const T &scalar) const
        {
            Tensor result(forma);
            for (size_t i = 0; i < datos.size(); ++i)
                result.datos[i] = datos[i] - scalar;
            return result;
        }

        friend Tensor operator-(const T &scalar, const Tensor &t)
        {
            Tensor result(t.forma);
            for (size_t i = 0; i < t.datos.size(); ++i)
                result.datos[i] = scalar - t.datos[i];
            return result;
        }

        Tensor operator/(const T &scalar) const
        {
            Tensor result(forma);
            for (size_t i = 0; i < datos.size(); ++i)
                result.datos[i] = datos[i] / scalar;
            return result;
        }

        friend Tensor operator/(const T &scalar, const Tensor &t)
        {
            Tensor result(t.forma);
            for (size_t i = 0; i < t.datos.size(); ++i)
                result.datos[i] = scalar / t.datos[i];
            return result;
        }

        Tensor operator*(const Tensor &other) const
        {
            if (!shapes_compatible_for_multiplication(forma, other.forma))
                throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            std::array<size_t, N> result_shape;
            for (size_t i = 0; i < N; ++i)
                result_shape[i] = std::max(forma[i], other.forma[i]);

            Tensor result(result_shape);
            for (size_t i = 0; i < result.datos.size(); ++i)
            {
                std::array<size_t, N> idx = unflatten_index(i, result_shape);
                result.datos[i] = this->at_broadcast(idx) * other.at_broadcast(idx);
            }
            return result;
        }

        friend std::ostream &operator<<(std::ostream &os, const Tensor &t)
        {
            print_recursive(os, t.datos, t.forma, 0, 0);
            return os;
        }

        template <typename, size_t>
        friend class Tensor;

        template <typename, size_t>
        friend Tensor matrix_product(const Tensor &, const Tensor &);

        size_t flat_index(const std::array<size_t, N> &indices) const
        {
            size_t idx = 0, stride = 1;
            for (size_t i = N; i-- > 0;)
            {
                idx += indices[i] * stride;
                stride *= forma[i];
            }
            return idx;
        }

        std::array<size_t, N> unflatten_index(size_t idx, const std::array<size_t, N> &shape) const
        {
            std::array<size_t, N> indices;
            for (size_t i = N; i-- > 0;)
            {
                indices[i] = idx % shape[i];
                idx /= shape[i];
            }
            return indices;
        }

        T at_broadcast(const std::array<size_t, N> &idx) const
        {
            std::array<size_t, N> real_idx;
            for (size_t i = 0; i < N; ++i)
                real_idx[i] = (forma[i] == 1) ? 0 : idx[i];
            return call_operator_parenthesis(real_idx, std::make_index_sequence<N>());
        }

        static bool shapes_compatible_for_multiplication(const std::array<size_t, N> &a, const std::array<size_t, N> &b)
        {
            for (size_t i = 0; i < N; ++i)
                if (a[i] != b[i] && a[i] != 1 && b[i] != 1)
                    return false;
            return true;
        }

        static void print_recursive(std::ostream &os, const std::vector<T> &data, const std::array<size_t, N> &shape, size_t dim, size_t offset)
        {
            if (dim == N - 1)
            {
                for (size_t i = 0; i < shape[dim]; ++i)
                {
                    os << data[offset + i];
                    if (i + 1 < shape[dim])
                        os << " ";
                }
            }
            else
            {
                os << "{\n";
                size_t step = 1;
                for (size_t d = dim + 1; d < N; ++d)
                    step *= shape[d];
                for (size_t i = 0; i < shape[dim]; ++i)
                {
                    print_recursive(os, data, shape, dim + 1, offset + i * step);
                    if (i + 1 < shape[dim])
                        os << "\n";
                }
                os << "\n}";
            }
        }

        template <typename... Dims>
        static std::array<size_t, sizeof...(Dims)> to_array(Dims... dims)
        {
            return std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(dims)...};
        }

        template <typename Array, size_t... I>
        T &call_operator_parenthesis(Array &&arr, std::index_sequence<I...>)
        {
            return (*this)(arr[I]...);
        }
        template <typename Array, size_t... I>
        const T &call_operator_parenthesis(Array &&arr, std::index_sequence<I...>) const
        {
            return (*this)(arr[I]...);
        }
    };


    template <typename T, size_t N>
    Tensor<T, N> matrix_product(const Tensor<T, N> &a, const Tensor<T, N> &b)
    {
        if constexpr (N < 2)
            throw invalid_argument("Matrix dimensions are incompatible for multiplication");

        const auto &ashape = a.shape();
        const auto &bshape = b.shape();

        if (ashape[N - 1] != bshape[N - 2])
            throw invalid_argument("Matrix dimensions are incompatible for multiplication");

        for (size_t i = 0; i + 2 < N; ++i)
        {
            if (ashape[i] != bshape[i])
            {
                throw invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }
        }

        array<size_t, N> result_shape;
        for (size_t i = 0; i < N - 2; ++i)
            result_shape[i] = max(ashape[i], bshape[i]);
        result_shape[N - 2] = ashape[N - 2];
        result_shape[N - 1] = bshape[N - 1];

        Tensor<T, N> result(result_shape);

        array<size_t, N> idx = {};
        for (size_t i = 0; i < result.datos.size(); ++i)
        {
            array<size_t, N> coord = result.unflatten_index(i, result_shape);
            T sum = T();
            for (size_t k = 0; k < ashape[N - 1]; ++k)
            {
                auto a_idx = coord;
                a_idx[N - 1] = k;
                auto b_idx = coord;
                b_idx[N - 2] = k;
                sum += a.at_broadcast(a_idx) * b.at_broadcast(b_idx);
            }
            result[i] = sum;
        }

        return result;
    }

    template <typename T, size_t N>
    Tensor<T, N> transpose_2d(const Tensor<T, N> &tensor) {
        if constexpr (N < 2)
        {
            throw invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
        }

        auto shape = tensor.shape();
        array<size_t, N> new_shape = shape;

        swap(new_shape[N - 2], new_shape[N - 1]);
        Tensor<T, N> result(new_shape);

        for (size_t i = 0; i < result.datos.size(); ++i)
        {
            array<size_t, N> idx = result.unflatten_index(i, new_shape);
            array<size_t, N> original_idx = idx;

            swap(original_idx[N - 2], original_idx[N - 1]);

            result[i] = tensor.at_broadcast(original_idx);
        }

        return result;
    }

    template <typename Fn, typename T>
    Tensor<T, 2> apply(const Tensor<T, 2>& tensor, Fn&& fn) {
        Tensor<T, 2> result(tensor.shape());
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                result(i, j) = fn(tensor(i, j)); // Aplica la función a cada elemento
            }
        }
        return result;
    }

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
