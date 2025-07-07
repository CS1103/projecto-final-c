#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <functional>
#include <initializer_list>

namespace utec::algebra {


    template <typename T, size_t Rank>
    struct Tensor {
        std::array<size_t, Rank> dimensions;
        std::vector<T> data;

        template <typename... Idxs>
        size_t calculate_index(Idxs... idxs) const;

        void validate_initializer_list(const std::initializer_list<T>& init_list) const;

        // Verifica si hay dos tensores compatibles para broadcasting
        bool is_broadcastable(const Tensor& other) const {
            for (int i = Rank - 1; i >= 0; i--) {
                if (dimensions[i] != other.dimensions[i] &&
                    dimensions[i] != 1 && other.dimensions[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        // Obtiene índice con broadcasting para un índice lineal dado
        std::array<size_t, Rank> get_broadcasted_index(size_t linear_idx, const std::array<size_t, Rank>& target_shape) const {
            std::array<size_t, Rank> indices;

            size_t temp = linear_idx;
            for (int i = Rank - 1; i >= 0; i--) {
                indices[i] = temp % target_shape[i];
                temp /= target_shape[i];
            }

            for (size_t i = 0; i < Rank; i++) {
                if (dimensions[i] == 1 && target_shape[i] > 1) {
                    indices[i] = 0;
                }
            }

            return indices;
        }

        size_t get_linear_index(const std::array<size_t, Rank>& indices) const {
            size_t index = 0;
            size_t stride = 1;

            for (int i = Rank - 1; i >= 0; i--) {
                index += indices[i] * stride;
                stride *= dimensions[i];
            }

            return index;
        }

        // Función auxiliar para calcular el tamaño totl
        static size_t compute_total_size(const std::array<size_t, Rank>& shape) {
            return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                                 std::multiplies<size_t>{});
        }

        // Operación binaria elemento por elemento con broadcasting
        template <typename Op>
        Tensor elementwise_binary_op(const Tensor& other, Op op) const {
            std::array<size_t, Rank> result_shape{};
            for (size_t i = 0; i < Rank; i++) {
                size_t a_dim = dimensions[i];
                size_t b_dim = other.dimensions[i];
                if (a_dim == b_dim) {
                    result_shape[i] = a_dim;
                } else if (a_dim == 1) {
                    result_shape[i] = b_dim;
                } else if (b_dim == 1) {
                    result_shape[i] = a_dim;
                } else {
                    throw std::invalid_argument(
                        "Shapes do not match and they are not compatible for broadcasting");
                }
            }
            Tensor result(result_shape);
            size_t result_total = compute_total_size(result_shape);
            for (size_t idx = 0; idx < result_total; idx++) {
                auto idx_multi = linear_to_multi(idx, result_shape);
                std::array<size_t, Rank> idxA{}, idxB{};
                for (size_t i = 0; i < Rank; i++) {
                    idxA[i] = (dimensions[i] == 1 ? 0 : idx_multi[i]);
                    idxB[i] = (other.dimensions[i] == 1 ? 0 : idx_multi[i]);
                }
                T a_val = this->operator()(idxA);
                T b_val = other.operator()(idxB);
                result.data[idx] = op(a_val, b_val);
            }
            return result;
        }

        // Constructores
        explicit Tensor(const std::array<size_t, Rank>& shape) : dimensions(shape) {
            size_t total_size = compute_total_size(dimensions);
            data.resize(total_size);
        }

        // Constructor variádico
        template <typename... Dims, typename = std::enable_if_t<sizeof...(Dims) == Rank>>
explicit Tensor(Dims... dims) {
            std::array<size_t, Rank> shp = {static_cast<size_t>(dims)...};
            dimensions = shp;
            size_t total_size = compute_total_size(dimensions);
            data.resize(total_size);
        }

        // Declaraciones friend para acceso a miembros privados desde funciones externas
        template <typename U, size_t R>
        friend Tensor<U, R> transpose_2d(const Tensor<U, R>& t);

        template <typename U, size_t R>
        friend Tensor<U, R> matrix_product(const Tensor<U, R>& A, const Tensor<U, R>& B);


        size_t size() const { return data.size(); }


        Tensor<T,2> slice(size_t start, size_t end) const {
            if (Rank != 2) throw std::runtime_error("Slice only implemented for 2D tensors");
            if (end > dimensions[0]) end = dimensions[0];

            Tensor<T,2> result({end - start, dimensions[1]});
            for (size_t i = start; i < end; ++i) {
                for (size_t j = 0; j < dimensions[1]; ++j) {
                    result(i-start, j) = operator()(i,j);
                }
            }
            return result;
        }

        // Operadores de acceso
        template <typename... Idxs>
        T& operator()(Idxs... idxs) {
            std::vector<size_t> idxVec = { static_cast<size_t>(idxs)... };
            if (idxVec.size() != Rank) {
                throw std::invalid_argument("Number of indices do not match with " + std::to_string(Rank));
            }
            std::array<size_t, Rank> indices;
            for (size_t i = 0; i < Rank; i++) {
                indices[i] = idxVec[i];
            }
            return data[calculate_index_from_array(indices)];
        }

        template <typename... Idxs>
        const T& operator()(Idxs... idxs) const {
            std::vector<size_t> idxVec = { static_cast<size_t>(idxs)... };
            if (idxVec.size() != Rank) {
                throw std::invalid_argument("Number of indices do not match with " + std::to_string(Rank));
            }
            std::array<size_t, Rank> indices;
            for (size_t i = 0; i < Rank; i++) {
                indices[i] = idxVec[i];
            }
            return data[calculate_index_from_array(indices)];
        }

        // Acceso basado en array normal y const
        T& operator()(const std::array<size_t, Rank>& indices) {
            return data[calculate_index_from_array(indices)];
        }

        const T& operator()(const std::array<size_t, Rank>& indices) const {
            return data[calculate_index_from_array(indices)];
        }

        // Acceso basado en vector y const
        T& operator()(const std::vector<size_t>& indices) {
            if (indices.size() != Rank) {
                throw std::invalid_argument("Number of indices do not match with " + std::to_string(Rank));
            }
            std::array<size_t, Rank> arr_indices;
            for (size_t i = 0; i < Rank; ++i) arr_indices[i] = indices[i];
            return data[calculate_index_from_array(arr_indices)];
        }

        const T& operator()(const std::vector<size_t>& indices) const {
            if (indices.size() != Rank) {
                throw std::invalid_argument("Number of indices do not match with " + std::to_string(Rank));
            }
            std::array<size_t, Rank> arr_indices;
            for (size_t i = 0; i < Rank; ++i) arr_indices[i] = indices[i];
            return data[calculate_index_from_array(arr_indices)];
        }


        // Operador [] para tensores 2D
        T& operator[](size_t idx) {
            if (Rank != 2) {
                throw std::invalid_argument("operator[] available only for 2D tensors");
            }
            if (idx >= data.size()) {
                throw std::invalid_argument("Index out of range");
            }
            size_t cols = dimensions[1];
            size_t i = idx / cols;
            size_t j = idx % cols;
            return operator()(i, j);
        }

        const T& operator[](size_t idx) const {
            if (Rank != 2) {
                throw std::invalid_argument("operator[] available only for 2D tensors");
            }
            if (idx >= data.size()) {
                throw std::invalid_argument("Index out of range");
            }
            size_t cols = dimensions[1];
            size_t i = idx / cols;
            size_t j = idx % cols;
            return operator()(i, j);
        }

        size_t calculate_index_from_array(const std::array<size_t, Rank>& indices) const {
            size_t index = 0;
            size_t stride = 1;

            for (int i = Rank - 1; i >= 0; i--) {
                if (indices[i] >= dimensions[i]) {
                    throw std::invalid_argument("Index out of range in dimension " + std::to_string(i));
                }
                index += indices[i] * stride;
                stride *= dimensions[i];
            }

            return index;
        }

        // Informa acerca de las dimensiones
        std::array<size_t, Rank> shape() const noexcept { return dimensions; }

        // Función de reshape
        template <typename... Dims>
        void reshape(Dims... dims) {
            std::vector<size_t> dimsVec = { static_cast<size_t>(dims)... };
            if (dimsVec.size() != Rank) {
                throw std::invalid_argument("Number of dimensions reshape do not match with " + std::to_string(Rank));
            }
            std::array<size_t, Rank> new_shape;
            for (size_t i = 0; i < Rank; i++) {
                new_shape[i] = dimsVec[i];
            }
            size_t new_total = compute_total_size(new_shape);
            size_t current_total = data.size();

            if (new_total == current_total) {
                dimensions = new_shape;
                return;
            }
            if (new_total < current_total) {
                std::vector<T> temp(new_total); // usamos temporales
                for (size_t i = 0; i < new_total; i++) {
                    temp[i] = data[i];
                }
                data = std::move(temp);
                dimensions = new_shape;
                return;
            }
            // Si el nuevo tamaño es mayor, redimensionar y mantener datos existentes
            data.resize(new_total);
            dimensions = new_shape;
        }

        void reshape(const std::array<size_t, Rank>& new_shape) {
            size_t new_total = compute_total_size(new_shape);
            size_t current_total = data.size();

            if (new_total == current_total) {
                dimensions = new_shape;
                return;
            }
            if (new_total < current_total) {
                std::vector<T> temp(new_total);
                for (size_t i = 0; i < new_total; i++) {
                    temp[i] = data[i];
                }
                data = std::move(temp);
                dimensions = new_shape;
                return;
            }
            data.resize(new_total);
            dimensions = new_shape;
        }

        void fill(const T& value) noexcept {
            std::fill_n(data.begin(), data.size(), value);
        }

        Tensor& operator=(std::initializer_list<T> init_list) {
            if (init_list.size() != data.size()) {
                throw std::invalid_argument(
                    "Initializer list size (" + std::to_string(init_list.size()) +
                    ") does not match tensor size (" + std::to_string(data.size()) + ")");
            }
            std::copy(init_list.begin(), init_list.end(), data.begin());
            return *this;
        }

        // En la clase Tensor, añade:
        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;



        // Operaciones aritméticas ---------------------------------------------------------
        Tensor operator+(const Tensor& other) const {
            return elementwise_binary_op(other, std::plus<>{});
        }

        Tensor operator-(const Tensor& other) const {
            return elementwise_binary_op(other, std::minus<>{});
        }

        Tensor operator*(const Tensor& other) const {
            return elementwise_binary_op(other, std::multiplies<>{});
        }

        //-------------------------------------------------------------------------------------

        // Operaciones con escalares--------------------------------------------------------------------------
        Tensor operator+(const T& scalar) const {
            Tensor result(dimensions);
            for (size_t i = 0; i < data.size(); i++) {
                result.data[i] = data[i] + scalar;
            }
            return result;
        }

        Tensor operator-(const T& scalar) const {
            Tensor result(dimensions);
            for (size_t i = 0; i < data.size(); i++) {
                result.data[i] = data[i] - scalar;
            }
            return result;
        }

        Tensor operator*(const T& scalar) const {
            Tensor result(dimensions);
            for (size_t i = 0; i < data.size(); i++) {
                result.data[i] = data[i] * scalar;
            }
            return result;
        }

        Tensor operator/(const T& scalar) const {
            Tensor result(dimensions);
            for (size_t i = 0; i < data.size(); i++) {
                result.data[i] = data[i] / scalar;
            }
            return result;
        }

        friend Tensor operator+(const T& scalar, const Tensor& t) {
            return t + scalar;
        }

        friend Tensor operator-(const T& scalar, const Tensor& t) {
            Tensor result(t.dimensions);
            for (size_t i = 0; i < t.data.size(); i++) {
                result.data[i] = scalar - t.data[i];
            }
            return result;
        }

        friend Tensor operator*(const T& scalar, const Tensor& t) {
            return t * scalar;
        }

        friend Tensor operator/(const T& scalar, const Tensor& t) {
            Tensor result(t.dimensions);
            for (size_t i = 0; i < t.data.size(); i++) {
                result.data[i] = scalar / t.data[i];
            }
            return result;
        }

        //-----------------------------------------------------------------------------------------

        auto begin() noexcept { return data.begin(); }
        auto end() noexcept { return data.end(); }
        auto cbegin() const noexcept { return data.cbegin(); }
        auto cend() const noexcept { return data.cend(); }

        static std::array<size_t, Rank> linear_to_multi(size_t linear, const std::array<size_t, Rank>& shape) {
            std::array<size_t, Rank> indices{};
            size_t remainder = linear;
            size_t stride = compute_total_size(shape);
            for (size_t dim = 0; dim < Rank; ++dim) {
                stride /= shape[dim];
                indices[dim] = remainder / stride;
                remainder %= stride;
            }
            return indices;
        }

        // Función de impresión recursiva para tensores de alto rango (Rank > 2)
        void print_recursive(std::ostream& os, std::array<size_t, Rank>& indices, size_t dim, const std::string& indent = "") const {
            const auto& shape = dimensions;
            if (dim == Rank - 2) {
                os << indent << "{" << endl;
                for (size_t i = 0; i < shape[dim]; i++) {
                    indices[dim] = i;
                    os << indent << "  ";
                    for (size_t j = 0; j < shape[dim + 1]; j++) {
                        indices[dim + 1] = j;
                        if (j != 0) os << " ";
                        os << operator()(indices);
                    }
                    if (i != shape[dim] - 1) os << endl;
                }
                os << endl << indent << "}";
            } else {
                os << indent << "{"<<endl;
                for (size_t i = 0; i < shape[dim]; i++) {
                    indices[dim] = i;
                    print_recursive(os, indices, dim + 1, indent + "  ");
                    if (i != shape[dim] - 1) os << endl;
                }
                os << endl << indent << "}";
            }
        }

        // Operador de salida
        friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
            const auto& shape = t.shape();
            if constexpr (Rank == 1) {
                os << "";
                for (size_t i = 0; i < shape[0]; i++) {
                    if (i != 0) os << " ";
                    os << t(i);
                }
                os << ""<<endl;
            } else if constexpr (Rank == 2) {
                os << "{\n";
                for (size_t i = 0; i < shape[0]; i++) {
                    os << "  ";
                    for (size_t j = 0; j < shape[1]; j++) {
                        if (j != 0) os << " ";
                        os << t(i, j);
                    }
                    if (i != shape[0] - 1) os << endl;
                }
                os << endl<<"}";
            } else {
                std::array<size_t, Rank> indices{};
                t.print_recursive(os, indices, 0, "");
            }
            return os;
        }


        template<typename U, size_t R, typename F>
friend auto apply(const Tensor<U, R>& tensor, F&& func);



    };




    template<typename U, size_t R, typename F>
auto apply(const Tensor<U, R>& tensor, F&& func) {
        Tensor<U, R> result(tensor.shape());
        for (size_t i = 0; i < tensor.size(); ++i) {
            result.data[i] = func(tensor.data[i]);
        }
        return result;
    }
template <typename T, size_t Rank>
template <typename... Idxs>
size_t Tensor<T, Rank>::calculate_index(Idxs... idxs) const {
    static_assert(sizeof...(Idxs) == Rank,
                  "Number of indices must match tensor rank");

    std::array<size_t, Rank> indices{static_cast<size_t>(idxs)...};
    return calculate_index_from_array(indices);
}

template <typename T, size_t Rank>
void Tensor<T, Rank>::validate_initializer_list(const std::initializer_list<T>& init_list) const {
    if (init_list.size() != data.size()) {
        throw std::invalid_argument("Data size does not match tensor size");
    }
}

// Función de transposición 2D
    template <typename T, size_t R>
    Tensor<T, R> transpose_2d(const Tensor<T, R>& t) {
    if constexpr (R < 2) {
        throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
    }
    auto old_shape = t.shape();
    std::array<size_t, R> new_shape = old_shape;
    std::swap(new_shape[R - 2], new_shape[R - 1]);
    Tensor<T, R> result(new_shape);

    std::vector<size_t> idx(R, 0);
    size_t total = result.data.size();

    for (size_t flat = 0; flat < total; ++flat) {
        size_t rem = flat;
        for (int d = R - 1; d >= 0; --d) {
            idx[d] = rem % new_shape[d];
            rem /= new_shape[d];
        }
        std::array<size_t, R> idx_src;
        for (size_t i = 0; i < R; i++) idx_src[i] = idx[i];
        std::swap(idx_src[R - 2], idx_src[R - 1]);
        result(idx) = t(idx_src);
    }
    return result;
}

    template <typename T, size_t R>
    Tensor<T, R> matrix_product(const Tensor<T, R>& A, const Tensor<T, R>& B) {

    static_assert(R >= 2, "Rank must be at least 2 for matrix multiplication");
    auto a_shape = A.shape();
    auto b_shape = B.shape();

    for (size_t i = 0; i < R - 2; i++) {
        if (a_shape[i] != b_shape[i]) {
            throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
        }
    }

    size_t M = a_shape[R - 2];
    size_t K = a_shape[R - 1];
    size_t K2 = b_shape[R - 2];
    size_t N = b_shape[R - 1];

    if (K != K2) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }

    std::array<size_t, R> result_shape = a_shape;
    result_shape[R - 1] = N;
    result_shape[R - 2] = M;
    Tensor<T, R> result(result_shape);

    std::vector<size_t> idx(R, 0);
    size_t total = result.data.size();

    for (size_t flat = 0; flat < total; ++flat) {
        size_t rem = flat;
        for (int d = R - 1; d >= 0; --d) {
            idx[d] = rem % result_shape[d];
            rem /= result_shape[d];
        }
        std::array<size_t, R> idxA, idxB;
        for (size_t d = 0; d < R; ++d) {
            idxA[d] = idx[d];
            idxB[d] = idx[d];
        }


        T sum = T{};
        for (size_t k = 0; k < K; k++) {
            idxA[R - 1] = k;
            idxB[R - 2] = k;
            result.data[flat] += A(idxA) * B(idxB);
        }
    }
    return result;
}

}

#endif // TENSOR_H