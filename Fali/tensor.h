#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <ostream>
#include <vector>
#include <array>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>
#include <iterator>



namespace utec::algebra {

template<typename T, size_t N>
class Tensor {
private:
    std::array<size_t, N> shape_;
    std::vector<T> data_;
    std::array<size_t, N> strides_;

    size_t total_size(const std::array<size_t, N> &shape) const {
        return accumulate( shape.begin(), shape.end(), size_t{1},std::multiplies<size_t>());
    }
    void compute_strides() {
        strides_[N - 1] = 1;
        if constexpr (N >= 2) {
            for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }
    }
    template <typename... Args>
    size_t compute_index(Args... args) const {
        static_assert(sizeof...(Args) == N, "Number of dimensions do not match with");
        std::array<size_t, N> indices{ static_cast<size_t>(args)... };
        size_t index = 0;
        for (size_t i = 0; i < N; ++i)
            index += strides_[i] * indices[i];
        return index;
    }

    //5
    size_t compute_broadcast_index(const std::array<size_t, N>& shape,
                                   const std::array<size_t, N>& strides,
                                   const std::array<size_t, N>& idx) const {
        size_t offset = 0;
        for (size_t i = 0; i < N; ++i) {
            size_t index_in_dim = (shape[i] == 1) ? 0 : idx[i];
            offset += index_in_dim * strides[i];
        }
        return offset;
    }
    //
    static bool shapes_compatible_for_broadcasting(const std::array<size_t, N>& a, const std::array<size_t, N>& b) {
        for (size_t i = 0; i < N; ++i) {
            if (a[i] != b[i] && a[i] != 1 && b[i] != 1)
                return false;
        }
        return true;
    }
    static void linear_to_multi_index(size_t linear_idx, const std::array<size_t, N>& shape, std::array<size_t, N>& idx) {
        for (int i = N - 1; i >= 0; --i) {
            idx[i] = linear_idx % shape[i];
            linear_idx /= shape[i];
        }
    }
    //Impresión
    static void print_tensor_recursive(std::ostream& os, const std::vector<T>& data,
                                       const std::array<size_t, N>& shape,
                                       const std::array<size_t, N>& strides,
                                       size_t dim, size_t offset) {
        if (dim == N - 1) {
            for (size_t i = 0; i < shape[dim]; ++i) {
                os << data[offset + i];
                if (i + 1 < shape[dim]) os << " ";
            }
        } else {
            os << "{\n";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i != 0) os << "\n";
                print_tensor_recursive(os, data, shape, strides, dim + 1, offset + i * strides[dim]);
            }
            os << "\n}";
        }
    }


public:
    template <typename... Args>
    explicit Tensor(Args... dims) {
        if constexpr (sizeof...(Args) == N) {
            shape_ = {static_cast<size_t>(dims)...};
            compute_strides();
            data_.resize(total_size(shape_));
        } else {
            throw std::runtime_error("Number of dimensions do not match with " + std::to_string(N));
        }
    }

    const std::array<size_t, N>& shape() const {
        return shape_;
    }
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    Tensor() {
        shape_.fill(0);
        strides_.fill(0);
    }
    explicit Tensor(const std::array<size_t, N>& shape) {
        shape_ = shape;
        compute_strides();
        data_.resize(total_size(shape_));
    }

// Acceso con operador ()
    template <typename... Args>
    T& operator()(Args... args) {
        return data_[compute_index(args...)];
    }

    template <typename... Args>
    const T& operator()(Args... args) const {
        return data_[compute_index(args...)];
    }

    T operator()(const std::array<size_t, N>& idx) const {
        size_t index = 0;
        for (size_t i = 0; i < N; ++i)
            index += strides_[i] * idx[i];
        return data_[index];
    }

    T& operator()(const std::array<size_t, N>& idx) {
        size_t index = 0;
        for (size_t i = 0; i < N; ++i)
            index += strides_[i] * idx[i];
        return data_[index];
    }

// Asignación desde initializer list
    Tensor<T, N>& operator=(std::initializer_list<T> values) {
        if (values.size() != data_.size())
            throw std::runtime_error("Data size does not match tensor size");
        std::copy(values.begin(), values.end(), data_.begin());
        return *this;
    }
// Iteradores
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend() const { return data_.cend(); }
// size()
    auto size() const { return std::distance(data_.begin(), data_.end()); }

// Impresión
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        print_tensor_recursive(os, t.data_, t.shape_, t.strides_, 0, 0);
        return os;
    }




// Redimensionamiento       Pregunta 2
    template<typename... Args>
    void reshape(Args... newDims) {
        if constexpr (sizeof...(Args) == N) {
            std::array<size_t, N> newShape = {static_cast<size_t>(newDims)...};
            size_t newSize = total_size(newShape);

            if (newSize != data_.size()) {
                if (newSize < data_.size()) {
                    data_.resize(newSize); // recorta
                } else {
                    data_.resize(newSize, T{}); // rellena con ceros o T{}
                }
            }
            shape_ = newShape;
            compute_strides();

        } else {
            throw std::runtime_error("Number of dimensions do not match with " + std::to_string(N));
        }
    }
    void reshape(const std::array<size_t, N>& newShape) {   //para la pregunta 6
        size_t newSize = total_size(newShape);

        if (newSize != data_.size()) {
            if (newSize < data_.size()) {
                data_.resize(newSize); // recorta
            } else {
                data_.resize(newSize, T{}); // rellena con ceros o T{}
            }
        }
        shape_ = newShape;
        compute_strides();
    }

// ================================================ Pregunta 3 ==================================================

    //Suma Y Resta
    friend Tensor operator+(const Tensor& a, const Tensor& b) {
        if (a.shape() != b.shape() && !shapes_compatible_for_broadcasting(a.shape(), b.shape())) {
            throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
        }

        Tensor result = a;
        // Con Broadcasting     Pregunta 5
        if (a.shape() != b.shape() && shapes_compatible_for_broadcasting(b.shape(), a.shape())) {


            std::array<size_t, N> result_shape;
            for (size_t i = 0; i < N; ++i) {
                result_shape[i] = std::max(a.shape_[i], b.shape_[i]);
            }
            result.shape_ = result_shape;
            result.compute_strides();
            result.data_.resize(result.total_size(result_shape));

            std::array<size_t, N> idx{};
            size_t total = result.total_size(result_shape);

            for (size_t linear_idx = 0; linear_idx < total; ++linear_idx) {
                linear_to_multi_index(linear_idx, result_shape, idx);
                size_t a_idx = a.compute_broadcast_index(a.shape_, a.strides_, idx);
                size_t b_idx = b.compute_broadcast_index(b.shape_, b.strides_, idx);
                result.data_[linear_idx] = a.data_[a_idx] + b.data_[b_idx];
            }
            return result;
        }
    //Sin Broadcasting

        for (size_t i = 0; i < result.end() - result.begin(); ++i)
            result.begin()[i] += b.cbegin()[i];
        return result;
    }

    friend Tensor operator-(const Tensor& a, const Tensor& b) {
        if (a.shape() != b.shape() && !shapes_compatible_for_broadcasting(a.shape(), b.shape())) {
            throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
        }


        Tensor result = a;
        // Con Broadcasting     Pregunta 5
        if (a.shape() != b.shape() && shapes_compatible_for_broadcasting(a.shape(), b.shape())) {

            std::array<size_t, N> result_shape;
            for (size_t i = 0; i < N; ++i) {
                result_shape[i] = std::max(a.shape_[i], b.shape_[i]);
            }
            result.shape_ = result_shape;
            result.compute_strides();
            result.data_.resize(result.total_size(result_shape));

            std::array<size_t, N> idx;
            size_t total = result.total_size(result_shape);

            for (size_t linear_idx = 0; linear_idx < total; ++linear_idx) {
                linear_to_multi_index(linear_idx, result_shape, idx);
                size_t a_idx = a.compute_broadcast_index(a.shape_, a.strides_, idx);
                size_t b_idx = b.compute_broadcast_index(b.shape_, b.strides_, idx);
                result.data_[linear_idx] = a.data_[a_idx] - b.data_[b_idx];
            }
            return result;
        }
        // Sin Broadcasting
        for (size_t i = 0; i < result.end() - result.begin(); ++i)
            result.begin()[i] -= b.cbegin()[i];
        return result;
    }
// ================================================ Pregunta 4 ==================================================
    // Multiplicación elemento a elemento entre tensores
    friend Tensor operator*(const Tensor& a, const Tensor& b) {
        if (a.shape_ != b.shape_ && !shapes_compatible_for_broadcasting(a.shape_, b.shape_))
            throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");

        // Con Broadcasting     Pregunta 5
        if (a.shape_ != b.shape_ && shapes_compatible_for_broadcasting(a.shape_, b.shape_)) {
            std::array<size_t, N> result_shape;
            for (size_t i = 0; i < N; ++i) {
                result_shape[i] = std::max(a.shape_[i], b.shape_[i]);
            }

            Tensor result;
            result.shape_ = result_shape;
            result.compute_strides();
            result.data_.resize(result.total_size(result_shape));

            std::array<size_t, N> idx;
            size_t total = result.total_size(result_shape);

            for (size_t linear_idx = 0; linear_idx < total; ++linear_idx) {
                linear_to_multi_index(linear_idx, result_shape, idx);
                size_t a_idx = a.compute_broadcast_index(a.shape_, a.strides_, idx);
                size_t b_idx = b.compute_broadcast_index(b.shape_, b.strides_, idx);
                result.data_[linear_idx] = a.data_[a_idx] * b.data_[b_idx];
            }

            return result;
        }

        // Sin Broadcasting
        Tensor result = a;
        for (size_t i = 0; i < a.data_.size(); ++i)
            result.data_[i] = a.data_[i] * b.data_[i];
        return result;
    }

    // Escalar + Tensor
    friend Tensor operator+(const T& scalar, const Tensor& t) {
        Tensor result = t;
        for (auto& val : result.data_)
            val = scalar + val;
        return result;
    }

    // Tensor + Escalar
    friend Tensor operator+(const Tensor& t, const T& scalar) {
        return scalar + t;
    }

    // Tensor - Escalar
    friend Tensor operator-(const Tensor& t, const T& scalar) {
        Tensor result = t;
        for (auto& val : result.data_)
            val = val - scalar;
        return result;
    }

    // Escalar - Tensor
    friend Tensor operator-(const T& scalar, const Tensor& t) {
        Tensor result = t;
        for (auto& val : result.data_)
            val = scalar - val;
        return result;
    }

    // Tensor * Escalar
    friend Tensor operator*(const Tensor& t, const T& scalar) {
        Tensor result = t;
        for (auto& val : result.data_)
            val *= scalar;
        return result;
    }

    // Escalar * Tensor
    friend Tensor operator*(const T& scalar, const Tensor& t) {
        return t * scalar;
    }

    // Tensor / Escalar
    friend Tensor operator/(const Tensor& t, const T& scalar) {
        Tensor result = t;
        for (auto& val : result.data_)
            val /= scalar;
        return result;
    }

// ================================================ Pregunta 6 ==================================================
    friend Tensor transpose_2d(const Tensor& input) {
        if constexpr (N < 2) {
            throw std::runtime_error("Cannot transpose 1D tensor: need at least 2 dimensions");
        }

        std::array<size_t, N> new_shape = input.shape();
        std::swap(new_shape[N - 1], new_shape[N - 2]);

        Tensor result;
        result.shape_ = new_shape;
        result.compute_strides();
        result.data_.resize(result.total_size(new_shape));

        std::array<size_t, N> idx{};
        std::array<size_t, N> transposed_idx{};

        size_t total = input.data_.size();
        for (size_t linear_idx = 0; linear_idx < total; ++linear_idx) {
            linear_to_multi_index(linear_idx, input.shape_, idx);
            transposed_idx = idx;
            std::swap(transposed_idx[N - 1], transposed_idx[N - 2]);
            result(transposed_idx) = input(idx);
        }

        return result;
    }


    friend Tensor<T, N> matrix_product<T, N>(const Tensor& A, const Tensor& B);

};
    // ================================================ Pregunta 7 ==================================================
    template<typename T, size_t N>
    Tensor<T,N> matrix_product(const Tensor<T,N>& A, const Tensor<T,N>& B) {
        static_assert(N == 2 || N == 3, "Only 2D or 3D tensor matrix multiplication is supported.");//provisional¿?

        // 2D case
        if constexpr (N == 2) {
            auto [m, k1] = A.shape();
            auto [k2, n] = B.shape();

            if (k1 != k2)
                throw std::runtime_error("Matrix dimensions are incompatible for multiplication");

            Tensor<T, 2> result(m, n);
            result.fill(0);

            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j)
                    for (size_t k = 0; k < k1; ++k)
                        result(i, j) += A(i, k) * B(k, j);

            return result;
        }

        // 3D case (batched)
        if constexpr (N == 3) {
            auto [batch_a, m, k1] = A.shape();
            auto [batch_b, k2, n] = B.shape();

            if (k1 != k2)
                throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
            if (batch_a != batch_b)
                throw std::runtime_error("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");

            Tensor<T, 3> result(batch_a, m, n);
            result.fill(0);

            for (size_t b = 0; b < batch_a; ++b)
                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        for (size_t k = 0; k < k1; ++k)
                            result(b, i, j) += A(b, i, k) * B(b, k, j);

            return result;
        }
    }
}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
