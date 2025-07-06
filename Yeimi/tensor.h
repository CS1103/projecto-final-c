//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <initializer_list>
#include <functional>

namespace utec::algebra {
    template <typename T, size_t Rank>
    class Tensor {
    private:
        array<size_t, Rank> shape_{};
        vector<T> data_;
        template <typename U, size_t OtherRank> friend auto transpose_2d(const Tensor<U, OtherRank>& input);
        template <typename U> friend Tensor<U, 2> matrix_product(const Tensor<U, 2>&, const Tensor<U, 2>&);
        template <typename U> friend Tensor<U, 3> matrix_product(const Tensor<U, 3>&, const Tensor<U, 3>&);

        size_t total_size(const std::array<size_t, Rank>& shape) {
            return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
        }

        size_t get_index(const array<size_t, Rank>& indices) const {
            size_t index = 0;
            size_t stride = 1;
            for (size_t i = Rank; i-- > 0;) {
                index += indices[i] * stride;
                stride *= shape_[i];
            }
            return index;
        }

        static array<size_t, Rank> broadcast_shape(const array<size_t, Rank>& a, const array<size_t, Rank>& b) {
            array<size_t, Rank> result;
            for (size_t i = 0; i < Rank; ++i) {
                if (a[i] == b[i])
                    result[i] = a[i];
                else if (a[i] == 1)
                    result[i] = b[i];
                else if (b[i] == 1)
                    result[i] = a[i];
                else
                    throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }
            return result;
        }

        size_t broadcast_index(const array<size_t, Rank>& indices, const array<size_t, Rank>& shape) const {
            size_t index = 0;
            size_t stride = 1;
            for (size_t i = Rank; i-- > 0;) {
                size_t idx = (shape[i] == 1) ? 0 : indices[i];
                index += idx * stride;
                stride *= shape[i];
            }
            return index;
        }

    public:
        template<typename... Args, typename = std::enable_if_t<sizeof...(Args) >= Rank>>
        explicit Tensor(Args... args) {
            constexpr size_t num_args = sizeof...(Args);
            if (num_args != Rank) {
                cerr << "Number of dimensions do not match with " << Rank << endl;
                throw invalid_argument("Number of dimensions do not match with " + to_string(Rank));
            }

            array<size_t, sizeof...(Args)> full_shape{static_cast<size_t>(args)...};
            for (size_t i = 0; i < Rank; ++i) {
                shape_[i] = full_shape[i];
            }
            size_t total = total_size(shape_);
            data_.resize(total);
        }

        Tensor(std::initializer_list<size_t> shape, std::initializer_list<T> values = {}) {
            if (shape.size() != Rank) {
                throw std::invalid_argument("Number of dimensions in shape does not match tensor Rank");
            }

            std::copy(shape.begin(), shape.end(), shape_.begin());
            size_t total_elements = total_size(shape_);
            data_.resize(total_elements);

            if (!values.empty()) {
                if (values.size() != total_elements) {
                    throw std::invalid_argument("Number of values does not match tensor size");
                }
                std::copy(values.begin(), values.end(), data_.begin());
            }
        }

        explicit Tensor(std::initializer_list<T> values) {
            if constexpr (Rank == 1) {
                shape_[0] = values.size();
                data_.resize(values.size());
                std::copy(values.begin(), values.end(), data_.begin());
            } else {
                // Para tensores de rango > 1, shape por defecto sería [values.size(), 1, 1, ...]
                shape_.fill(1);
                shape_[0] = values.size();
                data_.resize(values.size());
                std::copy(values.begin(), values.end(), data_.begin());
            }
        }

        Tensor() = default;

        const array<size_t, Rank>& shape() const {
            return shape_;
        }
    public:
        void reshape(const array<size_t, Rank> new_shape) {
            auto new_size = total_size(new_shape);
            if (new_size > data_.size()) {
                throw std::invalid_argument("New shape does not match total size.");
            }
            data_.resize(new_size);
            shape_ = new_shape;
        }

        template <class... Args>
        void reshape(Args... args) {
            std::vector<size_t> dims{static_cast<size_t>(args)...};

            if (dims.size() > Rank)
                throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));


            std::array<size_t, Rank> new_shape{};
            std::copy(dims.begin(), dims.end(), new_shape.begin());

            reshape(new_shape);
        }

        void fill(const T& value) {
            std::fill(data_.begin(), data_.end(), value);
        }

        T& operator()(const array<size_t, Rank>& indices) {
            return data_[get_index(indices)];
        }

        const T& operator()(const array<size_t, Rank>& indices) const {
            return data_[get_index(indices)];
        }

        template <typename... Args>
        T& operator()(Args... args) {
            static_assert(sizeof...(Args) == Rank, "Número incorrecto de índices");
            array<size_t, Rank> indices = {static_cast<size_t>(args)...};
            return (*this)(indices);
        }

        template <typename... Args>
        const T& operator()(Args... args) const {
            static_assert(sizeof...(Args) == Rank, "Número incorrecto de índices");
            array<size_t, Rank> indices = {static_cast<size_t>(args)...};
            return (*this)(indices);
        }


        Tensor<T, Rank>& operator=(initializer_list<initializer_list<T>> values) {
            static_assert(Rank == 2, "Nested initializer list assignment only supported for 2D tensors.");
            size_t rows = shape_[0];
            size_t cols = shape_[1];

            if (values.size() != rows)
                throw invalid_argument("Row count does not match tensor shape.");

            size_t idx = 0;
            for (const auto& row : values) {
                if (row.size() != cols)
                    throw invalid_argument("Column count does not match tensor shape.");
                for (const auto& val : row)
                    data_[idx++] = val;
            }

            return *this;
        }

        Tensor<T, Rank>& operator=(initializer_list<T> values) {
            if (values.size() != data_.size())
                throw invalid_argument("Data size does not match tensor size");
            copy(values.begin(), values.end(), data_.begin());
            return *this;
        }

        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }
        auto cbegin() const { return data_.cbegin(); }
        auto cend() const { return data_.cend(); }


        friend std::ostream& operator<<(std::ostream& os, const Tensor<T, Rank>& t) {
            // Caso especial para tensores 1D
            if constexpr (Rank == 1) {
                for (size_t i = 0; i < t.shape_[0]; ++i) {
                    os << t.data_[i];
                    if (i + 1 < t.shape_[0]) os << " ";
                }
            }
            // Caso especial para tensores 2D
            else if constexpr (Rank == 2) {
                os << "{\n";
                size_t rows = t.shape_[0];
                size_t cols = t.shape_[1];
                for (size_t i = 0; i < rows; ++i) {
                    os << "  ";
                    for (size_t j = 0; j < cols; ++j) {
                        os << t.data_[i * cols + j];
                        if (j + 1 < cols) os << " ";
                    }
                    if (i + 1 < rows) os << "\n";
                }
                os << "\n}";
            }
            else {
                std::function<void(std::ostream&, size_t, size_t, size_t, size_t)> print;
                print = [&](std::ostream& os, size_t dim, size_t offset, size_t stride, size_t indent) {
                    string indent_str(indent, ' ');
                    if (dim == Rank - 1) {
                        os << indent_str;
                        for (size_t i = 0; i < t.shape_[dim]; ++i) {
                            os << t.data_[offset + i];
                            if (i + 1 < t.shape_[dim]) os << " ";
                        }
                    } else {
                        size_t inner_stride = 1;
                        for (size_t d = dim + 1; d < Rank; ++d)
                            inner_stride *= t.shape_[d];
                        os << indent_str << "{\n";
                        for (size_t i = 0; i < t.shape_[dim]; ++i) {
                            print(os, dim + 1, offset + i * inner_stride, inner_stride, indent + 2);
                            if (i + 1 < t.shape_[dim]) os << "\n";
                        }
                        os << "\n" << indent_str << "}";
                    }
                };
                print(os, 0, 0, t.data_.size(), 0);
            }
            return os;
        }

        friend Tensor<T, Rank> binary_op(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b, std::function<T(const T&, const T&)> op) {
            auto new_shape = broadcast_shape(a.shape_, b.shape_);
            Tensor<T, Rank> result;
            result.shape_ = new_shape;
            result.data_.resize(result.total_size(new_shape));

            array<size_t, Rank> indices;
            size_t total = result.total_size(new_shape);
            for (size_t i = 0; i < total; ++i) {
                size_t idx = i;
                for (size_t j = Rank; j-- > 0;) {
                    indices[j] = idx % new_shape[j];
                    idx /= new_shape[j];
                }
                result.data_[i] = op(
                    a.data_[a.broadcast_index(indices, a.shape_)],
                    b.data_[b.broadcast_index(indices, b.shape_)]
                );
            }
            return result;
        }

        friend Tensor<T, Rank> operator+(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
            return binary_op(a, b, std::plus<T>());
        }

        friend Tensor<T, Rank> operator-(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
            return binary_op(a, b, std::minus<T>());
        }

        friend Tensor<T, Rank> operator*(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
            return binary_op(a, b, std::multiplies<T>());
        }

        friend Tensor<T, Rank> operator+(const Tensor<T, Rank>& a, const T& scalar) {
            Tensor<T, Rank> result = a;
            for (size_t i = 0; i < a.data_.size(); ++i)
                result.data_[i] += scalar;
            return result;
        }

        friend Tensor<T, Rank> operator-(const Tensor<T, Rank>& a, const T& scalar) {
            Tensor<T, Rank> result = a;
            for (size_t i = 0; i < a.data_.size(); ++i)
                result.data_[i] -= scalar;
            return result;
        }

        friend Tensor<T, Rank> operator*(const Tensor<T, Rank>& a, const T& scalar) {
            Tensor<T, Rank> result = a;
            for (size_t i = 0; i < a.data_.size(); ++i)
                result.data_[i] *= scalar;
            return result;
        }

        friend Tensor<T, Rank> operator/(const Tensor<T, Rank>& a, const T& scalar) {
            Tensor<T, Rank> result = a;
            for (size_t i = 0; i < a.data_.size(); ++i)
                result.data_[i] /= scalar;
            return result;
        }

        // OPERACIONES ENTRE ESCALAR Y TENSOR (escalar op tensor)
        friend Tensor<T, Rank> operator+(const T& scalar, const Tensor<T, Rank>& a) {
            return a + scalar;
        }

        friend Tensor<T, Rank> operator-(const T& scalar, const Tensor<T, Rank>& a) {
            Tensor<T, Rank> result = a;
            for (size_t i = 0; i < a.data_.size(); ++i)
                result.data_[i] = scalar - result.data_[i];
            return result;
        }

        friend Tensor<T, Rank> operator*(const T& scalar, const Tensor<T, Rank>& a) {
            return a * scalar;
        }

        friend Tensor<T, Rank> operator/(const T& scalar, const Tensor<T, Rank>& a) {
            Tensor<T, Rank> result = a;
            for (size_t i = 0; i < a.data_.size(); ++i)
                result.data_[i] = scalar / a.data_[i];
            return result;
        }
    };

    template <typename T, size_t D>
    auto transpose_2d(const Tensor<T, D>& input) {
        if (D < 2) {
            throw std::runtime_error("Cannot transpose 1D tensor: need at least 2 dimensions");
        }

        if constexpr (D == 2) {
            Tensor<T, 2> result(input.shape()[1], input.shape()[0]);
            for (size_t i = 0; i < input.shape()[0]; ++i) {
                for (size_t j = 0; j < input.shape()[1]; ++j) {
                    result(j, i) = input(i, j);
                }
            }
            return result;
        }
        else {
            array<size_t, D> new_shape;
            for (size_t i = 0; i < D - 2; ++i) {
                new_shape[i] = input.shape()[i];
            }
            new_shape[D-2] = input.shape()[D-1];
            new_shape[D-1] = input.shape()[D-2];

            Tensor<T, D> result;
            result.shape_ = new_shape;
            result.data_.resize(input.data_.size());

            size_t batch_size = 1;
            for (size_t i = 0; i < D - 2; ++i) {
                batch_size *= input.shape()[i];
            }

            const size_t rows = input.shape()[D-2];
            const size_t cols = input.shape()[D-1];

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        array<size_t, D> original_idx;
                        array<size_t, D> transposed_idx;
                        size_t temp = b;
                        for (size_t k = D - 3; k != static_cast<size_t>(-1); --k) {
                            original_idx[k] = temp % input.shape()[k];
                            transposed_idx[k] = original_idx[k];
                            temp /= input.shape()[k];
                        }

                        original_idx[D-2] = i;
                        original_idx[D-1] = j;
                        transposed_idx[D-2] = j;
                        transposed_idx[D-1] = i;
                        result(transposed_idx) = input(original_idx);
                    }
                }
            }

            return result;
        }
    }

    template <typename T>
    Tensor<T, 2> matrix_product(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
        if (A.shape()[1] != B.shape()[0]) {
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        }

        size_t rows = A.shape()[0];
        size_t cols = B.shape()[1];
        size_t common = A.shape()[1];

        Tensor<T, 2> result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T sum = T();
                for (size_t k = 0; k < common; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T, 3> matrix_product(const Tensor<T, 3>& A, const Tensor<T, 3>& B) {
        if (A.shape()[0] != B.shape()[0] || A.shape()[2] != B.shape()[1]) {
            throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
        }

        const size_t batches = A.shape()[0];
        const size_t rows = A.shape()[1];
        const size_t cols = B.shape()[2];
        const size_t common = A.shape()[2];

        Tensor<T, 3> result;
        result.shape_[0] = batches;
        result.shape_[1] = rows;
        result.shape_[2] = cols;
        result.data_.resize(batches * rows * cols);

        for (size_t b = 0; b < batches; ++b) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    T sum = T();
                    for (size_t k = 0; k < common; ++k) {
                        sum += A(b, i, k) * B(b, k, j);
                    }
                    size_t index = b * rows * cols + i * cols + j;
                    result.data_[index] = sum;
                }
            }
        }
        return result;
    }
}

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H