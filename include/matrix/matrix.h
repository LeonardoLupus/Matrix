#pragma once

#include <iostream>
#include <concepts>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>

namespace concepttypes {
    template <typename Type>
    concept FPType = std::same_as<Type, float> ||
                     std::same_as<Type, double>;

    template <typename Type>
    concept NumberType = std::same_as<Type, int> ||
                         std::same_as<Type, float> ||
                         std::same_as<Type, double> ||
                         std::same_as<Type, int8_t> ||
                         std::same_as<Type, int16_t> ||
                         std::same_as<Type, int32_t> ||
                         std::same_as<Type, int64_t> ||
                         std::same_as<Type, uint8_t> ||
                         std::same_as<Type, uint16_t> ||
                         std::same_as<Type, uint32_t> ||
                         std::same_as<Type, uint64_t>;
}

template <concepttypes::NumberType T>
class Matrix;

namespace compare {
    template <concepttypes::FPType T>
    auto fp(T a, T b) -> int;

    template<concepttypes::FPType T>
    auto fp(T a, T b) -> int {
        constexpr auto epsilon = 1.0e-14;   //std::numeric_limits<T>::epsilon()
        if(std::fabs(a - b) <= epsilon) {return 0;}
        if(a - b > 0) {return 1;}
        return -1;
    }
}

template <concepttypes::NumberType T>
class Matrix {
public:
    Matrix();
    Matrix(const Matrix<T>& other);
    Matrix(size_t column, size_t row);
    explicit Matrix(std::vector<std::vector<T>> m);
    Matrix(std::vector<T> m, size_t column, size_t row);
    ~Matrix() = default;

    template<concepttypes::NumberType Tto>
    auto cast_copy() const -> Matrix<Tto>;

    auto identity_matrix() -> Matrix<T>&;

    auto fill(T data) -> Matrix<T>&;


    auto operator=(const Matrix<T>& other) -> Matrix<T>&;

    template<concepttypes::NumberType OtherT>
    auto operator+(const Matrix<OtherT>& m) const -> decltype(auto);

    template<concepttypes::NumberType OtherT>
    auto operator-(const Matrix<OtherT>& m) const -> decltype(auto);

    template<concepttypes::NumberType OtherT>
    auto operator*(const Matrix<OtherT>& m) const -> decltype(auto);

    template<concepttypes::NumberType OtherT>
    friend auto operator==(const Matrix<OtherT>& first, const Matrix<OtherT>& second) -> bool;

    template<concepttypes::NumberType TNum>
    [[maybe_unused]] auto multiply(TNum number) -> Matrix<T>;

    template<concepttypes::NumberType TNum>
    [[maybe_unused]] auto division(TNum number) -> Matrix<T>;


    auto at(size_t x, size_t y) -> T&;
    [[maybe_unused]] [[nodiscard]] auto at(size_t x, size_t y) const -> const T&;
    [[nodiscard]] auto number_col() const -> size_t;
    [[nodiscard]] auto number_row() const -> size_t;
    [[nodiscard]] auto size() const -> size_t;
    [[nodiscard]] auto is_square() const -> bool;
    [[nodiscard]] auto is_empty() const -> bool;


    auto resize(size_t column, size_t row) -> void;
    auto swap_column(size_t i, size_t j) -> void;
    auto swap_line(size_t i, size_t j) -> void;


    auto print() const -> void;


    auto transposition() -> Matrix<T>&;
    [[nodiscard]] auto t() const -> Matrix<T>;
    [[nodiscard]] auto det() const -> double;
    [[nodiscard]] auto minor(size_t x, size_t y) const -> Matrix<T>;
    [[nodiscard]] auto inverse() const -> Matrix<double>;

private:
    auto is_valid_index(size_t column, size_t row) const -> void;
    auto is_correct_to_sq_use() const -> void;

    std::vector<T> m_matrix;
    size_t m_col;
    size_t m_row;
    bool m_transposition{};
};


template<concepttypes::NumberType T>
template<concepttypes::NumberType Tto>
auto Matrix<T>::cast_copy() const -> Matrix<Tto> {
    std::vector<Tto> buffer(size());
    for (int i = 0; i < number_row(); ++i) {
        for (int j = 0; j < number_col(); ++j) {
            buffer[j + i*number_col()] = static_cast<Tto>(at(j, i));
        }
    }
    Matrix <Tto> c_m(buffer, number_col(), number_row());
    return c_m;
}

template<concepttypes::NumberType T>
Matrix<T>::Matrix() : m_col(0), m_row(0), m_transposition(false) {}

template<concepttypes::NumberType T>
Matrix<T>::Matrix(const Matrix<T> &other) {
    m_matrix.resize(other.size());
    for (int i = 0; i < m_matrix.size(); ++i) {
        m_matrix.at(i) = static_cast<T>(other.m_matrix.at(i));
    }
    m_col = other.m_col;
    m_row = other.m_row;
    m_transposition = other.m_transposition;
}

template<concepttypes::NumberType T>
Matrix<T>::Matrix(const size_t column, const size_t row):
        m_col(column),
        m_row(row),
        m_transposition(false)
{
    m_matrix.resize(column*row, 0);
}

template<concepttypes::NumberType T>
Matrix<T>::Matrix(std::vector<std::vector<T>> m):
    m_row(m.size()),
    m_col(m[0].size()),
    m_matrix(false)
{
    for(auto &i: m){
        for (auto &j: i) {
            m_matrix.push_back(j);
        }
    }
    if (size() != m_row * m_col) {
        std::cout << "\nWarning! The array length does not match the matrix size\n";
        m_matrix.resize(m_row * m_col);
    }
}

template<concepttypes::NumberType T>
Matrix<T>::Matrix(std::vector<T> m, const size_t column, const size_t row) :
        m_matrix(std::vector<T>(m)),
        m_col(column),
        m_row(row),
        m_transposition(false)
{
    if (column * row != m.size()) {
        std::cout << "\nWarning! The array length does not match the matrix size\n";
    }
    m_matrix.resize(column*row, 0);
}

template<concepttypes::NumberType T>
auto Matrix<T>::identity_matrix() -> Matrix<T>& {
    is_correct_to_sq_use();
    for (size_t i = 0; i < m_row; ++i) {
        for (size_t j = 0; j < m_col; ++j) {
            if(i == j) {at(j, i) = 1;}
            else {at(j, i) = 0;}
        }
    }
    m_transposition = false;
    return *this;
}

template<concepttypes::NumberType T>
auto Matrix<T>::fill(T data) -> Matrix<T>& {
    for (auto &i : m_matrix) {
        i = data;
    }
    return *this;
}



template<concepttypes::NumberType T>
auto Matrix<T>::operator=(const Matrix<T> &other) -> Matrix<T> & {
    if (this != &other){
        m_matrix.clear();
        m_matrix.resize(other.size());
        for (int i = 0; i < other.size(); ++i) {
            m_matrix[i] = other.m_matrix[i];
        }
        m_col = other.m_col;
        m_row = other.m_row;
        m_transposition = other.m_transposition;
    }
    return *this;
}

template<concepttypes::NumberType T>
template<concepttypes::NumberType OtherT>
auto Matrix<T>::operator+(const Matrix<OtherT>& m) const -> decltype(auto) {
    Matrix<decltype(at(0,0) + m.at(0,0))> matrix;
    if(number_row() != m.number_row() || number_col() != m.number_col()) {
        std::cout << "\nWarning! Addition of matrices of inappropriate sizes\n";
        return matrix;
    }
    matrix.resize(number_col(), number_row());
    for (int i = 0; i < number_row(); ++i) {
        for (int j = 0; j < number_col(); ++j) {
            matrix.at(j, i) = at(j, i) + m.at(j, i);
        }
    }
    return matrix;
}

template<concepttypes::NumberType T>
template<concepttypes::NumberType OtherT>
auto Matrix<T>::operator-(const Matrix<OtherT> &m) const -> decltype(auto) {
    Matrix<decltype(at(0,0) - m.at(0,0))> matrix;
    if(number_row() != m.number_row() || number_col() != m.number_col()) {
        std::cout << "\nWarning! Subtraction of matrices of inappropriate sizes\n";
        return matrix;
    }
    matrix.resize(number_col(), number_row());
    for (int i = 0; i < number_row(); ++i) {
        for (int j = 0; j < number_col(); ++j) {
            matrix.at(j, i) = at(j, i) - m.at(j, i);
        }
    }
    return matrix;
}

template<concepttypes::NumberType T>
template<concepttypes::NumberType OtherT>
auto Matrix<T>::operator*(const Matrix<OtherT> &m) const -> decltype(auto) {
    Matrix<decltype(at(0,0) * m.at(0,0))> matrix;
    if(number_col() != m.number_row()) {
        std::cout << "\nWarning! Multiplication of matrices of inappropriate sizes\n";
        return matrix;
    }
    matrix.resize(m.number_col(), number_row());
    for (size_t i = 0; i < number_row(); ++i) {
        for (size_t j = 0; j < m.number_col(); ++j) {
            for (size_t k = 0; k < number_col(); ++k) {
                matrix.at(j, i) += at(k, i) * m.at(j, k);
            }
            if (auto _ = compare::fp(matrix.at(j, i), 0.); _ == 0) {
                matrix.at(j, i) = 0;
            }
        }
    }
    return matrix;
}

template<concepttypes::NumberType OtherT>
auto operator==(const Matrix<OtherT> &first, const Matrix<OtherT> &second) -> bool {
    if (first.number_col() != second.number_col() || first.number_row() != second.number_row()) {
        return false;
    }
    for (size_t i = 0; i < first.number_row(); ++i) {
        for (size_t j = 0; j < first.number_col(); ++j) {
            auto a = static_cast<double>(first.at(j, i));
            auto b = static_cast<double>(second.at(j, i));
            if (compare::fp(a, b) != 0) {return false;}
        }
    }
    return true;
}

template<concepttypes::NumberType T>
template<concepttypes::NumberType TNum>
[[maybe_unused]] auto Matrix<T>::multiply(TNum number) -> Matrix<T> {
    for (auto &i : m_matrix) {
        i *= number;
        if (auto _ = compare::fp(i, 0.); _ == 0) {
            i = 0;
        }
    }
    return *this;
}

template<concepttypes::NumberType T>
template<concepttypes::NumberType TNum>
auto Matrix<T>::division(TNum number) -> Matrix<T> {
    for (auto &i : m_matrix) {
        i /= number;
        if (auto _ = compare::fp(i, 0.); _ == 0) {
            i = 0;
        }
    }
    return *this;
}



template<concepttypes::NumberType T>
auto Matrix<T>::at(const size_t x, const size_t y) -> T& {
    is_valid_index(x, y);
    if (m_transposition) {
        return m_matrix.at(y + x*m_row);
    }
    return m_matrix.at(x + y*m_col);
}

template<concepttypes::NumberType T>
[[maybe_unused]] auto Matrix<T>::at(const size_t x, const size_t y) const -> const T& {
    is_valid_index(x, y);
    if (m_transposition) {
        return m_matrix[y + x*m_row];
    }
    return m_matrix[x + y*m_col];
}

template<concepttypes::NumberType T>
auto Matrix<T>::number_col() const -> size_t {
    return m_col;
}

template<concepttypes::NumberType T>
auto Matrix<T>::number_row() const -> size_t {
    return m_row;
}

template<concepttypes::NumberType T>
auto Matrix<T>::size() const -> size_t {
    return m_matrix.size();
}

template<concepttypes::NumberType T>
auto Matrix<T>::is_square() const -> bool {
    return m_col == m_row;
}

template<concepttypes::NumberType T>
auto Matrix<T>::is_empty() const -> bool {
    return m_matrix.empty();
}



template<concepttypes::NumberType T>
auto Matrix<T>::resize(size_t column, size_t row) -> void {
    m_matrix.clear();
    m_col = column;
    m_row = row;
    m_matrix.resize(column*row);
    m_transposition = false;
}

template<concepttypes::NumberType T>
auto Matrix<T>::swap_column(size_t i, size_t j) -> void {
    if (i >= m_col || j >= m_col) {
        std::cout << "\nError! Attempt to access non-existent index (swap_column)\n";
        exit(-1);
    }
    for (int k = 0; k < m_row; ++k) {
        std::swap(at(i, k), at(j, k));
    }
}

template<concepttypes::NumberType T>
auto Matrix<T>::swap_line(size_t i, size_t j) -> void {
    if (i >= m_row || j >= m_row) {
        std::cout << "\nError! Attempt to access non-existent index (swap_line)\n";
        exit(-1);
    }
    if (i < 0 || i >= m_row || j < 0 || j >= m_row) {return;}
    for (int k = 0; k < m_col; ++k) {
        std::swap(at(k, i), at(k, j));
    }
}



template<concepttypes::NumberType T>
auto Matrix<T>::print() const -> void {
    std::cout << "\n";
    for (int i = 0; i < m_row; ++i) {
        for (int j = 0; j < m_col; ++j) {
            std::cout << at(j, i) << "\t";
        }
        std::cout << "\n";
    }
}



template<concepttypes::NumberType T>
auto Matrix<T>::is_valid_index(const size_t column, const size_t row) const -> void {
    if (column >= m_col || row >= m_row) {
        std::cout << "\nError! INVALID INDEX\n";
        exit(-1);
    }
}

template<concepttypes::NumberType T>
auto Matrix<T>::is_correct_to_sq_use() const -> void {
    if (is_empty() || (!is_square())) {
        std::cout << "\nERROR! Attempt to call a method on a non-square or empty matrix\n";
        exit(-1);
    }
}



template<concepttypes::NumberType T>
auto Matrix<T>::transposition() -> Matrix<T>& {
    if (m_matrix.size() < 2) {return *this;}
    m_transposition = not m_transposition;
    std::swap(m_col, m_row);
    return *this;
}

template<concepttypes::NumberType T>
auto Matrix<T>::t() const -> Matrix<T> {
    Matrix matrix = *this;
    matrix.transposition();
    return matrix;
}

template<concepttypes::NumberType T>
auto det_2x2(const Matrix<T>& m) -> double {
    double det = m.at(0, 0)*m.at(1, 1) - m.at(0,1)*m.at(1, 0);
    if (compare::fp(det, 0.) == 0) {det = 0;}
    return det;
}

template<concepttypes::NumberType T>
auto det_3x3(const Matrix<T>& m) -> double {
    double det = m.at(0, 0) * (m.at(1, 1)*m.at(2, 2) - m.at(1, 2)*m.at(2, 1)) -
            m.at(0, 1) * (m.at(1, 0)*m.at(2, 2) - m.at(1, 2)*m.at(2, 0)) +
            m.at(0, 2) * (m.at(1, 0)*m.at(2, 1) - m.at(1, 1)*m.at(2, 0));
    if (compare::fp(det, 0.) == 0) {det = 0;}
    return det;
}

template<concepttypes::NumberType T>
auto Matrix<T>::det() const -> double {
    is_correct_to_sq_use();
    switch (m_col) {
        case 1:
            return at(0,0);
        case 2:
            return det_2x2(*this);
        case 3:
            return det_3x3(*this);
        default:
            break;
    }

    Matrix<T> copy = *this;
    auto size = m_col;
    auto det = 1.0;
    for (size_t i = 0; i < size - 1; ++i) {
        if (copy.at(i ,i) == 0) {
            size_t j = i + 1;
            while (j < size && copy.at(j ,i) == 0) {
                ++j;
            }
            if (j == size) {
                return 0.0;
            }
            copy.swap_line(i, j);
            det *= -1;
        }
        for (auto j = i + 1; j < size; ++j) {
            double factor = copy.at(j, i)/ copy.at(i, i);
            for (auto k = i; k < size; ++k) {
                copy.at(j, k) -= factor * copy.at(i, k);
            }
        }
    }
    for (size_t i = 0; i < size; ++i) {
        det *= copy.at(i, i);
    }
    if (compare::fp(det, 0.) == 0) {det = 0;}
    return det;
}

template<concepttypes::NumberType T>
auto Matrix<T>::minor(size_t x, size_t y) const -> Matrix<T> {
    is_valid_index(x, y);
    Matrix<T> buffer;
    if(m_col == 1 || m_row == 1) {return buffer;}
    buffer.resize(m_col-1, m_row-1);
    for (size_t i = 0; i < m_row; ++i) {
        for (size_t j = 0; j < m_col; ++j) {
            if (i == y) {break;}
            if (j == x) {continue;}
            auto _x = j;
            auto _y = i;
            if (_x > x) {--_x;}
            if (_y > y) {--_y;}
            buffer.at(_x, _y) = at(j, i);
        }
    }
    return buffer;
}

template<concepttypes::NumberType T>
auto Matrix<T>::inverse() const -> Matrix<double> {
    is_correct_to_sq_use();
    if(m_col == 1) {return Matrix<T>({{1/at(0,0)}});}
    Matrix<double> buffer(m_col, m_row);
    auto d = det();
    if (compare::fp(d, 0.) == 0) {return buffer;}
    for (size_t i = 0; i < m_row; ++i) {
        for (size_t j = 0; j < m_col; ++j) {
            buffer.at(j, i) = minor(j, i).det() * ((j+i)%2==0?1:-1);
        }
    }
    buffer.transposition();
    buffer.division(d);
    return buffer;
}



namespace matrix {
    template<concepttypes::NumberType TEquation, concepttypes::NumberType TFree>
    auto solve_SLAE(const Matrix<TEquation>& equation, const Matrix<TFree>& free_members) -> Matrix<double> {
        return equation.inverse() * free_members.template cast_copy<double>();
    }

    template <concepttypes::NumberType T>
    auto identity_matrix(size_t rank) -> Matrix<T> {
        return Matrix<T>(rank, rank).identity_matrix();
    }
}


