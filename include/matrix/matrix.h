#pragma once

#include <stdexcept>
#include <algorithm>
#include <string>
#include <string_view>
#include <iostream>
#include <concepts>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <compare>

namespace detail
{
  auto warn(std::ostream& os, std::string_view message) -> std::ostream&;

  template <typename T>
  concept Number = std::integral<T> || std::floating_point<T>;

  template <Number T>
  auto compare(T a, T b, float precision = 1.0F) -> std::strong_ordering {
    if constexpr(std::is_floating_point_v<T>)
      return std::abs(a - b) <= (std::numeric_limits<T>::epsilon() * precision)
        ? std::strong_ordering::equal
        : (a < b
          ? std::strong_ordering::less
          : std::strong_ordering::greater
        );
    else
      return a == b
        ? std::strong_ordering::equal
        : (a < b
          ? std::strong_ordering::less
          : std::strong_ordering::greater
        );
  }

  template <Number T>
  auto is_null(T a, float precision = 1.0F) -> bool {
    return compare(a, .0, precision) == std::strong_ordering::equal;
  }

  // todo
  template <Number T>
  auto zero_if_null(T& t) -> void {
    if(is_null(t))
      t = T(0.0);
  }
} // namespace detail

template <detail::Number T>
class Matrix;

template <detail::Number T>
class Matrix {
public:
    Matrix();
    Matrix(const Matrix<T>& other);
    Matrix(size_t column, size_t row);
    explicit Matrix(const std::vector<std::vector<T>>& m);
    Matrix(std::vector<T> m, size_t column, size_t row);
    ~Matrix() = default;

    template<detail::Number Tto>
    auto cast_copy() const -> Matrix<Tto>;

    auto identity_matrix() -> Matrix<T>&;

    auto fill(T data) -> Matrix<T>&;


    auto operator=(const Matrix<T>& other) -> Matrix<T>&;

    template<detail::Number OtherT>
    auto operator+(const Matrix<OtherT>& m) const -> decltype(auto);

    template<detail::Number OtherT>
    auto operator-(const Matrix<OtherT>& m) const -> decltype(auto);

    template<detail::Number T2>
    auto operator*(const Matrix<T2>& m) const -> decltype(auto)
    {
      Matrix<decltype(this->at(0,0) * m.at(0,0))> matrix;
      if(number_col() != m.number_row()) {
        detail::warn(std::cout, "Multiplication of matrices of inappropriate sizes");
        return matrix;
      }
      matrix.resize(m.number_col(), number_row());
      for (size_t i = 0; i < number_row(); ++i) {
        for (size_t j = 0; j < m.number_col(); ++j) {
          for (size_t k = 0; k < number_col(); ++k) {
            matrix.at(j, i) += at(k, i) * m.at(j, k);
          }
          if(detail::compare(matrix.at(j, i), 0.) == std::strong_ordering::equal)
            matrix.at(j, i) = 0;
        }
      }
      return matrix;
    }

    template<detail::Number T2>
    [[nodiscard]] constexpr auto operator==(Matrix<T2> const& other) const -> bool {
      if (this->number_col() != other.number_col() || this->number_row() != other.number_row()) {
        return false;
      }
      for (size_t i = 0; i < this->number_row(); ++i) {
        for (size_t j = 0; j < this->number_col(); ++j) {
          auto a = static_cast<double>(this->at(j, i));
          auto b = static_cast<double>(other(j, i));
          if(detail::compare(a, b, 1e2) != std::strong_ordering::equal)
            return false;
        }
      }
      return true;
    }

    template<detail::Number TNum>
    [[maybe_unused]] auto multiplication(TNum number) -> Matrix<T>;

    template<detail::Number TNum>
    [[maybe_unused]] auto division(TNum number) -> Matrix<T>;


    [[nodiscard]] auto at(size_t x, size_t y) -> T&;
    [[nodiscard]] auto at(size_t x, size_t y) const -> const T&;
    [[nodiscard]] auto operator()(size_t x, size_t y) -> T& { return this->at(x, y); }
    [[nodiscard]] auto operator()(size_t x, size_t y) const -> const T& { return this->at(x, y); }

    [[nodiscard]] auto number_col() const -> size_t;
    [[nodiscard]] auto number_row() const -> size_t;
    [[nodiscard]] auto size() const -> size_t;
    [[nodiscard]] auto is_square() const -> bool;
    [[nodiscard]] auto is_empty() const -> bool;


    auto resize(size_t column, size_t row) -> void;
    auto swap_column(size_t i, size_t j) -> void;
    auto swap_line(size_t i, size_t j) -> void;

    auto transpose() -> Matrix<T>&;
    [[nodiscard]] auto Tr() const -> Matrix<T>;
    [[nodiscard]] auto det() const -> double;
    [[nodiscard]] auto minor(size_t x, size_t y) const -> Matrix<T>;
    [[nodiscard]] auto inverse() const -> Matrix<double>;

    template <detail::Number U>
    friend auto operator<<(std::ostream& os, const Matrix<U>& m) -> std::ostream&;

    template <detail::Number U>
    friend auto operator>>(std::istream& is, Matrix<U>& m) -> std::istream&;

private:
    auto is_valid_index(size_t column, size_t row) const -> void;
    auto is_correct_to_sq_use() const -> void;

    std::vector<T> m_matrix;
    size_t m_col;
    size_t m_row;
    bool m_transposition{};
};

template <detail::Number U>
auto operator<<(std::ostream& os, Matrix<U> const& m) -> std::ostream& {
  for(int i = 0; i < m.m_row; ++i) {
    for(int j = 0; j < m.m_col; ++j)
      os << "\t" << m.at(j, i);
    os << std::endl;
  }
  return os;
}

template <detail::Number U>
auto operator>>(std::istream& is, Matrix<U>& m) -> std::istream& {
    auto value = m.at(0, 0);
    auto i = 0;
    auto j = 0;
    while (j < m.number_row()) {
        is >> value;
        if (i >= m.number_col()) {
            throw std::runtime_error("Error! The format of the matrix in the file does not match the transmitted matrix!");
            break;
        }
        m.at(i, j) = value;
        ++i;
        if (is.peek() == '\n') {
            if (i != m.number_col()) {
                throw std::runtime_error("Error! The format of the matrix in the file does not match the transmitted matrix!");
                break;
            }
            i = 0;
            ++j;
            continue;
        }
    }
    
    return is;
}

template<detail::Number T>
template<detail::Number Tto>
auto Matrix<T>::cast_copy() const -> Matrix<Tto> {
    std::vector<Tto> buffer(this->size());
    std::transform(this->m_matrix.begin(), this->m_matrix.end(), buffer.begin(), [](T a) {
      return static_cast<Tto>(a);
    });
    Matrix <Tto> c_m(buffer, number_col(), number_row());
    return c_m;
}

template<detail::Number T>
Matrix<T>::Matrix() : m_col(0), m_row(0), m_transposition(false) {}

template<detail::Number T>
Matrix<T>::Matrix(const Matrix<T> &other) {
    m_matrix.resize(other.size());
    for (int i = 0; i < m_matrix.size(); ++i) {
        m_matrix.at(i) = static_cast<T>(other.m_matrix.at(i));
    }
    m_col = other.m_col;
    m_row = other.m_row;
    m_transposition = other.m_transposition;
}

template<detail::Number T>
Matrix<T>::Matrix(const size_t column, const size_t row):
        m_col(column),
        m_row(row),
        m_transposition(false)
{
    m_matrix.resize(column*row, 0);
}

template<detail::Number T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& m):
    m_row(m.size()),
    m_col(m[0].size()),
    m_transposition(false)
{
    for(auto &i: m){
        for (auto &j: i) {
            m_matrix.push_back(j);
        }
    }
    if (size() != m_row * m_col) {
        detail::warn(std::cout, "The array length does not match the matrix size");
        m_matrix.resize(m_row * m_col);
    }
}

template<detail::Number T>
Matrix<T>::Matrix(std::vector<T> m, const size_t column, const size_t row) :
        m_matrix(std::move(std::vector<T>(m))),
        m_col(column),
        m_row(row),
        m_transposition(false)
{
    if (column * row != m.size())
      detail::warn(std::cout, "The array length does not match the matrix size");

    m_matrix.resize(column*row, 0);
}

template<detail::Number T>
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

template<detail::Number T>
auto Matrix<T>::fill(T data) -> Matrix<T>& {
    for (auto &i : m_matrix) {
        i = data;
    }
    return *this;
}



template<detail::Number T>
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

template<detail::Number T>
template<detail::Number OtherT>
auto Matrix<T>::operator+(const Matrix<OtherT>& m) const -> decltype(auto) {
    Matrix<decltype(at(0,0) + m.at(0,0))> matrix;
    if(number_row() != m.number_row() || number_col() != m.number_col()) {
      detail::warn(std::cout, "Addition of matrices of inappropriate sizes");
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

template<detail::Number T>
template<detail::Number OtherT>
auto Matrix<T>::operator-(const Matrix<OtherT> &m) const -> decltype(auto) {
    Matrix<decltype(at(0,0) - m.at(0,0))> matrix;
    if(number_row() != m.number_row() || number_col() != m.number_col()) {
      detail::warn(std::cout, "Subtraction of matrices of inappropriate sizes");
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

template<detail::Number T>
template<detail::Number TNum>
[[maybe_unused]] auto Matrix<T>::multiplication(TNum number) -> Matrix<T>{
    auto buffer = *this;
    for (auto &i : buffer.m_matrix) {
        i *= number;
        if(detail::is_null(i))
          i = 0;
    }
    return buffer;
}

template<detail::Number T>
template<detail::Number TNum>
auto Matrix<T>::division(TNum number) -> Matrix<T> {
    auto buffer = *this;
    for (auto &i : buffer.m_matrix) {
        i /= number;
        if(detail::is_null(i)) // precision ? todo
          i = 0;
    }
    return buffer;
}

template<detail::Number T>
auto Matrix<T>::at(const size_t x, const size_t y) -> T& {
    is_valid_index(x, y);
    if (m_transposition) {
        return m_matrix.at(y + x*m_row);
    }
    return m_matrix.at(x + y*m_col);
}

template<detail::Number T>
[[maybe_unused]] auto Matrix<T>::at(const size_t x, const size_t y) const -> const T& {
    is_valid_index(x, y);
    if (m_transposition) {
        return m_matrix[y + x*m_row];
    }
    return m_matrix[x + y*m_col];
}

template<detail::Number T>
auto Matrix<T>::number_col() const -> size_t {
    return m_col;
}

template<detail::Number T>
auto Matrix<T>::number_row() const -> size_t {
    return m_row;
}

template<detail::Number T>
auto Matrix<T>::size() const -> size_t {
    return m_matrix.size();
}

template<detail::Number T>
auto Matrix<T>::is_square() const -> bool {
    return m_col == m_row;
}

template<detail::Number T>
auto Matrix<T>::is_empty() const -> bool {
    return m_matrix.empty();
}

template<detail::Number T>
auto Matrix<T>::resize(size_t column, size_t row) -> void {
    m_matrix.clear();
    m_col = column;
    m_row = row;
    m_matrix.resize(column*row);
    m_transposition = false;
}

template<detail::Number T>
auto Matrix<T>::swap_column(size_t i, size_t j) -> void {
    if (i >= m_col || j >= m_col)
      throw std::out_of_range("Error! Attempt to access non-existent index (swap_column)");
    for (int k = 0; k < m_row; ++k) {
        std::swap(at(i, k), at(j, k));
    }
}

template<detail::Number T>
auto Matrix<T>::swap_line(size_t i, size_t j) -> void {
    if (i >= m_row || j >= m_row)
      throw std::out_of_range("Error! Attempt to access non-existent index (swap_line)");
    if (i < 0 || i >= m_row || j < 0 || j >= m_row) {return;}
    for (int k = 0; k < m_col; ++k) {
        std::swap(at(k, i), at(k, j));
    }
}

template<detail::Number T>
auto Matrix<T>::is_valid_index(const size_t column, const size_t row) const -> void {
    if(column >= m_col || row >= m_row)
      throw std::out_of_range("invalid index (is_valid_index)");
}

template<detail::Number T>
auto Matrix<T>::is_correct_to_sq_use() const -> void {
    if(is_empty() || (!is_square()))
      throw std::invalid_argument("Error! Attempt to use an empty or non-square matrix");
}



template<detail::Number T>
auto Matrix<T>::transpose() -> Matrix<T>& {
    if (m_matrix.size() < 2) {return *this;}
    m_transposition = not m_transposition;
    std::swap(m_col, m_row);
    return *this;
}

template<detail::Number T>
auto Matrix<T>::Tr() const -> Matrix<T> {
    Matrix matrix = *this;
    matrix.transpose();
    return matrix;
}

template<detail::Number T>
auto det_2x2(const Matrix<T>& m) -> double {
    double det = m.at(0, 0)*m.at(1, 1) - m.at(0,1)*m.at(1, 0);
    if (detail::is_null(det))
      det = 0;
    return det;
}

template<detail::Number T>
auto det_3x3(const Matrix<T>& m) -> double {
    double det = m.at(0, 0) * (m.at(1, 1)*m.at(2, 2) - m.at(1, 2)*m.at(2, 1)) -
            m.at(0, 1) * (m.at(1, 0)*m.at(2, 2) - m.at(1, 2)*m.at(2, 0)) +
            m.at(0, 2) * (m.at(1, 0)*m.at(2, 1) - m.at(1, 1)*m.at(2, 0));
    if(detail::is_null(det))
      det = 0.0;
    return det;
}

template<detail::Number T>
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

    if (detail::is_null(det))
      det = 0.0;
    return det;
}

template<detail::Number T>
auto Matrix<T>::minor(size_t x, size_t y) const -> Matrix<T> {
    is_valid_index(x, y);
    Matrix<T> buffer;
    if(m_col <= 1 || m_row <= 1) {return buffer;}
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

template<detail::Number T>
auto Matrix<T>::inverse() const -> Matrix<double> {
    is_correct_to_sq_use();
    if(m_col == 1) {return Matrix<T>{{1/at(0,0)}, 1, 1};}
    Matrix<double> buffer;
    auto d = det();
    if (detail::is_null(d))
      return buffer;
    buffer.resize(m_col, m_row);
    for (size_t i = 0; i < m_row; ++i) {
        for (size_t j = 0; j < m_col; ++j) {
            buffer.at(j, i) = minor(j, i).det() * ((j+i)%2==0?1:-1);
        }
    }
  buffer.transpose();
    buffer = buffer.division(d);
    return buffer;
}



namespace matrix {
    template<detail::Number TEquation, detail::Number TFree>
    auto solve_SLAE(const Matrix<TEquation>& equation, const Matrix<TFree>& free_members) -> Matrix<double> {
        return equation.inverse() * free_members.template cast_copy<double>();
    }

    template <detail::Number T>
    auto identity_matrix(size_t rank) -> Matrix<T> {
        return Matrix<T>(rank, rank).identity_matrix();
    }
}