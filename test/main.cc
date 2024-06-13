#include <iostream>

#include <gtest/gtest.h>

#include "matrix/matrix.h"


int main() {

    Matrix b(std::vector<int>{
        3, 2, -1,
        2, -1, 5,
        1, 7, -1,

        }, 3, 3);

    Matrix <int> z({
                     {1, 2, 3},
                     {2, 34, 4},
                     {7, 4, 6},
                     {3, 5, 6}
    });

    Matrix c(std::vector<int>{
            4,
            23,
            5
    }, 1, 3);



    auto a = matrix::solve_SLAE(b, c);

    z.cast_copy<double>().print();


    std::cout << "\n" << b.det();


    return 0;
}
