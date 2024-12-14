#include <iostream>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <matrix/matrix.h>

// todo https://github.com/tmp-name-org/floppy/blob/euclid/include/floppy/euclid/scale.h
TEST(TestCreate, CreateMatrix) {
    Matrix <int> a;
    Matrix <double> b;
    Matrix <int> c (1, 1);
    Matrix <int> d ({1}, 1, 1);
    Matrix <int> e ({1, 2, 3, 4}, 2, 2);
    Matrix <int> f ({
                            {1, 2, 3},
                            {5, 6, 7}
    });
    Matrix <int> g ({1, 2}, 1, 2);
    Matrix <int> h ({1, 2}, 2, 1);
    Matrix i = d;

    ASSERT_EQ(a.size(), 0);
    ASSERT_EQ(b.size(), 0);
    ASSERT_EQ(c.size(), 1);
    ASSERT_EQ(d.size(), 1);
    ASSERT_EQ(e.size(), 4);
    ASSERT_EQ(f.size(), 6);

    ASSERT_EQ(c.at(0, 0), 0);
    ASSERT_EQ(d.at(0, 0), 1);
    ASSERT_EQ(e.at(0, 0), 1);
    ASSERT_EQ(e.at(1, 0), 2);
    ASSERT_EQ(e.at(0, 1), 3);
    ASSERT_EQ(e.at(1, 1), 4);

    ASSERT_EQ(f.at(0, 0), 1);
    ASSERT_EQ(f.at(1, 0), 2);
    ASSERT_EQ(f.at(2, 0), 3);
    ASSERT_EQ(f.at(0, 1), 5);
    ASSERT_EQ(f.at(1, 1), 6);
    ASSERT_EQ(f.at(2, 1), 7);

    ASSERT_EQ(g.at(0, 0), 1);
    ASSERT_EQ(g.at(0, 1), 2);

    ASSERT_EQ(h.at(0, 0), 1);
    ASSERT_EQ(h.at(1, 0), 2);

    ASSERT_EQ(i.at(0, 0), 1);

}

TEST(TestCast, CastMatrix) {
    Matrix <double> b;
    Matrix <int> d ({1}, 1, 1);
    Matrix <int> e ({1, 2, 3, 4}, 2, 2);
    Matrix <int> f ({1, 2, 3, 4}, 2, 2);

    b = d.cast_copy<double>();
    b.at(0, 0) = 1.6;

    ASSERT_DOUBLE_EQ(b.at(0, 0), 1.6);
}

TEST(TestIdentity, IdentityMatrix) {
    Matrix <int> d ({2}, 1, 1);
    Matrix <double> e ({1, 2, 3, 4}, 2, 2);

    d.identity_matrix();
    e.identity_matrix();

    ASSERT_EQ(d.at(0, 0), 1);

    ASSERT_DOUBLE_EQ(e.at(0, 0), 1);
    ASSERT_DOUBLE_EQ(e.at(1, 0), 0);
    ASSERT_DOUBLE_EQ(e.at(0, 1), 0);
    ASSERT_DOUBLE_EQ(e.at(1, 1), 1);
}

TEST(TestFill, FillMatrix) {
    Matrix <int> d ({2}, 1, 1);
    Matrix <double> e ({1, 2, 3, 4}, 2, 2);

    d.fill(2);
    e.fill(2);

    ASSERT_EQ(d.at(0, 0), 2);

    ASSERT_DOUBLE_EQ(e.at(0, 0), 2);
    ASSERT_DOUBLE_EQ(e.at(1, 0), 2);
    ASSERT_DOUBLE_EQ(e.at(0, 1), 2);
    ASSERT_DOUBLE_EQ(e.at(1, 1), 2);
}

TEST(TestOperators, SimpleOperatorsMatrix) {
    Matrix <int> d ({100, 200, 300, 400}, 2, 2);
    Matrix <double> e ({1.5, -2, -3.5, 0}, 2, 2);
    Matrix s = d + e;
    Matrix x = d - e;

    ASSERT_DOUBLE_EQ(s.at(0, 0), 101.5);
    ASSERT_DOUBLE_EQ(s.at(1, 0), 198);
    ASSERT_DOUBLE_EQ(s.at(0, 1), 296.5);
    ASSERT_DOUBLE_EQ(s.at(1, 1), 400);

    ASSERT_DOUBLE_EQ(x.at(0, 0), 98.5);
    ASSERT_DOUBLE_EQ(x.at(1, 0), 202);
    ASSERT_DOUBLE_EQ(x.at(0, 1), 303.5);
    ASSERT_DOUBLE_EQ(x.at(1, 1), 400);
}

TEST(TestOperators, MultiplyOperatorsMatrix) {
    Matrix <int> a ({1, 2, 3, 4}, 2, 2);
    Matrix <double> b ({0, 3, 4, 5}, 2, 2);
    Matrix <int> c ({1}, 1, 1);
    Matrix <double> d ({5}, 1, 1);
    Matrix <int> e ({1, 2, 3, 4, 5, 6}, 2, 3);
    Matrix <double> f ({0, 3, 1, 4.2, 5.5, 8}, 3, 2);
    Matrix <int> j ({1, 2, 0}, 3, 1);
    Matrix <double> h ({0, 4.2, 0}, 1, 3);
    Matrix <int> k ({1, 2, 3}, 1, 3);
    Matrix <double> l ({9, 8, 7}, 3, 1);


    Matrix s = a * b;
    Matrix t = c * d;
    Matrix x = j * h;
    Matrix y = e * f;
    Matrix z = k * l;

    ASSERT_DOUBLE_EQ(s.at(0, 0), 8);
    ASSERT_DOUBLE_EQ(s.at(1, 0), 13);
    ASSERT_DOUBLE_EQ(s.at(0, 1), 16);
    ASSERT_DOUBLE_EQ(s.at(1, 1), 29);

    ASSERT_DOUBLE_EQ(t.at(0, 0), 5);

    ASSERT_DOUBLE_EQ(x.at(0, 0), 8.4);

    ASSERT_DOUBLE_EQ(y.at(0, 0), 8.4);
    ASSERT_DOUBLE_EQ(y.at(1, 0), 14);
    ASSERT_DOUBLE_EQ(y.at(2, 0), 17);
    ASSERT_DOUBLE_EQ(y.at(0, 1), 16.8);
    ASSERT_DOUBLE_EQ(y.at(1, 1), 31);
    ASSERT_DOUBLE_EQ(y.at(2, 1), 35);
    ASSERT_DOUBLE_EQ(y.at(0, 2), 25.2);
    ASSERT_DOUBLE_EQ(y.at(1, 2), 48);
    ASSERT_DOUBLE_EQ(y.at(2, 2), 53);

    ASSERT_DOUBLE_EQ(z.at(0, 0), 9);
    ASSERT_DOUBLE_EQ(z.at(1, 0), 8);
    ASSERT_DOUBLE_EQ(z.at(2, 0), 7);
    ASSERT_DOUBLE_EQ(z.at(0, 1), 18);
    ASSERT_DOUBLE_EQ(z.at(1, 1), 16);
    ASSERT_DOUBLE_EQ(z.at(2, 1), 14);
    ASSERT_DOUBLE_EQ(z.at(0, 2), 27);
    ASSERT_DOUBLE_EQ(z.at(1, 2), 24);
    ASSERT_DOUBLE_EQ(z.at(2, 2), 21);
}

TEST(TestOperators, TranspositionMatrix) {
    Matrix <int> d ({100, 200, 300, 400}, 2, 2);
    Matrix <double> e ({1.5, -2, -3.5, 0}, 1, 4);
  d.transpose();
  e.transpose();

    ASSERT_DOUBLE_EQ(d.at(0, 0), 100);
    ASSERT_DOUBLE_EQ(d.at(1, 0), 300);
    ASSERT_DOUBLE_EQ(d.at(0, 1), 200);
    ASSERT_DOUBLE_EQ(d.at(1, 1), 400);

    ASSERT_DOUBLE_EQ(e.at(0, 0), 1.5);
    ASSERT_DOUBLE_EQ(e.at(1, 0), -2);
    ASSERT_DOUBLE_EQ(e.at(2, 0), -3.5);
    ASSERT_DOUBLE_EQ(e.at(3, 0), 0);
}

TEST(TestDet, DetMatrix) {
    Matrix <int> a ({100, 200, 300, 400}, 2, 2);
    Matrix <double> b ({1.5}, 1, 1);
    Matrix <double> c ({1, -4, 6.4, 0}, 2, 2);
    Matrix <double> d ({1, 2, 4, 5, 0, 4.4, 1, 10, 12.2}, 3, 3);
    Matrix <double> e ({1, 2, 4, 9, 0.5, 5, 78, 6, 1000000}, 3, 3);
    Matrix <double> f ({1, 2, 4, 78, 9, 7, 5, 24, 78, 6, 0.6, 8, 9, 99, 10, 1.5}, 4, 4);
    Matrix <uint16_t > g ({100, 200, 300, 400}, 2, 2);

    ASSERT_DOUBLE_EQ(a.det(), -20000);
    ASSERT_DOUBLE_EQ(b.det(), 1.5);
    ASSERT_DOUBLE_EQ(c.det(), 25.6);
    ASSERT_DOUBLE_EQ(d.det(), 42.8);
    ASSERT_DOUBLE_EQ(e.det(), -17499190);
    ASSERT_DOUBLE_EQ(f.det(), -1893260.8999999948);
    ASSERT_DOUBLE_EQ(g.det(), -20000);
}

TEST(TestInverse, InverseMatrix) {
    Matrix <double> _one(1, 1);
    Matrix <double> _two(2, 2);
    Matrix <double> _three(3, 3);
    Matrix <double> _four(4, 4);

    _one = matrix::identity_matrix<double>(1);
    _two.identity_matrix();
    _three.identity_matrix();
    _four.identity_matrix();

    Matrix <double> b ({2}, 1, 1);
    Matrix <double> c ({1, -4, 6.4, 0}, 2, 2);
    Matrix <double> d ({1, 2, 4, 5, 0, 4.4, 1, 10, 12.2}, 3, 3);
    Matrix <double> e ({1, 2, 4, 9, 0.5, 5, 78, 6, 1000000}, 3, 3);
    Matrix <double> f ({1, 2, 4, 78, 9, 7, 5, 24, 78, 6, 0.6, 8, 9, 99, 10, 1.5}, 4, 4);

    ASSERT_TRUE(b*b.inverse() == matrix::identity_matrix<double>(1));
    ASSERT_TRUE(c*c.inverse() == matrix::identity_matrix<double>(2));
    ASSERT_TRUE(d*d.inverse() == matrix::identity_matrix<double>(3));
    ASSERT_TRUE(f*f.inverse() == matrix::identity_matrix<double>(4));

}

auto main() -> int {

    ::testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
