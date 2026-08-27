#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>
#include <vector>
#include <string>
#include <cmath>

namespace py = pybind11;
using CD  = std::complex<double>;
using Mat = Eigen::MatrixXcd;
using Vec = Eigen::VectorXcd;

static const double PI = std::acos(-1.0);

// Numpy normalized sinc: sin(pi*x)/(pi*x), returns 1 at x=0
static double np_sinc(double x) {
    if (std::abs(x) < 1e-15) return 1.0;
    return std::sin(PI * x) / (PI * x);
}

// Fix sign of Q so that propagating modes have positive criterion
static void fix_sign_Q(Vec& Q, bool is_metal) {
    for (int i = 0; i < (int)Q.size(); i++) {
        double crit = is_metal ? Q[i].imag() : (Q[i].real() + Q[i].imag());
        if (crit <= 0.0) Q[i] = -Q[i];
    }
}

static std::pair<py::array_t<double>, py::array_t<double>>
Rcwa1d(const std::string& pol, double lambda0, double kx0, double period,
       py::tuple layer_py, int norder)
{
    int nl = (int)layer_py.size();

    // ---------- Parse layer tuple ----------
    std::vector<double>              depth(nl, 0.0);
    std::vector<bool>                metal(nl, false);
    std::vector<std::vector<CD>>     refra(nl);
    std::vector<std::vector<double>> filfac(nl);
    std::vector<int>                 nsect(nl);

    for (int j = 0; j < nl; j++) {
        py::tuple lj  = layer_py[j].cast<py::tuple>();
        int raw       = (int)lj.size() / 2;
        nsect[j]      = (j == 0 || j == nl - 1) ? 1 : raw;
        depth[j]      = lj[0].cast<double>();
        refra[j].resize(nsect[j]);
        filfac[j].resize(nsect[j]);
        for (int i = 0; i < nsect[j]; i++) {
            py::object obj = lj[2 * i + 1];
            try         { refra[j][i] = obj.cast<CD>(); }
            catch (...) { refra[j][i] = CD(obj.cast<double>(), 0.0); }
            if (std::abs(refra[j][i].imag()) > 1e-100) metal[j] = true;
            filfac[j][i] = lj[2 * i + 2].cast<double>();
        }
        if (j == 0 || j == nl - 1) filfac[j][0] = 1.0;
    }

    // ---------- Basic parameters ----------
    lambda0 += 1e-6;
    double k0 = 2.0 * PI / lambda0;
    CD kc = k0 * refra[nl - 1][0];  // incident medium (last layer)
    CD ks = k0 * refra[0][0];        // substrate (first layer)
    int p = norder / 2;              // index of 0th order
    int M = norder - 1;
    double K = 2.0 * PI / period;

    Vec kx(norder);
    for (int i = 0; i < norder; i++)
        kx[i] = CD(kx0 + (double)(i - p) * K, 0.0);

    // kzc, kzs: principal sqrt (matches Python behavior)
    Vec kzc(norder), kzs(norder);
    for (int i = 0; i < norder; i++) {
        kzc[i] = std::sqrt(kc * kc - kx[i] * kx[i]);
        kzs[i] = std::sqrt(ks * ks - kx[i] * kx[i]);
    }

    Vec kxk0 = kx / k0;  // diagonal elements of Kx matrix

    // ---------- Fourier matrices for each layer ----------
    std::vector<Mat> EpsX(nl), AlpX(nl);

    for (int kk = 0; kk < nl; kk++) {
        if (nsect[kk] > 1) {
            int Mlen = 2 * M + 1;
            Vec vX(Mlen), ivX(Mlen);
            vX.setZero(); ivX.setZero();

            for (int jj = 0; jj < nsect[kk]; jj++) {
                double disp = 0.0;
                for (int k = 0; k <= jj; k++) disp += filfac[kk][k];
                disp -= filfac[kk][jj] * 0.5;

                CD eps = refra[kk][jj] * refra[kk][jj];
                double ff = filfac[kk][jj];

                for (int m = -M; m <= M; m++) {
                    CD ph  = std::exp(CD(0.0, -2.0 * PI * disp * (double)m));
                    double as = (m == 0) ? ff : ff * np_sinc(ff * std::abs((double)m));
                    vX[M + m]  += eps       * as * ph;
                    ivX[M + m] += (1.0/eps) * as * ph;
                }
            }

            // Build Toeplitz: T[i,j] = vX[M + i - j]
            EpsX[kk].resize(norder, norder);
            AlpX[kk].resize(norder, norder);
            for (int i = 0; i < norder; i++)
                for (int j = 0; j < norder; j++) {
                    EpsX[kk](i, j) = vX[M + i - j];
                    AlpX[kk](i, j) = ivX[M + i - j];
                }
        } else {
            CD eps0 = refra[kk][0] * refra[kk][0];
            EpsX[kk] = Mat::Identity(norder, norder) * eps0;
            AlpX[kk] = Mat::Identity(norder, norder) / eps0;
        }
    }

    // ---------- S-matrix recursion ----------
    Mat Eye  = Mat::Identity(norder, norder);
    Mat Rud  = Mat::Zero(norder, norder);
    Mat Tdd  = Eye;
    Mat Phip = Mat::Zero(norder, norder);
    Mat W0, V0, W1, V1, W00;

    for (int ii = 0; ii < nl; ii++) {
        CD epsr = refra[ii][0] * refra[ii][0];
        Vec Eig(norder);

        if (nsect[ii] > 1) {
            Mat A;
            if (pol == "s") {
                // A = Kx^2 - EpsX  (eq. 5.14)
                A = -EpsX[ii];
                for (int i = 0; i < norder; i++)
                    A(i, i) += kxk0[i] * kxk0[i];
            } else {
                // A = inv(AlpX) * ( Kx * inv(EpsX) * Kx - I )  (eq. 5.39)
                Mat tmp = EpsX[ii].inverse();
                for (int i = 0; i < norder; i++) tmp.row(i) *= kxk0[i];
                for (int j = 0; j < norder; j++) tmp.col(j) *= kxk0[j];
                A = AlpX[ii].inverse() * (tmp - Eye);
            }
            Eigen::ComplexEigenSolver<Mat> eig(A, true);
            Eig = eig.eigenvalues();
            W1  = eig.eigenvectors();
        } else {
            W1 = Eye;
            for (int i = 0; i < norder; i++)
                Eig[i] = kxk0[i] * kxk0[i] - epsr;
        }

        if (ii == 0) W00 = W1;

        Vec Q(norder);
        for (int i = 0; i < norder; i++) Q[i] = std::sqrt(-Eig[i]);
        fix_sign_Q(Q, metal[ii]);

        // V1 (eq. 5.20 / 5.47)
        if (pol == "s") {
            V1 = W1 * Q.asDiagonal();
        } else {
            if (nsect[ii] > 1) {
                V1 = AlpX[ii] * W1 * Q.asDiagonal();
            } else {
                V1 = Mat::Zero(norder, norder);
                V1.diagonal() = Q / epsr;
            }
        }

        if (ii > 0) {
            Mat RudTilde = Phip * Rud * Phip;      // eq. 5.110
            Mat TddTilde = Tdd  * Phip;             // eq. 5.111
            Mat W1inv    = W1.inverse();
            Mat V1inv    = V1.inverse();
            Mat Q1 = W1inv * W0;                    // eq. 5.118
            Mat Q2 = V1inv * V0;
            Mat F  = Q1 * (Eye + RudTilde);         // eq. 5.116
            Mat G  = Q2 * (Eye - RudTilde);         // eq. 5.117
            Mat FpGinv = (F + G).inverse();
            if (pol == "s") {
                Rud = Eye - 2.0 * G * FpGinv;      // eq. 5.120
                Tdd = 2.0 * TddTilde * FpGinv;      // eq. 5.121
            } else {
                Mat Tau = 2.0 * FpGinv;
                Rud = Eye - G  * Tau;
                Tdd = TddTilde * Tau;
            }
        }

        if (ii != nl - 1) {
            Vec ph(norder);
            for (int i = 0; i < norder; i++)
                ph[i] = std::exp(CD(0.0, 1.0) * k0 * Q[i] * depth[ii]);
            Phip = Mat::Zero(norder, norder);
            Phip.diagonal() = ph;   // eq. 5.25
            W0 = W1;
            V0 = V1;
        }
    }

    // Final similarity transform (eq. 5.131 / 5.132)
    Mat W1inv = W1.inverse();
    Rud = W1  * Rud * W1inv;
    Tdd = W00 * Tdd * W1inv;

    Vec Rc = Rud.col(p);
    Vec Tc = Tdd.col(p);

    // ---------- Diffraction efficiencies ----------
    py::array_t<double> IR(norder), IT(norder);
    auto ir = IR.mutable_unchecked<1>();
    auto it = IT.mutable_unchecked<1>();

    double kzcp = kzc[p].real();
    if (pol == "s") {
        for (int i = 0; i < norder; i++) {
            ir(i) = std::norm(Rc[i]) * kzc[i].real() / kzcp;
            it(i) = std::norm(Tc[i]) * kzs[i].real() / kzcp;
        }
    } else {
        CD eps_inc = refra[nl - 1][0] * refra[nl - 1][0];
        CD eps_sub = refra[0][0]       * refra[0][0];
        double denom = (kzc[p] / eps_inc).real();
        for (int i = 0; i < norder; i++) {
            ir(i) = std::norm(Rc[i]) * kzc[i].real()              / kzcp;
            it(i) = std::norm(Tc[i]) * (kzs[i] / eps_sub).real()  / denom;
        }
    }

    return {IR, IT};
}

PYBIND11_MODULE(rcwa_mh, m) {
    m.doc() = "1D RCWA/FMM C++ backend (pybind11 + Eigen)";
    m.def("Rcwa1d", &Rcwa1d,
          py::arg("pol"), py::arg("lambda0"), py::arg("kx0"),
          py::arg("period"), py::arg("layer"), py::arg("norder"));
}
