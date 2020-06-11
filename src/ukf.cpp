#include "ukf.h"
#include "Eigen/Dense"

#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // matrix to hold sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // define spread parameter
  lambda_ = 3.0 - n_aug_;

  // initialize sigma weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 0; i < 2 * n_aug_; ++i) {
    weights_(i + 1) = 0.5 / (lambda_ + n_aug_);
  }

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0,
      0, std_radrd_ * std_radrd_;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_(0),
          meas_package.raw_measurements_(1), 0, 0, 0;
      VectorXd initial_var(5);
      initial_var << std_laspx_ * std_laspx_, std_laspy_ * std_laspy_, 1, 1, 1;
      P_ = initial_var.asDiagonal();
      std::cout << "Initialized with LIDAR message\n" << x_ << std::endl;
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // set initial state using radar values (convert ρ,φ -> px, py, v)
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      double px = rho * std::cos(phi);
      double py = rho * std::sin(phi);
      double vx = rho_dot * std::cos(phi);
      double vy = rho_dot * std::sin(phi);
      double v = sqrt(vx * vx + vy * vy);
      x_ << px, py, v, rho, rho_dot;
      Eigen::VectorXd initial_var(5);
      initial_var << std_radr_ * std_radr_, std_radr_ * std_radr_,
          std_radrd_ * std_radrd_, std_radphi_, std_radphi_;
      P_ = initial_var.asDiagonal();
      std::cout << "Initialized with RADAR message:\n" << x_ << std::endl;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  Prediction(dt);
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR &&
             use_radar_) {
    UpdateRadar(meas_package);
  }
  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  // create augmented state
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd Xsig_aug = calculateSigmaPoints(x_aug, P_aug, lambda_);

  // predict sigma points (process model)
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double x = Xsig_aug(0, i);
    double y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yaw_rate = Xsig_aug(4, i);
    double a = Xsig_aug(5, i);
    double yaw_dd = Xsig_aug(6, i);

    // px and py
    if (fabs(yaw_rate) > 0.001) {
      Xsig_pred_(0, i) =
          x + v / yaw_rate * (sin(yaw + yaw_rate * delta_t) - sin(yaw));
      Xsig_pred_(1, i) =
          y + v / yaw_rate * (-cos(yaw + yaw_rate * delta_t) + cos(yaw));
    } else {
      Xsig_pred_(0, i) = x + v * cos(yaw) * delta_t;
      Xsig_pred_(1, i) = y + v * sin(yaw) * delta_t;
    }
    // velocity
    Xsig_pred_(2, i) = v;
    // yaw
    Xsig_pred_(3, i) = yaw + yaw_rate * delta_t;
    // yaw_rate
    Xsig_pred_(4, i) = yaw_rate;

    // add noise
    Xsig_pred_(0, i) += delta_t * delta_t / 2. * cos(yaw) * a;
    Xsig_pred_(1, i) += delta_t * delta_t / 2. * sin(yaw) * a;
    Xsig_pred_(2, i) += delta_t * Xsig_aug(5, i);
    Xsig_pred_(3, i) += delta_t * delta_t / 2. * yaw_dd;
    Xsig_pred_(4, i) += delta_t * yaw_dd;
  }

  x_ = calculateState(Xsig_pred_);
  P_ = calculateCovariance(Xsig_pred_, [&](const VectorXd &Xsig_col) {
    VectorXd x_diff = Xsig_col - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;
    return x_diff;
  });
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Zsig(0, i) = Xsig_pred_(0, i); // px
    Zsig(1, i) = Xsig_pred_(1, i); // py
  }

  VectorXd z_pred = calculateState(Zsig);
  MatrixXd S = calculateCovariance(
      Zsig, [&](const VectorXd &Zsig_col) { return Zsig_col - z_pred; });
  S = S + R_laser_;

  // update
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate kalman gain
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // update NIS
  NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // predict sigma points
  for(int i=0; i<2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    // calculate vx and vy
    double vx = cos(yaw) * v;
    double vy = sin(yaw) * v;

    // store values that will be reused for efficiency
    double divisor = sqrt(px*px + py*py);

    // measurement model for radar
    Zsig(0, i) = divisor;
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px*vx + py*vy) / divisor;
  }

  VectorXd z_pred = calculateState(Zsig);
  MatrixXd S = calculateCovariance(Zsig, [&](VectorXd Zsig_col) {
    VectorXd diff = Zsig_col - z_pred;
    // angle normalization
    while (diff(1)> M_PI) diff(1)-=2.*M_PI;
    while (diff(1)<-M_PI) diff(1)+=2.*M_PI;
    return diff;
  });
  S = S + R_radar_;

  // update
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for(int i=0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate kalman gain
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // update NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

Eigen::MatrixXd UKF::calculateSigmaPoints(const VectorXd &x, const MatrixXd P,
                                          double lambda) {
  const int n = x.size();
  // create sigma point matrix
  MatrixXd Xsig = MatrixXd(n, 2 * n + 1);
  MatrixXd A = P.llt().matrixL(); // calculate sqrt P
  Xsig.col(0) = x;
  for (int i = 0; i < n; i++) {
    Xsig.col(i + 1) = x + sqrt(lambda + n) * A.col(i);
    Xsig.col(i + 1 + n) = x - sqrt(lambda + n) * A.col(i);
  }
  return Xsig;
}

Eigen::VectorXd UKF::calculateState(const MatrixXd &Xsig) {
  double n_states = Xsig.rows();
  VectorXd x = VectorXd::Zero(n_states);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x = x + weights_(i) * Xsig.col(i);
  }
  return x;
}

Eigen::MatrixXd UKF::calculateCovariance(
    MatrixXd Xsig,
    const std::function<VectorXd(const VectorXd &)> &diffFn) {
  double n_states = Xsig.rows();
  MatrixXd P = MatrixXd::Zero(n_states, n_states);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // calculate diff between sigma col and x
    VectorXd diff = diffFn(Xsig.col(i));
    P = P + weights_(i) * diff * diff.transpose();
  }
  return P;
}
