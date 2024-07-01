#ifndef DEKF_TYPES_HPP
#define DEKF_TYPES_HPP

#include <concepts>
#include <Eigen/Dense>

template <typename FloatT, int N>
using Vector = Eigen::Vector<FloatT, N>;

template <typename V>
using Covariance = Eigen::Matrix<typename V::Scalar, V::RowsAtCompileTime, V::RowsAtCompileTime>;

template <typename VY, typename VX>
using Jacobian = Eigen::Matrix<typename VY::Scalar, VY::RowsAtCompileTime, VX::RowsAtCompileTime>;

template <typename StateV, typename MeasurementV>
using KalmanGain = Eigen::Matrix<typename StateV::Scalar, StateV::RowsAtCompileTime, MeasurementV::RowsAtCompileTime>;

template <typename V>
struct VectorCovariancePair {
  V estimate;
  Covariance<V> covariance;
};

template<typename T>
concept SystemModel = requires (const T& mdl, typename T::State x, typename T::Parameters p, typename T::Control u) {
  requires std::derived_from<typename T::State, Vector<typename T::State::Scalar, T::State::RowsAtCompileTime>>;
  requires std::derived_from<typename T::Parameters, Vector<typename T::State::Scalar, T::Parameters::RowsAtCompileTime>>;
  requires std::derived_from<typename T::Control, Vector<typename T::State::Scalar, T::Control::RowsAtCompileTime>>;

  { mdl.f(x, p, u) } -> std::same_as<typename T::State>;
  { mdl.jacobian_wrt_state(x, p, u) } -> std::same_as<Jacobian<typename T::State, typename T::State>>;
  { mdl.jacobian_wrt_parameters(x, p, u) } -> std::same_as<Jacobian<typename T::State, typename T::Parameters>>;
  { mdl.state_process_noise() } -> std::same_as<Covariance<typename T::State>>;
  { mdl.parameters_process_noise() } -> std::same_as<Covariance<typename T::State>>;
};

template<typename T, typename SysT>
concept MeasurementModel = requires (const T& mdl, typename T::MeasurementType::Measurement z, typename SysT::State x, typename SysT::Parameters p) {
  requires SystemModel<SysT>;
  requires std::derived_from<typename T::MeasurementType::Measurement, Vector<typename SysT::State::Scalar, T::MeasurementType::Measurement::RowsAtCompileTime>>;

  { mdl.h(x, p) } -> std::same_as<typename T::MeasurementType::Measurement>;
  { mdl.jacobian_wrt_parameters(x, p) } -> std::same_as<Jacobian<typename T::MeasurementType::Measurement, typename SysT::Parameters>>;
};

#endif