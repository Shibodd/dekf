#ifndef DEKF_TYPES_HPP
#define DEKF_TYPES_HPP

#include <concepts>
#include <Eigen/Dense>

template <typename FloatT, int N>
using Vector = Eigen::Vector<FloatT, N>;

template <typename FloatT, int R, int C>
using Matrix = Eigen::Matrix<FloatT, R, C>;

#define VECTOR_MEMBER(name, enum_name, idx) \
  static constexpr size_t enum_name = idx; \
  FloatT name() const { return (*this)[ enum_name ]; } \
  FloatT& name() { return (*this)[ enum_name ]; }

template <typename V>
using Covariance = Matrix<typename V::Scalar, V::RowsAtCompileTime, V::RowsAtCompileTime>;

template <typename VY, typename VX>
using Jacobian = Matrix<typename VY::Scalar, VY::RowsAtCompileTime, VX::RowsAtCompileTime>;

template <typename StateV, typename MeasurementV>
using KalmanGain = Matrix<typename StateV::Scalar, StateV::RowsAtCompileTime, MeasurementV::RowsAtCompileTime>;

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
  { mdl.state_process_noise() } -> std::same_as<Covariance<typename T::State>>;
  { mdl.parameters_process_noise() } -> std::same_as<Covariance<typename T::Parameters>>;
};

template<typename T, typename SysT>
concept MeasurementModel = requires (const T& mdl, typename T::MeasurementType::Measurement z, typename SysT::State x, typename SysT::Parameters p, typename T::MeasurementType meastype) {
  requires SystemModel<SysT>;
  requires std::derived_from<typename T::MeasurementType::Measurement, Vector<typename SysT::State::Scalar, T::MeasurementType::Measurement::RowsAtCompileTime>>;

  { mdl.h(x, p) } -> std::same_as<typename T::MeasurementType::Measurement>;
  { mdl.jacobian_wrt_state(x, p) } -> std::same_as<Jacobian<typename T::MeasurementType::Measurement, typename T::State>>;
  { mdl.jacobian_wrt_parameters(x, p) } -> std::same_as<Jacobian<typename T::MeasurementType::Measurement, typename SysT::Parameters>>;

  { meastype.measurement() } -> std::same_as<typename T::MeasurementType::Measurement>;
  { meastype.covariance() } -> std::same_as<Covariance<typename T::MeasurementType::Measurement>>;
};

#endif