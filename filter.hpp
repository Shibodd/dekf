#include <optional>
#include <tuple>

#include "lateral_ekf/dekf_types.hpp"

/**
* @param P Old covariance
* @param F State transition function Jacobian
* @param Q Process noise
*/
template <typename StateT>
static Covariance<StateT> kf_covariance_predict(
  const Covariance<StateT>& P,
  const Jacobian<StateT, StateT>& F,
  const Covariance<StateT>& Q
) {
  return F * P.covariance * F.transpose() + Q;
}

/**
 * @param x State estimate
 * @param expected_measurement The result of applying the observation model to the current state
 * @param measurement The actual observation
 * @param H The observation model Jacobian
 */
template <typename StateT, typename MeasurementT>
static void kf_update(
  VectorCovariancePair<StateT>& x,
  const MeasurementT& expected_measurement,
  const VectorCovariancePair<MeasurementT>& measurement,
  const Jacobian<MeasurementT, StateT>& H
) {
  Covariance<MeasurementT> S = H * x.covariance * H.transpose() + measurement.covariance();
  KalmanGain<StateT, MeasurementT> K = x.covariance() * H.transpose() * S.inverse();

  x.estimate += K * (measurement.estimate() - expected_measurement);
  x.covariance -= K * H * x.covariance;
}

template <SystemModel SysT, MeasurementModel<SysT>... MeasTs>
class DEKF {
  using State = typename SysT::State;
  using Parameters = typename SysT::Parameters;
  using Control = typename SysT::Control;

  VectorCovariancePair<State> m_state;
  VectorCovariancePair<Parameters> m_parameters;

  SysT m_system_model;
  std::tuple<MeasTs...> m_measurement_models;

  std::tuple<std::optional<typename MeasTs::MeasurementType>...> m_measurements;

  VectorCovariancePair<Parameters> predict_parameters(const VectorCovariancePair<Parameters>& old_estimate) {
    // Parameters are modeled as a random walk
    return VectorCovariancePair<Parameters> {
      .estimate = old_estimate.estimate, // constant parameters
      .covariance = old_estimate.covariance + m_system_model.parameter_process_noise()
    };
  }

  VectorCovariancePair<State> predict_state(const VectorCovariancePair<State>& old_estimate, const Parameters& prior_parameters, const Control& u) {
    const State fx = m_system_model.f(old_estimate.estimate, prior_parameters, u);
    const Jacobian<State, State> F = m_system_model.jacobian_wrt_state(old_estimate.estimate, prior_parameters, u);

    return VectorCovariancePair<State> {
      .estimate = fx,
      .covariance = ekf_covariance_predict(old_estimate.covariance, F, m_system_model.state_process_noise())
    };
  }

  template <MeasurementModel<SysT> MeasT>
  void update_parameters(MeasT& model, VectorCovariancePair<Parameters>& parameters, const VectorCovariancePair<State>& prior_state, const MeasT::MeasurementType& z) {
    using Measurement = MeasT::MeasurementType::Measurement;
    const Measurement hx = model.h(prior_state.estimate, parameters.estimate);

    // TODO: Compute this properly.
    // - https://ntrs.nasa.gov/api/citations/20170005722/downloads/20170005722.pdf
    // - https://ieeexplore.ieee.org/abstract/document/9564571
    const Jacobian<Measurement, Parameters> G = model.jacobian_wrt_parameters(prior_state.estimate(), parameters.estimate());
   
    ekf_update(parameters, hx, z.measurement(), G, z.covariance());
  }

  template <MeasurementModel<SysT> MeasurementModelT>
  void update_state(MeasurementModelT& model, VectorCovariancePair<State>& state, const VectorCovariancePair<Parameters>& prior_parameters, const typename MeasurementModelT::MeasurementType::Measurement& z) {
    using Measurement = typename MeasurementModelT::MeasurementType::Measurement;

    const Measurement hx = model.h(state.estimate, prior_parameters.estimate);
    const Jacobian<Measurement, State> H = model.jacobian_wrt_state(state.estimate, prior_parameters.estimate);

    ekf_update(state, hx, z.measurement(), H, z.covariance());
  }

public:
  DEKF(SysT system_model, VectorCovariancePair<State> initial_state, VectorCovariancePair<Parameters> initial_parameters,  MeasTs... measurement_models)
      : m_state(initial_state),
        m_parameters(initial_parameters),
        m_system_model(system_model),
        m_measurement_models(measurement_models...) 
  {
    
  }

  template <size_t... Idxs>
  void process_measurements(std::index_sequence<Idxs...>, const VectorCovariancePair<State>& prior_state, const VectorCovariancePair<Parameters>& prior_parameters) {
    ([&]{
      // For each measurement model, check if there's a pending measurement.
      const auto& z = std::get<Idxs>(m_measurements);
      if (z.has_value()) {
        // Retrieve the measurement model
        auto& measurement_model = std::get<Idxs>(m_measurement_models);

        // Process the pending measurement.
        // Note that parameter update uses state prior and viceversa (not posteriors!!)
        update_parameters(measurement_model, m_parameters, prior_state, *z);
        update_state(measurement_model, m_state, prior_parameters, *z);
        z.reset();
      }
    }(), ...);
  }

  void step(Control u) {
    // Compute priors from the best guess of the previous step
    VectorCovariancePair<Parameters> prior_parameters = predict_parameters(m_parameters);
    VectorCovariancePair<State> prior_state = predict_state(m_state, prior_parameters, u);

    // If there are no available measurements, the priors are our best guess  
    m_parameters = prior_parameters;
    m_state = prior_state;

    // If we have measurements, compute posteriors - they become our best guess
    process_measurements(std::index_sequence_for<MeasTs...>{}, prior_state, prior_parameters);
  }

  template <size_t N>
  void on_measurement(const typename std::tuple_element<N, decltype(m_measurement_models)>::type::MeasurementType& z) {
    // Template magic on z to get the N-th measurement model's measurement type

    // Store the measurement in the corresponding model measurement slot
    // Note: assumption that there's at most one measurement per tick!
    std::get<N>(m_measurements) = z;
  }
};