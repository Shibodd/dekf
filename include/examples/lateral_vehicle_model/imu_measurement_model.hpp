#ifndef IMU_MEASUREMENT_MODEL_HPP
#define IMU_MEASUREMENT_MODEL_HPP

#include "dekf/types.hpp"
#include "examples/lateral_vehicle_model/system_model.hpp"

namespace LateralVehicleModel {

template <typename FloatT>
class IMUMeasurementModel {
  constexpr static FloatT G = 9.821; // Gravity specific force [m / s^2]

  FloatT m_imu_x;

public:
  class MeasurementType {
  public:
    class Measurement : public Vector<FloatT, 3> {
      VECTOR_MEMBER(ay, AY, 0);
      VECTOR_MEMBER(psidot, PSIDOT, 1);
      VECTOR_MEMBER(rolldot, ROLLDOT, 2);
    };

  private:
    Measurement m_z;

  public:
    MeasurementType(FloatT ay, FloatT psidot, FloatT rolldot) : m_z(ay, psidot, rolldot) { }
    
    Measurement measurement() const { return m_z; }
    Covariance<Measurement> covariance() const {
      return Measurement{ 0.5, 0.05, 0.05 }.asDiagonal();
    }
  };

  using System = SystemModel<FloatT>;
  using State = typename System::State;
  using Parameters = typename System::Parameters;
  using Measurement = typename MeasurementType::Measurement;

  IMUMeasurementModel(FloatT imu_x) : m_imu_x(imu_x) { }

  Measurement h(const State& x, const Parameters& p) const {
    FloatT kin_acc = p.vx() * x.psidot();
    FloatT gay = G * x.roll(); // Lateral specific force due to gravity (linearized)
    FloatT euler_ay = (m_imu_x - p.xcom()) * x.psidotdot();

    Measurement ans;
    ans.ay() = x.ay() + kin_acc + gay + euler_ay;
    ans.psidot() = x.psidot();
    ans.rolldot() = x.rolldot();
    
    return ans;
  }

  Jacobian<Measurement, State> jacobian_wrt_state(const State& x, const Parameters& p) const {
    Jacobian<Measurement, State> ans;
    
    ans.setZero();
    ans( Measurement::AY, State::AY ) = 1;
    ans( Measurement::AY, State::PSIDOT ) = p.vx();
    ans( Measurement::AY, State::ROLL ) = G;
    ans( Measurement::AY, State::PSIDOTDOT ) = m_imu_x - p.xcom();
    ans( Measurement::PSIDOT, State::PSIDOT ) = 1;
    ans( Measurement::ROLLDOT, State::ROLLDOT ) = 1;

    return ans;
  }

  Jacobian<Measurement, Parameters> jacobian_wrt_parameters(const State& x, const Parameters& p) const {
    Jacobian<Measurement, Parameters> ans;

    ans.setZero();
    ans( Measurement::AY, Parameters::VX ) = x.psidot();
    ans( Measurement::AY, Parameters::XCOM ) = -x.psidotdot();

    return ans;
  }
};

}

#endif