#ifndef SYSTEM_MODEL_HPP
#define SYSTEM_MODEL_HPP

#include "dekf/types.hpp"

namespace LateralVehicleModel {

template <typename FloatT>
class SystemModel {
public:
  class State : public Vector<FloatT, 6> {
    VECTOR_MEMBER(vy, VY, 0); // Lateral velocity
    VECTOR_MEMBER(psidot, PSIDOT, 1); // Yaw rate
    VECTOR_MEMBER(ay, AY, 2); // Lateral acceleration
    VECTOR_MEMBER(psidotdot, PSIDOTDOT, 3); // Yaw angular acceleration
    VECTOR_MEMBER(roll, ROLL, 4); // bank + roll (gravity compensation)
    VECTOR_MEMBER(rolldot, ROLLDOT, 5); // Roll rate
  };

  class Parameters : public Vector<FloatT, 5> {
    VECTOR_MEMBER(iz, IZ, 0);
    VECTOR_MEMBER(xcom, XCOM, 1);
    VECTOR_MEMBER(cf, CF, 2);
    VECTOR_MEMBER(cr, CR, 3);
    VECTOR_MEMBER(vx, VX, 4);
  };

  class Control : public Vector<FloatT, 1> {
    VECTOR_MEMBER(delta, DELTA, 0);
  };

  const FloatT m_mass;
  const FloatT m_wheelbase;
  const FloatT m_deltat;
  const Parameters m_initial_parameters;

  SystemModel(FloatT deltat, FloatT wheelbase, FloatT mass, Parameters initial_parameters) 
    : m_deltat(deltat),
      m_wheelbase(wheelbase),
      m_mass(mass),
      m_initial_parameters(initial_parameters)
    {}

  State f(const State& x, const Parameters& p, const Control& u) const {
    FloatT vx = p.vx();
    FloatT iz = p.iz();

    FloatT m = m_mass;

    FloatT twocf = 2 * p.cf();
    FloatT twocr = 2 * p.cr();

    FloatT lf = m_wheelbase / 2 - p.xcom();
    FloatT lr = m_wheelbase / 2 + p.xcom();

    FloatT mvx = m * vx;
    FloatT izvx = iz * vx;

    FloatT twocflf = twocf * lf;
    FloatT twocrlr = twocr * lr;

    Matrix<FloatT, 2, 3> mat {
      { -(twocf + twocr) / mvx, -(twocflf - twocrlr) / mvx - vx, twocf / m },
      { -(twocflf - twocrlr) / izvx, -(twocflf * lf + twocrlr * lr) / izvx, twocflf / iz }
    };

    Vector<FloatT, 2> ay_psidotdot = mat * Vector3(x.vy(), x.psidot(), u.delta());

    State ans(x);
    ans.vy() += x.ay() * m_deltat;
    ans.psidot() += x.psidotdot() * m_deltat;
    ans.roll() += x.rolldot() * m_deltat;
    ans.ay() = ay_psidotdot(0);
    ans.psidotdot() = ay_psidotdot(1);
    return ans;
  }

  Jacobian<State, State> jacobian_wrt_state(const State& x, const Parameters& p, const Control& u) const {
    FloatT vx = p.vx();
    FloatT iz = p.iz();

    FloatT m = m_mass;

    FloatT twocf = 2 * p.cf();
    FloatT twocr = 2 * p.cr();

    FloatT lf = m_wheelbase / 2 - p.xcom();
    FloatT lr = m_wheelbase / 2 + p.xcom();

    FloatT mvx = m * vx;
    FloatT izvx = iz * vx;

    FloatT twocflf = twocf * lf;
    FloatT twocrlr = twocr * lr;

    Jacobian<State, State> ans;
    ans.setZero();

    ans( State::VY, State::VY ) = 1;
    ans( State::VY, State::AY ) = m_deltat;

    ans( State::PSIDOT, State::PSIDOT ) = 1;
    ans( State::PSIDOT, State::PSIDOTDOT ) = m_deltat;

    ans( State::ROLL, State::ROLL ) = 1;
    ans( State::ROLL, State::ROLLDOT ) = m_deltat;

    ans( State::AY, State::VY ) = -(twocf + twocr) / mvx;
    ans( State::AY, State::PSIDOT ) = -(twocflf - twocrlr) / mvx - vx;

    ans( State::PSIDOTDOT, State::VY ) = -(twocflf - twocrlr) / izvx;
    ans( State::PSIDOTDOT, State::PSIDOT ) = -(twocflf * lf + twocrlr * lr) / izvx;
  }
  
  Covariance<State> state_process_noise() const {
    State ans;
    ans.vy() = 0.1;
    ans.psidot() = 0.1;
    ans.ay() = 0.2;
    ans.psidotdot() = 0.2;
    ans.roll() = 0.1;
    ans.rolldot() = 0.2;
    return ans.asDiagonal();
  }
  Covariance<Parameters> parameters_process_noise() const {
    return m_deltat * m_initial_parameters.asDiagonal();
  }
};

}

#endif