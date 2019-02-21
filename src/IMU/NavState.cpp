//
// Created by jh on 19-1-16.
//

#include "NavState.hpp"

namespace ssvo
{
NavState::NavState()
{
    //todo 为什么把这里的删除了？
    //_qR.setIdentity();     // rotation
    _P.setZero();         // position
    _V.setZero();         // velocity

    _BiasGyr.setZero();   // bias of gyroscope
    _BiasAcc.setZero();   // bias of accelerometer

    _dBias_g.setZero();
    _dBias_a.setZero();
}

// if there's some other constructor, normalizeRotation() is needed
NavState::NavState(const NavState &_ns):
        _P(_ns._P), _V(_ns._V), _R(_ns._R),
        _BiasGyr(_ns._BiasGyr), _BiasAcc(_ns._BiasAcc),
        _dBias_g(_ns._dBias_g), _dBias_a(_ns._dBias_a)
{
}

}
