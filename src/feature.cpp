#include "keyframe.hpp"
#include "feature.hpp"
#include "utils.hpp"

namespace ssvo{

//! Feature
Feature::Feature(const Vector2d &px, const Vector3d &fn, const int level, const std::shared_ptr<MapPoint> &mpt):
    px(px), fn(fn), level(level), mpt(mpt)
{
    assert(fn[2] == 1);
    assert(mpt);
}

std::ostream& operator<<(std::ostream &out, const Feature &ft)
{
    Vector3d xyz = ft.mpt->pose();
    out << "{ px: [" << ft.px[0] << ", " << ft.px[1] << "],"
        << " fn: [" << ft.fn[0] << ", " << ft.fn[1] << ", " << ft.fn[2] << "],"
        << " level: " << ft.level
        << " mpt: " << ft.mpt->id_ << ", [" << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << "] "
        << " }";

    return out;
}

//! Seed
Seed::Seed(const KeyFrame::Ptr &kf, const Vector2d &px, const Vector3d &fn, const int level, double depth_mean, double depth_min) :
    kf(kf), fn_ref(fn), px_ref(px), level_ref(level), px_cur(px), level_cur(level),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36)
{
    assert(fn_ref[2] == 1);
}

double Seed::computeTau(
    const SE3d& T_ref_cur,
    const Vector3d& f,
    const double z,
    const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double f_norm = f.norm();
    double alpha = acos(f.dot(t)/(t_norm*f_norm)); // dot product
    double beta = acos(-a.dot(t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    z_plus /= f_norm;

    double tau = z_plus - z;
    return 0.5 * (1.0/MAX(0.0000001, z-tau) - 1.0/(z+tau));
}

double Seed::computeVar(const SE3d &T_cur_ref, const double z, const double delta)
{
    const Vector3d &t(T_cur_ref.translation()); // from cur->ref in cur's frame
    Vector3d xyz_r(fn_ref*z);
    Vector3d f_c(T_cur_ref * xyz_r);
    Vector3d f_r(f_c-t);

    double t_norm = t.norm();
    double f_c_norm = f_c.norm();
    double f_r_norm = f_r.norm();

//    double alpha = acos(f_r.dot(-t)/f_r_norm/t_norm);

    double epslion = atan(0.5*delta/f_c_norm/f_c[2])*2.0;
//    epslion  = 0.0021867665614925609;
    double beta = acos(f_c.dot(t)/(f_c_norm*t_norm));
    double gamma = acos(f_c.dot(f_r)/(f_c_norm*f_r_norm));

    double z1 = t_norm * sin(beta+epslion) / sin(gamma-epslion);
    z1 /= f_r_norm;

    return 0.5 * (1.0/MAX(0.0000001, 2*z-z1) - 1.0/(z1));
}

void Seed::update(const double x, const double tau2)
{
    double norm_scale = sqrt(sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;

    double s2 = 1./(1./sigma2 + 1./tau2);
    double m = s2*(mu/sigma2 + x/tau2);
    double C1 = a/(a+b) * utils::normal_distribution<double>(x, mu, norm_scale);
    double C2 = b/(a+b) * 1./z_range;
    double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    double f = C1*(a+1.)/(a+b+1.) + C2*a/(a+b+1.);
    double e = C1*(a+1.)*(a+2.)/((a+b+1.)*(a+b+2.))
        + C2*a*(a+1.0f)/((a+b+1.0f)*(a+b+2.0f));

    // update parameters
    double mu_new = C1*m+C2*mu;
    sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new;
    mu = mu_new;
    a = (e-f)/(f-e/f);
    b = a*(1.0f-f)/f;

    history.emplace_back(x, 1.0/mu);
}

}
