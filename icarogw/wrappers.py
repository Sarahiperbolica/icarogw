from .cupy_pal import get_module_array, get_module_array_scipy, np
from .cosmology import alphalog_astropycosmology, cM_astropycosmology, extraD_astropycosmology, Xi0_astropycosmology, astropycosmology, eps0_astropycosmology
from .cosmology import  md_rate, md_gamma_rate, powerlaw_rate, beta_rate, beta_rate_line
from .priors import LowpassSmoothedProb, LowpassSmoothedProbEvolving, PowerLaw, BetaDistribution, TruncatedBetaDistribution, TruncatedGaussian, Bivariate2DGaussian, SmoothedPlusDipProb, BrokenPowerLawMultiPeak
from .priors import PowerLawGaussian, BrokenPowerLaw, PowerLawTwoGaussians, conditional_2dimpdf, conditional_2dimz_pdf, piecewise_constant_2d_distribution_normalized,paired_2dimpdf
from .priors import PowerLawStationary, PowerLawLinear, GaussianStationary, GaussianLinear, _mixed_linear_function, _mixed_double_sigmoid_function
from .priors import BrokenPowerLawTripleMultiPeak, get_gaussian_norm
import copy
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM
from .PBH_spin_dists import chi1_analytical_fit, chi2_analytical_fit

modgravity_wrappers = ['eps0_mod_wrap','Xi0_mod_wrap','extraD_mod_wrap',
                      'cM_mod_wrap','alphalog_mod_wrap']


lcdm_wrappers = ['FlatLambdaCDM_wrap','FlatwCDM_wrap',
                'Flatw0waCDM_wrap']

    
# A parent class for the rate
# LVK Reviewed
class rate_default(object):
    def evaluate(self,z):
        return self.rate.evaluate(z)
    def log_evaluate(self,z):
        return self.rate.log_evaluate(z)
    
# LVK Reviewed
class rateevolution_PowerLaw(rate_default):
    def __init__(self):
        self.population_parameters=['gamma']
    def update(self,**kwargs):
        self.rate=powerlaw_rate(**kwargs)

# LVK Reviewed
class rateevolution_Madau(rate_default):
    def __init__(self):
        self.population_parameters=['gamma','kappa','zp']
    def update(self,**kwargs):
        self.rate=md_rate(**kwargs)

class rateevolution_Madau_gamma(rate_default):
    def __init__(self):
        self.population_parameters=['gamma','kappa','zp','a','b','c']
    def update(self,**kwargs):
        self.rate=md_gamma_rate(**kwargs)

class rateevolution_beta(rate_default):
    def __init__(self):
        self.population_parameters=['a','b','c']
    def update(self,**kwargs):
        self.rate=beta_rate(**kwargs)

class rateevolution_beta_line(rate_default):
    def __init__(self):
        self.population_parameters=['a','b','c','d']
    def update(self,**kwargs):
        self.rate=beta_rate_line(**kwargs)


# LVK Reviewed
class FlatLambdaCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatLambdaCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

# LVK Reviewed
class FlatwCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0','w0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatwCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))


class Flatw0waCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0','w0','wa']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=Flatw0waCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

class eps0_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['eps0']
        self.cosmology=eps0_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),eps0=kwargs['eps0'])

# LVK Reviewed
class Xi0_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['Xi0','n']
        self.cosmology=Xi0_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),Xi0=kwargs['Xi0'],n=kwargs['n'])

# LVK Reviewed
class extraD_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['D','n','Rc']
        self.cosmology=extraD_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),D=kwargs['D'],n=kwargs['n'],Rc=kwargs['Rc'])

# LVK Reviewed
class cM_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['cM']
        self.cosmology=cM_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),cM=kwargs['cM'])

# LVK Reviewed
class alphalog_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['alphalog_1','alphalog_2','alphalog_3']
        self.cosmology=alphalog_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),alphalog_1=kwargs['alphalog_1']
                                       ,alphalog_2=kwargs['alphalog_2'],alphalog_3=kwargs['alphalog_3'])

# A parent class for the standard 1D mass probabilities
class pm_prob(object):
    def pdf(self,mass_1_source):
        return self.prior.pdf(mass_1_source)
    def log_pdf(self,mass_1_source):
        return self.prior.log_pdf(mass_1_source)

class mass_ratio_prior_Gaussian(pm_prob):
    def __init__(self):
        self.population_parameters=['mu_q','sigma_q']
    def update(self,**kwargs):
        p1=TruncatedGaussian(kwargs['mu_q'],kwargs['sigma_q'],0.,1.)
        self.prior=p1

class mass_ratio_prior_Powerlaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_q']
    def update(self,**kwargs):
        self.prior=PowerLaw(0.,1.,kwargs['alpha_q'])

class lowSmoothedwrapper(pm_prob):
   def __init__(self, mw):
        self.population_parameters = ['delta_m'] + mw.population_parameters
        self.mw = mw
   def update(self,**kwargs):
        self.mw.update(**{key:kwargs[key] for key in self.mw.population_parameters})
        self.prior = LowpassSmoothedProb(self.mw.prior,kwargs['delta_m'])
 
# A parent class for the standard mass probabilities
# LVK Reviewed
class pm1m2_prob(object):
    def pdf(self,mass_1_source,mass_2_source):
        return self.prior.pdf(mass_1_source,mass_2_source)
    def log_pdf(self,mass_1_source,mass_2_source):
        return self.prior.log_pdf(mass_1_source,mass_2_source)
    
class pm1m2z_prob(object):
    def pdf(self,mass_1_source,mass_2_source,z):
        return self.prior.pdf(mass_1_source,mass_2_source,z)
    def log_pdf(self,mass_1_source,mass_2_source,z):
        return self.prior.log_pdf(mass_1_source,mass_2_source,z)

class massprior_PowerLaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax']
    def update(self,**kwargs):
        self.prior=PowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'])
        
class massprior_PowerLawPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g','sigma_g','lambda_peak']
    def update(self,**kwargs):
        self.prior=PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g'])
        
class massprior_BrokenPowerLaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b']
    def update(self,**kwargs):
        self.prior=BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'])
        
class massprior_MultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g']
    def update(self,**kwargs):
        self.prior=PowerLawTwoGaussians(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high'])


class massprior_BrokenPowerLawMultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g']
    def update(self,**kwargs):
        self.prior=BrokenPowerLawMultiPeak(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high'])

# New class for TripleMultipeak \(multipopulation model\)
class massprior_BrokenPowerLawTripleMultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b','mu_g_1','sigma_g_1','lambda_g','mu_g_2','sigma_g_2','lambda_1','mu_g_3','sigma_g_3','lambda_2']
    def update(self,**kwargs):
        self.prior=BrokenPowerLawTripleMultiPeak(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'],
                                                 kwargs['lambda_g'],kwargs['lambda_1'],kwargs['lambda_2'],
                                                 kwargs['mu_g_1'],kwargs['sigma_g_1'],kwargs['mmin'],kwargs['mu_g_1']+5*kwargs['sigma_g_1'],
                                                 kwargs['mu_g_2'],kwargs['sigma_g_2'],kwargs['mmin'],kwargs['mu_g_2']+5*kwargs['sigma_g_2'],
                                                 kwargs['mu_g_3'],kwargs['sigma_g_3'],kwargs['mmin'],kwargs['mu_g_3']+5*kwargs['sigma_g_3'])


class m1m2_conditioned(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta'])
        self.prior=conditional_2dimpdf(p1,p2)

class m1m2_conditioned_lowpass_m2(pm1m2z_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimz_pdf(p1,p2)

class m1m2_conditioned_lowpass(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta','delta_m']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = LowpassSmoothedProb(self.wrapper_m.prior,kwargs['delta_m'])
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)


class m1m2_paired_massratio_dip(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        def pairing_function(m1,m2,beta=kwargs['beta']):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.power(q,beta)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)


class m1m2_paired_massratio_dip_general(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],
                            rightdip=kwargs['rightdip']):
            
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=rightdip
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>rightdip
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)


class m1m2_paired_massratio_bpl_dip_farah_2022(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLaw()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        kwargs['b'] = (kwargs['leftdip']-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top']):
            
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=5.
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>5.
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)

# Class for the BPLMultipopTriplePeak_dip
class m1m2_paired_bpl_triplepeak_dip(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawTripleMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m

    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],mbreak=mbreak):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=mbreak
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>mbreak
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)

class m1m2_paired_massratio_bplmulti_dip(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],mbreak=mbreak):
            # The motivation for using only m2 for beta top and bottom is that if m2 is a NS for sure 
            # it is more probable that the binary comes from isolated stellar binaries.
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=mbreak
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>mbreak
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)



class m1m2_paired_massratio_bplmulti_dip_conditioned(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p1 = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        # Equivalent to a broken power law distribution in q = m2/m1
        bpl = BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta_bottom'],kwargs['beta_top'],kwargs['b'])
        p2 = LowpassSmoothedProb(bpl,kwargs['bottomsmooth'])
        
        self.prior=conditional_2dimpdf(p1,p2)

class m1m2_paired(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
    
        def pairing_function(m1,m2,beta=kwargs['beta']):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.power(q,beta)
            toret[q>1] = 0.
            return toret
        self.prior=paired_2dimpdf(self.wrapper_m.prior,pairing_function)


class massprior_BinModel2d(pm1m2_prob):
    def __init__(self, n_bins_1d):
        self.population_parameters=['mmin','mmax']
        n_bins_total = int(n_bins_1d * (n_bins_1d + 1) / 2)
        self.bin_parameter_list = ['bin_' + str(i) for i in range(n_bins_total)]
        self.population_parameters += self.bin_parameter_list
    def update(self,**kwargs):
        kwargs_bin_parameters = np.array([kwargs[key] for key in self.bin_parameter_list])
        
        pdf_dist = piecewise_constant_2d_distribution_normalized(
            kwargs['mmin'], 
            kwargs['mmax'],
            kwargs_bin_parameters
        )
        
        self.prior=pdf_dist


# ----------- #
# Spin models #
# ----------- #

class spinprior_default_evolving_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi','sigma_chi','mu_dot','sigma_dot'
                                    ,'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):
        self.mu_chi = kwargs['mu_chi']
        self.sigma_chi = kwargs['sigma_chi']
        self.mu_dot = kwargs['mu_dot']
        self.sigma_dot = kwargs['sigma_dot']     
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):

        xp = get_module_array(chi_1)
        sx = get_module_array_scipy(chi_1)
 
        #evolving means/sigmas (notice the dependence on the source frame mass)
        mu_chi_1 = self.mu_chi + self.mu_dot*mass_1_source
        sigma_chi_1 = self.sigma_chi + self.sigma_dot*mass_1_source
        mu_chi_2 = self.mu_chi + self.mu_dot*mass_2_source
        sigma_chi_2 = self.sigma_chi + self.sigma_dot*mass_2_source

        #compute the truncated gaussian referred to chi1
        a, b = (0. - mu_chi_1) / sigma_chi_1, (1. - mu_chi_1) / sigma_chi_1 
        g1 = sx.stats.truncnorm.pdf(chi_1,a,b,loc=mu_chi_1,scale=sigma_chi_1)

        #compute the truncated gaussian referred to chi2
        a, b = (0. - mu_chi_2) / sigma_chi_2, (1. - mu_chi_2) / sigma_chi_2 
        g2 = sx.stats.truncnorm.pdf(chi_2,a,b,loc=mu_chi_2,scale=sigma_chi_2)

        #implement eq. 12 of https://arxiv.org/pdf/2406.01679 
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))

        #log(p(chi1,chi2|m1,m2,Lambda))
        out = xp.log(g1)+xp.log(g2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))

class spinprior_default_beta_window_gaussian(object):
    def __init__(self):

        #population parameters
        self.population_parameters= ['mt', 
                                     'delta_mt','mix_f',
                                     'alpha_chi','beta_chi',
                                     'mu_chi','sigma_chi',
                                     'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
    

    def update(self,**kwargs):
        
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        #condition for physical solutions
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
        self.beta_pdf_chi = BetaDistribution(self.alpha_chi,self.beta_chi)
        
        self.mu_chi = kwargs['mu_chi']
        self.sigma_chi = kwargs['sigma_chi']
        self.csi_spin = kwargs['csi_spin']
        self.gaussian_pdf_chi = TruncatedGaussian(kwargs['mu_chi'],kwargs['sigma_chi'],0.,1.)

        self.mt, self.delta_mt, self.mix_f = kwargs['mt'], kwargs['delta_mt'], kwargs['mix_f']

        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        
        xp = get_module_array(chi_1)
        # FIXME: The sigmoid function implementation has been changed. Check it is correct.
        wz_1 = _mixed_double_sigmoid_function(mass_1_source, self.mix_f, 0., self.mt, self.delta_mt)
        wz_2 = _mixed_double_sigmoid_function(mass_2_source, self.mix_f, 0., self.mt, self.delta_mt)


        #compute p(chi_i|m_i,Lambda)
        pdf_1 = wz_1*self.beta_pdf_chi.pdf(chi_1)+(1-wz_1)*self.gaussian_pdf_chi.pdf(chi_1)
        pdf_2 = wz_2*self.beta_pdf_chi.pdf(chi_2)+(1-wz_2)*self.gaussian_pdf_chi.pdf(chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        out = xp.log(pdf_1)+xp.log(pdf_2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))


class spinprior_default_beta_window_beta(object):
    def __init__(self):
        self.population_parameters= ['mt', 
                                     'delta_mt','mix_f',
                                     'alpha_chi_low','beta_chi_low',
                                     'alpha_chi_high','beta_chi_high',
                                     'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
    

    def update(self,**kwargs):
        
        self.alpha_chi_low = kwargs['alpha_chi_low']
        self.beta_chi_low = kwargs['beta_chi_low']
        self.alpha_chi_high = kwargs['alpha_chi_high']
        self.beta_chi_high = kwargs['beta_chi_high']
        self.csi_spin = kwargs['csi_spin']
        
        if (self.alpha_chi_low <= 1) | (self.beta_chi_low <= 1) | (self.alpha_chi_high <= 1) | (self.beta_chi_high <= 1):
            raise ValueError('Alpha and Beta must be > 1') 
        
        self.beta_pdf_chi_low = BetaDistribution(self.alpha_chi_low,self.beta_chi_low)
        self.beta_pdf_chi_high = BetaDistribution(self.alpha_chi_high,self.beta_chi_high)

        self.mt, self.delta_mt, self.mix_f = kwargs['mt'], kwargs['delta_mt'], kwargs['mix_f']

        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        
        xp = get_module_array(chi_1)
        # FIXME: The sigmoid function implementation has been changed. Check it is correct.
        wz_1 = _mixed_double_sigmoid_function(mass_1_source, self.mix_f, 0., self.mt, self.delta_mt)
        wz_2 = _mixed_double_sigmoid_function(mass_2_source, self.mix_f, 0., self.mt, self.delta_mt)

        pdf_1 = wz_1*self.beta_pdf_chi_low.pdf(chi_1)+(1-wz_1)*self.beta_pdf_chi_high.pdf(chi_1)
        pdf_2 = wz_2*self.beta_pdf_chi_low.pdf(chi_2)+(1-wz_2)*self.beta_pdf_chi_high.pdf(chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        out = xp.log(pdf_1)+xp.log(pdf_2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))

        
class spinprior_default(object):
    def __init__(self):
        self.population_parameters=['alpha_chi','beta_chi','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
        self.name='DEFAULT'

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
    
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        return self.beta_pdf.log_pdf(chi_1)+self.beta_pdf.log_pdf(chi_2)+log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))


class spinprior_default_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi_1','mu_chi_2','sigma_chi_1','sigma_chi_2','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):        
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        self.g1 = TruncatedGaussian(kwargs['mu_chi_1'],kwargs['sigma_chi_1'],0.,1.)
        self.g2 = TruncatedGaussian(kwargs['mu_chi_2'],kwargs['sigma_chi_2'],0.,1.)
    
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        return self.g1.log_pdf(chi_1)+self.g2.log_pdf(chi_2)+log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))



########### TGR Implementation ######################
class pseobprior_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_domega220','sigma_domega220','mu_dtau220','sigma_dtau220','rho_pseob']
    def update(self,**kwargs):
        # Note, the bounds below are set to be consistent with Lorenzo's injections
        self.pdf_evaluator=Bivariate2DGaussian(x1min=-10,x1max=10.,x1mean=kwargs['mu_domega220'],
                                               x2min=-10,x2max=10.,x2mean=kwargs['mu_dtau220'],
                                               x1variance=kwargs['sigma_domega220']**2.,x12covariance=kwargs['rho_pseob']*kwargs['sigma_domega220']*kwargs['sigma_dtau220'],
                                               x2variance=kwargs['sigma_dtau220']**2.)
    def log_pdf(self,domega220,dtau220):
        return self.pdf_evaluator.log_pdf(domega220,dtau220)
    def pdf(self,domega220,dtau220):
        xp = get_module_array(domega220)
        return xp.exp(self.log_pdf(domega220,dtau220))
#####################################################


# LVK Reviewed

class spinprior_gaussian(object):
    def __init__(self):

        self.population_parameters=['mu_chi_eff','sigma_chi_eff','mu_chi_p','sigma_chi_p','rho']
        self.event_parameters=['chi_eff','chi_p']
        self.name='GAUSSIAN'
    def update(self,**kwargs):
        self.pdf_evaluator=Bivariate2DGaussian(x1min=-1.,x1max=1.,x1mean=kwargs['mu_chi_eff'],
                                               x2min=0.,x2max=1.,x2mean=kwargs['mu_chi_p'],
                                               x1variance=kwargs['sigma_chi_eff']**2.,x12covariance=kwargs['rho']*kwargs['sigma_chi_eff']*kwargs['sigma_chi_p'],
                                               x2variance=kwargs['sigma_chi_p']**2.)
    def log_pdf(self,chi_eff,chi_p):
        return self.pdf_evaluator.log_pdf(chi_eff,chi_p)
    def pdf(self,chi_eff,chi_p):
        xp = get_module_array(chi_eff)
        return xp.exp(self.log_pdf(chi_eff,chi_p))
    def sample(self, N):
        return self.pdf_evaluator.sample(N)

class spinprior_ECOs_totally_reflective(object):
    def __init__(self,q=1.):
        # q=1 is the polar case, q = 2 is the axial case, m=2 fixed
        self.q=q
        self.population_parameters=['alpha_chi','beta_chi','eps', 'f_eco', 'sigma_chi_ECO']
        self.event_parameters=['chi_1','chi_2'] 
        self.name='DEFAULT'
        
    def get_chi_crit(self, eps):
        xp = get_module_array(eps)   
        return xp.pi*(1.+self.q)/(2*xp.abs(xp.log(eps)))

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.eps = kwargs['eps']
        self.f_eco = kwargs['f_eco']
        self.sigma = kwargs['sigma_chi_ECO']
        self.chi_crit = self.get_chi_crit(self.eps)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
            
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
        self.truncatedbeta_pdf = TruncatedBetaDistribution(self.alpha_chi,self.beta_chi,self.chi_crit)
        self.truncatedgaussian_pdf = TruncatedGaussian(self.chi_crit, self.sigma, 0., self.chi_crit)
        self.lambda_eco = 1-self.beta_pdf.cdf(np.array([self.get_chi_crit(self.eps)]))[0]
        
        
    def pdf(self,chi_1,chi_2):
        p_chi_1 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_1) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_1)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_1) 
        p_chi_2 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_2) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_2)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_2) 
        return p_chi_1*p_chi_2
        
        
    def log_pdf(self,chi_1,chi_2):
        xp = get_module_array(chi_1)
        return xp.log(self.pdf(chi_1,chi_2))
    

# ------------------------ #
# Redshift evolving models #
# ------------------------ #

class PowerLaw_PowerLaw():
    '''
        Class implementing the mass function model for two stationary PowerLaws.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'mix']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a = kwargs['alpha_a']
        self.mmin_a  = kwargs['mmin_a']
        self.mmax_a  = kwargs['mmax_a']
        self.alpha_b = kwargs['alpha_b']
        self.mmin_b  = kwargs['mmin_b']
        self.mmax_b  = kwargs['mmax_b']
        self.mix     = kwargs['mix']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a, self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b, self.mmax_b)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix > 1) or (self.mix < 0):
            return xp.nan
        else:
            return self.mix * powerlaw_part_a + (1-self.mix) * powerlaw_part_b
    
    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_PowerLaw_PowerLaw():
    '''
        Class implementing the mass function model for three stationary PowerLaws.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'alpha_c', 'mmin_c', 'mmax_c', 'mix_alpha', 'mix_beta']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b', 'delta_m_c']

    def update(self,**kwargs):

        self.alpha_a   = kwargs['alpha_a']
        self.mmin_a    = kwargs['mmin_a']
        self.mmax_a    = kwargs['mmax_a']
        self.alpha_b   = kwargs['alpha_b']
        self.mmin_b    = kwargs['mmin_b']
        self.mmax_b    = kwargs['mmax_b']
        self.alpha_c   = kwargs['alpha_c']
        self.mmin_c    = kwargs['mmin_c']
        self.mmax_c    = kwargs['mmax_c']
        self.mix_alpha = kwargs['mix_alpha']
        self.mix_beta  = kwargs['mix_beta']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']
            self.delta_m_c = kwargs['delta_m_c']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a, self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b, self.mmax_b)
        powerlaw_class_c = PowerLawStationary(self.alpha_c, self.mmin_c, self.mmax_c)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
            powerlaw_class_c = LowpassSmoothedProb(powerlaw_class_c, self.delta_m_c)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)
        powerlaw_part_c = powerlaw_class_c.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix_alpha > 1) or (self.mix_alpha < 0) or (self.mix_beta > 1) or (self.mix_beta < 0) or (self.mix_alpha + self.mix_beta > 1):
            return xp.nan
        else:
            return self.mix_alpha * powerlaw_part_a + self.mix_beta * powerlaw_part_b + (1 - self.mix_beta - self.mix_alpha) * powerlaw_part_c

    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_PowerLaw_Gaussian():
    '''
        Class implementing the mass function model for two stationary PowerLaws
        and a Gaussian peak.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'mu_g', 'sigma_g', 'mix_alpha', 'mix_beta']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a   = kwargs['alpha_a']
        self.mmin_a    = kwargs['mmin_a']
        self.mmax_a    = kwargs['mmax_a']
        self.alpha_b   = kwargs['alpha_b']
        self.mmin_b    = kwargs['mmin_b']
        self.mmax_b    = kwargs['mmax_b']
        self.mu_g      = kwargs['mu_g']
        self.sigma_g   = kwargs['sigma_g']
        self.mix_alpha = kwargs['mix_alpha']
        self.mix_beta  = kwargs['mix_beta']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a,  self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b,  self.mmax_b)
        gaussian_class   = GaussianStationary(self.mu_g,    self.sigma_g, self.mmin_a)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)
        gaussian_part   = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix_alpha > 1) or (self.mix_alpha < 0) or (self.mix_beta > 1) or (self.mix_beta < 0) or (self.mix_alpha + self.mix_beta > 1):
            return xp.nan
        else:
            return self.mix_alpha * powerlaw_part_a + self.mix_beta * powerlaw_part_b + (1 - self.mix_beta - self.mix_alpha) * gaussian_part
    
    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for a stationary PowerLaw and a redshift linearly-dependent Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 1, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha', 'mmin', 'mmax', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha    = kwargs['alpha']
        self.mmin     = kwargs['mmin']
        self.mmax     = kwargs['mmax']
        self.mu_z0    = kwargs['mu_z0']
        self.mu_z1    = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']
        self.mix_z0   = kwargs['mix_z0']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = self.mix_z0

        powerlaw_class = PowerLawStationary(self.alpha, self.mmin, self.mmax)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProb(powerlaw_class, self.delta_m)
        gaussian_class = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin)
        powerlaw_part  = powerlaw_class.pdf(m)
        gaussian_part  = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * powerlaw_part + (1-wz) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))
    

class PowerLaw_GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for a stationary PowerLaw and two redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the functions for the redshift transition
            between the PowerLaw and the two Gaussian peaks. The transition is
            the same between the PowerLaw and the first Gaussian a, and the two
            Gaussians a and b.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 1, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha', 'mmin', 'mmax', 'mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha        = kwargs['alpha']
        self.mmin         = kwargs['mmin']
        self.mmax         = kwargs['mmax']
        self.mu_a_z0      = kwargs['mu_a_z0']
        self.mu_a_z1      = kwargs['mu_a_z1']
        self.sigma_a_z0   = kwargs['sigma_a_z0']
        self.sigma_a_z1   = kwargs['sigma_a_z1']
        self.mu_b_z0      = kwargs['mu_b_z0']
        self.mu_b_z1      = kwargs['mu_b_z1']
        self.sigma_b_z0   = kwargs['sigma_b_z0']
        self.sigma_b_z1   = kwargs['sigma_b_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(         z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(         z, self.mix_beta_z0 , self.mix_beta_z1 )
            elif self.redshift_transition == 'sigmoid':
                wz_alpha = _mixed_double_sigmoid_function( z, self.mix_alpha_z0, self.mix_alpha_z1, self.zt, self.delta_zt)
                wz_beta  = _mixed_double_sigmoid_function( z, self.mix_beta_z0 , self.mix_beta_z1 , self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = self.mix_alpha_z0
            wz_beta  = self.mix_beta_z0

        powerlaw_class = PowerLawStationary(self.alpha, self.mmin, self.mmax)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProb(powerlaw_class, self.delta_m)
        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin)
        powerlaw_part    = powerlaw_class.pdf(m)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part + wz_beta * gaussian_a_part + (1 - wz_beta - wz_alpha) * gaussian_b_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class PowerLawRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for both a redshift linearly-dependent PowerLaw and Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_z0', 'alpha_z1', 'mmin_z0', 'mmin_z1', 'mmax_z0', 'mmax_z1', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha_z0 = kwargs['alpha_z0']
        self.alpha_z1 = kwargs['alpha_z1']
        self.mmin_z0  = kwargs['mmin_z0']
        self.mmin_z1  = kwargs['mmin_z1']
        self.mmax_z0  = kwargs['mmax_z0']
        self.mmax_z1  = kwargs['mmax_z1']
        self.mu_z0    = kwargs['mu_z0']
        self.mu_z1    = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']
        self.mix_z0   = kwargs['mix_z0']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = self.mix_z0

        powerlaw_class = PowerLawLinear(z, self.alpha_z0, self.alpha_z1, self.mmin_z0, self.mmin_z1, self.mmax_z0, self.mmax_z1)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProbEvolving(powerlaw_class, self.delta_m)
        gaussian_class = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin_z0)
        powerlaw_part  = powerlaw_class.pdf(m)
        gaussian_part  = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * powerlaw_part + (1-wz) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))
    

class PowerLawRedshiftLinear_PowerLawRedshiftLinear_PowerLawRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for three redshift linearly-dependent PowerLaws.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaws (a, b and c).
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_a_z0', 'alpha_a_z1', 'mmin_a_z0', 'mmin_a_z1', 'mmax_a_z0', 'mmax_a_z1', 'alpha_b_z0', 'alpha_b_z1', 'mmin_b_z0', 'mmin_b_z1', 'mmax_b_z0', 'mmax_b_z1', 'alpha_c_z0', 'alpha_c_z1', 'mmin_c_z0', 'mmin_c_z1', 'mmax_c_z0', 'mmax_c_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.population_parameters += ['delta_m_a', 'delta_m_b', 'delta_m_c']

    def update(self,**kwargs):

        self.alpha_a_z0   = kwargs['alpha_a_z0']
        self.alpha_a_z1   = kwargs['alpha_a_z1']
        self.mmin_a_z0    = kwargs['mmin_a_z0']
        self.mmin_a_z1    = kwargs['mmin_a_z1']
        self.mmax_a_z0    = kwargs['mmax_a_z0']
        self.mmax_a_z1    = kwargs['mmax_a_z1']
        self.alpha_b_z0   = kwargs['alpha_b_z0']
        self.alpha_b_z1   = kwargs['alpha_b_z1']
        self.mmin_b_z0    = kwargs['mmin_b_z0']
        self.mmin_b_z1    = kwargs['mmin_b_z1']
        self.mmax_b_z0    = kwargs['mmax_b_z0']
        self.mmax_b_z1    = kwargs['mmax_b_z1']
        self.alpha_c_z0   = kwargs['alpha_b_z0']
        self.alpha_c_z1   = kwargs['alpha_b_z1']
        self.mmin_c_z0    = kwargs['mmin_c_z0']
        self.mmin_c_z1    = kwargs['mmin_c_z1']
        self.mmax_c_z0    = kwargs['mmax_c_z0']
        self.mmax_c_z1    = kwargs['mmax_c_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.delta_m_a    = kwargs['delta_m_a']
            self.delta_m_b    = kwargs['delta_m_b']
            self.delta_m_c    = kwargs['delta_m_c']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(z, self.mix_beta_z0,  self.mix_beta_z1 )
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = self.mix_alpha_z0
            wz_beta  = self.mix_beta_z0

        powerlaw_class_a = PowerLawLinear(z, self.alpha_a_z0, self.alpha_a_z1, self.mmin_a_z0, self.mmin_a_z1, self.mmax_a_z0, self.mmax_a_z1)
        powerlaw_class_b = PowerLawLinear(z, self.alpha_b_z0, self.alpha_b_z1, self.mmin_b_z0, self.mmin_b_z1, self.mmax_b_z0, self.mmax_b_z1)
        powerlaw_class_c = PowerLawLinear(z, self.alpha_c_z0, self.alpha_c_z1, self.mmin_c_z0, self.mmin_c_z1, self.mmax_c_z0, self.mmax_c_z1)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProbEvolving(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProbEvolving(powerlaw_class_b, self.delta_m_b)
            powerlaw_class_c = LowpassSmoothedProbEvolving(powerlaw_class_c, self.delta_m_c)
        powerlaw_part_a  = powerlaw_class_a.pdf(m)
        powerlaw_part_b  = powerlaw_class_b.pdf(m)
        powerlaw_part_c  = powerlaw_class_c.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part_a + wz_beta * powerlaw_part_b + (1 - wz_beta - wz_alpha) * powerlaw_part_c
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class PowerLawRedshiftLinear_PowerLawRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for both two redshift linearly-dependent PowerLaws and a Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaws (a and b) and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_a_z0', 'alpha_a_z1', 'mmin_a_z0', 'mmin_a_z1', 'mmax_a_z0', 'mmax_a_z1', 'alpha_b_z0', 'alpha_b_z1', 'mmin_b_z0', 'mmin_b_z1', 'mmax_b_z0', 'mmax_b_z1', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a_z0   = kwargs['alpha_a_z0']
        self.alpha_a_z1   = kwargs['alpha_a_z1']
        self.mmin_a_z0    = kwargs['mmin_a_z0']
        self.mmin_a_z1    = kwargs['mmin_a_z1']
        self.mmax_a_z0    = kwargs['mmax_a_z0']
        self.mmax_a_z1    = kwargs['mmax_a_z1']
        self.alpha_b_z0   = kwargs['alpha_b_z0']
        self.alpha_b_z1   = kwargs['alpha_b_z1']
        self.mmin_b_z0    = kwargs['mmin_b_z0']
        self.mmin_b_z1    = kwargs['mmin_b_z1']
        self.mmax_b_z0    = kwargs['mmax_b_z0']
        self.mmax_b_z1    = kwargs['mmax_b_z1']
        self.mu_z0        = kwargs['mu_z0']
        self.mu_z1        = kwargs['mu_z1']
        self.sigma_z0     = kwargs['sigma_z0']
        self.sigma_z1     = kwargs['sigma_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.delta_m_a    = kwargs['delta_m_a']
            self.delta_m_b    = kwargs['delta_m_b']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(z, self.mix_beta_z0,  self.mix_beta_z1 )
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = self.mix_alpha_z0
            wz_beta  = self.mix_beta_z0

        powerlaw_class_a = PowerLawLinear(z, self.alpha_a_z0, self.alpha_a_z1, self.mmin_a_z0, self.mmin_a_z1, self.mmax_a_z0, self.mmax_a_z1)
        powerlaw_class_b = PowerLawLinear(z, self.alpha_b_z0, self.alpha_b_z1, self.mmin_b_z0, self.mmin_b_z1, self.mmax_b_z0, self.mmax_b_z1)
        gaussian_class   = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin_a_z0)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProbEvolving(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProbEvolving(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a  = powerlaw_class_a.pdf(m)
        powerlaw_part_b  = powerlaw_class_b.pdf(m)
        gaussian_part    = gaussian_class.pdf(m)
    
        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part_a + wz_beta * powerlaw_part_b + (1 - wz_beta - wz_alpha) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for two redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_redshift_mixture = 1):
        
        self.population_parameters = ['mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mix_z0', 'mmin_g']
        self.redshift_transition   = redshift_transition
        self.flag_redshift_mixture = flag_redshift_mixture
        
        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']

    def update(self,**kwargs):

        self.mu_a_z0    = kwargs['mu_a_z0']
        self.mu_a_z1    = kwargs['mu_a_z1']
        self.sigma_a_z0 = kwargs['sigma_a_z0']
        self.sigma_a_z1 = kwargs['sigma_a_z1']
        self.mu_b_z0    = kwargs['mu_b_z0']
        self.mu_b_z1    = kwargs['mu_b_z1']
        self.sigma_b_z0 = kwargs['sigma_b_z0']
        self.sigma_b_z1 = kwargs['sigma_b_z1']
        self.mix_z0     = kwargs['mix_z0']
        self.mmin_g     = kwargs['mmin_g']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = self.mix_z0

        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin_g)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin_g)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * gaussian_a_part + (1-wz) * gaussian_b_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianRedshiftLinear_GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for three redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the three Gaussian peaks. The transition is the same between the
            first two Gaussians a and b, and the other two Gaussians b and c.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_redshift_mixture = 1):

        self.population_parameters = ['mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mu_c_z0', 'mu_c_z1', 'sigma_c_z0', 'sigma_c_z1', 'mix_alpha_z0', 'mix_beta_z0', 'mmin_g']
        self.redshift_transition   = redshift_transition
        self.flag_redshift_mixture = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']

    def update(self,**kwargs):

        self.mu_a_z0      = kwargs['mu_a_z0']
        self.mu_a_z1      = kwargs['mu_a_z1']
        self.sigma_a_z0   = kwargs['sigma_a_z0']
        self.sigma_a_z1   = kwargs['sigma_a_z1']
        self.mu_b_z0      = kwargs['mu_b_z0']
        self.mu_b_z1      = kwargs['mu_b_z1']
        self.sigma_b_z0   = kwargs['sigma_b_z0']
        self.sigma_b_z1   = kwargs['sigma_b_z1']
        self.mu_c_z0      = kwargs['mu_c_z0']
        self.mu_c_z1      = kwargs['mu_c_z1']
        self.sigma_c_z0   = kwargs['sigma_c_z0']
        self.sigma_c_z1   = kwargs['sigma_c_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']
        self.mmin_g       = kwargs['mmin_g']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(         z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(         z, self.mix_beta_z0 , self.mix_beta_z1 )
            elif self.redshift_transition == 'sigmoid':
                wz_alpha = _mixed_double_sigmoid_function( z, self.mix_alpha_z0, self.mix_alpha_z1, self.zt, self.delta_zt)
                wz_beta  = _mixed_double_sigmoid_function( z, self.mix_beta_z0 , self.mix_beta_z1 , self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = self.mix_alpha_z0
            wz_beta  = self.mix_beta_z0

        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin_g)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin_g)
        gaussian_c_class = GaussianLinear(z, self.mu_c_z0, self.mu_c_z1, self.sigma_c_z0, self.sigma_c_z1, self.mmin_g)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)
        gaussian_c_part  = gaussian_c_class.pdf(m)

        # Impose the rate to be between [0,1].
        if   (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * gaussian_a_part + wz_beta * gaussian_b_part + (1 - wz_beta - wz_alpha) * gaussian_c_part

    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianEvolving():
    '''
        Class implementing the mass function model conditioned on redshift p(m|z), for
        one redshift evolving Gaussian peak with arbitrary polynomial redshift expansion.

        Some options are available:
            - order sets the order of the redshift expansion. The population
            parameters [mu_zx, sigma_zx] are automatically defined from the
            selected order.
            Ex. order = 2 gives mu_z = mu_z0 + mu_z1*z + mu_z2*z^2

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, order = 1):

        self.order = order
        mu_list    = ['mu_z{}'.format(   i) for i in range(order + 1)]
        sigma_list = ['sigma_z{}'.format(i) for i in range(order + 1)]
        self.population_parameters = mu_list + sigma_list

    def update(self, **kwargs):

        for par in kwargs.keys():
            globals()['self.%s' % par] = kwargs[par]

    def polynomial(self, expansion_order, x, variable):
        
        pol = 0
        for i in range(expansion_order + 1):
            par = '{}_z{}'.format(variable, i)
            pol += globals()['self.%s' % par] * x ** i
        return pol

    def log_pdf(self, m, z):

        xp = get_module_array(m)
        sx = get_module_array_scipy(m)
        self.muz    = self.polynomial(self.order, z, 'mu')
        self.sigmaz = self.polynomial(self.order, z, 'sigma')
        a, b = (0. - self.muz) / self.sigmaz, (self.muz + 6*self.muz - self.muz) / self.sigmaz  # Truncte the Gaussian at zero and mu+6*sigma. This improve the numerical stability.
        gaussian = xp.log( sx.stats.truncnorm.pdf(m, a, b, loc = self.muz, scale = self.sigmaz) )
        return gaussian

    def pdf(self, m, z):
        xp = get_module_array(m)
        return xp.exp(self.log_pdf(m, z))
    
    def return_mu_sigma(self):
        return self.muz, self.sigmaz
   
# ABH pop pdf: correlated gaussian -> correlated gaussian
class spinprior_default_corr_gaussian_window_corr_gaussian(object):
    def __init__(self):

        # Population parameters
        self.population_parameters = [
            'Mt', 'delta_Mt', 'mix_f_M',
            'mu_chi_eff_l', 'sigma_chi_eff_l', 'mu_chi_p_l', 'sigma_chi_p_l', 'rho_l',
            'mu_chi_eff_h', 'sigma_chi_eff_h', 'mu_chi_p_h', 'sigma_chi_p_h', 'rho_h'
        ]
        self.event_parameters = ['chi_eff', 'chi_p']

        # Initialize Gaussian priors for low and high mass
        self.name='corr_gaussian_window_corr_gaussian'

    def update(self, **kwargs):
        
        # Update logistic function parameters
        self.Mt = kwargs['Mt']
        self.delta_Mt = kwargs['delta_Mt']
        self.mix_f_M = kwargs['mix_f_M']

        # Extract low Gaussian parameters
        low_params = {
            'mu_chi_eff': kwargs['mu_chi_eff_l'],
            'sigma_chi_eff': kwargs['sigma_chi_eff_l'],
            'mu_chi_p': kwargs['mu_chi_p_l'],
            'sigma_chi_p': kwargs['sigma_chi_p_l'],
            'rho': kwargs['rho_l']
        }
        
        # Extract high Gaussian parameters
        high_params = {
            'mu_chi_eff': kwargs['mu_chi_eff_h'],
            'sigma_chi_eff': kwargs['sigma_chi_eff_h'],
            'mu_chi_p': kwargs['mu_chi_p_h'],
            'sigma_chi_p': kwargs['sigma_chi_p_h'],
            'rho': kwargs['rho_h']
        }

        # Instantiate the Gaussian distributions
        self.gaussian_low = spinprior_gaussian()
        self.gaussian_low.update(**low_params)

        self.gaussian_high = spinprior_gaussian()
        self.gaussian_high.update(**high_params)


    def log_pdf(self,chi_eff, chi_p, mass_1_source,mass_2_source):
        
        xp = get_module_array(chi_eff)

        #compute logistic function for M
        M=mass_1_source+mass_2_source
        wz_M = _mixed_sigmoid_function(M, self.Mt, self.delta_Mt, self.mix_f_M)

        #compute p(chi_p, chi_eff|M,Lambda)
        pdf = wz_M*self.gaussian_low.pdf(chi_eff, chi_p)+(1-wz_M)*self.gaussian_high.pdf(chi_eff, chi_p)

        return xp.log(pdf)

    def pdf(self,chi_eff,chi_p, mass_1_source,mass_2_source):
        xp = get_module_array(chi_eff)
        return xp.exp(self.log_pdf(chi_eff,chi_p, mass_1_source,mass_2_source))
    

    def sample(self, N, mass_1_source, mass_2_source):
        """
        Sample (chi_eff, chi_p) given mass_1_source and mass_2_source arrays.
        
        Parameters
        ----------
        N : int
            Number of samples
        mass_1_source, mass_2_source : array-like or float
            Source-frame masses

        Returns
        -------
        chi_eff_samples, chi_p_samples : arrays
        """
        xp = get_module_array(mass_1_source)
        mass_1_source = xp.asarray(mass_1_source)
        mass_2_source = xp.asarray(mass_2_source)

        # Compute total mass
        M = mass_1_source + mass_2_source

        # Logistic weight between low and high Gaussians
        wz_M = _mixed_sigmoid_function(M, self.Mt, self.delta_Mt, self.mix_f_M)

        # Draw random uniform values to choose low/high Gaussian per sample
        choose_low = xp.random.uniform(0, 1, size=N) < wz_M

        # Allocate arrays
        chi_eff_samples = xp.empty(N)
        chi_p_samples = xp.empty(N)

        # Sample from low and high components accordingly
        n_low = xp.sum(choose_low)
        n_high = N - n_low

        if n_low > 0:
            eff_l, p_l = self.gaussian_low.sample(n_low)
            chi_eff_samples[choose_low] = eff_l
            chi_p_samples[choose_low] = p_l

        if n_high > 0:
            eff_h, p_h = self.gaussian_high.sample(n_high)
            chi_eff_samples[~choose_low] = eff_h
            chi_p_samples[~choose_low] = p_h

        return chi_eff_samples, chi_p_samples


def log_tgaussian(x,xmin,xmax,mu,sigma):
    xp = get_module_array(x)
    norm = get_gaussian_norm(xmin,xmax,mu,sigma)
    toret = xp.where((x>=xmin) & (x<=xmax),
                     -xp.log(sigma)-0.5*xp.log(2*xp.pi)-0.5*xp.power((x-mu)/sigma,2.)-xp.log(norm),
                     -xp.inf
                     )
    return toret

def log_uniform(x,xmin,xmax):
    xp = get_module_array(x)
    out = xp.where((x>=xmin) & (x<=xmax),-xp.log(xmax-xmin),-xp.inf)
    return out

# PBH pop pdf: gaussian on chi_1, chi_2 with small sigma and mean value dependent from the masses    
class spinprior_HBH(object):
    def __init__(self):

        #population parameters
        self.population_parameters= ['chi_max_1g','mean_chi_2g','sigma_chi_2g',
                                     'f_1g_m_0','f_1g_m_inf','f_1g_mt','f_1g_delta_mt']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2','mass_1_source','mass_2_source']
    
        # Initialize smeared Gaussians for chi_1, chi_2
        self.name='spinprior_HBH'

    def update(self,**kwargs):
        
        self.chi_max_1g = kwargs['chi_max_1g']
        self.mean_chi_2g = kwargs['mean_chi_2g']
        self.sigma_chi_2g = kwargs['sigma_chi_2g']
        self.f_1g_m_0 = kwargs['f_1g_m_0']
        self.f_1g_m_inf = kwargs['f_1g_m_inf']
        self.f_1g_mt = kwargs['f_1g_mt']
        self.f_1g_delta_mt = kwargs['f_1g_delta_mt']

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):

        #choose module between numpy and cupy
        xp = get_module_array(chi_1)

        p1g1g = log_tgaussian(chi_1,0.,1.,0.,self.chi_max_1g) + log_tgaussian(chi_2,0.,1.,0.,self.chi_max_1g) - xp.log(2.) - xp.log(2.)
        
        p2g1g = log_tgaussian(chi_1,0.,1.,self.mean_chi_2g,self.sigma_chi_2g) + log_tgaussian(chi_2,0.,1.,0.,self.chi_max_1g) - xp.log(2.) - xp.log(2.)

        p1g2g = log_tgaussian(chi_2,0.,1.,self.mean_chi_2g,self.sigma_chi_2g) + log_tgaussian(chi_1,0.,1.,0.,self.chi_max_1g) - xp.log(2.) - xp.log(2.)

        p2g2g = log_tgaussian(chi_2,0.,1.,self.mean_chi_2g,self.sigma_chi_2g) + log_tgaussian(chi_1,0.,1.,self.mean_chi_2g,self.sigma_chi_2g) - xp.log(2.) - xp.log(2.)
        
        f1gm1 = _mixed_double_sigmoid_function(mass_1_source,self.f_1g_mt,self.f_1g_delta_mt,self.f_1g_m_0,self.f_1g_m_inf)
        f1gm2 = _mixed_double_sigmoid_function(mass_2_source,self.f_1g_mt,self.f_1g_delta_mt,self.f_1g_m_0,self.f_1g_m_inf)

        pout = f1gm1*f1gm2*xp.exp(p1g1g) + f1gm1*(1-f1gm2)*xp.exp(p1g2g) + (1-f1gm1)*f1gm2*xp.exp(p2g1g) + (1-f1gm1)*(1-f1gm2)*xp.exp(p2g2g)  

        return xp.log(pout)
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))   



# PBH pop pdf: gaussian on chi_1, chi_2 with small sigma and mean value dependent from the masses    
class spinprior_PBH_smeared(object):
    def __init__(self):

        #population parameters
        self.population_parameters= ['sigma_chi_pbh', 'z_co']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2','mass_1_source','mass_2_source']
    
        # Initialize smeared Gaussians for chi_1, chi_2
        self.name='PBH_distribution_smearing'

    def update(self,**kwargs):
        
        self.sigma_chi_PBH = kwargs['sigma_chi_pbh']
        self.z_co = kwargs['z_co']
        
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):

        #choose module between numpy and cupy
        xp = get_module_array(chi_1)
        sx = get_module_array_scipy(chi_1)
 
        q = mass_2_source/mass_1_source
        mu_chi_1 = chi1_analytical_fit(mass_1_source, q, self.z_co)
        mu_chi_2 = chi2_analytical_fit(mass_1_source, q, self.z_co)

        # Second line is the angular part isotropic in spins
        out = log_tgaussian(chi_1,0.,1.,mu_chi_1,self.sigma_chi_PBH) + log_tgaussian(chi_2,0.,1.,mu_chi_2,self.sigma_chi_PBH) + \
        xp.log(0.5) + xp.log(0.5)
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))   
    

# IBH pop pdf: flat prior on chi and theta, evolving in mass
class spinprior_IBH(object):
    def __init__(self,mindelta=0.1,minchi=0.05):
        # Minimum value of delta and chi for stability
        self.mindelta, self.minchi = mindelta, minchi
        # Population-level parameters
        self.population_parameters = ['delta_0', 'delta_dot_0','chi_ibh_0','chi_dot_ibh_0']
        self.event_parameters = ['chi_1', 'chi_2', 'cos_t_1', 'cos_t_2','mass_1_source','mass_2_source']
        # Label for the distribution
        self.name = 'IBH_distribution'

    def update(self, **kwargs):
        
        self.delta_0 = kwargs['delta_0']       
        self.delta_dot_0 = kwargs['delta_dot_0']
        self.chi_IBH_0 = kwargs['chi_ibh_0']
        self.chi_dot_IBH_0 = kwargs['chi_dot_ibh_0']

    def log_pdf(self, chi_1, chi_2, cos_t_1, cos_t_2, mass_1_source, mass_2_source):
        # Select array module (NumPy or CuPy)
        xp = get_module_array(chi_1)
        sx = get_module_array_scipy(chi_1)

        delta_1 = self.delta_0+self.delta_dot_0*(mass_1_source/30.)
        delta_2 = self.delta_0+self.delta_dot_0*(mass_2_source/30.)
        chi_IBH_max_1 = self.chi_IBH_0+self.chi_dot_IBH_0*(mass_1_source/30.)
        chi_IBH_max_2 = self.chi_IBH_0+self.chi_dot_IBH_0*(mass_2_source/30.)

        # Clip to minimal and maximum values for numerical stability
        delta_1 = np.clip(delta_1,self.mindelta,4.)
        delta_2 = np.clip(delta_2,self.mindelta,4.)
        chi_IBH_max_1 = np.clip(chi_IBH_max_1,self.minchi,2.)
        chi_IBH_max_2 = np.clip(chi_IBH_max_2,self.minchi,2.)

        c1 = log_tgaussian(chi_1,0.,1.,0.,chi_IBH_max_1)
        c2 = log_tgaussian(chi_1,0.,1.,0.,chi_IBH_max_2)
        a1 = log_tgaussian(cos_t_1,-1.,1.,1.,1.-delta_1)
        a2 = log_tgaussian(cos_t_2,-1.,1.,1.,1.-delta_2)
        out = a1 + a2 + c1 + c2

        return out

    def pdf(self, chi_1, chi_2, cos_t_1, cos_t_2, mass_1_source, mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1, chi_2, cos_t_1, cos_t_2, mass_1_source, mass_2_source))
