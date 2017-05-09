import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import scipy.io.wavfile
#import numpy.fft as fft_module
import scipy.fftpack as fft_module
from scipy.signal import chirp

"""
Created on July 2016

@author: ravikiran, ENS Paris
"""

def _ispow2(N):
    """ Checks if N is a power of 2"""
    return 0 == (N & (N - 1))


def get_mother_frequency(nfo):
    """Function returns the dimensionless mother frequency

    Parameters
    ----------
    nfo : Number of Filters per Octave

    Returns
    -------
    mother_frequency : dimensionaless mother center frequency

    Notes
    -----

    The dimensionless mother center frequency xi (corresponding to a log period
    \gamma=0) is computed as the midpoint between the center frequency of the 
    second center frequency xi*2^(-1/nfo) (corresponding to \gamma
    equals 1) and the negative mother center frequency (1-xi). Hence the eqn.
    2 xi = xi*2^(-1/nfo) + (1-xi), from which we derive : 
    xi = 1 / (3 - 2^(1/nfo)). This formula is valid only when the 
    wavelet is a symmetric bump in the Fourier domain.

    References
    ----------
    .. [1] https://github.com/lostanlen/WaveletScattering.jl
           Implementation of scattering transform in Julia by V. Lostanlen
    """

    mother_frequency = 1.0 / (3.0 - 2.0**(-1.0 / nfo))
    return mother_frequency


def get_wavelet_filter_specs(nfo, quality_factor, nOctaves):
    """Create wavelet filter specs : centerfrequecy, bandwidth 
    Wavelet filter specs are independent of the signal length and resolution.

    Parameters
    ----------
    nfo : (scalar or list of size M)
        number of wavelets per octave in the fourier domain
    quality_factor : (scalar or list of size M)
        This is the ratio of center freq. to bandwidth
    nOctaves (scalar or list of size scattering order)
        number of octaves covering the fourier domain
    M : scattering order
        the order of scattering transform (max tested = 2)

    Returns
    -------

    psi_specs[gamma] : gamma indexed dictionary that contains the tuple 
                       (centerfrequency, bandwidth) 
                       #gammas = nfo * nOctaves

    Notes
    -----
    
    
    To be sure that we calculate valid wavelet filters in higher orders we need 
    to check the following condition :

    xi_1 2^(-\gamma_1/Q1) / Q1   >   xi_2 2^(-\gamma_2/Q2)

    Bandwidth of the wavelet filter (@some gamma) for order 
    M  > Center frequency for wavelet filter(@gamma) for order M+1.

    at Q1=Q2=nfo=1 (dyadic) we have xi_1 = xi_2
    we have the resulting condition to be :

    \gamma_1 < \gamma_2
    
    Though this does not hold for other values of Q1, Q2
    
    References
    ----------
    .. [1] https://github.com/lostanlen/WaveletScattering.jl
           Implementation of scattering transform in Julia by V. Lostanlen
           
    """
    #check there are only 2 values of nfo possible
    assert(len(nfo)<=2)
    psi_spec_order = {}
    for order in range(len(nfo)):
        mother_frequency = get_mother_frequency(nfo[order])
        psi_specs = {}
        fc_vec = []
        bw_vec = []
        
        for j in range(nOctaves):
            for q in range(0, nfo[order]):
                gamma = j * nfo[order] + q           
                resolution = np.power(2, -gamma / nfo[order])
                centerfrequency = mother_frequency * resolution
                bandwidth = centerfrequency / quality_factor
                psi_specs[(j,q)] = (centerfrequency, bandwidth)
                fc_vec.append(centerfrequency)
                bw_vec.append(bandwidth)
        
        psi_spec_order[order] = (psi_specs, fc_vec, bw_vec)
   
        if(display_flag):
            plt.figure()
            plt.plot(fc_vec)
            plt.plot(bw_vec)
            plt.title('Normalized bw and Fc for order =' +repr(order))
            plt.xlabel('Filter index')
            plt.ylabel('Normalized Center Frequency f in [0,1]')
            plt.legend(('fc','bw'))
        
    return psi_spec_order


def filterbank_morlet_1d(N, psi_specs, nOctaves):
    """Function returns complex morlet 1d wavelet filter bank 

    Parameters
    ----------
    N : integer > 0
        length of input signal that is a power of 2 (after chunking)
    nOctaves : integer > 0
        number of octaves/scales covering the frequency domain, 
        nOctaves <= log2(N) 
    nfo : integer > 0 
        number of wavelet filters per octave

    Returns
    -------
    psi : dictionary 
        dictionary of morlet wavelet filters indexed with different gamma 
    phi : array_like
        low pass filter
    lp : array_like 
        Littlewood payley function : Measure of quality of the filter bank for 
        signal representation, it should be as close as possible to 1.

    References
    ----------

    .. [1] Anden, J., Mallat S. 'Deep Scattering Spectrum'.  
          IEEE Transactions on Signal Processing 2014 

    Notes
    -----
    calculate bandpass filters \psi for size N and J different center 
    frequencies and low pass filter \phi and resolutions res < log2(N)
    
    The max. and min. of the littlewood payley function needs to be close to 1
    This preserves norm and ensures contractive operator

    TODO : correct the bandwidth of low pass filter (add this at the end)
    TODO : Morlet with corrections: _corrected_morlet_1d

    """

    psi = {} #wavelet
    lp = np.zeros(shape=(N)) #little-wood payley   
    lp_afternorm = np.zeros(shape=(N)) #little-wood payley
    FWHM_factor = 10 * np.log(2) #fullwidth at half max factor

    for index in psi_specs:
        fc, bandwidth_psi = psi_specs[index]
        den = bandwidth_psi**2 / FWHM_factor
        psi[index] = _morlet_1d(N, fc, den)
        lp = lp + np.square(np.abs(psi[index]))
    
    phi = _gaussian(N, nOctaves)
    
    lp = lp + np.square(np.abs(phi[1]))
    lp[1::] = (lp[1::] + lp[-1:0:-1]) * 0.5    
    
    normalizer_lp = np.max(np.sqrt(lp))
    
    for index in psi:
        psi[index] = psi[index] / normalizer_lp

    phi = phi / normalizer_lp

    for index in psi_specs:
        lp_afternorm = lp_afternorm + np.square(np.abs(psi[index]))

    lp_afternorm = lp_afternorm + np.square(np.abs(phi[1]))
    lp_afternorm[1::] = (lp_afternorm[1::] + lp_afternorm[-1:0:-1]) * 0.5
    
    if(display_flag):        
        plt.figure()
        plt.title('Littlewood Payley function')
        plt.plot(lp)
        plt.plot(lp_afternorm)
        plt.legend(('Before Norm', 'After Norm'))
        
    filters = dict(phi=phi, psi=psi)

    return (filters, lp)

def filterbank_to_multiresolutionfilterbank(filters, max_resolution):
    """Converts a filter bank into a multiresolution filterbank
    For every filter in the filter bank, compute different resolutions 
    (differnt support). The input filters are assumed to be in the 
    This precalculated Multiresolution filter bank will speed up
    calculation of convolution of signal at the output of different
    wavelet filters and different resolutions.


    Parameters
    ----------
    filters : dictionary
        Set of filters stored in a dictionary in the following way:
         - filters['phi'] : Low pass filter at resolutions nOctaves to J
         - filters['psi'] : Band pass filter (Morlet) 
              where 'j' indexes the scale and 'q' indexes the nfo 
              of a single filter.

    max_resolution : int
        number of resolutions to compute for every filter 

    filterbank_to_multiresolutionfilterbank iterates first through 
    j in range(J) and then through the resolution based on current J. 
    This changes the structure of wavelet_filters
    
        for j in range(nOctaves):
            for q in range(nfo):
                for res in range(0,max(j-1,0)+1):
                    get_filter_at_resoluion(wavelet_filters['psi][(j,q)],res)                

    Returns
    -------
    filters_multires : dictionary
        Set of filters in the Fourier domain, at different scales & resolutions
        See multiresolution_filter_bank_morlet1d for more details 

    """

    keys_jq = max(list(filters['psi'].keys()))
    nOctaves = keys_jq[0] + 1 
    nfo = keys_jq[1] + 1
    

    Phi_multires = []
    Psi_multires = {}

    for res in range(0,max_resolution):
        Phi_multires.append(_get_filter_at_resolution(filters['phi'],res))
    
    for j in range(nOctaves):
        for q in range(nfo):
            Psi_multires[(j,q)] = {}
            filt = filters['psi'][(j,q)]
            for res in range(0,max(j-1,0)+1):  
                Psi_multires[(j,q)][res] = _get_filter_at_resolution(filt,res)


    filters_multires = dict(phi=Phi_multires, psi=Psi_multires)
    
    return filters_multires

def _gaussian(N, nOctaves):
    """Function calculates the gaussian function of length N with bandwidth
    0.4 * 2**(-nOctaves+1). This creates the low pass filter that is used
    to average the wavelet filtered signal at different scales.
    
    Parameters
    ----------
    
    N : length of signal
    nOctaves : Number of octaves
    
    Returns
    -------
    phi : array like
        low pass filter in fourier domain
    """
    f = np.arange(0, N, dtype=float) / N # normalized frequency domain        
    bandwidth_phi = 0.4 * 2**(-nOctaves+1)#is this the right bandwidth?    
    phi = np.exp(-np.square(f) * 10 * np.log(2) / bandwidth_phi**2)
    return phi
    
def _morlet_1d(N, fc, den):
    """Morlet wavelet at center frequency and bandwidth 
    
    Parameters
    ----------
    N : integer
        length of filter
    fc : float
        center frequency
    den : denominator of gaussian with sigma**2
    
    Returns
    -------
    Morlet filter with these parameters of length N
    
    """
    # normalized frequency axis
    f = np.arange(0, N, dtype=float) / N  
    return 2 * np.exp(- np.square(f - fc) / den).transpose()
    
def _get_filter_at_resolution(filt,j):
    """Computes filter 'filt' at resolution 'j'
    
    Parameters
    ----------
    filt : array
        Filter in the Fourier domain.
    j : int
        Resolution to be computed
    
    Returns
    -------
    filt_multires : array
    Filter 'filt' at the resolution j, in the Fourier domain
    
    """
    

    N = filt.shape[0]  
    
    assert _ispow2(N), 'Filter size must be an integer power of 2.'
    
    # Truncation in fourier domain and suming over responses from other bands
    # back into the truncated fourier domain (make sure there are no or 
    # neglible responses in these frequencies, otherwise leads to aliasing)
    mask = np.hstack((np.ones(int(N / 2 ** (1 + j))), 0.5, \
            np.zeros(int(N - N / 2 ** (j + 1) - 1)))) \
            + np.hstack((np.zeros(int(N - N / 2 ** (j + 1))), \
            0.5, np.ones(int(N / 2 ** (1 + j) - 1))))
    
    #truncation by using mask and reshape
    filt_lp = np.complex64(filt*mask)
    
    fold_size = (int(2 ** j), int(N / 2 ** j))
    filt_multires = filt_lp.reshape(fold_size).sum(axis=0)
    
    return filt_multires
    
def scattering(x,wavelet_filters=None,wavelet_filters_order2=None,M=2):
    """Compute the scattering transform of a signal using the filter bank.

    
    Parameters
    ----------
    x : array_like
        input signal 
        Length of x needs to be a power of 2. 

    wavelet_filters : Dictionary 
        Multiresolution wavelet filter bank 
    M : int
        Order of the scattering transform, which can be 0, 1 or 2.


    Returns
    -------
    S : 2D array_like
        Scattering transform of the x signals, of size (num_coeffs, time). 

    U : array_like
        Result before applying the lowpass filter and subsampling.

    S_tree : dictionary
        Dictionary that allows to access the scattering coefficients (S) 
        according to the layer and indices. More specifically:
    
    Zero-order layer: The only available key is 0:
    S_tree[0] : 0th-order scattering transform
    S_tree[1] : 1st-order coefficients nOctaves*nfo \times window_size matrix
    S_tree[2] : 2nd-order coefficients with nOctaves, nfo, nfo2 as params:
    nOctaves*nfo + nOctaves * (nOctaves - 1) *nfo*nfo2 // 2 \times window_size
    matrix

    References
    ----------
    .. [1] Anden, J., Mallat S. 'Deep Scattering Spectrum'.  
            IEEE Transactions on Signal Processing 2014 
    .. [2] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'. 
            IEEE Transactions on PAMI, 2012.
    Examples
    --------
    
    """
    
    if(not _ispow2(len(x))):
        max_J = int(np.ceil(np.log2(len(x))))
        x = np.append(x, np.zeros(2**max_J-len(x)))
        
    if(wavelet_filters==None):#build filters 
        N = len(x)        
        nfo = [12, 1]
        nOctaves = 10
        quality_factor = 4 #defaults
        
        psi_specs_order = get_wavelet_filter_specs(nfo, \
                            quality_factor, nOctaves)
        psi_specs, _, _ = psi_specs_order[0] 
        filters, _ = filterbank_morlet_1d(N, psi_specs, nOctaves)
        wavelet_filters = \
        filterbank_to_multiresolutionfilterbank(filters, nOctaves)
        if(M==2 and wavelet_filters_order2==None):
            psi_specs_2, _, _ = psi_specs_order[1]
            filters_order2, _ = filterbank_morlet_1d(N, psi_specs_2, nOctaves)
            wavelet_filters_order2 = \
            filterbank_to_multiresolutionfilterbank(filters_order2, nOctaves)
            
    keys_jq = max(list(wavelet_filters['psi'].keys()))
    nOctaves = keys_jq[0] + 1 
    nfo = keys_jq[1] + 1
    
    #for second order    
    if(M==2):
        keys_jq_order2 = max(list(wavelet_filters_order2['psi'].keys()))
        nfo2 = keys_jq_order2[1] + 1
    else:
        nfo2 = nfo
    
    num_coefs = {
        0: int(1),
        1: int(1 + nOctaves*nfo),
        2: int(1 + nOctaves*nfo + nOctaves * (nOctaves - 1) * nfo * nfo2 // 2)
    }.get(M, -1)

    window_size = int(x.shape[0]/2**(nOctaves-1)) # #coeffecients in time
    
    oversample = 1  # subsample at rate a bit lower than the critic frequency

    U = []
    v_resolution = []
    current_resolution = 0
    #output coefficients matrix
    S = np.zeros((num_coefs,window_size)) 
    S_tree = {} 
    
    Xf = fft_module.fft(x) # precompute the fourier transform of the signal
    
    ds2 = len(Xf)//window_size
    lp_filter = wavelet_filters['phi'][current_resolution]
    S[0, :] = ds2* np.abs(fft_module.ifft(Xf*lp_filter))[::ds2]
    
    S_tree[0] = S[0, :].view()

    
    if M > 0: #First order scattering coeffs
        num_order1_coefs = nOctaves*nfo
        S1 = S[1:num_order1_coefs+1,:].view()
        S1.shape=(num_order1_coefs,window_size)
        indx = 0
        
        if(nfo==1 and display_flag): 
            #display only when Q =1 otherwise too many signals to plot    
            fig, axarr = plt.subplots(nOctaves, sharex=True)
            fig2, axarr2 = plt.subplots(nOctaves, sharex=True)
            fig3, axarr3 = plt.subplots(nOctaves, sharex=True)
            fig_title = "Filtered O/p & Mask for fourier Truncation (Fourier)"
            fig2_title = "Lowpass filters with non-zero support (Fourier)"
            fig3_title = "Filtered signal and its absolute value (Time)"
            fig.suptitle(fig_title, fontsize=14)
            fig2.suptitle(fig2_title, fontsize=14)
            fig3.suptitle(fig3_title, fontsize=14)
        
        for j in range(nOctaves):
            resolution = max(j-oversample, 0)
            v_resolution.append(resolution) # resolution for the next order
            ds =  2**resolution
            lp_filter = wavelet_filters['phi'][resolution]
            for q in range(nfo):
                filtersjq = wavelet_filters['psi'][(j,q)][current_resolution]
                #Fourier truncate eqs subsample in time
                x_conv_f = Xf*filtersjq
                len_x_conv_f = len(x_conv_f)
                x_conv_f_truncate = x_conv_f[:len_x_conv_f//ds]
                x_conv = fft_module.ifft(x_conv_f_truncate)
                x_conv_mod = np.abs(x_conv)
                x_conv_mod_f = fft_module.fft(x_conv_mod)
                ds2 = len(x_conv_mod_f)//window_size
                S1[indx, :] = ds2* np.abs(fft_module.ifft(x_conv_mod_f*lp_filter))[::ds2]
                U.append(x_conv_mod)
                
                if(print_flag):
                    x_conv_time_sub = ds*fft_module.ifft(x_conv_f)[::ds]
                    disp_str = '-->j, q, res = ' + repr((j,q,resolution)) + \
                                '-Fourier_trunc==subsample_time =' + \
                                repr(np.allclose(x_conv_time_sub,x_conv))
                    disp_str2 = '--phi_len = ' + repr(len(lp_filter)) + \
                                '--Max='+repr(max(S[indx,:]))
                    print(disp_str + disp_str2)
                    
                if(nfo==1 and display_flag):
                    mask = np.zeros(x_conv_f.shape)
                    mask[:len_x_conv_f//ds] = max(np.abs(x_conv_f))
                    axarr[indx].plot(np.abs(x_conv_f))
                    axarr[indx].plot(mask,'k--')
                    axarr[indx].set_title('Mask Length =' + repr(sum(mask>0)))
                    
                    phi_support_len = np.sum(np.abs(lp_filter)!=0)
                    plot_xlen = max(phi_support_len, window_size)
                    mask_phi = np.zeros((plot_xlen))
                    mask_phi[:phi_support_len] = max(np.abs(lp_filter))
                    mask_window_size = np.zeros((plot_xlen))
                    mask_window_size[:plot_xlen] = max(mask_phi) + 1
                    
                    axarr2[indx].plot(lp_filter[:plot_xlen])
                    axarr2[indx].plot(mask_phi,'k--')
                    axarr2[indx].plot(mask_window_size,'r--')
                    title_2 = 'Non-zero support of lowpass = ' 
                    axarr2[indx].set_title(title_2 + repr(phi_support_len))
                    axarr3[indx].plot(U[indx]) 
                    axarr3[indx].plot(x_conv)
#                    axarr3[indx].set_title('Max_val ='+repr(max(S1[indx, :])))
                
                indx = indx + 1
        
        S_tree[1] = S1.view()
        
        if(display_flag):
            plt.figure()
            plt.imshow(S1, aspect='auto', cmap='jet')
            plt.title('First Order Coeffs : ' + repr(S1.shape) )
            plt.xlabel('Time window')
            plt.ylabel('octave J')
            plt.colorbar()

    if M > 1: 
        #Smaller nfo2 largely reduces the number of coefficients.
        num_order2_coefs = nOctaves*(nOctaves-1)*nfo*nfo2//2
        S2 = S[num_order1_coefs+1:num_coefs, :].view()  # view of the data
        S2.shape = (num_order2_coefs, window_size)
        indx = 0
        for j1 in range(nOctaves):
            #pick resolution of filtered signal stored during U1 calculation.
            current_resolution = v_resolution[j1]
            lp_filter = wavelet_filters_order2['phi'][current_resolution]
            for q1 in range(nfo):
                Ujq = fft_module.fft(U[j1*nfo+q1])  
                for j2 in range(j1+1,nOctaves):
                    for q2 in range(nfo2):
                        # | U_lambda1 * Psi_j2 | * phi
                        filtersj2q2 = wavelet_filters_order2['psi'][(j2,q2)][current_resolution].view()
                        #Subsampling is only required in order 1 to set the resolution of the signal decided by the wavelet bandpass filters 
                        x_conv = np.abs(fft_module.ifft(Ujq*filtersj2q2))
                        x_conv_f = fft_module.fft(x_conv) 
                        ds2 = len(x_conv_f)//window_size 
                        Uj2 = ds2* np.abs(fft_module.ifft(x_conv_f*lp_filter))[::ds2] 
                        S2[indx, :] = Uj2
                        indx = indx+1
        
        S_tree[2] = S2.view()
        
        if(display_flag):
            plt.figure()
            plt.imshow(S2, aspect='auto', cmap='jet')
            plt.title('Second Order Coeffs')
            plt.colorbar()

    return S, U, S_tree
   
def test_scattering(nfo, quality_factor, nOctaves, N, M):
    """Test scattering transform 
    Parameters
    ----------
    nfo : integer
        Number of wavelet filters per octave in the fourier domain.
    quality_factor : integer
        The quality factor for all wavelet filters (by default 1)
    nOctaves : integer
        Number of Octaves or scales 
    N : integer
        length of input signal ( this is just for testing)
    M : integer
        scattering order (can be 1 or 2)
    
    """
    
    def get_audio_test():
        """ load test file from librosa/tests/data/test1_22050.wav """
        (sr, y) = scipy.io.wavfile.read('./data/0506001a2.mp3.wav')
        y = y.mean(axis=1)
        return (y, sr)
        
    def get_chirp(N):
        """Chirp signal of length N with 0 start freq and f1 at time t1 """
        t = np.linspace(0, 20, N)
        y = chirp(t, f0=0, f1=10, t1=10, method='linear')
        return y
        
    def get_dirac(N, loc):
        """ Create dirac function at location loc of length N """
        y = np.zeros(N)
        y[loc] = 1
        return y    
    #Read / Create signals 
    
    # uncomment to test chirp and dirac signals
    (y, fs) = get_audio_test()
#    y = get_dirac(N, int(N/8))
#    y = get_chirp(N)

    
    if(not _ispow2(len(y))):
        max_J = int(np.ceil(np.log2(len(y))))
        y = np.append(y, np.zeros(2**max_J-len(y)))
    
    N = len(y)
    
    assert(nOctaves < np.log2(N))
    
    #create wavelet filters
    psi_specs_order = get_wavelet_filter_specs(nfo, \
                            quality_factor, nOctaves)
    psi_specs, _, _ = psi_specs_order[0] 
    filters, _ = filterbank_morlet_1d(N, psi_specs, nOctaves)
    wavelet_filters = filterbank_to_multiresolutionfilterbank(filters, nOctaves)
    if(M==2):
        psi_specs_2, _, _ = psi_specs_order[1]
        filters_order2, _ = filterbank_morlet_1d(N, psi_specs_2, nOctaves)
        wavelet_filters_order2 = \
        filterbank_to_multiresolutionfilterbank(filters_order2, nOctaves)        
    else:
        wavelet_filters_order2 = None
        
    scat,u,scat_tree = scattering(y, wavelet_filters=wavelet_filters, \
                        wavelet_filters_order2=wavelet_filters_order2, M=M)
    coef_index, spatial = scat.shape    
    
       
    return scat

global display_flag, print_flag
display_flag, print_flag = 0, 0

#psi_specs = get_wavelet_filter_specs(nfo, 1, nOctaves)
#filters, lp = filterbank_morlet_1d(N, psi_specs, nOctaves)
#wavelet_filters = filterbank_to_multiresolutionfilterbank(filters, nOctaves)
#scat = test_scattering(**test_args)

            
