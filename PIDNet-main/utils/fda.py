import torch
import numpy as np

# new version of ftt 
def FDA_source_to_target(src_img, trg_img, L=0.03):  #L is beta in the paper, controls the size of the low frequency window to be replaced.
    # Trasformata di Fourier bidimensionale per source e target
    fft_src = torch.fft.fft2(src_img.clone(), dim=(-2, -1))  # Trasformata 2D
    fft_trg = torch.fft.fft2(trg_img.clone(), dim=(-2, -1))

    # Estrai ampiezza e fase
    amp_src, pha_src = extract_ampl_phase(fft_src)
    amp_trg, _ = extract_ampl_phase(fft_trg)

    # Sostituisci le basse frequenze dell'ampiezza
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # Ricomponi la trasformata di Fourier
    fft_src_ = torch.polar(amp_src_, pha_src)  # Ricostruisci il tensore complesso

    # Trasformata inversa per ottenere l'immagine modificata
    src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1)).real  # Prendi solo la parte reale

    return src_in_trg

def extract_ampl_phase(fft_im):
    # Estrai ampiezza e fase da un tensore complesso
    fft_amp = torch.abs(fft_im)  # Ampiezza
    fft_pha = torch.angle(fft_im)  # Fase
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, L=0.03):
    _, _, h, w = amp_src.size()
    b = int(np.floor(min(h, w) * L))  # Calcola la dimensione della regione a bassa frequenza

    # Sostituisci le basse frequenze
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]  # Alto sinistra
    amp_src[:, :, 0:b, w-b:w] = amp_trg[:, :, 0:b, w-b:w]  # Alto destra
    amp_src[:, :, h-b:h, 0:b] = amp_trg[:, :, h-b:h, 0:b]  # Basso sinistra
    amp_src[:, :, h-b:h, w-b:w] = amp_trg[:, :, h-b:h, w-b:w]  # Basso destra

    return amp_src

#---------------
def low_freq_mutate_np( amp_src, amp_trg, L=0.05 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src


def FDA_source_to_target_np( src_img, trg_img, L=0.05 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg