import torch
import numpy as np

# new version of ftt 
def FDA_source_to_target(src_img, trg_img, L=0.1):
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

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = int(np.floor(min(h, w) * L))  # Calcola la dimensione della regione a bassa frequenza

    # Sostituisci le basse frequenze
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]  # Alto sinistra
    amp_src[:, :, 0:b, w-b:w] = amp_trg[:, :, 0:b, w-b:w]  # Alto destra
    amp_src[:, :, h-b:h, 0:b] = amp_trg[:, :, h-b:h, 0:b]  # Basso sinistra
    amp_src[:, :, h-b:h, w-b:w] = amp_trg[:, :, h-b:h, w-b:w]  # Basso destra

    return amp_src