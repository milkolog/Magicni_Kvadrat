from magicni_kvadrat import *
import pandas as pd
import matplotlib.pyplot as plt


def monte_carlo_simulacije(n=3, broj_pocetnih_stanja=100):
    sve_gramzive_ciljne_fje = list()
    sve_sa_ciljne_fje = list()
    sve_beam_ciljne_fje = list()
    sve_ga_ciljne_fje = list()
    sve_nasumicne_ciljne_fje = list()

    for i in range(broj_pocetnih_stanja):
        MK = MagicniKvadrat(n)
        pocetno_resenje = MK.generisanje_pocetnog_resenja()

        # generisanje svih objekata
        MK_nasumicno = MagicniKvadratNasumicno(n)
        MK_gamzi = MagicniKvadratGramzivaPretraga(n)
        MK_sa = MagicniKvadratSimuliranoKaljenje(n)
        MK_beam = MagicniKvadratSnop(n)
        MK_ga = MagicniKvadratGenetskiAlgoritam(n)

        # trazimo sva resenja
        resenje_nasumicno = MK_nasumicno.trazenje_najboljeg_resenja()
        resenje_gramzi, _ = MK_gamzi.trazenje_najboljeg_resenja(pocetno_resenje)
        resenje_sa, _, __ = MK_sa.trazenje_najboljeg_resenja(pocetno_resenje)
        resenje_beam = MK_beam.trazi_najbolje_resenje(pocetno_resenje)
        resenje_ga = MK_ga.trazi_najboljeg_naslednika()

        # trazimo sve ciljne fje
        ciljna_nasumicno = MK_nasumicno.izracunavanje_vrednosti_ciljne_funkcije(resenje_nasumicno)
        ciljna_gramzi = MK_gamzi.izracunavanje_vrednosti_ciljne_funkcije(resenje_gramzi)
        ciljna_sa = MK_sa.izracunavanje_vrednosti_ciljne_funkcije(resenje_sa)
        ciljna_beam = MK_beam.izracunavanje_vrednosti_ciljne_funkcije(resenje_beam)
        ciljna_ga = MK_ga.izracunavanje_vrednosti_ciljne_funkcije(resenje_ga)

        # dodajemo u niz da bismo mogli da ih uporedimo
        sve_nasumicne_ciljne_fje.append(ciljna_nasumicno)
        sve_gramzive_ciljne_fje.append(ciljna_gramzi)
        sve_sa_ciljne_fje.append(ciljna_sa)
        sve_beam_ciljne_fje.append(ciljna_beam)
        sve_ga_ciljne_fje.append(ciljna_ga)

    srvr_nasumicno = np.mean(np.array(sve_nasumicne_ciljne_fje))
    std_nasumicno = np.std(np.array(sve_nasumicne_ciljne_fje))

    srvr_gramzi = np.mean(np.array(sve_gramzive_ciljne_fje))
    std_gramzi = np.std(np.array(sve_gramzive_ciljne_fje))

    srvr_sa = np.mean(np.array(sve_sa_ciljne_fje))
    std_sa = np.std(np.array(sve_sa_ciljne_fje))

    srvr_beam = np.mean(np.array(sve_beam_ciljne_fje))
    std_beam = np.std(np.array(sve_beam_ciljne_fje))

    srvr_ga = np.mean(np.array(sve_ga_ciljne_fje))
    std_ga = np.std(np.array(sve_ga_ciljne_fje))

    srvr = np.array([srvr_nasumicno, srvr_gramzi, srvr_sa, srvr_beam, srvr_ga])
    std = np.array([std_nasumicno, std_gramzi, std_sa, std_beam, std_ga])
    nazivi = ['Nasumicna', 'Gramziva', 'Simulirano kaljenje', 'Po snopu', 'Genetski algoritam']

    df = pd.DataFrame({
        'Srednja vrednost': srvr,
        'Standardna devijacija': std
    }, index=nazivi)

    print(df)


def vizualizacija_gramzi():
    MK = MagicniKvadratGramzivaPretraga(3)
    pocetno_resenje = MK.generisanje_pocetnog_resenja()
    resenje, niz_ciljnih_fja = MK.trazenje_najboljeg_resenja(pocetno_resenje)
    mean = []
    std = []
    j = []
    broj_iteracija = MK.br

    for i in range(1, broj_iteracija):
        m = np.mean(niz_ciljnih_fja[:i])
        s = np.std(niz_ciljnih_fja[:i])
        mean.append(m)
        std.append(s)
        j.append(i)

    fig, axs = plt.subplots(2)
    axs[0].plot(j, mean)
    axs[0].set_title('Srednja vrednost')
    axs[1].plot(j, std)
    axs[1].set_title('Standardna devijacija')
    plt.show()


def vizualizacija_sa():
    MK = MagicniKvadratSimuliranoKaljenje(3)
    pocetno_resenje = MK.generisanje_pocetnog_resenja()
    resenje, niz_ciljnih_fja, vrv, t, brojac = MK.trazenje_najboljeg_resenja(pocetno_resenje)
    mean = []
    std = []
    j = []

    for i in range(1, brojac):
        m = np.mean(niz_ciljnih_fja[:i])
        s = np.std(niz_ciljnih_fja[:i])
        mean.append(m)
        std.append(s)
        j.append(i)

    fig, axs = plt.subplots(4)
    axs[0].plot(j, mean)
    axs[0].set_ylabel('Sr_vr')
    axs[1].plot(j, std)
    axs[1].set_ylabel('Std')
    axs[2].plot(range(brojac), vrv)
    axs[2].set_ylabel('p')
    axs[3].plot(range(brojac), t)
    axs[3].set_ylabel('T')
    plt.show()


def vizualizacija_prosecna_i_minimalna_vrednost_ciljne_fje_u_populaciji(prosecna, minimalna, broj_iteracija):
    brojac = []
    for i in range(broj_iteracija):
        brojac.append(i)
    fig, axs = plt.subplots(2)
    axs[0].plot(brojac, prosecna)
    axs[0].set_title('Prosecna vrednost populacije')
    axs[1].plot(brojac, minimalna)
    axs[1].set_title('Minimalna vrednost populacije')
    plt.show()


if __name__ == "__main__":
    # monte_carlo_simulacije()
    # vizualizacija_gramzi()
    # vizualizacija_sa()

    MK_beam = MagicniKvadratSnop(3)
    pocetno_resenje = MK_beam.generisanje_pocetnog_resenja()
    _, prosecna_beam, minimalna_beam, broj_iteracija_beam = MK_beam.trazi_najbolje_resenje(pocetno_resenje)

    # vizualizacija_prosecna_i_minimalna_vrednost_ciljne_fje_u_populaciji(prosecna_beam,
                                                                        #minimalna_beam, broj_iteracija_beam)

    MK_ga = MagicniKvadratGenetskiAlgoritam(3)
    _, prosecna_ga, min_ga, br_iter_ga = MK_ga.trazi_najboljeg_naslednika()

    # vizualizacija_prosecna_i_minimalna_vrednost_ciljne_fje_u_populaciji(prosecna_ga, min_ga, br_iter_ga)
