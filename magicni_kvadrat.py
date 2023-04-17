import numpy as np


class MagicniKvadrat:
    """
    Magicni kvadrat reda nxn je matrica sa sledecim svojstvima:
    -elementi su brojevi od 1 do n^2
    -svi elementsi su razliciti (svaki broj se pojavljuje tacno jednom)
    -suma elemenata svake vreste, kolone, i duz obe dijagonale je ista i iznosi n*(n^2+1)/2

    Ideja je kroz algoritme doci do najpribliznijeg resenja magicnom kvadratu, zbog cega algoritme pravimo kao
    naslednike ove klase
    """

    def __init__(self, n):
        self.n = n
        self.idealna_suma = n * (n * n + 1) / 2

    def generisanje_pocetnog_resenja(self):
        """
        Generisanje random nxn matrice od koje cemo krenuti da trazimo najbolje resenje.
        Matrica se pravi prvobitno kao niz brojeva do n^2, a zatim reshape-uje u nxn oblik.
        :return: nxn matrica sa n^2 razlicitih brojeva
        """
        pocetni_niz = np.arange(1, self.n * self.n + 1)
        np.random.shuffle(pocetni_niz)
        pocetno_resenje = pocetni_niz.reshape((self.n, self.n))
        return pocetno_resenje

    def izracunavanje_vrednosti_ciljne_funkcije(self, nn_kvadrat):
        """
        Na osnovu unete nxn matrice sa n^2 razlicitih brojeva, trazimo ciljnu funkciju kao ukupnu apsolutnu gresku,
        odnosno sumu apsolutnih r5azlika stvarnih i idealih suma.
        :param nn_kvadrat: nxn matrica sa n^2 razlicitih brojeva
        :return: ciljnu funkciju
        """
        zbir_po_kolonama = np.sum(nn_kvadrat, axis=0)
        zbir_po_redovima = np.sum(nn_kvadrat, axis=1)
        zbir_glavna_dijagonala = sum(nn_kvadrat[i][i] for i in range(self.n))
        zbir_sporedna_dijagonala = sum(nn_kvadrat[self.n - i - 1][self.n - i - 1] for i in range(self.n))

        ciljne_kolone = np.sum(np.abs(zbir_po_kolonama - self.idealna_suma))
        ciljni_redovi = np.sum(np.abs(zbir_po_redovima - self.idealna_suma))
        ciljna_glavna_dijagonala = np.sum(np.abs(zbir_glavna_dijagonala - self.idealna_suma))
        ciljna_sporedna_dijagonala = np.sum(np.abs(zbir_sporedna_dijagonala - self.idealna_suma))

        ciljna_funkcija = ciljne_kolone + ciljni_redovi + ciljna_glavna_dijagonala + ciljna_sporedna_dijagonala

        return ciljna_funkcija

    def generisanje_slucajnog_naslednika_resenja(self, kvadrat):
        """
        Slucajni naslednik jedne matrice se generise tako sto se na slucajan nacin izaberu dva polja iz matrice i zamene
        svoje vrednosti
        :param kvadrat: nxn matrica na n^2 razlicitih brojeva
        :return: kvadrat sa zamenjenim random clanovima
        """
        nn_kvadrat = kvadrat.copy()
        i = np.random.randint(0, self.n)
        j = np.random.randint(0, self.n)
        m = np.random.randint(0, self.n)
        n = np.random.randint(0, self.n)
        if [i, j] != [m, n]:
            pom = nn_kvadrat[i][j]
            nn_kvadrat[i][j] = nn_kvadrat[m][n]
            nn_kvadrat[m][n] = pom

        return nn_kvadrat

    def traznje_srednje_vrednosti_i_standardne_devijacije(self, niz_ciljnih_fja):
        """
        Na osnovu ciljnih funkcija dobijene u toku iteracija algoritama, trazenje srednje vrednosti i standardne
        devijacije
        :param niz_ciljnih_fja:
        :return: srednja vrednost, standarda devijacija
        """
        sr_vr = np.mean(niz_ciljnih_fja)
        std = np.std(niz_ciljnih_fja)

        return sr_vr, std


class MagicniKvadratNasumicno(MagicniKvadrat):
    """
    Algoritam za nasumicno postavljanje magicnog kvadrata u k iteracija.
    """

    def __init__(self, n, broj_iteracija=1024):
        self.br = broj_iteracija
        super().__init__(n)

    def trazenje_najboljeg_resenja(self):
        resenje = self.generisanje_pocetnog_resenja()
        ciljna_fja = self.izracunavanje_vrednosti_ciljne_funkcije(resenje)

        if ciljna_fja == 0:
            return resenje

        for i in range(self.br):
            novo_resenje = self.generisanje_pocetnog_resenja()
            nova_ciljna_fja = self.izracunavanje_vrednosti_ciljne_funkcije(novo_resenje)
            if nova_ciljna_fja == 0:
                return novo_resenje
            elif nova_ciljna_fja < ciljna_fja:
                resenje = novo_resenje
                ciljna_fja = nova_ciljna_fja

        return resenje


class MagicniKvadratGramzivaPretraga(MagicniKvadrat):
    """
    Gramziva pretraga u svakoj iteraciji generise sve naslednike trenutnog resenja, i bira ono cija je ciljna funkcija
    minimanla.
    AKA penjanje uzbrdo
    """

    def __init__(self, n):
        super(MagicniKvadratGramzivaPretraga, self).__init__(n)

        self.br_naslednika = int(self.n * self.n * (self.n * self.n - 1) / 2)
        # broj iteracija nam zavisi od broja naslednika ako zelimo da kasnije poredjenja budu fer
        self.br = 1024 // self.br_naslednika

    def generisanje_svih_naslednika_resenja(self, nn_kvadrat):
        niz_naslednika = np.zeros((self.br_naslednika, self.n, self.n))
        brojac = 0

        for i in range(self.n * self.n - 1):
            for j in range(i + 1, self.n * self.n):
                naslednik = nn_kvadrat.copy().flatten()
                pom = naslednik[i]
                naslednik[i] = naslednik[j]
                naslednik[j] = pom
                niz_naslednika[brojac, :, :] = naslednik.reshape((self.n, self.n))
                brojac += 1

        return niz_naslednika

    def generisanje_ciljnih_funkcija_svih_resenja(self, niz_naslednika):
        niz_ciljnih_fja = np.zeros(self.br_naslednika)
        for i in range(self.br_naslednika):
            niz_ciljnih_fja[i] = self.izracunavanje_vrednosti_ciljne_funkcije(niz_naslednika[i, :, :])

        return niz_ciljnih_fja

    def trazenje_najboljeg_resenja(self, pocetno_resenje):
        resenje = pocetno_resenje
        ciljna_fja = self.izracunavanje_vrednosti_ciljne_funkcije(resenje)
        niz_ciljinih_fja = [ciljna_fja]

        if ciljna_fja == 0:
            return resenje

        for i in range(self.br):
            sva_resenja = self.generisanje_svih_naslednika_resenja(resenje)
            sve_ciljne_fje = self.generisanje_ciljnih_funkcija_svih_resenja(sva_resenja)
            min_ciljna_fja = min(sve_ciljne_fje)
            niz_ciljinih_fja.append(min_ciljna_fja)
            indeks_minimuma = np.argmin(sve_ciljne_fje)
            optimalno_resenje = sva_resenja[indeks_minimuma, :, :]

            resenje = optimalno_resenje
            if min_ciljna_fja == 0:
                break

        return resenje, niz_ciljinih_fja


class MagicniKvadratSimuliranoKaljenje(MagicniKvadrat):

    def __init__(self, n, temperatura=1000, alfa=0.75, broj_iteracija=1024):
        super(MagicniKvadratSimuliranoKaljenje, self).__init__(n)
        self.T = temperatura
        self.br = broj_iteracija
        self.a = alfa

    def trazenje_najboljeg_resenja(self, pocetno_resenje):
        resenje = pocetno_resenje
        ciljna_fja = self.izracunavanje_vrednosti_ciljne_funkcije(resenje)
        niz_ciljnih_fja = [ciljna_fja]

        i = 0
        t = []
        verovatnoca = 0
        vrv_niz = []
        if ciljna_fja == 0:
            return resenje, niz_ciljnih_fja, vrv_niz, t, i

        while self.T > 0:

            if i == self.br:
                break
            # ovaj deo je isti kao nasumicna, da li moze da se optimizuje?
            novo_resenje = self.generisanje_slucajnog_naslednika_resenja(resenje)
            nova_ciljna_fja = self.izracunavanje_vrednosti_ciljne_funkcije(novo_resenje)
            if nova_ciljna_fja == 0:
                niz_ciljnih_fja.append(nova_ciljna_fja)
                return novo_resenje, niz_ciljnih_fja, vrv_niz, t, i
            elif nova_ciljna_fja < ciljna_fja:
                resenje = novo_resenje
                ciljna_fja = nova_ciljna_fja
                niz_ciljnih_fja.append(nova_ciljna_fja)
                verovatnoca += 1
            elif nova_ciljna_fja > ciljna_fja:
                dh = nova_ciljna_fja - ciljna_fja
                p = np.exp(-dh / self.T)
                q = np.random.uniform(0, 1)
                if p > q:
                    resenje = novo_resenje
                    ciljna_fja = nova_ciljna_fja
                    niz_ciljnih_fja.append(nova_ciljna_fja)
                    verovatnoca += 1

            i += 1
            self.T *= self.a
            t.append(self.T)
            verovatnoca = verovatnoca / i
            vrv_niz.append(verovatnoca)

        return resenje, niz_ciljnih_fja, vrv_niz, t, i


class MagicniKvadratSnop(MagicniKvadrat):
    """
    Pretraga po snopu resenje trazi tako sto iz dobijene populacije bira n najboljih, i za svakog od n najboljih
    generise m naslednih stanja, cime dobijamo novu populaciju.
    """

    def __init__(self, n, broj_iteracija=300, sirina=4, cvorovi=8):
        super(MagicniKvadratSnop, self).__init__(n)
        self.br = broj_iteracija
        self.W = sirina
        self.N = cvorovi

    def nadji_n_naslednika(self, nn_kvadrat):
        # treba mi n random naslednika inace ce da vrti iste nizove
        niz_N_naslednika = list()

        for i in range(self.N):
            naslednik = self.generisanje_slucajnog_naslednika_resenja(nn_kvadrat)
            niz_N_naslednika.append(naslednik)

        return niz_N_naslednika

    def biraj_w_najboljih(self, niz_n_naslednika):
        # todo: pravljenje dictinari-ja nam smanjujuje niz_n_naslednika na onoliko koliko imamo razlicitih ciljnih fja
        # treba sortirati preko argsort i listi, ali nam ovde ne pravi problem
        niz_ciljnih_fja = np.zeros(len(niz_n_naslednika))
        for i in range(len(niz_n_naslednika)):
            niz_ciljnih_fja[i] = self.izracunavanje_vrednosti_ciljne_funkcije(niz_n_naslednika[i])

        d = {}
        for A, B in zip(niz_n_naslednika, niz_ciljnih_fja):
            d[B] = A
        sortirani_dict = sorted(d.items())
        d = dict(sortirani_dict)

        najboljihW = dict(list(d.items())[:self.W])

        ciljne_fje_W = np.array(list(najboljihW.keys()))
        naslednici_W = np.array(list(najboljihW.values()))

        return ciljne_fje_W, naslednici_W

    def trazi_najbolje_resenje(self, pocetno_resenje):
        resenje = list([pocetno_resenje])
        ciljna_fja = list([self.izracunavanje_vrednosti_ciljne_funkcije(pocetno_resenje)])
        prosecna_vrednost_populacije = ciljna_fja.copy()
        minimalna_vredmost_populacije = ciljna_fja.copy()

        if ciljna_fja == 0:
            return resenje[0], prosecna_vrednost_populacije, minimalna_vredmost_populacije, 1

        for i in range(2, self.br):
            novo_resenje = list()
            for j in range(len(resenje)):
                novih_N_naslednika = self.nadji_n_naslednika(resenje[j])
                novo_resenje.extend(novih_N_naslednika)

            niz_ciljnih_fja_populacije = []
            for j in range(len(novo_resenje)):
                ciljna_fja_clana_populacije = self.izracunavanje_vrednosti_ciljne_funkcije(novo_resenje[j])
                niz_ciljnih_fja_populacije.append(ciljna_fja_clana_populacije)
            prosecna_vrednost_populacije.append(np.mean(niz_ciljnih_fja_populacije))

            W_ciljnih_fja, W_najboljih = self.biraj_w_najboljih(novo_resenje)
            minimalna_vredmost_populacije.append(W_ciljnih_fja[0])

            if W_ciljnih_fja[0] == 0:
                return W_najboljih[0], prosecna_vrednost_populacije, minimalna_vredmost_populacije, i

            resenje = W_najboljih

        return resenje[0], prosecna_vrednost_populacije, minimalna_vredmost_populacije, i


class MagicniKvadratGenetskiAlgoritam(MagicniKvadrat):

    def __init__(self, n, populacija=8, roditelji=4, broj_iteracija=300):
        super(MagicniKvadratGenetskiAlgoritam, self).__init__(n)
        self.populacija = populacija
        self.roditelji = roditelji
        self.br = broj_iteracija

    def inicijalizacija_populacije(self):
        """
        Generisemo onoliko clanova koliko zelimo da imamo u svakoj populaciji
        :return:
        """
        niz_populazije = list()
        for i in range(self.populacija):
            clan = self.generisanje_pocetnog_resenja()
            niz_populazije.append(clan)

        return np.array(niz_populazije)

    def biranje_roditelja(self, populacija):
        """
        Roditelji se biraju kao k najboljih iz populaiije. Na osnovu sortiranih ciljnih funkcija populacije, sortiramo
        populaciju, a zatim izaberemo k elemenata.
        :param populacija:
        :return:
        """

        niz_ciljnih_fja_populacije = np.zeros(len(populacija))
        for i in range(len(populacija)):
            niz_ciljnih_fja_populacije[i] = self.izracunavanje_vrednosti_ciljne_funkcije(populacija[i])

        indeksi = np.argsort(niz_ciljnih_fja_populacije)

        populacija_sortirano = np.zeros((len(niz_ciljnih_fja_populacije), self.n, self.n))
        ciljne_sortirano = np.zeros((len(niz_ciljnih_fja_populacije)))
        for i in range(len(niz_ciljnih_fja_populacije)):
            populacija_sortirano[i] = populacija[indeksi[i]]
            ciljne_sortirano[i] = niz_ciljnih_fja_populacije[indeksi[i]]

        roditelji = populacija_sortirano[:self.roditelji]

        return roditelji, ciljne_sortirano[0]

    def pravljenje_dece(self, roditelj1, roditelj2):
        """
        Bazirano na PMS algoritmu (partilly mapped crossover).
        PMX za 1D slucaj:
            biramo 2 podniza od svakog roditelja. Podniz prvog roditelja postavljamo na mesto drugog deteta (na istim
            indeksima) a podniz drugog roditelja postavljamo na prvo dete. Zatim, elemente prvog deteta popunjavamo
            elementima prvog roditelja; ako se desi da se ponovi broj koji se nalazi u podnizu koji je vec unet u dete
            tada taj broj menjamo sa elementom iz drugog podniza, na istoj lokaciji (ponavljamo dok ne nadjemo element
            koji se ne nalazi u podnizu deteta). Analogno za drugo dete.
        :param roditelj1:
        :param roditelj2:
        :return: Dve nxn matrice proistekle iz roditelja 1 i roditelja2
        """
        roditelj1 = roditelj1.flatten()
        roditelj2 = roditelj2.flatten()

        tacka1 = np.random.randint(0, self.n * self.n - 1)
        tacka2 = np.random.randint(tacka1 + 1, self.n * self.n)

        podniz1 = roditelj1[tacka1:tacka2]
        podniz2 = roditelj2[tacka1:tacka2]

        dete1 = np.zeros(self.n * self.n)
        dete2 = np.zeros(self.n * self.n)

        dete1[tacka1:tacka2] = podniz2
        dete2[tacka1:tacka2] = podniz1

        # dete1
        for i in range(tacka1):
            if roditelj1[i] not in set(podniz2):
                dete1[i] = roditelj1[i]
            else:
                while dete1[i] == 0:
                    for j in range(len(podniz2)):
                        if roditelj1[i] == podniz2[j]:
                            roditelj1[i] = podniz1[j]
                            if roditelj1[i] not in set(podniz2):
                                dete1[i] = roditelj1[i]
                                break

        for i in range(tacka2, self.n * self.n):
            if roditelj1[i] not in set(podniz2):
                dete1[i] = roditelj1[i]
            else:
                while dete1[i] == 0:
                    for j in range(len(podniz2)):
                        if roditelj1[i] == podniz2[j]:
                            roditelj1[i] = podniz1[j]
                            if roditelj1[i] not in set(podniz2):
                                dete1[i] = roditelj1[i]
                                break

        # dete2
        for i in range(tacka1):
            if roditelj2[i] not in set(podniz1):
                dete2[i] = roditelj2[i]
            else:
                while dete2[i] == 0:
                    for j in range(len(podniz1)):
                        if roditelj2[i] == podniz1[j]:
                            roditelj2[i] = podniz2[j]
                            if roditelj2[i] not in set(podniz1):
                                dete2[i] = roditelj2[i]
                                break

        for i in range(tacka2, self.n * self.n):
            if roditelj2[i] not in set(podniz1):
                dete2[i] = roditelj2[i]
            else:
                while dete2[i] == 0:
                    for j in range(len(podniz1)):
                        if roditelj2[i] == podniz1[j]:
                            roditelj2[i] = podniz2[j]
                            if roditelj2[i] not in set(podniz1):
                                dete2[i] = roditelj2[i]
                                break

        return dete1.reshape((3, 3)), dete2.reshape((3, 3))

    def trazi_najboljeg_naslednika(self):

        populacija = self.inicijalizacija_populacije()
        prosecna_ciljna_populacije = []
        minimalna_ciljna_populacije = []

        niz_ciljnih_populacije = []
        for i in range(len(populacija)):
            # prvo proveravamo da li bilo ko iz populacije zadovoljava uslov magicnog kvadrata
            ciljna_clana = self.izracunavanje_vrednosti_ciljne_funkcije(populacija[i])
            niz_ciljnih_populacije.append(ciljna_clana)
            if ciljna_clana == 0:
                resenje = populacija[i]
                prosecna_ciljna_populacije.append(np.mean(niz_ciljnih_populacije))
                return resenje, prosecna_ciljna_populacije, [0], 1
        prosecna_ciljna_populacije.append(np.mean(niz_ciljnih_populacije))

        for m in range(2, self.br):
            deca = list()

            roditelji, minimalna = self.biranje_roditelja(populacija)
            minimalna_ciljna_populacije.append(minimalna)

            niz_ciljnih_populacije = []
            for j in range(2):

                # posto hocemo da nam populacija bude 8, imamo 4 roditelja, a svaki par generise dvoje dece, moramo da
                # prodjemo kro ovo 2 puta
                # todo: ovo mora da se de_hardcode-uje
                for i in range(0, len(roditelji) - 1, 2):
                    roditelj1 = roditelji[i]
                    roditelj2 = roditelji[i + 1]

                    dete1, dete2 = self.pravljenje_dece(roditelj1, roditelj2)
                    ciljna_dete1 = self.izracunavanje_vrednosti_ciljne_funkcije(dete1)
                    ciljna_dete2 = self.izracunavanje_vrednosti_ciljne_funkcije(dete2)

                    niz_ciljnih_populacije.append(ciljna_dete1)
                    niz_ciljnih_populacije.append(ciljna_dete2)
                    if ciljna_dete1 == 0:
                        resenje = dete1
                        prosecna_ciljna_populacije.append(np.mean(niz_ciljnih_populacije))
                        minimalna_ciljna_populacije.append(0)
                        return resenje, prosecna_ciljna_populacije, minimalna_ciljna_populacije, m
                    elif ciljna_dete2 == 0:
                        resenje = dete2
                        prosecna_ciljna_populacije.append(np.mean(niz_ciljnih_populacije))
                        minimalna_ciljna_populacije.append(0)
                        return resenje, prosecna_ciljna_populacije, minimalna_ciljna_populacije, m

                    # mutacija ce nam biti random zamena 2 broja na svkaoj 5oj iteraciji
                    if i % 5 == 0:
                        dete1 = self.generisanje_slucajnog_naslednika_resenja(dete1)

                    deca.append(dete1)
                    deca.append(dete2)

            populacija = np.array(deca)
            prosecna_ciljna_populacije.append(np.mean(niz_ciljnih_populacije))

        niz_ciljnih_fja_populacije = np.zeros(len(populacija))
        # ako nismo nasli decu, trazimo najbolje resenje u poslednjoj populaciji
        # sortiranjem kao i sortiranjem kod biranja roditelja
        for i in range(len(populacija)):
            niz_ciljnih_fja_populacije[i] = self.izracunavanje_vrednosti_ciljne_funkcije(populacija[i])

        indeksi = np.argsort(np.array(niz_ciljnih_fja_populacije))

        niz_ciljnih_fja_populacije_sortirano = np.zeros(len(niz_ciljnih_fja_populacije))
        populacija_sortirano = np.zeros((len(niz_ciljnih_fja_populacije), self.n, self.n))
        for i in range(len(niz_ciljnih_fja_populacije)):
            populacija_sortirano[i] = populacija[indeksi[i]]
            niz_ciljnih_fja_populacije_sortirano[i] = niz_ciljnih_fja_populacije[indeksi[i]]

        resenje = populacija_sortirano[0]
        ciljna_fja = niz_ciljnih_fja_populacije[0]
        minimalna_ciljna_populacije.append(ciljna_fja)

        return resenje, prosecna_ciljna_populacije, minimalna_ciljna_populacije, m
