import argparse
import itertools
import re
import numpy as np
import scipy.stats as stats


def wczytaj_nagrody(sciezka_pliku):
    """Funkcja wyciąga nagrody z pliku tekstowego zachowując ich kolejność."""
    try:
        with open(sciezka_pliku, "r", encoding="utf-8") as f:
            tresc = f.read()
            znalezione = re.findall(r"reward\s+([\d\.]+)", tresc)
            if not znalezione:
                print(
                    f"Ostrzeżenie: Nie znaleziono żadnych nagród w pliku: {sciezka_pliku}"
                )
            return np.array([float(x) for x in znalezione])
    except FileNotFoundError:
        print(f"Błąd: Nie można odnaleźć pliku pod ścieżką: {sciezka_pliku}")
        exit(1)
    except Exception as e:
        print(f"Nieoczekiwany błąd podczas odczytu pliku {sciezka_pliku}: {e}")
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Analiza statystyczna wyników ewaluacji dla wielu konfiguracji sieci (próby zależne)."
    )
    # narg='+' pozwala na przekazanie 2, 3 lub więcej plików jako lista
    parser.add_argument(
        "pliki",
        type=str,
        nargs="+",
        help="Ścieżki do plików z wynikami ewaluacji (minimum 2 pliki)",
    )

    args = parser.parse_args()

    # Walidacja minimalnej liczby plików
    if len(args.pliki) < 2:
        print("Błąd: Musisz podać przynajmniej 2 pliki do porównania!")
        exit(1)

    # Wczytanie wszystkich danych do słownika
    baza_danych = {}
    for sciezka in args.pliki:
        baza_danych[sciezka] = wczytaj_nagrody(sciezka)

    # Walidacja długości wektorów danych
    dlugosci = [len(v) for v in baza_danych.values()]
    if len(set(dlugosci)) > 1 or dlugosci[0] == 0:
        print("\n" + "!" * 50)
        print("BŁĄD WALIDACJI DANYCH:")
        print("Wszystkie pliki muszą zawierać dokładnie tę samą liczbę prób!")
        for sciezka, dane in baza_danych.items():
            print(f"  - {sciezka}: {len(dane)} prób")
        print("!" * 50 + "\n")
        exit(1)

    N_prob = dlugosci[0]
    liczba_plikow = len(args.pliki)

    print("=" * 60)
    print("             ZAAWANSOWANA ANALIZA STATYSTYCZNA              ")
    print("=" * 60)
    print(f"Liczba porównywanych konfiguracji: {liczba_plikow}")
    print(f"Liczba par testowych (N): {N_prob}")
    print("-" * 60)

    # =========================================================================
    # SCENARIUSZ 1: DOKŁADNIE 2 PLIKI (Zachowanie bez zmian)
    # =========================================================================
    if liczba_plikow == 2:
        plik_1, plik_2 = args.pliki[0], args.pliki[1]
        roznice = baza_danych[plik_2] - baza_danych[plik_1]

        stat_shapiro, p_shapiro = stats.shapiro(roznice)
        print(f"1. Test normalności Shapiro-Wilka dla różnic:")
        print(f"   - Wartość p: {p_shapiro:.4f}")

        normalny = p_shapiro > 0.05
        print(
            f"   - Wniosek: Różnice {'Mają' if normalny else 'NIE mają'} rozkładu normalnego."
        )
        print("-" * 60)

        t_stat, p_paired = stats.ttest_rel(
            baza_danych[plik_2], baza_danych[plik_1]
        )
        wilc_stat, p_wilcoxon = stats.wilcoxon(
            baza_danych[plik_2], baza_danych[plik_1]
        )

        wybrane_p = p_paired if normalny else p_wilcoxon
        nazwa_testu = (
            "t-Studenta parowany" if normalny else "kolejności par Wilcoxona"
        )

        print(f"2. Wyniki testów istotności:")
        print(f"   - Test t-Studenta parowany:  p-value = {p_paired:.4f}")
        print(f"   - Test parowany Wilcoxona:   p-value = {p_wilcoxon:.4f}")
        print("-" * 60)

        print("3. Podsumowanie wniosków:")
        if wybrane_p < 0.05:
            print(
                f"   [SUKCES] Różnica jest ISTOTNA statystycznie (p = {wybrane_p:.4f}, {nazwa_testu})."
            )
        else:
            print(
                f"   [BRAK ISTOTNOŚCI] Różnica NIE JEST istotna statystycznie (p = {wybrane_p:.4f}, {nazwa_testu})."
            )

    # =========================================================================
    # SCENARIUSZ 2: 3 LUB WIĘCEJ PLIKÓW (Test Friedmana + Post-Hoc)
    # =========================================================================
    else:
        print("1. Globalny test nieparametryczny Friedmana:")
        # Przygotowanie list danych do testu omnibus
        lista_danych = [baza_danych[sciezka] for sciezka in args.pliki]

        stat_friedman, p_friedman = stats.friedmanchisquare(*lista_danych)
        print(f"   - Wartość p testu Friedmana: {p_friedman:.4f}")

        if p_friedman >= 0.05:
            print("-" * 60)
            print("2. Podsumowanie wniosków:")
            print(
                f"   [BRAK ISTOTNOŚCI] Globalny test Friedmana nie wykazał różnic między grupami (p = {p_friedman:.4f})."
            )
            print(
                "   Żadna z konfiguracji nie wyróżnia się w sposób istotny statystycznie. Koniec analizy."
            )
        else:
            print(
                "   - Wniosek: Istnieją ISTOTNE globalne różnice między grupami. Uruchamiam testy post-hoc."
            )
            print("-" * 60)

            # Generowanie wszystkich unikalnych par do porównań
            pary = list(itertools.combinations(args.pliki, 2))
            liczba_porownan = len(pary)

            # Poprawka Bonferroniego: nowy próg istotności to alfa / liczba porównań
            alfa_bazowa = 0.05
            alfa_bonferroni = alfa_bazowa / liczba_porownan

            print(f"2. Testy Post-Hoc (Parowane testy Wilcoxona):")
            print(f"   - Liczba porównań parami: {liczba_porownan}")
            print(
                f"   - Skorygowany próg istotności (Bonferroni) α = {alfa_bonferroni:.4f}\n"
            )

            istotne_pary = []

            for plik_a, plik_b in pary:
                _, p_wilc = stats.wilcoxon(baza_danych[plik_a], baza_danych[plik_b])
                status = (
                    "ISTOTNA (SUKCES)"
                    if p_wilc < alfa_bonferroni
                    else "nieistotna"
                )

                print(f"   * {plik_a} vs {plik_b}:")
                print(
                    f"     p-value = {p_wilc:.4f} -> Różnica jest {status}"
                )

                if p_wilc < alfa_bonferroni:
                    istotne_pary.append((plik_a, plik_b, p_wilc))

            print("-" * 60)
            print("3. Podsumowanie wniosków do pracy naukowej:")
            if istotne_pary:
                print(
                    "   Testy post-hoc potwierdziły statystycznie istotne różnice dla par:"
                )
                for pa, pb, p_val in istotne_pary:
                    print(f"   [+] {pa} oraz {pb} (p = {p_val:.4f})")
            else:
                print(
                    "   Mimo pozytywnego testu globalnego, konserwatywna poprawka Bonferroniego"
                )
                print(
                    "   nie pozwoliła jednoznacznie wskazać konkretnej, wygrywającej pary."
                )

    print("=" * 60)


if __name__ == "__main__":
    main()