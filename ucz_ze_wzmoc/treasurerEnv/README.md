# TREASURE HUNTER - Test w Konsoli

Animowana wizualizacja gry Treasure Hunter w konsoli z wytrenowanymi agentami!

## Struktura plikow

```
treasurerEnv/
├── test_game.py           # Glowny skrypt testowy (uruchom to!)
├── treasure_env.py        # Srodowisko gry
├── save_agents_helper.py  # Pomocnik do zapisywania agentow
├── README.md             # Ten plik
├── agent1.pkl            # Wytrenowany agent 1 (tworzony przez notebook)
└── agent2.pkl            # Wytrenowany agent 2 (tworzony przez notebook)
```

## Jak uruchomic

### Krok 1: Wytrenuj agentow w notebooku

W notebooku `treasuerHunter_v2.ipynb`:
1. Uruchom trening agentow (komorki z `play_and_train_multi`)
2. Uruchom komorke zapisujaca agentow (na koncu notebooka)

Alternatywnie mozesz dodac wlasny kod:
```python
import pickle

with open('treasurerEnv/agent1.pkl', 'wb') as f:
    pickle.dump(agent1, f)

with open('treasurerEnv/agent2.pkl', 'wb') as f:
    pickle.dump(agent2, f)
```

### Krok 2: Uruchom test w konsoli

```powershell
python treasurerEnv/test_game.py
```

## Co zobaczysz

```
============================================================
            TREASURE HUNTER - Krok 15
============================================================

  0123456789
0 A........T
1 ..#..#..#.
2 .H..T...H.
3 .#..#*.#..
4 T........2

Legenda:
  A/B = Bazy agentow | T = Skarb | H = Pulapka | # = Sciana
  1/2 = Agenci | */+ = Agent trzyma skarb | X = Kolizja

Status:
  Punkty:  Agent 1: 3  |  Agent 2: 0
  Skarby:  Na mapie: 2  |  A1 trzyma: True  |  A2 trzyma: False

Ostatni ruch:
  Agent 1: > PRAWO -> Nagroda: -1
  Agent 2: v DOL -> Nagroda: -1
============================================================
```

## Opcje

Program pyta o:
- **Tryb**: Zaladuj wytrenowanych agentow (1) lub stworz nowych bez treningu (2)
- **Liczba gier**: Ile rund rozegrac (domyslnie 3)
- **Opoznienie**: Czas miedzy ruchami w sekundach (domyslnie 0.5s)

## Funkcje

- Czyszczenie konsoli - kazdy krok aktualizuje ekran
- Kolorowa plansza - agenci (1/2), skarby (T), pulapki (H), sciany (#)
- Status na zywo - punkty, skarby, co trzymaja agenci
- Historia ruchow - pokazuje ostatnie akcje i nagrody
- Eventy specjalne - [PICK] podnoszenie, [OK] oddawanie skarbow
- Wynik koncowy - kto wygral i o ile punktow

## Wymagania

- Python 3.7+
- Moduly: `os`, `time`, `pickle` (standardowe)
- Plik `agents.py` w katalogu glownym (dla trybu 2)

## Wskazowki

**Zbyt szybko?**
```bash
python treasurerEnv/test_game.py
# Wybierz wieksze opoznienie (np. 1.0 lub 2.0 sekund)
```

**Zbyt wolno?**
```bash
python treasurerEnv/test_game.py
# Wybierz mniejsze opoznienie (np. 0.1 lub 0.2 sekund)
```

**Chcesz testowac nowych agentow?**
- Wybierz opcje 2 w menu
- Agenci beda robic losowe ruchy (brak treningu)

## Troubleshooting

**Blad: "Nie znaleziono plikow agent1.pkl"**
→ Uruchom najpierw komorke zapisujaca w notebooku!

**Blad: "Import treasure_env could not be resolved"**
→ Upewnij sie ze uruchamiasz z katalogu `ucz ze wzmoc/`

**Konsola sie nie czysci**
→ Skrypt automatycznie wykrywa system (Windows/Linux/Mac)
→ Na Windowsie uzywa `cls`, na Unix `clear`

## Notatki

- Agent 1 = zazwyczaj SARSA-lambda (niebieski)
- Agent 2 = zazwyczaj Double Q-Learning (czerwony)
- Mozna zapisac rozne pary agentow i testowac ich strategie!

## Milej zabawy!

Obserwuj jak wytrenowani agenci konkuruja o skarby w czasie rzeczywistym!
