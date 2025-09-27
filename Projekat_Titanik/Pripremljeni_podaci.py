import os
import pandas as pd

# Ulazni i izlazni fajlovi
IN = "processed_data.csv"
OUT = "processed_data_reduced.csv"

# Proveravamo da li postoji ulazni fajl
if not os.path.exists(IN):
    raise FileNotFoundError(f"{IN} not found")

# Učitavamo CSV u DataFrame
df = pd.read_csv(IN)

# Lista kolona koje želimo da uklonimo jer:
# - nisu korisne za predikciju
# - imaju bolju zamenu (redundantne su)
# - ili su previše slabe
kolone_za_izbacivanje = [
    "PassengerId",  # ID putnika – nije relevantan za predikciju
    "SibSp",        # Zamenjeno sa "FamilySize"
    "Parch",        # Zamenjeno sa "FamilySize"
    "Fare",         # Zamenjeno sa "FarePerPerson"
    "FarePerTicketPerson", # Zamenjeno sa FarePerPerson
    "Cabin",        # Transformisano u "has_cabin" i "Deck"
    "Title_Other",  # Jedna od one-hot kolona titula, smanjujemo dimenzionalnost
    "Title_Master", # Isto kao gore
    "Embarked_C",   # One-hot kodiranje luke ukrcavanja, smanjujemo dimenzionalnost
    "Embarked_Q",   # Isto
    "Embarked_S"    # Isto
]

# Filtriramo samo one kolone koje zaista postoje u DataFrame-u
postojeci_za_brisanje = [col for col in kolone_za_izbacivanje if col in df.columns]

# Ako postoje, brišemo ih
if postojeci_za_brisanje:
    df = df.drop(columns=postojeci_za_brisanje)
    df.to_csv(OUT, index=False)  # Čuvamo novi DataFrame u CSV fajl
    print("------------------------------------------------------------------------------------------------")
    print(f"Sačuvan fajl {OUT}")
    print(f"Uklonjene kolone: [{', '.join(postojeci_za_brisanje)}].")
    print("------------------------------------------------------------------------------------------------")
else:
    # Ako ni jedna od predviđenih kolona ne postoji – ništa se ne briše
    print("Nijedna od ciljanih kolona nije pronađena; ništa nije uklonjeno.")

# Ispis trenutnih kolona koje su ostale u DataFrame-u
print("Trenutne kolone u DataFrame-u:", df.columns.tolist())
