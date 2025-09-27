import os
import numpy as np
import pandas as pd

IN = "data.csv"
OUT_PROC = "processed_data.csv"

# Funkcija za ekstrakciju titule iz imena putnika
def extract_title(name):
    s = str(name)
    if "," in s:
        _, after_comma = s.split(",", 1)
        title, _, _ = after_comma.partition(".")
        title = title.strip()
        # Čuvamo samo osnovne titule, sve ostalo ide pod "Other"
        return title if title in ("Mr", "Mrs", "Miss", "Master") else "Other"
    return "Other"

# Funkcija za ekstrakciju deck-a iz kabine
def extract_deck(cabin):
    if pd.isna(cabin):
        return "U"   # U = Unknown
    s = str(cabin).strip()
    if s == "":
        return "U"
    return s[0]  # prvi karakter označava kabinu

# Glavna funkcija za obradu podataka
def process_dataframe(df):
    # Ekstrakcija titula iz imena
    df["Title"] = df["Name"].apply(extract_title)
    
    # One-hot encoding za Title (pravi posebne kolone Title_Mr, Title_Mrs...)
    df = pd.get_dummies(df, columns=["Title"], drop_first=False)
    
    # Ekstrakcija Deck iz kolone Cabin
    df["Deck"] = df["Cabin"].apply(extract_deck)
    deck_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'U':0}
    df["Deck"] = df["Deck"].map(deck_map)
    df["Deck"] = df["Deck"].fillna(df["Deck"].median())

    # Indikator da li putnik ima kabinu ili ne
    df["has_cabin"] = df["Cabin"].notna().astype(int)

    # Family size = SibSp + Parch + 1 (sam putnik)
    df["FamilySize"] = df["SibSp"].astype(int) + df["Parch"].astype(int) + 1
    # Indikator da li je putnik sam
    df["IsAlone"] = df["FamilySize"].apply(lambda x: 1 if x == 1 else 0)

    # Ticket group size (koliko ljudi ima istu kartu → grupna karta)
    df["Ticket"] = df["Ticket"].astype(str).str.strip()
    counts = df["Ticket"].value_counts()
    df["TicketGroupSize"] = df["Ticket"].map(counts).astype(int)

    # Cena po osobi (na osnovu veličine porodice)
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Cena po osobi (na osnovu veličine grupe sa istom kartom)
    df["FarePerTicketPerson"] = df["Fare"] / df["TicketGroupSize"]

    # Popunjavanje nedostajućih godina (Age)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if "Title" in df.columns:
        # Median godina po tituli (npr. Mr, Mrs, Miss...)
        age_med_by_title = df.groupby("Title")["Age"].median() if "Title" in df.columns else None
    else:
        age_med_by_title = None
    global_age_med = df["Age"].median()
    if age_med_by_title is not None:
        df["Age"] = df["Age"].fillna(df["Title"].map(age_med_by_title)).fillna(global_age_med)
    else:
        df["Age"] = df["Age"].fillna(global_age_med)

    # Popunjavanje nedostajućih vrednosti za Embarked
    if "Embarked" in df.columns:
        sizes = df.groupby("Embarked").size()
        fill_val = sizes.idxmax() if not sizes.empty else "S"  # popunjavanje sa "S"
        df["Embarked"] = df["Embarked"].fillna(fill_val)

    # One-hot encoding za Embarked (Embarked_C, Embarked_Q, Embarked_S)
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=False)

    # Kodiranje pola: muški = 1, ženski = 0
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    return df

if __name__ == "__main__":
    if not os.path.exists(IN):
        raise FileNotFoundError(f"{IN} not found")
        
    df = pd.read_csv(IN)
    df = process_dataframe(df)

    # Prikupljanje svih kolona za Title i Embarked (posle one-hot encodinga)
    title_cols = [col for col in df.columns if col.startswith("Title_")]
    embarked_cols = [col for col in df.columns if col.startswith("Embarked_")]

    # Kolone koje želimo da sačuvamo u fajlu
    cols_to_save = [
        "PassengerId","Survived","Pclass","Sex","Age","SibSp","Parch",
        "FamilySize","IsAlone","TicketGroupSize","Fare","FarePerPerson",
        "FarePerTicketPerson",
        "Cabin","Deck","has_cabin"
    ] + title_cols + embarked_cols

    # Prebrojavanje preživelih i nepreživelih za uvid u balans skupa
    if "Survived" in df.columns:
        survived_counts = df["Survived"].value_counts()
        print("\nBroj putnika po ishodu (Survived):")
        print(survived_counts.to_string())
    
    print("---------------------------------------------")
    print("Kolone koje se trenutno koriste:", cols_to_save)
    df[cols_to_save].to_csv(OUT_PROC, index=False)
    print("---------------------------------------------")
    print(f"Podaci su sačuvani u {OUT_PROC}")
    print("---------------------------------------------")