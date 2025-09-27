import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Funkcija za crtanje korelacione matrice ---
def draw_correlation_heatmap(df, title):
    # Računanje korelacione matrice za numeričke kolone
    corr_matrix = df.corr() 
    plt.figure(figsize=(10, 8))
    # Heatmap sa vrednostima
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title(title)
    plt.show()

# --- Funkcija za crtanje različitih survival grafika ---
def draw_survival_plots(df):
    # 1. Raspodela starosti za preživele i nepreživele
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, palette="Set1")
    plt.title("Preživljavanje u odnosu na godine")
    plt.xlabel("Godine")
    plt.ylabel("Broj putnika")
    plt.legend(title="Preživeo", labels=["Ne", "Da"])
    plt.show()

    # 2. Preživljavanje po klasi
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set2")
    plt.title("Preživljavanje po klasi")
    plt.xlabel("Klasa")
    plt.ylabel("Broj putnika")
    plt.legend(title="Preživeo", labels=["Ne", "Da"])
    plt.show()

    # 3. Preživljavanje po polu (Sex je kodiran 0/1 → pretvaramo u labelu)
    sex_map = {0: "Žena", 1: "Muškarac"}
    df["Sex_label"] = df["Sex"].map(sex_map)
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="Sex_label", hue="Survived", palette="Set1")
    plt.title("Preživljavanje po polu")
    plt.xlabel("Pol")
    plt.ylabel("Broj putnika")
    plt.legend(title="Preživeo", labels=["Ne", "Da"])
    plt.show()
    # Brišemo pomoćnu kolonu
    df.drop(columns=["Sex_label"], inplace=True)

    # 4. Preživljavanje po veličini porodice
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x="FamilySize", hue="Survived", palette="pastel")
    plt.title("Preživljavanje po veličini porodice")
    plt.xlabel("Veličina porodice (broj članova)")
    plt.ylabel("Broj putnika")
    plt.legend(title="Preživeo", labels=["Ne", "Da"])
    plt.show()

    # 5. Raspodela FarePerPerson u odnosu na preživljavanje
    plt.figure(figsize=(10,6))
    sns.histplot(
        data=df, 
        x="FarePerPerson", 
        hue="Survived", 
        multiple="stack", 
        bins=40,               # malo više binova da se bolje vidi raspodela
        palette="Set2",        # blaže boje
        edgecolor="black"      # da se stubići jasnije vide
    )
    plt.title("Preživljavanje u odnosu na cenu karte po osobi", fontsize=14)
    plt.xlabel("Cena karte po osobi", fontsize=12)
    plt.ylabel("Broj putnika", fontsize=12)
    plt.legend(title="Preživeo", labels=["Ne", "Da"])
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xlim(0, df["FarePerPerson"].quantile(0.95))  # iseče ekstremno skupe karte da graf bude pregledniji
    plt.show()


    # 6. Uticaj veličine grupe putnika (isti tiket) na preživljavanje
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, x="TicketGroupSize", hue="Survived", palette="muted")
    plt.title("Uticaj veličine grupe putnika (isti tiket) na preživljavanje")
    plt.xlabel("Veličina grupe (isti tiket)")
    plt.ylabel("Broj putnika")
    plt.legend(title="Preživeo", labels=["Ne", "Da"])
    plt.show()

    # 7. Da li posedovanje kabine utiče na preživljavanje
    cabin_map = {0: "Ne", 1: "Da"}
    df["has_cabin_label"] = df["has_cabin"].map(cabin_map)
    plt.figure(figsize=(6,5))
    sns.countplot(data=df, x="has_cabin_label", hue="Survived", palette="Set2")
    plt.title("Uticaj posedovanja kabine na verovatnoću preživljavanja")
    plt.xlabel("Ima kabinu")
    plt.ylabel("Broj putnika")
    plt.legend(title="Preživeo", labels=["Ne", "Da"])
    plt.show()
    df.drop(columns=["has_cabin_label"], inplace=True)

    # 8. Preživljavanje po palubi (Deck)
    deck_map = {0: "U", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}
    df["Deck_label"] = df["Deck"].map(deck_map)
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x="Deck_label", hue="Survived", palette="Set1")
    plt.title("Uticaj palube (Deck) na verovatnoću preživljavanja")
    plt.xlabel("Kabina")
    plt.ylabel("Broj putnika")
    plt.legend(title="Preživeo", labels=["Ne", "Da"])
    plt.show()
    df.drop(columns=["Deck_label"], inplace=True)

# --- Glavni deo programa ---

# Proveravamo da li postoji obrađeni dataset
if not os.path.exists("processed_data.csv"):
    raise FileNotFoundError("processed_data.csv not found")
df = pd.read_csv("processed_data.csv") 

if not os.path.exists("processed_data_reduced.csv"):
    raise FileNotFoundError("processed_data_reduced.csv not found")
df_reduced = pd.read_csv("processed_data_reduced.csv")

# Pretvaranje bool kolona u int (0/1), da se mogu koristiti za korelaciju
for col in df.select_dtypes(include=["bool"]).columns:
    df[col] = df[col].astype(int)

for col in df_reduced.select_dtypes(include=["bool"]).columns:
    df_reduced[col] = df_reduced[col].astype(int)

# Zadržavamo samo numeričke kolone za korelacionu analizu
df_numericke = df.select_dtypes(include=["number"])
print("Numeričke kolone (df):", df_numericke.columns)

df_reduced_numericke = df_reduced.select_dtypes(include=["number"])
print("Numeričke kolone (df_reduced):", df_reduced_numericke.columns)

# Crtanje korelacionih matrica
draw_correlation_heatmap(df_numericke, "Korelaciona matrica (sve kolone)")
draw_correlation_heatmap(df_reduced_numericke, "Korelaciona matrica (izabrane kolone)")

# Crtanje survival grafika (samo za full dataset)
draw_survival_plots(df)

print("Analiza podataka završena.")