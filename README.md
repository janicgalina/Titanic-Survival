Razvoj modela za klasifikaciju preživelih putnika sa Titanika

Obrada i priprema podataka
	Pre treniranja modela mašinskog učenja, bilo je neophodno izvršiti obradu sirovih podataka kako bi bili pripremljeni za analizu. Obrada je realizovana u Python skripti Obrada_podataka.py, koja učitava originalni fajl data.csv i generiše obrađeni skup podataka, sačuvan u fajlu processed_data.csv koji je dalje obrađen u okviru redukcije skupa podataka.
Otkrivanje i obrada nedostajućih vrednosti 
1. Kreiranje atributa Title i obrada nedostajućih vrednosti u koloni Age
	Iz kolone Name izdvojena je titula svakog putnika (npr. Mr, Mrs, Miss, Master) korišćenjem funkcije extract_title(). Primenjen je one-hot encoding kako bi se kategorijske vrednosti razložile u binarne promenljive. Ova transformacija omogućava modelu da efikasno koristi informacije iz Title bez uvođenja implicitnog redosleda među kategorijama, što poboljšava performanse modela.
Ove titule često predstavljaju socijalni status, pol ili starosnu grupu putnika. Na primer:
•	Mr – odrasli muškarac
•	Mrs – udata žena
•	Miss – neudata žena ili devojka
•	Master – dečak, obično mlađi od 14 godina
	Manje uobičajene titule (poput Don, Lady, Col, Dr itd.) grupisane su pod zajednički naziv Other kako bi se smanjila složenost modela i izbegla prekomerna segmentacija podataka. Dobijene titule iskorišćene su za procenu i popunjavanje nedostajućih vrednosti u koloni Age. Na primer, očekuje se da osobe sa titulom Master imaju znatno nižu prosečnu starost u odnosu na one sa titulom Mr ili Mrs.
	Nedostajuće vrednosti u koloni Age popunjene medianom starosti unutar svake pojedinačne titule. Ukoliko to nije bilo moguće, neka titula ima premalo uzoraka, za popunjavanje nedostajućih vrednosti korišćena je globalna medijana za celu kolonu Age. Na ovaj način omogućeno je da se informacije o tituli direktno iskoriste za što preciznije određivanje godina, bez uvođenja značajne pristrasnosti.
2. Obrada i transformacija kolone Cabin u kolonu Deck
	Iz kolone Cabin je izvedena kolona Deck. Kolona Cabin je obrađena tako što je iz nje izdvojen prvi karakter, koji označava kabinu u kojoj se nalazio putnik (npr. A, B, C...). Ove kabine su zatim kodirane pomoću ordinalnog kodiranja (ordinal encoding), pri čemu su kabine mapirane na numeričke vrednosti: A = 1, B = 2, C = 3, D = 4, E = 5, F = 6, dok je nepoznata kabina označena kao U = 0.
	Na taj način je omogućeno modelima da koriste informacije o kabini u numeričkom obliku. Redosled dodeljenih vrednosti je značajan jer nam daje informaciju o tome kako su kabine raspoređene u odnosu na palubu — kabina A je najbliža palubi, dok su kabine sa većim brojevima dalje po položaju. Sada model može da koristi informacije iz kolone Cabin na način koji odražava značaj položaja kabine u analizi preživljavanja putnika.
3. Indikator da li putnik ima podatak o kabini (has_cabin)
	Dodata je binarna kolona has_cabin koja označava da li putnik ima upisanu kabinu (vrednost 1) ili ne (vrednost 0). Ovaj indikator može pružiti dodatne informacije o klasnoj pripadnosti i statusu putnika, a posebno je koristan za modele poput stabala odlučivanja, ansambala  i logističke regresije, koji efikasno koriste diskretne i binarne karakteristike za poboljšanje tačnosti predviđanja.
4. Veličina porodice i da li je putnik bio sam (FamilySize, IsAlone)
	Dodate su dve kolone: FamilySize i isAlone. Ukupna veličina porodice izračunata je sabiranjem vrednosti kolona SibSp (broj braće, sestara i supružnika) i Parch (broj roditelja i dece), uz dodatak 1 za samog putnika. Na osnovu toga je dodata nova binarna kolona IsAlone, koja označava da li je putnik putovao sam (vrednost 1) ili nije (vrednost 0). Ove kolone su kreirane kako bi se stvorili korisniji i lakše interpretabilni signali u odnosu na pojedinačne kolone SibSp i Parch.
5.Grupisanje putnika po kartama (TicketGroupSize)
	Izračunat je broj putnika koji dele isti broj karte, što je ubačeno kao nova kolona TicketGroupSize. Ona je važna jer omogućava modelu da prepozna ljude koji su putovali u grupama, koje su mogle imati drugačiji obrazac preživljavanja usled međusobne povezanosti, zajedničkih resursa ili prioriteta pri evakuaciji. Uključivanjem ove informacije model bolje razume socijalne faktore koji su uticali na ishod.
6. Proračun cene po osobi (FarePerPerson,FarePerTicketPerson)
	Obe kolone, FarePerPerson i FarePerTicketPerson, služe za procenu socijalno-ekonomskog statusa putnika, ali na različite načine. FarePerPerson izražava prosečnu cenu karte po osobi unutar porodice, dok FarePerTicketPerson uzima u obzir broj putnika koji dele istu kartu, nezavisno od porodičnih veza. Ovim pristupom model može bolje da razume kako finansijski resursi utiču na šanse za preživljavanje, kako na nivou porodice, tako i na nivou grupe putnika.
8. Obrada nedostajućih podataka i transformacija kolone Embarked
	Za redove gde nije bila navedena luka ukrcavanja (Embarked), nedostajuća vrednost je zamenjena najčešćom kategorijom u skupu podataka – „S“. Nakon toga, kolona Embarked je transformisana primenom one-hot encodinga, čime je svaka kategorija (S, C, Q) pretvorena u posebnu binarnu promenljivu (Embarked_S, Embarked_C, Embarked_Q). 
9. Numerička reprezentacija pola (Sex)
	Kolona Sex je enkodirana korišćenjem label encodinga, pri čemu su vrednosti "male" zamenjene sa 1, a vrednosti "female" sa 0, čime je pol predstavljen numerički za potrebe modela mašinskog učenja.
Analiza podataka i vizuelizacija
	Analiza podatka i vizuelizacija je odrađena u Python skripti Vizuelizacija_podataka.py. Glavni cilj sprovedene analize bio je da se prikažu korelacije između numeričkih promenljivih, kao i zavisnost preživljavanja od različitih atributa putnika sa Titanika. Na ovaj način se olakšava identifikacija kombinacije kolona koja daje najbolje rezultate pri učenju modela.
Korelaciona matrica
	Iz oba skupa podataka (processed_data.csv i processed_data_reduced.csv) izdvojene su numeričke kolone. Pomoću funkcije draw_correlation_heatmap izračunate su i prikazane korelacione matrice, koje omogućavaju uvid u međusobnu povezanost promenljivih.
 
Slika 1 Korelaciona matrica za sve kolone
	Na slici je prikazana korelaciona matrica za sve numeričke promenljive u skupu podataka. Heatmap omogućava da se vizuelno uoči stepen i smer povezanosti između različitih atributa. Crvena boja označava pozitivnu korelaciju, dok plava označava negativnu korelaciju.
Korelacije ciljne promenljive Survived
	Najjača negativna korelacija uočena je sa polom (Sex, –0.54), što znači da su žene imale veću šansu za preživljavanje.	Postoji negativna povezanost i sa klasom (Pclass, –0.34), što ukazuje da su putnici iz nižih klasa imali manju verovatnoću da prežive.
	Pozitivna korelacija se javlja sa posedovanjem kabine (has_cabin, 0.32) i dodeljenom palubom (Deck, 0.30), što sugeriše da su putnici sa kabinama imali veće šanse za preživljavanje. Cena karte po osobi (FarePerPerson) takođe pokazuje pozitivnu korelaciju (0.22), što znači da je skuplja karta često bila povezana sa većim šansama za preživljavanje.
	Atributi izvedeni iz imena, posebno Title_Miss (0.33) i Title_Mrs (0.34), pokazuju pozitivnu povezanost sa preživljavanjem, dok Title_Mr ima negativnu korelaciju (–0.55), što potvrđuje razlike u preživljavanju među polovima i statusima putnika.
Međusobne korelacije atributa
	SibSp, Parch i FamilySize su u jakoj pozitivnoj korelaciji (0.78–0.89), što je očekivano jer predstavljaju slične informacije o broju članova porodice.
	Fare, FarePerPerson i FarePerTicketPerson su visoko povezani (0.74–0.84) jer su izvedeni i međusobno slični atributi.
	Deck i has_cabin imaju veoma jaku korelaciju (0.89) jer se vrednost kabine može dobiti samo ako je kabina poznata.
	Titule izvedene iz imena pokazuju umerenu do jaku korelaciju međusobno (npr. Title_Miss i Sex, –0.69) jer su povezane sa polom.
	Neki atributi poput PassengerId i Age nemaju značajnu korelaciju ni sa jednom promenljivom.
	Korelaciona matrica pokazuje da su pol, klasa putovanja, cena karte, posedovanje kabine i titule među najvažnijim faktorima povezanima sa preživljavanjem putnika. Takođe, vidljivo je da postoje jake međusobne zavisnosti između određenih izvedenih promenljivih, što je uzeto u obzir prilikom izbora atributa za obuku modela, kako bi se izbegla multikolinearnost.
Vizuelizacija preživljavanja
	Na osnovu celokupnog skupa podataka generisani su grafici koji prikazuju raspodelu preživelih i stradalih putnika u odnosu na različite atribute radi boljeg razumevanja njihovih uticaja:
 
Slika 2 Uticaj starosti na verovatnoću preživljavanja
	Analizom podataka o preživelima sa broda Titanik uočavaju se jasne razlike između različitih grupa putnika. Godine putnika su imale značajnu ulogu – mlađi putnici, naročito deca, imali su veću šansu za preživljavanje, dok su stariji putnici (posebno iznad 50 godina) imali manju verovatnoću da se spasu.

 
Slika 3 Uticaj klase na verovatnoću preživljavanja
	Analiza po klasama jasno pokazuje da su putnici iz 1. klase imali najveću šansu da prežive. U 2. klasi šanse su bile srednje, dok su putnici iz 3. klase bili ubedljivo najugroženiji i imali su najmanju stopu preživljavanja.
 
Slika 5 Uticaj pola na verovatnoću preživljavanja
	Kada se posmatra pol, razlike su još izraženije. Žene su imale daleko veću stopu preživljavanja u odnosu na muškarce. 
 
Slika 6 Uticaj veličine porodice na verovatnoću preživljavanja
	Veličina porodice takođe je bila značajna. Najveći broj preživelih bio je među putnicima koji su putovali sami, ali to je i očekivano jer je takvih putnika bilo najviše. Međutim, posmatrajući procente preživljavanja u odnosu na veličinu grupe, putnici koji su putovali u paru ili u manjim porodicama (2–3 člana) imali su najveću šansu za preživljavanje. Suprotno tome, članovi većih grupa i porodica (5 i više) imali su najmanju verovatnoću da se spasu, što se vidi i na slici ispod.
 
Slika 7 Uticaj veličine grupe putnika na verovatnoću preživljavanja

 
Slika 8 Uticaj cene karte po osobi na verovatnoću preživljavanja
	Cena karte je značajan faktor koji je uticao na verovatnoću preživljavanja. Na grafikonu se vidi da je najveći broj putnika platio relativno nisku cenu karte (do oko 10 funti). U toj grupi nalazimo i najveći broj preživelih, ali i najveći broj stradalih, što je posledica činjenice da je tu bila koncentrisana većina putnika treće klase. Sa porastom cene karte, učešće preživelih postaje veće, što ukazuje da su putnici viših klasa, koji su plaćali skuplje karte, imali povoljniju poziciju i bolji pristup čamcima za spasavanje.
 
Slika 8 Uticaj posedovanja kabine na verovatnoću preživljavanja
	Posedovanje kabine pokazalo se kao bitan pokazatelj. Putnici kojima je bila dodeljena kabina imali su veću verovatnoću preživljavanja u poređenju sa onima koji nisu imali kabinu. Osim toga, lokacija kabine igrala je ulogu – putnici čije su kabine bile bliže palubi imali su veću šansu da se evakuišu na vreme, dok su oni bez precizno dodeljene kabine ili sa udaljenijih paluba imali manju verovatnoću preživljavanja.
 
Slika 9 Uticaj udaljenosti kabine od palube na verovatnoću preživljavanja
Redukcija skupa podataka
	Redukcija skupa podatak je odrađena u Python skripti Pripremljeni_podaci.py. Na osnovu prethodne korelacione matrice izvršena je selekcija atributa i uklonjene su kolone koje su se pokazale kao redundantne ili slabo korisnim. Prikazana matrica korelacija odnosi se isključivo na odabrane karakteristike koje su korišćene za obuku modela, čime se jasno sagledavaju međusobni odnosi zadržanih atributa i njihova povezanost sa ciljnim ishodom (Survived).
 
Slika 10 Korelaciona matrica za kolone korišćene za obuku modela
Izbačene kolone:
•	PassengerId – jedinstveni identifikator putnika, koji nema uticaja na verovatnoću preživljavanja.
•	SibSp i Parch – predstavljaju broj braće/sestara/supružnika, odnosno roditelja/dece na brodu. Umesto njih, korišćena je izvedena kolona FamilySize, koja bolje obuhvata porodičnu strukturu putnika.
•	Fare – originalna cena karte zamenjena je sa FarePerPerson što normalizuje cenu u odnosu na broj članova grupe ili porodice.
•	FarePerTicketPerson – dodatno izvedena kolona koja se pokazala redundantnom u odnosu na FarePerPerson.
•	Cabin – tekstualna oznaka kabine, koja je transformisana u binarnu promenljivu has_cabin i kategorijalnu promenljivu Deck , radi korisnije upotrebe u modelu.
•	Title_Other – veoma retka kategorija izvedena iz imena, sa malim brojem primera, što otežava učenje modela i može unositi šum.
•	Title_Master – titula koja pokazuje jaku povezanost sa polom i godinama, informacije koje su već eksplicitno sadržane u kolonama Sex i Age, pa je redundantna.
•	Embarked_C, Embarked_Q i Embarked_S – dummy promenljive koje predstavljaju luku ukrcavanja. Analiza je pokazala da nemaju značajan uticaj na preživljavanje, pa su uklonjene radi smanjenja dimenzionalnosti ulaznih parametara.
	Nakon uklanjanja navedenih kolona, dobijen je redukovan skup podataka sačuva  u processed_data_reduced.csv sa zadržanim samo najrelevantnijim karakteristikama za izgradnju modela. Time je postignuta brža i stabilnija obuka modela i smanjen je rizik od prenaučenosti (overfitting).
Obuka modela 

•	Ulazne promenljive (X):                                     	
Pclass
Sex
Age
FamilySize
IsAlone
TicketGroupSize
FarePerPerson
Deck
has_cabin
Title_Miss
Title_Mr
Title_Mrs

•	Ciljana promenljiva (y): Survived (0 = nije preživeo, 1 = preživeo)
•	Podaci su podeljeni na trening (80%) i test (20%) skup.
Za  trening i poređenje performansi korišćeni su sledeći klasifikacioni modeli:
1.	Decision Tree 
2.	Random Forest 
3.	Gradient Boosting
4.	XGBoost
5.	Logistic Regression 
Navedeni modeli su imali sledeće performanse:
Model	Accuracy	Precision (Survived)	  Recall 
   (Survived)	       F1-score
       (Survived)
Decision Tree 	   0.76	           0.72	        0.61	           0.66
Random Forest	   0.82	           0.81	        0.68	           0.74
Gradient Boosting 	   0.81	           0.77	        0.72	           0.75
XGBoost 	   0.82	           0.79	        0.72	           0.76
Logistic Regression	   0.82	           0.74	        0.80	           0.77
Tabela 1 Performanse default modela
	Iz dobijenih rezultata vidimo da su se najbolje pokazali Random Forest, XGBoost i Logistic Regression, sa tačnošću od 0.82. Sa druge strane, Decision Tree je imao slabije rezultate u poređenju sa ostalim modelima.
Optimizacija hiperparametara
	Za svaki model sprovedena je optimizacija hiperparametara pomoću GridSearchCV metode sa unakrsnom validacijom od 5 podela (5-fold cross-validation). Ovim podešavanjima modeli su postigli najbolje rezultate na validacionom skupu. 
Najbolji dobijeni hiperparametri za:
•	Decision Tree: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
•	Random Forest: {'class_weight': 'balanced', 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
•	Gradient Boosting: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}
•	XGBoost: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}
•	Logistic Regression: {'C': 10, 'max_iter': 5000, 'penalty': 'l2', 'solver': 'lbfgs'}
 
Evaluacija modela

	Za svaki model su prvo izračunate i međusobno upoređivane performance na test skupu (Accuracy, Precision, Recall i F1-score) koji su u nastavku prikazane preko tabela.
Model	Precision (0)	Recall (0)	F1-score (0)	Precision (1)	Recall (1)	F1-                score (1)	Accuracy
Decision Tree	0.85	0.92	0.88	0.85	0.74	0.79	0.85
Random Forest	0.85	0.92	0.88	0.85	0.74	0.79	0.85
Gradient Boosting	0.85	0.88	0.87	0.80	0.75	0.78	0.83
XGBoost	0.84	0.92	0.88	0.85	0.72	0.78	0.84
Logistic Regression	0.87	0.82	0.85	0.74	0.81	0.77	0.82
Tabela 2 Classification report za sve modele posle optimizacije hiperparametara
	Rezultati pokazuju da Decision Tree i Random Forest postižu identične performanse, sa tačnošću od 0.85. Gradient Boosting i XGBoost ostvaruju nešto niže rezultate (0.83 i 0.84). Logistic Regression je imao najnižu ukupnu tačnost (0.82), ali se ističe najvišim odzivom za klasu „Survived“ (0.81), što znači da je bio najuspešniji u prepoznavanju putnika koji su preživeli, iako uz slabiju preciznost.
	Posle dodavanja class_weight = 'balanced' u parametre za optimizaciju modela Decision Tree dobijeni su sledeći rezultati:
Scenario	Precision (0)	Recall (0)	F1-score (0)	Precision (1)	Recall (1)	F1-score (1)	Accuracy
Bez class_weight='balanced'	0.85	0.92	0.88	0.85	0.74	0.79	0.85
Sa class_weight='balanced'	0.88	0.89	0.88	0.82	0.80	0.81	0.85
Table 3 Uporedna tabela za performanse modela Decision Tree
	Dodavanjem parametra class_weight='balanced' u Decision Tree model postignut je ravnomerniji odnos između klasa. Preciznost za klasu 1 (preživele) se blago smanjila sa 0.85 na 0.82, ali je odziv porastao sa 0.74 na 0.80, što znači da model uspešnije prepoznaje veći broj preživelih putnika. Ukupna tačnost modela ostala je nepromenjena (0.85), ali je balans između klasa značajno poboljšan. Na ovaj način model postaje pravedniji prema manjinskoj klasi.

Zaključak o najboljim modelima
	
	Rezultati pokazuju da su se kao najbolji modeli izdvojili Decision Tree sa podešenim parametrom class_weight='balanced' i Random Forest, oba sa ukupnom tačnošću od 0.85. Iako postižu isti nivo tačnosti, njihova primena ima različite prednosti. Decision Tree je jednostavniji za tumačenje i daje bolji balans između klasa, posebno u prepoznavanju preživelih (klase 1). S druge strane, Random Forest kao ansambl model nudi veću stabilnost i pouzdanost predikcija, jer kombinuje više stabala i samim tim smanjuje rizik od prenaučenosti. U zavisnosti od ciljeva, oba modela se mogu smatrati najboljim izborom: Decision Tree zbog interpretabilnosti, a Random Forest zbog robusnosti i konzistentnih performansi.
	Za oba model na slikama ispod su prikazane konfuzione matrice, radi vizuelnog uvida u ispravne i pogrešne klasifikacije.
	Na osnovu konfuzione matrice za Random Forest model može se uočiti da je od ukupno 110 putnika koji nisu preživeli, model ispravno klasifikovao 101, dok je 9 pogrešno svrstao u klasu preživelih. Kada je reč o putnicima koji su preživeli (69), tačno je identifikovano 51, dok je 18 klasifikovano kao da nisu preživeli. Ovi rezultati pokazuju da model bolje prepoznaje putnike koji nisu preživeli, dok se češće javlja greška prilikom klasifikacije onih koji jesu preživeli.

 
Slika 11 Confusion Matix za Random Forest
	Na osnovu konfuzione matrice za Decision Tree model može se uočiti da  je model od ukupno 110 putnika koji nisu preživeli, ispravno prepoznao 98, dok je 12 pogrešno svrstao u grupu preživelih. Kada je reč o putnicima koji su preživeli (69), model je tačno identifikovao 55, dok je 14 klasifikovao kao da nisu preživeli. 
 
Slika 12 Confusion Matrix za Decision Tree
Mogućnosti za unapređenje modela
Moguća unapređenja modela uključuju:
-	primenu tehnika balansiranja klasa (npr. SMOTE) –  na XGBoost za rešavanje nesrazmerne podele podataka između broja preživelih i onih koji nisu preživeli 
-	dodatni feature engineering (npr. kreiranje novih atributa na osnovu postojećih) –  na osnovu grafikona značaja atributa, prikazani ispod, može se uočiti da određene promenljive imaju znatno veći uticaj na donošenje odluka modela. Kod oba modela najdominantniji atribut je Title_Mr, dok kod Random Forest modela veći broj atributa pokazuje značajan uticaj. Ova analiza ukazuje da bi dodatni feature engineering (kreiranje novih atributa ili izbacivanje i zamena nekih) mogao doprineti boljoj reprezentaciji podataka i potencijalno unaprediti performanse modela. 

 
 Slika 13 Feature Importance za Decision Tree
 
Slika 14 Feature Importance za Random Forest
-	optimizaciju hiperparametara sa većim opsegom vrednosti – u radu je već sprovedena optimizacija, ali proširivanjem opsega pretrage hiperparametara moguće je pronaći još stabilnije i bolje konfiguracije modela.
