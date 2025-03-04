# # UE12 Projet programmation élémentaire Tristan Chariot

# ## Table des matières
# ### I. Analyse du fichier cities.csv
# #### 1) Epuration du fichier
# #### 2) Analyse des données
# ### II. Analyse du fichier countries.csv
# #### 1) Epuration du fichier
# #### 2) Analyse des données
# ### III. Analyse du fichier dailiy-weather-cities.csv
# #### 3) Epuration du fichier
# #### 4) Analyse des données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import geodatasets

# ## I. Analyse du fichier cities.csv

df = pd.read_csv('cities.csv')
df

df.shape

# La dataframe possède 1245 lignes et 7 colonnes indiquant pour chaque ville de la table son numéro unique associé (station_id), sa latitude, sa longitude, le pays et l'état dans laquelle elle se trouve aves ses code ISO2 et ISO3 associés.
# On va dès à présent vérifier qu'il ne manque pas de données dans la dataframe.

# ### 1) Epuration de la dataframe cities.csv

df_incomplet = df[df.isna().any(axis=1)]
df_incomplet

# On constate qu'il manque des valeurs dans l'iso2 de la dataframe mais toutes les valeurs de l'iso3 sont renseignées. Or l'iso3 est plus précis que l'iso2 car on a trois lettres au lieu de deux pour désigner un pays, on peut donc suprimer la colonne iso2.

df.drop(['iso2'], axis=1, inplace=True)
df

df_incompletb = df[df.isna().any(axis=1)]
df_incompletb

# On retire la colonne state à laquelle car il y a beaucoup trop de valeurs manquantes et la ligne 911 où le nom de la ville n'est pas indiqué.

df.drop(['state'], axis=1, inplace=True)
df.drop([911], axis=0, inplace=True)
df[df.isna().any(axis=1)]

# Concentrons nous maintenant sur les doublons dans notre dataframe.

print(df.shape, (df.city_name.unique()).shape)

df_doublonsvilles1 = df[df.duplicated(['city_name'])]
df_doublonsvilles1

df_doublonsvilles2 = df[df.duplicated(['city_name'], keep='last')]
df_doublonsvilles2

# On constate que les 10 villes ayant le même nom sont dans des pays différents. Il n'y a donc pas de doublons. Le nom de la ville n'est donc pas adapté comme index des colonnes.
# Regardons si l'indice station_id est plus approprié.

print(df.shape, (df.station_id.unique()).shape)

df_station_id1 = df[df.duplicated(['station_id'])]
print(df_station_id1.shape)
df_station_id1

df_station_id2 = df[df.duplicated(['station_id'], keep='last')]
print(df_station_id2.shape)
df_station_id2

# Dans la dataframe, il y a 18 doublons d'identifiant de stations on suprime 18 lignes de la dataframe correspondante. On a ensuite une unicité des station_id et peut donc déclarer cette variable comme index de ligne de la dataframe.

df = df.drop([36, 42, 69, 148, 195, 239, 265, 267, 311, 320, 325, 333, 348, 375, 519, 584, 905, 1056])

df.set_index('station_id', inplace=True)
df


# ### 2) Utilisation des données de cities.csv

def carte_villes(longitude, latitude):
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    world.plot(color='white', edgecolor='black')
    plt.scatter(longitude, latitude, s=0.1, color='g')
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.title('Villes du monde présentes dans la dataframe')
    plt.show()


carte_villes(df['longitude'], df['latitude'])


def pays_plusrepresente() :
    dico = {}
    liste_pays = df['country'].unique()
    for pays in liste_pays :
        dico[pays] = int((df['country'] == pays).sum())
    dico_trie = sorted(dico.items(), key=lambda x: x[1], reverse=True)
    top_5 = np.array(dico_trie[:5])
    pays = top_5[:5, 0]
    valeurs = np.array(top_5[:5, 1], dtype=int)
    print(pays)
    print(valeurs)
    plt.bar(x=pays, height=valeurs, width=0.1, align='center', color='blue', orientation='vertical')
    plt.title('Graphique représentant les 5 pays ayant le plus de villes dans la dataframe par ordre décroissant')
    plt.ylabel('nombre de villes pour chaque pays')
    plt.show()


pays_plusrepresente()

# On a obtenu un histogramme représentant les 5 pays ayant les plus de villes dans la dataframe.

# ## II. Analyse du fichier country.csv

df1 = pd.read_csv('countries.csv')
df1

df1.shape

# La dataframe contient 214 pays avec son nom, son nom dans la langue du pays, son iso2 et iso3, sa population, sa surface, le nom de sa capitale ainsi que ses coordonées géographiques, la région du monde et le continent dans lequel il se trouve.

# ### 1) Epuration de la dataframe countries.csv

df_incomplet1 = df1[df1.isna().any(axis=1)]
df_incomplet1

df1.drop(['iso2'], axis=1, inplace=True)
df1.head(10)

# On supprime la colonne iso2 car elle est moins précise que l'iso3.

df1_incomplet = df1[df1.isna().any(axis=1)]
df1_incomplet

df1 = df1.drop([10, 63, 66, 71, 123, 156, 157, 173, 180, 208])
df1


# ### 2) Utilisation des données de la datframe countries.csv

def carte_capitales(longitude, latitude):
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    world.plot(color='white', edgecolor='black')
    plt.scatter(longitude, latitude, s=0.2, color='b')
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.title('Capitales du monde présentes dans la dataframe')
    plt.show()


carte_capitales(df1['capital_lng'], df1['capital_lat'])


def pays_pluspeuple() :
    dico = {}
    liste_pays = df1['country'].unique()
    for pays in liste_pays :
        Ls = df1[df1['country'] == pays]
        dico[pays] = int(Ls['population'].iloc[0]) 
    dico_trie = sorted(dico.items(), key=lambda x: x[1], reverse=True)
    top_5 = np.array(dico_trie[:5])
    pays = top_5[:5, 0]
    pop = np.array(top_5[:5, 1], dtype=int)
    print(pays)
    print(pop)
    plt.bar(x=pays, height=pop, width=0.1, align='center', color='orange', orientation='vertical')
    plt.title('Graphique représentant les 5 pays les plus peuplés dans la dataframe par ordre décroissant')
    plt.ylabel("nombre d'habitants pour chaque pays")
    plt.show()


pays_pluspeuple()


# Les 5 pays les plus peuplés de la datframe sont représentés sur cet histogramme avec pour chacun son nombre d'habitant. La chine possède 1,37 milliard d'habitants.

def graph_continent():
    nbAsia = (df1['continent'] == 'Asia').sum()
    nbEurope = (df1['continent'] == 'Europe').sum()
    nbAfrica = (df1['continent'] == 'Africa').sum()
    nbOceania = (df1['continent'] == 'Oceania').sum()
    nbNorthAmerica = (df1['continent'] == 'NorthAmerica').sum()
    nbSouthAmerica = (df1['continent'] == 'SouthAmerica').sum()
    nbAntartica = (df1['continent'] == 'Antartica').sum()
    nbtot = nbAsia + nbEurope + nbAfrica + nbOceania + nbNorthAmerica + nbSouthAmerica + nbAntartica
    labels = ['Asia', 'Europe', 'Africa', 'Oceania', 'NorthAmerica', 'SouththAmerica', 'Antarctica']
    sizes = [nbAsia, nbEurope, nbAfrica, nbOceania, nbNorthAmerica, nbSouthAmerica, nbAntartica]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'red', 'blue', 'green']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()
    print(labels)
    print(sizes)


graph_continent()

df1Amerique = df1[(df1['continent'] == 'NorthAmerica') | (df1['continent'] == 'SouthAmerica')]
print(df1Amerique.sum())


# A l'aide de ce graphique en camenbert, on constate que le continent n'est pas reseigné pour les pays d'Amérique du nord et du sud. En effet, il y en a bien aucun pays d'Amérique qui a son continent de renseigné.

def pays_plusetendu() :
    dico = {}
    liste_pays = df1['country'].unique()
    for pays in liste_pays :
        Ls = df1[df1['country'] == pays]
        dico[pays] = int(Ls['area'].iloc[0]) 
    dico_trie = sorted(dico.items(), key=lambda x: x[1], reverse=True)
    top_5 = np.array(dico_trie[:5])
    pays = top_5[:5, 0]
    area = np.array(top_5[:5, 1], dtype=int)
    print(pays)
    print(area)
    plt.bar(x=pays, height=area, width=0.1, align='center', color='green', orientation='vertical')
    plt.title('Graphique représentant les 5 pays avec le plus de superficie dans la dataframe par ordre décroissant')
    plt.ylabel("surface de chaque pays")
    plt.show()


pays_plusetendu()

# Les 5 pays les plus étendus de la datframe sont représentés sur cet histogramme avec pour chacun sa superficie. La Russie a une surperficie de 17 millions de km**2

dfint = pd.merge(df, df1, left_on='iso3', right_on='iso3')
dfint

# ## III. Analyse du fichier daily-weather-cicties.csv

df2 = pd.read_csv('daily-weather-cities.csv')
df2

# ### 1) Epuration de la dataframe

df2['Date'] = pd.to_datetime(df2['date'])

df2['Year'] = df2['Date'].dt.year
df2['Month'] = df2['Date'].dt.month
df2['Day'] = df2['Date'].dt.day

df2.drop(['date'], axis=1, inplace=True)

# On vient de convertir la créer une colonne Date conteant la date sous le format pandas datetime (yyyy-mm-dd) pour ensuite pouvoir créer des colonnes Year, Month et Day. Ensuite, on a supprimé la colonne date qui ne servait plus.

df2

df2.notna().sum()

# Toutes les colonnes ont au moins des valeurs de reseignée

df2_incompletb = df2[df2.isna().all(axis=1)]
df2_incompletb

# Aucune des lignes de la dataframe a toutes ses valeurs de non indiquées.

dfpluie2023Winter = df2[(df2['precipitation_mm'].notna()) & (df2['Year']==2023) & (df2['season']=='Winter')]
dfpluie2023Winter


# ### 2) Analyse données de la dataframe

def pluie_hiver_2023():
    dfp = df2[(df2['precipitation_mm'].notna()) & (df2['Year']==2023) & (df2['season']=='Winter')]
    dico = {}
    liste_ville = df2['city_name'].unique()
    for ville in liste_ville :
        Ls = dfp[dfp['city_name'] == ville]
        dico[ville] = int(Ls['precipitation_mm'].sum()) 
    dico_trie = sorted(dico.items(), key=lambda x: x[1], reverse=True)
    top_5 = np.array(dico_trie[:5])
    ville = top_5[:5, 0]
    pluietot = np.array(top_5[:5, 1], dtype=int)
    print(ville)
    print(pluietot.dtype)
    plt.bar(x=ville, height=pluietot, width=0.1, align='center', color='red', orientation='vertical', bottom=0) #.set_ylim(0,110)
    plt.title("Graphique représentant les 5 villes avec le plus de pécipitations pendant l'hiver 2023 par ordre décroissant")
    plt.ylabel("precipitations totale en mm")
    plt.show()


pluie_hiver_2023()


# La ville ayant le plus de précipitations cumulées durant l'hiver 2023 est Bruxelle avec plus de double de la quantité qui est tombée sur Paris durant la même période.

def carte_temp():
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    world.plot(color='white', edgecolor='black')
    dftot = pd.merge(df, df2, left_on='city_name', right_on='city_name')
    dft = dftot[(dftot['avg_temp_c'].notna()) & (dftot['Year']==2010) & (dftot['season']=='Summer')]
    plt.scatter(dft[(dft.avg_temp_c < 27) & (dft.avg_temp_c >=10)].longitude, dft[(dft.avg_temp_c < 27) & (dft.avg_temp_c >=10)].latitude, s=0.5, color='g')
    plt.scatter(dft[(dft.avg_temp_c < 28) & (dft.avg_temp_c >=27)].longitude, dft[(dft.avg_temp_c < 28) & (dft.avg_temp_c >=27)].latitude, s=0.5, color='y')
    plt.scatter(dft[(dft.avg_temp_c < 29) & (dft.avg_temp_c >=28)].longitude, dft[(dft.avg_temp_c < 29) & (dft.avg_temp_c >=28)].latitude, s=0.5, color='orange')
    plt.scatter(dft[(dft.avg_temp_c < 30) & (dft.avg_temp_c >=29)].longitude, dft[(dft.avg_temp_c < 30) & (dft.avg_temp_c >=29)].latitude, s=0.5, color='red')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    left = 450
    top = 120
    right = 575
    bottom = 250


carte_temp()


def pluie_Paris():
    dfp = df2[(df2['precipitation_mm'].notna()) & (df2['Year'].notna()) & (df2['city_name']=='Paris')]
    dico = {}
    liste_annee = df2['Year'].unique()
    print(liste_annee)
    for annee in liste_annee :
        Ls = dfp[dfp['Year'] == annee]
        dico[annee] = int(Ls['precipitation_mm'].sum()) 
    dico_trie = sorted(dico.items(), key=lambda x: x[1], reverse=True)
    dico_trie = np.array(dico_trie)
    annee = dico_trie[:, 0]
    pluietot = dico_trie[:, 1]
    print(annee)
    print(pluietot)
    plt.bar(x=annee, height=pluietot, width=0.1, align='center', color='purple', orientation='vertical')
    plt.title("Graphique représentant l'évolution au fil des années des précipitations totales sur l'année à Paris")
    plt.ylabel("precipitations totale en mm")
    plt.xlim(1940,2020)
    plt.show()


pluie_Paris()

# ### Fin du projet de programmation élémentaire




