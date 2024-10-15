---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# TP à envoyer dimanche 13 au soir

## On a des fichiers à *recoller*

on utilise les fonctions `pandas.merge` et `pandas.concat`

+++

2 manières de faire ce TP
1. vous utilisez le help de ces deux fonctions et vous faites directement les exercices suivants
2. vous allez à l'annexe 1 qui est un cours sur merge et concat, vous le lisez et vous revenez faire les exercices

```{code-cell} ipython3
import pandas as pd
```

## simple critère

+++

on a trois fichiers à recoller

```{code-cell} ipython3
:cell_style: split

df1 = pd.read_csv('data/collages1.csv')
df1
```

```{code-cell} ipython3
:cell_style: split

df2 = pd.read_csv('data/collages2.csv')
df2
```

```{code-cell} ipython3
df3 = pd.read_csv('data/collages3.csv')
df3
```

comment vous feriez pour recoller les morceaux ? il s'agit d'obtenir une dataframe de 5 élèves et 4 caractéristiques

+++

on peut envisager deux versions de l'exercice, selon qu'on choisit ou non d'indexer selon le prénom

+++

### sans index

```{code-cell} ipython3
# à vous
dfint = pd.merge(df1, df2)
dfint
df = pd.concat([dfint, df3], ignore_index=True)
df
```

```{code-cell} ipython3
dfint = pd.merge(df1, df2)
dfint
df = pd.concat([dfint.set_index('name'), df3.set_index('name')])
df
```

### avec index

+++

dans un premier temps, pour chacune des trois tables, adoptez la colonne `name` comme index;

puis recollez les morceaux comme dans le premier exercice

```{code-cell} ipython3
# à vous
df1 = df1.set_index('name')
df2 = df2.set_index('name')
df3 = df3.set_index('name')
dfint = pd.merge(df1, df2, left_on='name', right_on='name')
df = pd.concat([dfint, df3])
df
```

## multiples critères

même idée, mais on n'a plus unicité des prénoms

```{code-cell} ipython3
m1 = pd.read_csv("data/multi1.csv")
m1
```

```{code-cell} ipython3
m2 = pd.read_csv("data/multi2.csv")
m2
```

```{code-cell} ipython3
m3 = pd.read_csv("data/multi3.csv")
m3
```

```{code-cell} ipython3
# à vous - c'est vous qui décidez comment gérer les index
# juste, à la fin on voudrait un index "raisonnable"
m1.columns = ['first_name', 'last_name', 'age', 'height', 'sex',]
m3.columns = ['first_name', 'last_name', 'age', 'grade', 'height', 'sex',]
mint = pd.merge(m1, m2)
mint
mf = pd.concat([m3.set_index('age'), mint.set_index('age')])
mf
```

# Annexe 1

## cours merge/concat

+++ {"tags": ["framed_cell"], "jp-MarkdownHeadingCollapsed": true}

````{admonition} →
parfois on obtient les données par **plusieurs canaux**, qu'il faut agréger dans une seule dataframe

les outils à utiliser pour cela sont multiples  
pour bien choisir, il est utile de se poser en priorité la question de savoir 
si les différentes sources à assembler concernent les **mêmes colonnes** ou au contraire les **mêmes lignes**  (*)


illustrations:

* on recueille les données à propos du coronavirus, qui sont disponibles par mois  
  chaque fichier a la même structure - disons 2 colonnes: *deaths*, *confirmed*  
  l'assemblage consiste donc à agréger les dataframes **en hauteur**

* on recueille les notes des élèves d'une classe de 20 élèves  
  chaque prof fournit un fichier excel avec les notes de sa matière  
  chaque table contient 20 lignes  
  il faut cette fois agréger les dataframes **en largeur**

```{note}
cette présentation est simpliste, elle sert uniquement à fixer les idées
```
````

+++ {"tags": ["framed_cell"]}

### en hauteur `pd.concat()`

````{admonition} →
pour l'accumulation de données (en hauteur donc), préférez la fonction `pandas` suivante

* la fonction `pd.concat([df1, df2, ..])`  
  qui a vocation à accumuler des données en hauteur  
````

```{code-cell} ipython3
# exemple 1
# les deux dataframes ont les mêmes colonnes
# (ici on crée les dataframe à partir d'un dict décrivant les colonnes)
df1 = pd.DataFrame(
    data={
        'name': ['Bob', 'Lisa', 'Sue'],
        'group': ['Accounting', 'Engineering', 'HR']})

df2 = pd.DataFrame(
    data={
        'name': ['John', 'Mary', 'Andrew'],
        'group': ['HR', 'Accounting', 'Engineering',]})
```

```{code-cell} ipython3
:cell_style: split

df1
```

```{code-cell} ipython3
:cell_style: split

df2
```

```{code-cell} ipython3
# nous ne gardons pas les index de chaque sous-dataframe
pd.concat([df1, df2], ignore_index=True)
# pd.concat([df1, df2], axis=0) # by default concat rows
```

```{code-cell} ipython3
# nous indexons les dataframes par la colonne 'name'
pd.concat([df1.set_index('name'), df2.set_index('name')])
```

+++ {"tags": ["framed_cell"]}

### en largeur `pd.merge()`

````{admonition} →
pour la réconciliation de données, voyez cette fois

* la fonction `pd.merge(left, right)`  
  ou sous forme de méthode `left.merge(right)`  

* et la méthode `left.join(right)` - en fait une version simplifiée de `left.merge(...)`

grâce à ces outils, il est possible d'aligner des dataframes sur les valeurs de une ou plusieurs colonnes

````

+++

### alignements

dans les deux cas, `pandas` va ***aligner*** les données  
par exemple on peut concaténer deux tables qui ont les mêmes colonnes, même si elles sont dans le désordre

l'usage typique de `merge()`/`join()` est l'équivalent d'un JOIN en SQL (pour ceux à qui ça dit quelque chose)  

**sans indication**, `merge()` calcule les **colonnes communes** et se sert de ça pour aligner les lignes

```{code-cell} ipython3
# exemple 1
# les deux dataframes ont exactement une colonne en commun: 'name'

df1 = pd.DataFrame(
    data={
        'name': ['Bob', 'Lisa', 'Sue'],
        'group': ['Accounting', 'Engineering', 'HR']})  # une seule colonne

df2 = pd.DataFrame(
    data={
        'name': ['Lisa', 'Bob', 'Sue'],
        'hire_date': [2004, 2008, 2014]})
```

```{code-cell} ipython3
:cell_style: split

df1
```

```{code-cell} ipython3
:cell_style: split

df2
```

```{code-cell} ipython3
# sans rien préciser, on JOIN sur la colonne commune 'name'

df1.merge(df2)
```

```{code-cell} ipython3
# on peut aussi l'écrire comme ceci

pd.merge(df1, df2)
```

```{code-cell} ipython3
# exemple 2
# cette fois il faut aligner l'index de gauche
# avec la colonne 'name' à droite

df1 = pd.DataFrame(
    index = ['Bob', 'Lisa', 'Sue'],  # l'index
    data={'group': ['Accounting', 'Engineering', 'HR']})  # une seule colonne

df2 = pd.DataFrame(
    data = {'name': ['Lisa', 'Bob', 'Sue'],
            'hire_date': [2004, 2008, 2014]})
```

```{code-cell} ipython3
:cell_style: split

df1
```

```{code-cell} ipython3
:cell_style: split

df2
```

```{code-cell} ipython3
# du coup ici sans préciser de paramètres, ça ne fonctionnerait pas
# il faut être explicite

df1.merge(df2, left_index=True, right_on='name')
```

```{code-cell} ipython3
# ou encore

pd.merge(df1, df2, left_index=True, right_on='name')
```

## optionnel: plusieurs stratégies pour le merge/join

comme en SQL, on a à notre disposition plusieurs stratégies pour le `merge` (ou `join`, donc)
le paramètre `how` peut prendre les valeurs suivantes:

* `left`: on garde les clés qui sont dans la première dataframe
* `right`: on garde les clés qui sont dans la seconde dataframe
* `inner`: on garde les clés communes
* `outer`: on garde l'union des clés

(il y a aussi `cross`, mais c'est plus particulier comme usage..)

+++

### concat() *vs* merge()

````{admonition} concat() *vs* merge()
les deux fonctionnalités sont assez similaires sauf que

* `merge` est une opération **binaire**, alors que `concat` est **n-aire**  
   ce qui explique d'ailleurs la différence de signatures: `concat([d1, d2])` *vs* `merge(d1, d2)`

* seule `concat()` supporte un paramètre `axis=`
````

```{code-cell} ipython3

```
