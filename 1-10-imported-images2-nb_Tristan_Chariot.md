---
jupytext:
  cell_metadata_json: true
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

Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

+++

# TP images (2/2)

merci à Wikipedia et à stackoverflow

**le but de ce TP n'est pas d'apprendre le traitement d'image  
on se sert d'images pour égayer des exercices avec `numpy`  
(et parce que quand on se trompe ça se voit)**

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt
```

+++ {"tags": ["framed_cell"]}

````{admonition} → **notions intervenant dans ce TP**

* sur les tableaux `numpy.ndarray`
  * `reshape()`, masques booléens, *ufunc*, agrégation, opérations linéaires
  * pour l'exercice `patchwork`:  
    on peut le traiter sans, mais l'exercice se prête bien à l'utilisation d'une [indexation d'un tableau par un tableau - voyez par exemple ceci](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
  * pour l'exercice `sepia`:  
    ici aussi on peut le faire "naivement" mais l'utilisation de `np.dot()` peut rendre le code beaucoup plus court
* pour la lecture, l'écriture et l'affichage d'images
  * utilisez `plt.imread()`, `plt.imshow()`
  * utilisez `plt.show()` entre deux `plt.imshow()` si vous affichez plusieurs images dans une même cellule

  ```{admonition} **note à propos de l'affichage**
  :class: seealso dropdown admonition-small

  * nous utilisons les fonctions d'affichage d'images de `pyplot` par souci de simplicité
  * nous ne signifions pas là du tout que ce sont les meilleures!  
    par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
    alors que la fonction `save` de `PIL` le permet
  * vous êtes libres d'utiliser une autre librairie comme `opencv`  
    si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte...
  ```
````

+++

## Création d'un patchwork

+++

1. Le fichier `data/rgb-codes.txt` contient une table de couleurs:
```
AliceBlue 240 248 255
AntiqueWhite 250 235 215
Aqua 0 255 255
.../...
YellowGreen 154 205 50
```
Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.

```{code-cell} ipython3
# votre code
filename = 'data/rgb-codes.txt'
colors = dict()
with open(filename, 'r') as file:
    for line in file:
        colname, *l = line.split()
        colors[colname] = np.array([int(e) for e in l], dtype=np.uint8)
print(colors)
```

2. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
`'Red'`, `'Lime'`, `'Blue'`

```{code-cell} ipython3
# votre code
print(colors['Red'])
print(colors['Lime'])
print(colors['Blue'])
```

3. Faites une fonction `patchwork` qui  

   * prend une liste de couleurs et la structure donnant le code des couleurs RGB
   * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
   * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
   si besoin de compléter l'image mettez du blanc

+++

````{admonition} indices
:class: dropdown
  
* sont potentiellement utiles pour cet exo:
  * la fonction `np.indices()`
  * [l'indexation d'un tableau par un tableau](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
* aussi, ça peut être habile de couper le problème en deux, et de commencer par écrire une fonction `rectangle_size(n)` qui vous donne la taille du patchwork en fonction du nombre de couleurs  
  ```{admonition} et pour calculer la taille au plus juste
  :class: tip dropdown

  en version un peu brute, on pourrait utiliser juste la racine carrée;
  par exemple avec 5 couleurs créer un carré 3x3 - mais 3x2 c'est quand même mieux !

  voici pour vous aider à calculer le rectangle qui contient n couleurs

  n | rect | n | rect | n | rect | n | rect |
  -|-|-|-|-|-|-|-|
  1 | 1x1 | 5 | 2x3 | 9 | 3x3 | 14 | 4x4 |
  2 | 1x2 | 6 | 2x3 | 10 | 3x4 | 15 | 4x4 |
  3 | 2x2 | 7 | 3x3 | 11 | 3x4 | 16 | 4x4 |
  4 | 2x2 | 8 | 3x3 | 12 | 3x4 | 17 | 4x5 |
  ```
````

```{code-cell} ipython3
# votre code
def rectangle_size(n):
    c = int(np.sqrt(n))
    if c**2 == n:
        l = c
        L = c
    elif c*(c+1) >= n:
        L = c
        l = c+1
    else:
        L = c+1
        l = c+1
    return (L, l)
    
def patchwork(Ls, dico):
    L = rectangle_size(len(Ls))[0]
    l = rectangle_size(len(Ls))[1]
    RGB = []
    for elem in Ls:
        RGB.append(dico[elem])
    for k in range (L*l - len(Ls)):
        RGB.append([255,255,255])
    RGB = np.array(RGB)
    pattern = np.zeros((L, l), dtype='uint8')
    for i in range(L):
        for j in range(l):
            pattern[i,j] = l*i+j
    sortie = np.astype(RGB[pattern], 'uint8')
    return sortie
```

4. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.

```{code-cell} ipython3
# votre code
Ls = np.random.choice(list(colors.keys()), 20)
print(Ls)
patchwork(Ls, colors)
plt.imshow(patchwork(Ls, colors))
```

```{code-cell} ipython3
# 5. Sélectionnez toutes les couleurs à base de blanx et affichez leur patchwork
# même chose pour les jaunes
```

```{code-cell} ipython3
# votre code
couleurs_blanches = []
for cle in colors.keys():
    if 'White' in cle:
        couleurs_blanches.append(cle)
couleurs_blanches = np.array(couleurs_blanches)
print(couleurs_blanches)
couleurs_jaunes = []
for cle in colors.keys():
    if 'Yellow' in cle:
        couleurs_jaunes.append(cle)
couleurs_jaunes = np.array(couleurs_jaunes)
print(couleurs_jaunes)
plt.imshow(patchwork(couleurs_blanches, colors))
plt.show()
plt.imshow(patchwork(couleurs_jaunes, colors))
```

6. Appliquez la fonction à toutes les couleurs du fichier  
et sauver ce patchwork dans le fichier `patchwork.png` avec `plt.imsave`

```{code-cell} ipython3
# votre code
patchwork(colors.keys(), colors)
plt.imsave('patchwork.png', patchwork(colors.keys(), colors))
```

7. Relisez et affichez votre fichier  
   attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels

vous devriez obtenir quelque chose comme ceci

```{image} media/patchwork-all.jpg
:align: center
```

```{code-cell} ipython3
# votre code
a = plt.imread('patchwork.png')
plt.imshow(a)
```

## Somme dans une image & overflow

+++

0. Lisez l'image `data/les-mines.jpg`

```{code-cell} ipython3
# votre code
im = plt.imread('data/les-mines.jpg')
im.flags.writeable
copim = np.copy(plt.imread('data/les-mines.jpg'))
plt.imshow(copim)
print(copim)
```

1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
# votre code
tab0 = np.array(copim[:,:,0], dtype=np.uint16)
tab1 = np.array(copim[:,:,1], dtype=np.uint16)
tab2 = np.array(copim[:,:,2], dtype=np.uint16)
tab = tab0 + tab1 + tab2
print(tab)
```

2. Regardez le type de cette image-somme, et son maximum; que remarquez-vous?  
   Affichez cette image-somme; comme elle ne contient qu'un canal il est habile de l'afficher en "niveaux de gris" (normalement le résultat n'est pas terrible ...)


   ```{admonition} niveaux de gris ?
   :class: dropdown tip

   cherchez sur google `pyplot imshow cmap gray`
   ```

```{code-cell} ipython3
# votre code
print(np.max(tab), tab.shape, tab.dtype)
plt.imshow(tab, cmap='gray')
# Si on ne modifie pas le type d'éléments des tableaux en uint16 au lieu de uint8 alors le nombre maximal atteint est 255 (cela 
# reboucle sur zéro si on dépasse 255). Ainsi, sans ce changement de type d'éléments on n'a pas l'image souhaitée.
```

3. Créez un nouveau tableau `numpy.ndarray` en sommant mais cette fois **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
# votre code
tabS = copim.sum(axis=2)
print(tabS)
```

4. Comme dans le 2., regardez son maximum et son type, et affichez la

```{code-cell} ipython3
# votre code
print(np.max(tabS), tabS.shape, tabS.dtype)
plt.imshow(tabS, cmap='gray')
```

5. Les deux images sont de qualité très différente, pourquoi cette différence ? Utilisez le help `np.sum?`

```{code-cell} ipython3
# votre code / explication
help(np.sum)
# Explications : 
# La différence de qualité entre les deux images s'explique par l'absence du bon niveau de gris avec la méthode + si on oublie de
# changer le type des éléments car la valeur de la somme des trois composantes ne peut dans ce cas pas axcéder 255. Avec la méhode
# sum, choisit un dtype adaptée pour faire la somme le type uint64 ce qui permet de contenir tous le nombres entre 0 et 3 fois 255,
# Ainsi la qualité de l'image qui en résulte est meilleur.
```

6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
(de la manière que vous préférez)

```{code-cell} ipython3
# votre code
newtab = np.array(tabS, dtype=np.uint8)
plt.imshow(newtab, cmap='gray')
```

7. Remplacez dans l'image en niveaux de gris,  
les valeurs >= à 127 par 255 et celles inférieures par 0  
Affichez l'image avec une carte des couleurs des niveaux de gris  
vous pouvez utilisez la fonction `numpy.where`

```{code-cell} ipython3
# votre code
tabf = np.where(newtab>=127, 255, 0)
print(tabf)
plt.imshow(tabf, cmap='gray')
```

8. avec la fonction `numpy.unique`  
regardez les valeurs différentes que vous avez dans votre image en noir et blanc

```{code-cell} ipython3
# votre code
np.unique(tabf)
```

## Image en sépia

+++

Pour passer en sépia les valeurs R, G et B d'un pixel  
(encodées ici sur un entier non-signé 8 bits)  

1. on transforme les valeurs `R`, `G` et `B` par la transformation  
`0.393 * R + 0.769 * G + 0.189 * B`  
`0.349 * R + 0.686 * G + 0.168 * B`  
`0.272 * R + 0.534 * G + 0.131 * B`  
(attention les calculs doivent se faire en flottants pas en uint8  
pour ne pas avoir, par exemple, 256 devenant 0)  
1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
1. naturellement l'image doit être ensuite remise dans un format correct  
(uint8 ou float entre 0 et 1)

+++

````{tip}
jetez un coup d'oeil à la fonction `np.dot` 
qui est si on veut une généralisation du produit matriciel

dont voici un exemple d'utilisation:
````

```{code-cell} ipython3
# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

print(f"en partant des dimensions {A.shape} et {B.shape}")
print(f"on obtient un résultat de dimension {C.shape}")
print(f"et le nombre de termes dans chaque `sum()` est {A.shape[-1]} == {B.shape[-2]}")
```

**Exercice**

+++

1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
la fonction `numpy.dot` peut être utilisée si besoin, voir l'exemple ci-dessus

```{code-cell} ipython3
# votre code
def sepia(im):
    m = np.array([
        0.393, 0.769, 0.189,
        0.349, 0.686, 0.168,
        0.272, 0.534, 0.131 ]).reshape((3,3)).T
    @np.vectorize
    def f(x):
        if x>255 : return 255
        return int(x)
    return f(np.dot(im, m))
```

2. Passez votre patchwork de couleurs en sépia  
Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso

```{code-cell} ipython3
# votre code
im = np.copy(plt.imread('patchwork-all.jpg'))
sep = sepia(im)
plt.imshow(sep)
```

3. Passez l'image `data/les-mines.jpg` en sépia

```{code-cell} ipython3
# votre code
im = np.copy(plt.imread('data/les-mines.jpg'))
im_sepia = sepia(im)
plt.imshow(im_sepia)
```

## Exemple de qualité de compression

+++

1. Importez la librairie `Image`de `PIL` (pillow)  
(vous devez peut être installer PIL dans votre environnement)

```{code-cell} ipython3
# votre code
from PIL import Image
```

2. Quelle est la taille du fichier `data/les-mines.jpg` sur disque ?

```{code-cell} ipython3
file = "data/les-mines.jpg"
```

```{code-cell} ipython3
# votre code
im = plt.imread(file)
print(f"le fichier a une taille de {im.nbytes} octets")
```

3. Lisez le fichier 'data/les-mines.jpg' avec `Image.open` et avec `plt.imread`

```{code-cell} ipython3
# votre code
im1 = plt.imread('data/les-mines.jpg')
im2 = Image.open('data/les-mines.jpg')
```

4. Vérifiez que les valeurs contenues dans les deux objets sont proches

```{code-cell} ipython3
# votre code
plt.imshow(im1)
plt.show()
plt.imshow(im2)
```

5. Sauvez (toujours avec de nouveaux noms de fichiers)  
l'image lue par `imread` avec `plt.imsave`  
l'image lue par `Image.open` avec `save` et une `quality=100`  
(`save` s'applique à l'objet créé par `Image.open`)

```{code-cell} ipython3
# votre code
plt.imsave('im1.jpg', im1)
np.save('im2.npy', im2)
```

6. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
Que constatez-vous ?

```{code-cell} ipython3
# votre code
print(f"le fichier im1 a une taille de 134 kilo-octets")
print(f"le fichier im2 a une taille de 1250 kilo-octets")
#On constate que le fichier im1 a la même taille que le fichier d'origine et que le fichier im2 a une taille plus grande que 
#le fichier d'origine.Cela peut s'expliquer par la différence des extensions des fichiers (jpg, npy)
```

7. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence

```{code-cell} ipython3
# votre code
plt.imshow(im1)
plt.show()
plt.imshow(im2)
```
