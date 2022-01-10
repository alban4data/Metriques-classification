# Les principales métriques en classification

Cette article a pour objectif de décrire les principales métriques utilisées dans le cadre d'un projet de data science et plus particulièrement lors d'une classification. 

Lors de votre projet, un modèle de machine learning va être entraîné sur votre base d'apprentissage et vous avez besoin d'analyser les résultats. Plusieurs métriques permettent d'évaluer un modèle de machine learning. Avant de rentrer dans le détail de ces métriques, vous avez besoin de vous rappeler de la définition de la matrice de confusion.

<p align="center">
  <img src="/images/matrice confusion.png" />
</p>

**1- L'accuracy, le rappel et la précision**

**1-1 L'accuracy**

L'accuracy est une métrique très souvent utilisé pour évaluer un modèle de machine learning dans le cadre d'un problème de classification binaire ou multi-classe.

```markdown
Accuracy = (TP + TN) / (TP + FP + FN + TN)
```

C'est une première métrique intéressante lorsque les modalités de la variable cible sont équilibrées. En effet, si vous avez 90% de 0 dans votre cible, vous aurez 90% d'accuracy. Cela ne signifie pas que votre modèle a de bonnes performances. 

Par exemple, si nous prenons 100 individus à classer à un seuil de probabilité égal à.5 :

```markdown
TN = 88, FP = 2, FN = 8, TP = 2
```

Vous voyez bien que votre accuracy est de 90 %et pourtant le modèle ne parvient pas à bien prédire les positifs.

**1-2 La précision**

Cet indicateur permet de savoir le nombre de bien classés parmi les individus prédits positifs.

```markdown
Précision = (TP) / (TP + FP)
```

Dans certains cas, il est intéressant d'avoir une bonne précision. En effet, vous ne souhaitez pas que certains individus soient contactés à tort dans le cadre de campagnes marketing. De plus, cela coûte de l'argent à l'entreprise de se tromper.

Dans d'autres cas, il est également préférable de prédire davantage de cas positifs. En effet, dans le cas de la détection de cancer, vous ne souhaitez pas passer à côté de cas positifs. Il vaut prédire davantage d'individus au risque de se tromper en demandant aux faux positifs des examens complémentaires.

**1-3 Le rappel**

Cet indicateur permet de savoir le nombre de bien classés parmi les individus réellement positifs.

```markdown
Rappel = (TP) / (TP + FN)
```

Dans le cas où vous souhaitez détecter si une personne a un cancer, vous voulez détecter la maladie même si vous n'êtes pas sûr. Le rappel vaut 1 si toute votre population est prédite positive.

Ainsi, vous voyez l'intérêt de trouver un compromis entre la précision et le rappel.

**2- Le F1 score**

Lorsque vous souhaitez un modèle avec à la fois un bon rappel et une bonne précision, il est recommandé d'utiliser le F1 Score. Cette métrique permet d'avoir un équilibre entre la précision et le rappel.

Si votre rappel ou précision est faible, votre F1 score sera faible.

Exemple d'utilisation du F1 score : Si vous êtes un policier et que vous devez attraper des criminels, vous voulez sûr que l'individu est un criminel (Précision) et vous voulez attraper le plus de criminels possible (Rappel). Le F1 score permet de trouver un équilibre entre ces deux métriques.

Un des inconvénients du F1 score est qu'il accorde le même poids au rappel et à la précision. Parfois, vous souhaiterez peut-être avoir davantage de rappel ou de précision.

Pour palier à cet inconvénient,vous pouvez créer un F1 score pondéré de la manière ci-dessous :
<p align="center">
  <img src="/images/f1_pondere.PNG" />
</p>

Avec Beta compris entre 0 et 1. Ici, vous accorder Beta fois plus d'importance au rappel qu'à la précision.

Ainsi, le F1 score s'écrit de la manière suivante : 
<p align="center">
  <img src="/images/f1_score_calcul.PNG" />
</p>

Le F1 score est toujours simplement une moyenne harmonique de la précision et du rappel. On peut donc se demander le sens de cette moyenne harmonique. Pour expliquer, considérez par exemple, quelle est la moyenne de 30 mph et 40 mph est ? si vous conduisez pendant 1 heure à chaque vitesse, la vitesse moyenne sur les 2 heures est en effet la moyenne arithmétique, 35 mph. Cependant, si vous conduisez sur la même distance à chaque vitesse - disons 10 miles - alors la vitesse moyenne sur 20 miles est la moyenne harmonique de 30 et 40, environ 34,3 mph. La raison en est que pour que la moyenne soit valide, vous avez vraiment besoin que les valeurs soient dans les mêmes unités mises à l'échelle. Les miles par heure doivent être comparés sur le même nombre d'heures; pour comparer sur le même nombre de miles, vous devez plutôt faire la moyenne des heures par mile, ce qui est exactement ce que fait la moyenne harmonique.

La moyenne harmonique de N valeurs est le nombre dont l'inverse est la moyenne arithmétique des inverses des dites valeurs. C'est donc l'inverse de la moyenne arithmétique de l'inverse des termes. La moyenne harmonique permet de calculer des moyennes sur des fractions si le dénominateurs change.

Il faut également retenir que lorsque F1 = 0.5, vous obtiendrez 
```markdown
2TP = FN +FP
```
Ainsi, lorsque deux individus sont bien classés, vous aurez deux individus mal classés.

**3- L'AUC (Area Under the Roc)**

L'AUC correspond à l'aire sous la courbe ROC.

Qu'est-ce que la courbe ROC ?

A ce moment, vous avez obtenu les probabilités issues de votre modèle de machine learning.
Vous pouvez représenter graphiquement à différents seuils de probabilité (en faisant varier le seuil de 0 à 1) la sensitivité (TPR) et (1-Sensitivité) (FPR) et vous obtiendrez la courbe ROC.

La spécificité correspond à la formule suivante :
```markdown
Sensitivité = TPR = Rappel = TP / (TP + FN)
```

La spécificité correspond à la formule suivante :
```markdown
1 - Spécificité = FPR = FP / (TN + FP)
```

<p align="center">
  <img src="/images/roc_curve.PNG" />
</p>

L'AUC est une métrique invariante contrairement au F1 score. En effet, l'AUC ne dépend pas du seuil de probabilité contrairement au F1 score.  
L'AUC se base seulement sur des ratios par rapport aux valeurs réelles. Dans le monde réel, le F1 score est davantage utilisé car les échantillons positifs et négatifs peuvent être très inégaux. Il faut donc se poser la question de l'utilisation que l'on souhaite faire de la métrique. Chaque métrique a un usage différent. 
Il est préférable d'utiliser le F1 score dans le cadre de données déséquilibrées. En effet, si vous avez 98% de négatifs dans votre population, votre taux de faux de faux positifs sera très faibles. Cependant, cela ne signifiera pas que votre modèle est un bon classifieur.

**3- Le MCC (Matthews Correlation Coefficient)**

Le MCC a des valeurs comprises entre -1 et +1 où -1 indique un très mauvais classifieur et +1 un parfait classifieur. Le MCC permet de quantifier les performances du modèle. Le MCC prend en compte les vrais et faux positifs et négatifs. On peut utiliser cette métrique même si les données sont déséquilibrées.

La formule s'écrit :
<p align="center">
  <img src="/images/mcc.PNG" />
</p>

Un des avantages du MCC est qu'il prend en compte dans le calcul toutes les cellules de la matrice de confusion au numérateur.

