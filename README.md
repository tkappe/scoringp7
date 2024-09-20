# Projet 7 - Parcours Data Scientist OpenClassrooms
C’est projet c’est la mise en place d’un outil de « scoring credit » pour calculer la probabilité de défaillance d’un client.
La société financière « Prêt à dépenser » propose des crédits  à la consommation pour des personnes ayant peu ou pas d’historiques de prêt.

# Implémentation d’un modèle de scoring : API de prédiction
Il s’agit ici de construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d’un client de façon automatique. Mettre en production le modèle de scoring de prédiction à l’aide d’une API.
L’API que nous avons construit prend en entrée l’identifiant d’un client (Client ID), récupère via cet ID l’ensemble des données du client, se sert du modèle construit pour faire la prédiction et renvoie en sortie la probabilité de défaut du client(Probability of default), sa classe (0 ou 1 )ainsi que la décision obtenue sur sa demande de prêt (request accepted ou request refused).

Les données originales sont téléchargeables sur Kaggle ici : https://www.kaggle.com/c/home-credit-default-risk/data

L’API de prédiction est déployé sur le cloud à cette adresse : https://scoringappp7-60437480951c.herokuapp.com/

L'ensemble des packages utilisés est listé dans le fichier requirements.txt