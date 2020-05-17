Pour lancer l'automate cellulaire, lancer dans python "automate_2D_commande.py"
Pour obtenir de l'aide, taper "automate_2D_commande.py --help" ou "automate_2D_commande.py -h"
Sans arguments optionnels sera lancé par défaut l'automate cellulaire en V1 avec comme structure le sablier.
Les arguments optionnels sont les suivants:
* "--version" ou "-v" permet de choisir la version utilisée de l'automate cellulaire, avec au choix :
	*"1" permet de choisir la première version
	*"2" permet de choisir la deuxième version
* "--structure" ou "-s" permet de choisir la structure utilisée dans l'automate cellulaire, avec au choix:
	*"flow" ou "f" permet de choisir un écoulement simple de sable
	*"avalanche" ou "a" permet de choisir un tas de sable avec un grain tombant (de préférence utiliser la V2)
	*"hourglass" ou "h" permet de choisir un sablier dans lequel tombe du sable
	*"galton" ou "g" permet de choisir une planche de Galton (de préférence utiliser la V1)