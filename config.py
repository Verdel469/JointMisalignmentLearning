import os
import json
from pathlib import Path

# TODO: pouvoir executer config pour pouvoir faire un config.manip = et rerun le fichier
# utile pour les notebook ou pour run les code sur plusieurs manip en meme temps

# working directory 
dirname = os.path.dirname(__file__)
# Manip to treat, it must be correspond to a manip folder name 
manip = "Transp_2DDL"
# manip = "manip_main"
print("****\nWorking on " + manip + " data\n****")
manip_path = os.path.join(dirname, manip)


# Seuil de coupure utilisé pour les profil de vitesse des données cinématiques
cutoff_velocity_treshold = 5 # rad/s # Ne marche pas pour compute_index
cutoff_velocity_treshold_is_rel = True
plot_velocity_cutting = False

take_overshoot_into_account = False

# delai electromecanique (utilisé pour les bornes des emgs)
electromechanical_delay = 0.1 # 100 ms

# chemin relatif vers les données organisées
data_path = os.path.join(manip_path, "data")
# chemin relatif vers les mouvements découpés
movements_path = './Transp_2DDL/data/movements'
# chemnin relatif vers les données brutes
raw_data_path = os.path.join(manip_path, "raw_data")
# Label des sujets - correspond au nom de dossier des sujets

# Nom des sujets. Utile pour organiser les données Vicon notamment.

with open(os.path.join(manip_path, 'config.json')) as config_file:
    spec = json.load(config_file)
# after import config, access to manip config specifities with config.spec.get("name_of_param")
# see config.json in manip folder
subjects_label = ["S" + str(i) for i in range(0, spec.get("subject_number"))] # S1...
for subj_lab in subjects_label:
    # Create subject folder in the data folder
    Path(os.path.join(data_path,subj_lab)).mkdir(parents=True, exist_ok=True)
