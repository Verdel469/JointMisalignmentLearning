"""This script is made to save the participants individual data."""

# author : Simon Bastide
# mail : simon.bastide@outlook.com

import os
import numpy as np
import pandas as pd
import config
import sys
import clean_import
import motor_control_tools as mct
from pathlib import Path

# Creating dictionary
participants_dict = {'subject' :['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'],
                     'Age'     :[24  , 25  , 27  , 25  , 23  , 25  , 27  , 18  , 21  , 23  ],
                     'Poids'   :[70  , 60  , 90  , 73  , 74  , 67  , 72  , 60  , 70  , 63  ],
                     'Taille ' :[169 , 173 , 178 , 180 , 176 , 177 , 180 , 172 , 177 , 176 ],
                     'L_bras'  :[30.5, 29.5, 33  , 34  , 32.5, 32  , 31  , 29  , 30  , 32  ],
                     'L_avbras':[24.5, 26.5, 27.5, 30  , 26.5, 26.5, 26.5, 24  , 24.5, 25  ],
                     'L_main'  :[20  , 22  , 21.5, 22  , 20  , 19.5, 21  , 20  , 20.5, 20  ],
                     'Sexe'    :['M' , 'M' , 'M' , 'M' , 'M' , 'F' , 'M' , 'M' , 'M' , 'F' ],
                     'm_coeff' :[1.75, 2.14, 2.37, 2.08, 2.01, 1.59, 1.93, 1.94, 2.16, 1.59],
                     'phi_h'   :[0.35, 0.35, 0.43, 0.27, 0.39, 0.31, 0.30, 0.35, 0.37, 0.26]}
# Transforming into dataframe
participants_data = pd.DataFrame(data = participants_dict)
# Saving into ".csv"
participants_data.to_csv("./all_params/participants_data.csv", index = False)


            
