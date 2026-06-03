import shutil
import os
import pickle


# src_root = './ComputedData/'
# dst_root = './data_CALIB/'

# subjList = ['S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17']
# src_conds = ['Calib1', 'Calib2']

# filesToCopyNames = ['exoPositions.csv', 'exoVelocities.csv', 'wristSensor.csv']
# filesToRemoveNames = ['emgFilt.csv', 'emgRaw.csv', 'muscleTorque.csv']

# for subject in subjList:
#     src_subj = src_root + subject + '/'
#     dst_subj = dst_root + subject + '/'

#     for cond in src_conds:
#         src_cond = src_subj + cond + '/'
#         if cond == 'Calib1':
#             dst_cond = dst_subj + 'SJ/'
#         else:
#             dst_cond = dst_subj + 'MJ/'

#         if subject != 'S01' or cond != 'Calib2':
#             list_src_trials = [d for d in os.listdir(src_cond) if os.path.isdir(os.path.join(src_cond, d))]
#             count = len(list_src_trials)
            
#             for i in range(0,10):
#                 src_trial_name = 'T' + str(count - 10 + i)
#                 src_trial = src_cond + src_trial_name + '/'
#                 dst_trial = dst_cond + 'T' + str(i) + '/'

#                 for fileToCopy in filesToCopyNames:
#                     src_file = src_trial + fileToCopy
#                     shutil.copy2(src_file, dst_trial)
                
#                 for fileToRemove in filesToRemoveNames:
#                     rm_fileName = dst_trial + fileToRemove
#                     os.remove(rm_fileName)

src_root = './SegmentedData/'
# dst_root = './data_CALIB/'

subjList = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S09', 'S10', 'S11', 'S12', 'S14']
src_conds = ['EG', 'ES', 'T']
src_phases = ['PE', 'PS', 'RE', 'RS']

nb_present = 0
nb_missing = 0
list_missing = []

for subject in subjList:
    src_subj = src_root + subject + '/Assist2/'

    for cond in src_conds:
        src_cond = src_subj + cond + '/'
        list_src_trials = [d for d in os.listdir(src_cond) if os.path.isdir(os.path.join(src_cond, d))]

        for trial in list_src_trials:
            src_trial = src_cond + trial + '/'

            for phase in src_phases:
                src_phase = src_trial + phase + '/'
                pathExoPos = src_phase + 'exoVelocities.csv'

                if os.path.isfile(pathExoPos):
                    nb_present += 1
                else:
                    nb_missing += 1
                    miss_file = subject + '_' + cond + '_' + trial + '_' + 'phase'
                    list_missing.append(miss_file)

percent_pres = nb_present / (nb_present + nb_missing) * 100
percent_miss = nb_missing / (nb_present + nb_missing) * 100
print('Number of present position files: ' + str(nb_present) + ' (' + str(percent_pres) + '%)')
print('Number of missing position files: ' + str(nb_missing) + ' (' + str(percent_miss) + '%)')

# with open('missingPosFiles.pkl', 'wb') as file:
#     pickle.dump(list_missing, file)