from user import *
from config import *
from utils import *
import os


def abp_ensemble(Influence1, Preference1, Influence2, Preference2):
    
    for label in Influence2.keys():
        for source_word in Influence2[label].keys():
            if source_word in Influence1[label].keys():
                for target_word in Influence2[label][source_word].keys():
                    if target_word in Influence1[label][source_word].keys():
                        Influence1[label][source_word][target_word] +=  Influence2[label][source_word][target_word]
                    else:
                        Influence1[label][source_word][target_word] =  Influence2[label][source_word][target_word]
            else:
                Influence1[label][source_word] = Influence2[label][source_word]
    
    for label in Preference2.keys():
        for source_word in Preference2[label].keys():
            if source_word in Preference1[label].keys():
                Preference1[label][source_word] += Preference2[label][source_word]
            else:
                Preference1[label][source_word] = Preference2[label][source_word]
    
    return Influence1, Preference1
    

def main():
    setup_seed(2070)
    dir1 = "bert-mr-pwws"
    dir2 = "albert-mr-pwws"
    dir3 = "lstm-mr-pwws"
    res_dir = "ens-mr-pwws"


    with open('result/abp/{}/Influence.json'.format(dir1), 'r', encoding='utf-8') as f:
        Influence1 = json.load(f)
    with open('result/abp/{}/Preference.json'.format(dir1), 'r', encoding='utf-8') as f:
        Preference1 = json.load(f)
    

    with open('result/abp/{}/Influence.json'.format(dir2), 'r', encoding='utf-8') as f:
        Influence2 = json.load(f)
    with open('result/abp/{}/Preference.json'.format(dir2), 'r', encoding='utf-8') as f:
        Preference2 = json.load(f)

    with open('result/abp/{}/Influence.json'.format(dir3), 'r', encoding='utf-8') as f:
        Influence3 = json.load(f)
    with open('result/abp/{}/Preference.json'.format(dir3), 'r', encoding='utf-8') as f:
        Preference3 = json.load(f)

    Influence1, Preference1 = abp_ensemble(Influence1, Preference1, Influence2, Preference2)

    Influence1, Preference1 = abp_ensemble(Influence1, Preference1, Influence3, Preference3)


    if not os.path.exists('result/abp/{}'.format(res_dir)):
        os.makedirs('result/abp/{}'.format(res_dir))

    with open('result/abp/{}/Influence.json'.format(res_dir), 'w', encoding='utf-8') as f:
        json.dump(Influence1, f)
    
    with open('result/abp/{}/Preference.json'.format(res_dir), 'w', encoding='utf-8') as f:
        json.dump(Preference1, f)


if __name__ == "__main__":
    main()
