import json
import glob

def find_last_checkpoint(path):
    checkpoint = sorted([int(x.split('/')[-1].split('-')[-1]) for x in glob.glob(path+'/checkpoint*')])
    return str(checkpoint[-1])  

def find_top_checkpoints(filename, top=10):
    with open(filename, 'r')as f:
        data = json.loads(f.read())
    result = []
    log = data['log_history']
    for Epoch in log:
        if 'eval_f1' in Epoch:
            result.append([Epoch['step'], Epoch['epoch'], Epoch['eval_f1']])
    result = sorted(result, key=lambda x:x[2], reverse=True)
    return result[:top]

if __name__ == "__main__":
    model_path = 'model'
    result_path = '../pipeline_step4/output'
    ensemble_list = []
    for M in glob.glob(model_path+'/[rR]o*a*'):
        M_name = M.split('/')[-1]
        if 'LS' not in M_name:  # only use label smoothing models
            continue    
        if 'RS' in M_name:
            ensemble_list.append(result_path+'/'+M_name)
        else:
            last_checkpoint = find_last_checkpoint(M)
            e = find_top_checkpoints(M+'/checkpoint-'+last_checkpoint+'/trainer_state.json')
            for checkpoint in e:
                ensemble_list.append(result_path+'/'+M_name+'/checkpoint-'+str(checkpoint[0]))

    with open('ensemble_list_runTest.dat','w') as f:
        f.writelines('\n'.join(ensemble_list)+'\n')

