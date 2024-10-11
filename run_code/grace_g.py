from utils import *
from wei_utils import *
# from wei_script import prepare
import GCL.augmentors as A
import itertools
import random
import os
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune')
    parser.add_argument('--num_layers', type=int, default = 2, help = 'number of layers')
    parser.add_argument('--did', type=int, default = 0, help='Learning rate for the Finetune model')
    parser.add_argument('--num_try', type=int, default = 20, help='number of random trials')
    parser.add_argument('--batch', type=int, default = 5, help="Number of independent run of the training")
    parser.add_argument('--approx_rank', type=int, default = 0, help='The rank for the low-rank approximation.')
    parser.add_argument('--aug_type', type=str, default='EdgeAdding_Cross')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--random_dataset', action='store_true')
    parser.add_argument('--same_setup', action='store_true')
    parser.add_argument('--eps1', type=float, default=0)
    parser.add_argument('--eps2', type=float, default=0)
    parser.add_argument('--reweight1', type=float, default=1)
    parser.add_argument('--reweight2', type=float, default=1)
    parser.add_argument('--random_reweight', action='store_true')
    parser.add_argument('--linear_reweight', action='store_true')
    parser.add_argument('--low_eps_bound', type=float, default=0.5)
    parser.add_argument('--high_eps_bound', type=float, default=2.0)
    parser.add_argument('--save_graph', action='store_true')
    parser.add_argument('--perturb_ratio', type=float, default=-1)
    parser.add_argument('--min_perturb_l2', type=float, default=0.5)
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--stage', type=str, default='all')#test



    #
    args = parser.parse_args()
    args.method = 'grace_g'

    ## tunable
    para_range = np.linspace(0.1, 1, num = 10)

    did = args.did
    num_layers = args.num_layers

    edge_adding_upper = 1


    dataset_names = ['MUTAG',
                     'PROTEINS',
                     'IMDB-BINARY',
                     'IMDB-MULTI',
                     'NCI1',
                     'DD']

    # Configs
    Anorm = {
        'norm_A': False,
        'improved' : False, 
        'add_self_loops' : False, 
        'flow' : "source_to_target",
        'sample_on_ori' : True
    }

    Lnorm = {'norm_L': 'sym'}

    default_settings = {
        'dim': 512,
        'lr': 0.001,
        'epoch': args.epochs,
        'batch': args.batch,
        'graph_save_dir': get_graph_store_path(args) if args.save_graph else None,
        'encoder':args.encoder
    }

    if args.save_graph:
        os.makedirs(default_settings['graph_save_dir'], exist_ok=True)

    save_path = get_store_path(args)

    datasets = {}
    for dataset_name in dataset_names:
        data_info = {
            'path': 'datasets',
            'name': dataset_name,
        }
        datasets[dataset_name] = load_dataset(data_info)

    if args.approx_rank > 0:
        Lnorm['rank'] = args.approx_rank
    else:
        Lnorm['rank'] = -1

    reweight1 = args.reweight1
    reweight2 = args.reweight2

    if args.perturb_ratio < 0:
        for nt in range(args.num_try):
            set_seed(nt+5) #nt+5
            dataset_name = dataset_names[did]
            if args.random_dataset:
                dataset_name = random.choice(dataset_names)

            if args.aug_type == 'EdgeRemoving':
                eps1 = random.uniform(0, 1)
                eps2 = random.uniform(0, 1)
                if args.same_setup:
                    eps2 = eps1
                # aug1 = A.Identity()
                aug1 = A.EdgeRemoving(pe=eps1)
                aug2 = A.EdgeRemoving(
                    pe=eps2
                )
            elif args.aug_type == 'EdgeAdding':
                eps1 = random.uniform(0.1, edge_adding_upper)
                eps2 = random.uniform(0.1, edge_adding_upper)
                if args.same_setup:
                    eps2 = eps1
                aug1=Edge_Adding_pergraph(pe=eps1)
                aug2=Edge_Adding_pergraph(pe=eps2)
            elif args.aug_type == 'GIP':
                eps1 = random.uniform(0, edge_adding_upper)
                eps2 = random.uniform(0, edge_adding_upper)
                if args.same_setup:
                    eps2 = eps1
                aug1 = GIP(
                    pe=eps1
                )                
                aug2 = GIP(
                    pe=eps2
                )
            elif args.aug_type == 'FeatureMasking':
                eps1 = random.uniform(0, edge_adding_upper)
                eps2 = random.uniform(0, edge_adding_upper)
                if args.same_setup:
                    eps2 = eps1
                aug1 = A.FeatureMasking(pf=eps1)
                aug2 = A.FeatureMasking(pf=eps2)
            elif args.aug_type == 'RWS':
                eps1 = int(random.uniform(0, 1)*20)
                eps2 = int(random.uniform(0, 1)*20)
                if args.same_setup:
                    eps2 = eps1
                aug1 = A.RWSampling(num_seeds=1000, walk_length=eps1)
                aug2 = A.RWSampling(num_seeds=999, walk_length=eps1)

            key = str((dataset_name, eps1, eps2, num_layers, reweight2))
            result_dict = {}
            result_dict['key'] = key
            if args.save_graph:
                default_settings['graph_save_dir'] = os.path.join(get_graph_store_path(args), key)
                os.makedirs(default_settings['graph_save_dir'], exist_ok=True)
            try:
                for seed in range(default_settings['batch']):
                    set_seed(seed)
                    original_save_dir = default_settings['graph_save_dir']
                    if default_settings['graph_save_dir'] is not None:
                        default_settings['graph_save_dir'] = os.path.join(default_settings['graph_save_dir'], f'seed_index_{seed}')
                        os.makedirs(default_settings['graph_save_dir'], exist_ok=True)
                    test_result = accuracy_eval_graceg(
                        datasets[dataset_name], 
                        aug1, 
                        aug2, 
                        num_layers,
                        'cuda', 
                        dim=default_settings['dim'], 
                        lr=default_settings['lr'], 
                        epoch=default_settings['epoch'],
                        batch_size=128,
                        graph_save_dir=default_settings['graph_save_dir'],
                        encoder=default_settings['encoder'],
                        stage=args.stage
                    )
                    default_settings['graph_save_dir'] = original_save_dir
                    result_dict[seed] = test_result
                print(result_dict)

                with open(os.path.join(
                    save_path
                ), 'a') as file:
                    json_string = json.dumps(result_dict)
                    file.write(json_string + '\n')
            except Exception as e:
                if args.aug_type == 'EdgeAdding':
                    print(f"Error: {e}")
                    edge_adding_upper = max(eps1, eps2)
                    print(f"New edge_adding_upper: {edge_adding_upper}")
                    nt -= 1
                else:
                    raise e
    else:
        ckpt_path = get_graph_store_path(args)
        keys = os.listdir(ckpt_path)
        keys = [k for k in keys if len(os.listdir(osp.join(ckpt_path, k))) == default_settings['batch'] and eval(k)[0] == dataset_names[args.did]]
        for key in keys:
            dataset_name = eval(key)[0]
            key_path = osp.join(ckpt_path, key)
            trials = os.listdir(key_path)
            augs1 = []
            augs2 = []
            for trial in trials:
                trial_path = osp.join(key_path, trial)
                aug1_graphs, aug2_graphs = load_graphs(trial_path)
                from_graph_aug1 = FromGraphsAugmentorNormal(aug1_graphs, perturb_ratio=args.perturb_ratio, min_perturb_l2=args.min_perturb_l2)
                from_graph_aug2 = FromGraphsAugmentorNormal(aug2_graphs, perturb_ratio=args.perturb_ratio, min_perturb_l2=args.min_perturb_l2)
                augs1.append(from_graph_aug1)
                augs2.append(from_graph_aug2)

            result_dict = {}
            result_dict['key'] = key
            for seed in range(default_settings['batch']):
                set_seed(seed)
                original_save_dir = default_settings['graph_save_dir']
                if default_settings['graph_save_dir'] is not None:
                    default_settings['graph_save_dir'] = os.path.join(default_settings['graph_save_dir'], f'seed_index_{seed}')
                    os.makedirs(default_settings['graph_save_dir'], exist_ok=True)
                test_result = accuracy_eval_graceg(
                    datasets[dataset_name], 
                    augs1[seed], 
                    augs2[seed], 
                    num_layers,
                    'cuda', 
                    dim=default_settings['dim'], 
                    lr=default_settings['lr'], 
                    epoch=default_settings['epoch'],
                    batch_size=128,
                    graph_save_dir=default_settings['graph_save_dir'],
                    encoder=default_settings['encoder'],
                )
                default_settings['graph_save_dir'] = original_save_dir
                result_dict[seed] = test_result
                result_dict[seed]['aug1_avg_l2'] = sum(augs1[seed].l2s) / len(augs1[seed].l2s)
                result_dict[seed]['aug2_avg_l2'] = sum(augs2[seed].l2s) / len(augs2[seed].l2s)
            print(result_dict)

            with open(save_path, 'a') as file:
                json_string = json.dumps(result_dict)
                file.write(json_string + '\n')
        
    
