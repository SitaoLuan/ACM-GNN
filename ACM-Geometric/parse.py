from models import GCN

def parse_method(args, n, c, d, device):
    model = GCN(nfeat = d, 
                nhid = args.hidden_channels, 
                nclass = c, 
                nlayers = args.num_layers, 
                nnodes = n, 
                dropout=args.dropout,  
                model_type = args.method, 
                structure_info = args.structure_info,
                variant = args.variant).to(device)    
    return model

# training parameters
def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str,
                        help='Dataset name.', default = 'snap-patents')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--method', type=str, 
                        help='name of model (gcn, sgc, graphsage, snowball, gcnII, acmgcn, acmgcnp, acmgcnpp, acmsgc, acmgraphsage, acmsnowball, mlp)', 
                        default = 'acmgcnp')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default= 1e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--hops', type=int, default=1,
                        help='Number of hops we use, k= 1,2')                        
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers, i.e. network depth')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use adam instead of adamW')
    parser.add_argument('--rand_split', action='store_true', default=False,
                        help='use random splits')    
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')  
    parser.add_argument('--rocauc', action='store_true', default=False,
                        help='set the eval function to rocauc')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')                      
    ## 
    parser.add_argument('--param_tunning', action='store_true', default=True,
                    help='Parameter fine-tunning mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--num_splits', type=int, help='number of training/val/test splits ', default = 1)
    # parser.add_argument('--early_stopping', type=float, default=200, help='early stopping used in GPRGNN')
    # parser.add_argument('--fixed_splits', type=float, default=0, help='0 for random splits in GPRGNN, 1 for fixed splits in GeomGCN')
    parser.add_argument('--variant', type=int, default=1, help='Indicate ACM, GCNII variant models.')
    parser.add_argument('--structure_info', type=int, default=0, help='1 for using structure information in acmgcn+, 0 for not')
